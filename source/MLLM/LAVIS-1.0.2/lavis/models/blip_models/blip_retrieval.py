"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from copy import deepcopy # 导入 deepcopy 函数，用于创建对象的深拷贝

import torch # 导入 PyTorch 库
import torch.nn.functional as F # 导入 PyTorch 的函数式 API，通常用于激活函数、损失函数等
from lavis.common.registry import registry # 从 lavis.common.registry 导入 registry，用于注册模型
from lavis.models.albef_models import compute_sim_matrix # 从 lavis.models.albef_models 导入 compute_sim_matrix 函数
from lavis.models.base_model import ( # 从 lavis.models.base_model 导入基类和辅助函数
    MomentumDistilationMixin, # 动量蒸馏 Mixin 类
    SharedQueueMixin, # 共享队列 Mixin 类
    all_gather_with_grad, # 带梯度的 all_gather 操作
    concat_all_gather, # 连接所有 rank 的张量
)
from lavis.models.blip_models.blip import BlipBase # 从 lavis.models.blip_models.blip 导入 BlipBase 基类
from lavis.models.blip_models.blip_outputs import ( # 从 lavis.models.blip_models.blip_outputs 导入 BLIP 模型的输出类
    BlipOutput, # BLIP 模型的主要输出类
    BlipSimilarity, # BLIP 模型的相似度输出类
    BlipIntermediateOutput, # BLIP 模型的中间输出类
)
from lavis.models.med import XBertEncoder # 从 lavis.models.med 导入 XBertEncoder
from lavis.models.vit import VisionTransformerEncoder # 从 lavis.models.vit 导入 VisionTransformerEncoder
from torch import nn # 导入 PyTorch 的神经网络模块

# 使用 registry.register_model 装饰器注册模型，模型名称为 "blip_retrieval"
@registry.register_model("blip_retrieval")
# 定义 BlipRetrieval 类，继承自 BlipBase, MomentumDistilationMixin, SharedQueueMixin
class BlipRetrieval(BlipBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    BLIP retrieval model. # BLIP 检索模型

    Supported model types: # 支持的模型类型
        - coco: fine-tuned BLIP base model on COCO dataset (Karpathy split). # 在 COCO 数据集 (Karpathy 分割) 上微调的 BLIP base 模型
        - flickr: fine-tuned BLIP base model on Flickr30k dataset. # 在 Flickr30k 数据集上微调的 BLIP base 模型

    Usage: # 用法
        >>> from lavis.models import load_model # 从 lavis.models 导入 load_model
        >>> model = load_model("blip_retrieval", "coco") # 加载 coco 类型的 blip_retrieval 模型
        >>> model = load_model("blip_retrieval", "flickr") # 加载 flickr 类型的 blip_retrieval 模型
    """

    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "coco": "configs/models/blip_retrieval_coco.yaml", # coco 模型的配置文件路径
        "flickr": "configs/models/blip_retrieval_flickr.yaml", # flickr 模型的配置文件路径
    }

    # 类的初始化方法
    def __init__(
        self,
        image_encoder, # 图像编码器
        text_encoder, # 文本编码器
        queue_size, # 队列大小
        alpha=0.4, # alpha 参数，用于控制动量蒸馏的权重
        embed_dim=256, # 嵌入维度
        momentum=0.995, # 动量参数
        negative_all_rank=False, # 是否在所有 rank 中采样负样本
        max_txt_len=35, # 最大文本长度
    ):
        """ """ # 初始化方法的文档字符串
        super().__init__() # 调用父类的初始化方法

        self.tokenizer = self.init_tokenizer() # 初始化分词器

        self.visual_encoder = image_encoder # 设置视觉编码器

        self.text_encoder = text_encoder # 设置文本编码器

        # creating projection layers for ITC # 为 ITC (Image-Text Contrastive) 创建投影层
        text_width = text_encoder.config.hidden_size # 获取文本编码器的隐藏层大小
        vision_width = image_encoder.vision_width # 获取图像编码器的视觉宽度

        self.vision_proj = nn.Linear(vision_width, embed_dim) # 创建视觉投影层
        self.text_proj = nn.Linear(text_width, embed_dim) # 创建文本投影层

        self.itm_head = nn.Linear(text_width, 2) # 创建 ITM (Image-Text Matching) 头，输出维度为 2 (匹配/不匹配)

        # create the momentum encoder # 创建动量编码器
        self.visual_encoder_m = deepcopy(self.visual_encoder) # 创建视觉编码器的动量版本
        self.text_encoder_m = deepcopy(self.text_encoder) # 创建文本编码器的动量版本

        self.vision_proj_m = deepcopy(self.vision_proj) # 创建视觉投影层的动量版本
        self.text_proj_m = deepcopy(self.text_proj) # 创建文本投影层的动量版本

        # 将当前模型和动量模型的对应模块组成对
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m], # 视觉编码器对
            [self.text_encoder, self.text_encoder_m], # 文本编码器对
            [self.vision_proj, self.vision_proj_m], # 视觉投影层对
            [self.text_proj, self.text_proj_m], # 文本投影层对
        ]
        self.copy_params() # 复制参数到动量模型

        # create the queue # 创建队列
        # 注册 image_queue 作为 buffer，用于存储图像特征队列
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        # 注册 text_queue 作为 buffer，用于存储文本特征队列
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        # 注册 idx_queue 作为 buffer，用于存储队列中样本的索引
        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))
        # 注册 queue_ptr 作为 buffer，用于记录队列的当前指针位置
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 对图像队列进行 L2 归一化
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # 对文本队列进行 L2 归一化
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size # 设置队列大小
        self.momentum = momentum # 设置动量参数
        # 将温度参数 temp 注册为可学习参数，初始值为 0.07
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.alpha = alpha # 设置 alpha 参数
        self.max_txt_len = max_txt_len # 设置最大文本长度

        self.negative_all_rank = negative_all_rank # 设置是否在所有 rank 中采样负样本

    # 计算 ramp-up 因子，用于控制 alpha 的权重随训练进程变化
    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        # 计算当前迭代次数
        current_iters = epoch * num_iters_per_epoch + iters
        # 计算 ramp-up 因子，在前两个 epoch 内线性增加到 1
        return min(1, current_iters / (2 * num_iters_per_epoch))

    # 模型的前向传播方法
    def forward(self, samples):
        """
        Args: # 参数
            samples (dict): A dictionary containing the following keys: # 一个字典，包含以下键：
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images. # 形状为 (batch_size, 3, H, W) 的张量。输入图像。
                - text_input (list): A list of length batch_size, each element is a string of text/caption. # 长度为 batch_size 的列表，每个元素是文本/标题字符串。
                - image_id (torch.Tensor): A tensor of shape (batch_size, ). The image ids, used to identify same images in batch. # 形状为 (batch_size, ) 的张量。图像 ID，用于识别 batch 中的相同图像。
                - epoch (int): The current epoch. # 当前 epoch。
                - iters (int): The current iteration. # 当前迭代次数。
                - num_iters_per_epoch (int): The number of iterations per epoch. # 每个 epoch 的迭代次数。

        Returns: # 返回值
            BlipOutput: A BlipOutput object. See ``lavis.models.blip_models.blip_outputs.BlipOutput`` for more details. # 一个 BlipOutput 对象。更多详情请参阅 ``lavis.models.blip_models.blip_outputs.BlipOutput``。

        Examples: # 示例
            >>> import torch # 导入 torch
            >>> from lavis.models import load_model # 从 lavis.models 导入 load_model
            >>> model = load_model("blip_retrieval", "coco") # 加载 coco 类型的 blip_retrieval 模型
            >>> images = torch.randn(4, 3, 384, 384) # 创建随机图像张量
            >>> text_input = ["caption of image 1", "another caption of image 1", "caption of image 2", "caption of image 3"] # 创建文本输入列表
            >>> image_id = torch.tensor([1, 1, 2, 3]) # 创建图像 ID 张量
            >>> samples = {"image": images, "text_input": text_input, "image_id": image_id, "epoch": 0, "iters": 0, "num_iters_per_epoch": 100} # 创建 samples 字典
            >>> output = model(samples) # 调用模型进行前向传播
            >>> output.keys() # 查看输出的键
            odict_keys(['sims', 'intermediate_output', 'loss', 'loss_itc', 'loss_itm']) # 输出的键
        """
        image = samples["image"] # 从 samples 中获取图像
        caption = samples["text_input"] # 从 samples 中获取文本标题
        idx = samples["image_id"] # 从 samples 中获取图像 ID

        # 计算当前迭代的 alpha 值，随训练进程 ramp-up
        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"], # 当前 epoch
            iters=samples["iters"], # 当前迭代次数
            num_iters_per_epoch=samples["num_iters_per_epoch"], # 每个 epoch 的迭代次数
        )

        with torch.no_grad(): # 在 no_grad 模式下执行
            self.temp.clamp_(0.001, 0.5) # 将温度参数 temp 限制在 [0.001, 0.5] 范围内

        # 使用视觉编码器提取图像特征
        image_embeds = self.visual_encoder.forward_features(image)
        # 创建图像注意力掩码，所有位置都设置为 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device # 将注意力掩码移动到图像所在的设备
        )
        # 对图像的 [CLS] token 特征进行视觉投影并 L2 归一化
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # 使用分词器处理文本输入
        text = self.tokenizer(
            caption, # 文本标题
            padding="max_length", # 填充到最大长度
            truncation=True, # 截断超过最大长度的文本
            max_length=self.max_txt_len, # 最大文本长度
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(image.device) # 将文本张量移动到图像所在的设备

        # 使用文本编码器处理文本
        text_output = self.text_encoder.forward_text(text)
        # 获取文本编码器的最后一层隐藏状态
        text_embeds = text_output.last_hidden_state
        # 对文本的 [CLS] token 特征进行文本投影并 L2 归一化
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # Image-text Contrastive Learning # 图像-文本对比学习
        idx = idx.view(-1, 1) # 将图像 ID 转换为列向量
        # 将当前 batch 的图像 ID 与队列中的图像 ID 连接起来
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        # 计算当前 batch 的图像 ID 与所有图像 ID (当前 batch + 队列) 的相等性，得到正样本掩码
        pos_idx = torch.eq(idx, idx_all).float()
        # 计算相似度目标，正样本位置为 1 / 正样本数量，其余为 0
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features # 获取动量特征
        with torch.no_grad(): # 在 no_grad 模式下执行
            self._momentum_update() # 更新动量模型的参数
            # 使用动量视觉编码器提取图像特征
            image_embeds_m = self.visual_encoder_m(image)
            # 对动量图像的 [CLS] token 特征进行动量视觉投影并 L2 归一化
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
            )
            # 将动量图像特征与队列中的图像特征连接起来
            image_feat_m_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            # 使用动量文本编码器处理文本
            text_output_m = self.text_encoder_m.forward_text(text)
            # 获取动量文本编码器的最后一层隐藏状态
            text_embeds_m = text_output_m.last_hidden_state
            # 对动量文本的 [CLS] token 特征进行动量文本投影并 L2 归一化
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            # 将动量文本特征与队列中的文本特征连接起来
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            # 计算动量图像特征与所有动量文本特征 (当前 batch + 队列) 的相似度
            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp
            # 计算动量文本特征与所有动量图像特征 (当前 batch + 队列) 的相似度
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp

            # 计算图像到文本的相似度目标，结合了动量模型的 softmax 输出和 sim_targets
            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            # 计算文本到图像的相似度目标，结合了动量模型的 softmax 输出和 sim_targets
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        # 计算当前图像特征与所有动量文本特征 (当前 batch + 队列) 的相似度
        sim_i2t = image_feat @ text_feat_m_all / self.temp
        # 计算当前文本特征与所有动量图像特征 (当前 batch + 队列) 的相似度
        sim_t2i = text_feat @ image_feat_m_all / self.temp

        # 计算图像到文本的对比损失
        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1 # 使用 log_softmax 和相似度目标计算损失
        ).mean() # 计算平均损失
        # 计算文本到图像的对比损失
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1 # 使用 log_softmax 和相似度目标计算损失
        ).mean() # 计算平均损失

        loss_itc = (loss_i2t + loss_t2i) / 2 # 计算 ITC 总损失，为 i2t 和 t2i 损失的平均值

        # 将当前 batch 的动量图像特征、动量文本特征和图像 ID 入队
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        # Image-text Matching # 图像-文本匹配
        encoder_input_ids = text.input_ids.clone() # 克隆文本的 input_ids
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id # 将第一个 token (通常是 [CLS]) 替换为编码器 token ID

        # forward the positve image-text pair # 前向传播正样本图像-文本对
        bs = image.size(0) # 获取 batch size
        # 使用文本编码器处理正样本对
        output_pos = self.text_encoder(
            encoder_input_ids, # 编码器输入 ID
            attention_mask=text.attention_mask, # 文本注意力掩码
            encoder_hidden_states=image_embeds, # 图像编码器的隐藏状态作为交叉注意力输入
            encoder_attention_mask=image_atts, # 图像注意力掩码
            return_dict=True, # 返回字典格式的输出
        )

        idxs = concat_all_gather(idx) # 收集所有 rank 的图像 ID
        if self.negative_all_rank: # 如果在所有 rank 中采样负样本
            # compute sample similarity # 计算样本相似度
            with torch.no_grad(): # 在 no_grad 模式下执行
                mask = torch.eq(idx, idxs.t()) # 计算当前 rank 的图像 ID 与所有 rank 的图像 ID 的相等性，得到掩码

                image_feat_world = concat_all_gather(image_feat) # 收集所有 rank 的图像特征
                text_feat_world = concat_all_gather(text_feat) # 收集所有 rank 的文本特征

                # 计算当前 rank 图像特征与所有 rank 文本特征的相似度
                sim_i2t = image_feat @ text_feat_world.t() / self.temp
                # 计算当前 rank 文本特征与所有 rank 图像特征的相似度
                sim_t2i = text_feat @ image_feat_world.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1) # 计算图像到文本相似度的 softmax 权重
                weights_i2t.masked_fill_(mask, 0) # 将正样本位置的权重设为 0

                weights_t2i = F.softmax(sim_t2i, dim=1) # 计算文本到图像相似度的 softmax 权重
                weights_t2i.masked_fill_(mask, 0) # 将正样本位置的权重设为 0

            image_embeds_world = all_gather_with_grad(image_embeds) # 收集所有 rank 的图像嵌入 (带梯度)

            # select a negative image (from all ranks) for each text # 为每个文本选择一个负样本图像 (来自所有 rank)
            image_embeds_neg = [] # 存储负样本图像嵌入的列表
            for b in range(bs): # 遍历 batch 中的每个样本
                neg_idx = torch.multinomial(weights_t2i[b], 1).item() # 根据权重采样一个负样本图像的索引
                image_embeds_neg.append(image_embeds_world[neg_idx]) # 将采样到的负样本图像嵌入添加到列表
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0) # 将列表转换为张量

            # select a negative text (from all ranks) for each image # 为每个图像选择一个负样本文本 (来自所有 rank)
            input_ids_world = concat_all_gather(encoder_input_ids) # 收集所有 rank 的编码器输入 ID
            att_mask_world = concat_all_gather(text.attention_mask) # 收集所有 rank 的文本注意力掩码

            text_ids_neg = [] # 存储负样本文本 ID 的列表
            text_atts_neg = [] # 存储负样本文本注意力掩码的列表
            for b in range(bs): # 遍历 batch 中的每个样本
                neg_idx = torch.multinomial(weights_i2t[b], 1).item() # 根据权重采样一个负样本文本的索引
                text_ids_neg.append(input_ids_world[neg_idx]) # 将采样到的负样本文本 ID 添加到列表
                text_atts_neg.append(att_mask_world[neg_idx]) # 将采样到的负样本文本注意力掩码添加到列表

        else: # 如果只在当前 rank 中采样负样本
            with torch.no_grad(): # 在 no_grad 模式下执行
                mask = torch.eq(idx, idx.t()) # 计算当前 rank 的图像 ID 的相等性，得到掩码

                # 计算当前 rank 图像特征与当前 rank 文本特征的相似度
                sim_i2t = image_feat @ text_feat.t() / self.temp
                # 计算当前 rank 文本特征与当前 rank 图像特征的相似度
                sim_t2i = text_feat @ image_feat.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1) # 计算图像到文本相似度的 softmax 权重
                weights_i2t.masked_fill_(mask, 0) # 将正样本位置的权重设为 0

                weights_t2i = F.softmax(sim_t2i, dim=1) # 计算文本到图像相似度的 softmax 权重
                weights_t2i.masked_fill_(mask, 0) # 将正样本位置的权重设为 0

            # select a negative image (from same rank) for each text # 为每个文本选择一个负样本图像 (来自当前 rank)
            image_embeds_neg = [] # 存储负样本图像嵌入的列表
            for b in range(bs): # 遍历 batch 中的每个样本
                neg_idx = torch.multinomial(weights_t2i[b], 1).item() # 根据权重采样一个负样本图像的索引
                image_embeds_neg.append(image_embeds[neg_idx]) # 将采样到的负样本图像嵌入添加到列表
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0) # 将列表转换为张量

            # select a negative text (from same rank) for each image # 为每个图像选择一个负样本文本 (来自当前 rank)
            text_ids_neg = [] # 存储负样本文本 ID 的列表
            text_atts_neg = [] # 存储负样本文本注意力掩码的列表
            for b in range(bs): # 遍历 batch 中的每个样本
                neg_idx = torch.multinomial(weights_i2t[b], 1).item() # 根据权重采样一个负样本文本的索引
                text_ids_neg.append(encoder_input_ids[neg_idx]) # 将采样到的负样本文本 ID 添加到列表
                text_atts_neg.append(text.attention_mask[neg_idx]) # 将采样到的负样本文本注意力掩码添加到列表

        text_ids_neg = torch.stack(text_ids_neg, dim=0) # 将负样本文本 ID 列表转换为张量
        text_atts_neg = torch.stack(text_atts_neg, dim=0) # 将负样本文本注意力掩码列表转换为张量

        # 将正样本和负样本的文本 ID 连接起来
        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        # 将正样本和负样本的文本注意力掩码连接起来
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        # 将负样本图像嵌入和正样本图像嵌入连接起来 (注意顺序，负样本在前)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        # 将图像注意力掩码连接起来 (负样本和正样本使用相同的掩码)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        # 使用文本编码器处理负样本对 (图像-文本) 和正样本对 (图像-文本)
        output_neg = self.text_encoder(
            text_ids_all, # 所有文本 ID (负样本 + 正样本)
            attention_mask=text_atts_all, # 所有文本注意力掩码
            encoder_hidden_states=image_embeds_all, # 所有图像嵌入 (负样本 + 正样本)
            encoder_attention_mask=image_atts_all, # 所有图像注意力掩码
            return_dict=True, # 返回字典格式的输出
        )

        # 将正样本和负样本的 [CLS] token 嵌入连接起来
        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :], # 正样本的 [CLS] token 嵌入
                output_neg.last_hidden_state[:, 0, :], # 负样本的 [CLS] token 嵌入
            ],
            dim=0, # 在维度 0 上连接
        )
        itm_logits = self.itm_head(vl_embeddings) # 使用 ITM 头计算 logits

        # 创建 ITM 标签：正样本为 1，负样本为 0
        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], # bs 个 1 (正样本)，2*bs 个 0 (负样本)
            dim=0, # 在维度 0 上连接
        ).to(self.device) # 将标签移动到设备
        loss_itm = F.cross_entropy(itm_logits, itm_labels) # 计算 ITM 损失 (交叉熵损失)

        # 返回 BlipOutput 对象，包含损失和中间结果
        return BlipOutput(
            loss=loss_itc + loss_itm, # 总损失为 ITC 损失和 ITM 损失之和
            loss_itc=loss_itc, # ITC 损失
            loss_itm=loss_itm, # ITM 损失
            sims=BlipSimilarity( # 相似度信息
                sim_i2t=sim_i2t, # 图像到文本相似度
                sim_t2i=sim_t2i, # 文本到图像相似度
                sim_i2t_m=sim_i2t_m, # 动量图像到文本相似度
                sim_t2i_m=sim_t2i_m, # 动量文本到图像相似度
                sim_i2t_targets=sim_i2t_targets, # 图像到文本相似度目标
                sim_t2i_targets=sim_t2i_targets, # 文本到图像相似度目标
            ),
            intermediate_output=BlipIntermediateOutput( # 中间输出信息
                image_embeds=image_embeds, # 图像嵌入
                image_embeds_m=image_embeds_m, # 动量图像嵌入
                text_embeds=text_embeds, # 文本嵌入
                text_embeds_m=text_embeds_m, # 动量文本嵌入
                encoder_output=output_pos, # 正样本编码器输出
                encoder_output_neg=output_neg, # 负样本编码器输出
                itm_logits=itm_logits, # ITM logits
                itm_labels=itm_labels, # ITM 标签
            ),
        )

    # 重置队列指针
    def reset_queue_ptr(self):
        self.queue_ptr = torch.zeros(1, dtype=torch.long) # 将队列指针设置为 0

    # 从配置字典创建模型
    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased' # 设置 from_pretrained=True 以加载 'bert-base-uncased' 的权重
        image_encoder = VisionTransformerEncoder.from_config(cfg) # 从配置创建视觉编码器
        text_encoder = XBertEncoder.from_config(cfg) # 从配置创建文本编码器

        embed_dim = cfg.get("embed_dim", 256) # 从配置获取嵌入维度，默认为 256
        momentum = cfg.get("momentum", 0.995) # 从配置获取动量参数，默认为 0.995
        alpha = cfg.get("alpha", 0.4) # 从配置获取 alpha 参数，默认为 0.4
        negative_all_rank = cfg.get("negative_all_rank", False) # 从配置获取 negative_all_rank 参数，默认为 False

        queue_size = cfg.get("queue_size", 0) # 从配置获取队列大小，默认为 0
        max_txt_len = cfg.get("max_txt_len", 35) # 从配置获取最大文本长度，默认为 35

        # 创建 BlipRetrieval 模型实例
        model = cls(
            image_encoder=image_encoder, # 视觉编码器
            text_encoder=text_encoder, # 文本编码器
            queue_size=queue_size, # 队列大小
            alpha=alpha, # alpha 参数
            embed_dim=embed_dim, # 嵌入维度
            momentum=momentum, # 动量参数
            negative_all_rank=negative_all_rank, # negative_all_rank 参数
            max_txt_len=max_txt_len, # 最大文本长度
        )

        model.load_checkpoint_from_config(cfg) # 从配置加载检查点
        model.reset_queue_ptr() # 重置队列指针

        return model # 返回创建的模型实例

    # 计算相似度矩阵
    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader. # 计算给定数据加载器的图像到文本 (i2t) 和文本到图像 (t2i) 相似度矩阵。
        """
        k_test = task_cfg.k_test # 从任务配置获取 k_test 参数

        # 调用 compute_sim_matrix 函数计算相似度矩阵
        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
