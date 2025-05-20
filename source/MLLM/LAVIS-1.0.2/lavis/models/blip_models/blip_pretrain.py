"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可证声明

from copy import deepcopy # 从 copy 模块导入 deepcopy 函数，用于深度复制对象

import torch # 导入 PyTorch 深度学习库
import torch.nn.functional as F # 导入 PyTorch 的函数式接口，通常用于激活函数、损失函数等
from lavis.common.registry import registry # 从 lavis.common.registry 导入 registry，用于注册模型
from lavis.models.base_model import MomentumDistilationMixin, SharedQueueMixin # 从 lavis.models.base_model 导入 MomentumDistilationMixin 和 SharedQueueMixin，提供动量蒸馏和共享队列的功能
from lavis.models.blip_models import tie_encoder_decoder_weights # 从 lavis.models.blip_models 导入 tie_encoder_decoder_weights 函数，用于绑定编码器和解码器的权重
from lavis.models.blip_models.blip import BlipBase # 从 lavis.models.blip_models.blip 导入 BlipBase 类，作为 BLIP 模型的基础类
from lavis.models.blip_models.blip_outputs import ( # 从 lavis.models.blip_models.blip_outputs 导入 BLIP 模型的输出类
    BlipOutput, # BLIP 模型的最终输出类
    BlipSimilarity, # BLIP 模型的相似度输出类
    BlipIntermediateOutput, # BLIP 模型的中间输出类
)
from lavis.models.med import XBertEncoder, XBertLMHeadDecoder # 从 lavis.models.med 导入 XBertEncoder 和 XBertLMHeadDecoder，用于文本编码器和解码器
from lavis.models.vit import VisionTransformerEncoder # 从 lavis.models.vit 导入 VisionTransformerEncoder，用于视觉编码器
from torch import nn # 导入 PyTorch 的神经网络模块

# 使用 registry 注册模型，名称为 "blip_pretrain"
@registry.register_model("blip_pretrain")
# 定义 BlipPretrain 类，继承自 BlipBase, SharedQueueMixin, MomentumDistilationMixin
class BlipPretrain(BlipBase, SharedQueueMixin, MomentumDistilationMixin):
    """
    BLIP pretrain model. # BLIP 预训练模型。

    Supported model types: # 支持的模型类型：
        - base: BLIP base model before pretraining. # base: 预训练前的 BLIP base 模型。
    """

    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_pretrain_base.yaml", # base 模型对应的配置文件路径
        # "large": "configs/models/blip_pretrain_large.yaml", # large 模型对应的配置文件路径 (注释掉)
    }

    # 类的初始化方法
    def __init__(
        self,
        image_encoder, # 图像编码器
        text_encoder, # 文本编码器
        text_decoder, # 文本解码器
        queue_size, # 队列大小
        alpha=0.4, # alpha 参数，用于动量蒸馏中的软目标权重
        embed_dim=256, # 嵌入维度
        momentum=0.995, # 动量参数
        tie_enc_dec_weights=True, # 是否绑定编码器和解码器的权重
        max_txt_len=30, # 最大文本长度
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化分词器
        self.tokenizer = self.init_tokenizer()

        # 调整文本编码器和解码器的 token 嵌入层大小，以匹配分词器的词汇表大小
        text_encoder.resize_token_embeddings(len(self.tokenizer))
        text_decoder.resize_token_embeddings(len(self.tokenizer))

        # 如果 tie_enc_dec_weights 为 True，则绑定编码器和解码器的权重
        if tie_enc_dec_weights:
            tie_encoder_decoder_weights(
                encoder=text_encoder, # 编码器
                decoder=text_decoder.bert, # 解码器 (使用其内部的 bert 模型)
                base_model_prefix="", # 基础模型前缀
                skip_key="/attention", # 跳过的键 (通常是注意力层的权重)
            )

        # 初始化视觉编码器
        self.visual_encoder = image_encoder

        # 初始化文本编码器和解码器
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder

        # 创建用于 ITC (Image-Text Contrastive) 任务的投影层
        text_width = text_encoder.config.hidden_size # 获取文本编码器的隐藏层大小
        vision_width = image_encoder.vision_width # 获取图像编码器的视觉宽度

        self.vision_proj = nn.Linear(vision_width, embed_dim) # 视觉投影层
        self.text_proj = nn.Linear(text_width, embed_dim) # 文本投影层

        # 创建用于 ITM (Image-Text Matching) 任务的头部
        self.itm_head = nn.Linear(text_width, 2) # ITM 头部，输出维度为 2 (匹配/不匹配)

        # 创建动量编码器，通过深度复制主编码器得到
        self.visual_encoder_m = deepcopy(self.visual_encoder) # 动量视觉编码器
        self.text_encoder_m = deepcopy(self.text_encoder) # 动量文本编码器

        # 创建动量投影层，通过深度复制主投影层得到
        self.vision_proj_m = deepcopy(self.vision_proj) # 动量视觉投影层
        self.text_proj_m = deepcopy(self.text_proj) # 动量文本投影层

        # 定义模型对，用于动量更新
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m], # 视觉编码器对
            [self.text_encoder, self.text_encoder_m], # 文本编码器对
            [self.vision_proj, self.vision_proj_m], # 视觉投影层对
            [self.text_proj, self.text_proj_m], # 文本投影层对
        ]
        # 复制参数，将主模型的参数复制到动量模型
        self.copy_params()

        # 创建队列
        # 注册 buffer，image_queue 用于存储图像特征队列，初始化为随机张量
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        # 注册 buffer，text_queue 用于存储文本特征队列，初始化为随机张量
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        # 注册 buffer，queue_ptr 用于记录队列的当前指针位置，初始化为零张量
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 对队列中的特征进行 L2 归一化
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        # 存储队列大小和动量参数
        self.queue_size = queue_size
        self.momentum = momentum
        # 定义温度参数 temp，用于对比学习，初始化为 0.07 的可学习参数
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # 存储 alpha 参数和最大文本长度
        self.alpha = alpha
        self.max_txt_len = max_txt_len

    # 计算 ramp-up 因子，用于控制 alpha 的变化
    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        # 计算当前迭代次数占总 ramp-up 迭代次数的比例，并限制在 [0, 1] 之间
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    # 模型的前向传播方法
    def forward(self, samples):

        """
        Args: # 参数：
            samples (dict): A dictionary containing the following keys: # 一个字典，包含以下键：
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images. Default: H=224, W=224. # 形状为 (batch_size, 3, H, W) 的张量。输入图像。默认：H=224, W=224。
                - text_input (list): A list of length batch_size, each element is a string of text/caption. # 长度为 batch_size 的列表，每个元素是一个文本/标题字符串。
                - epoch (int): The current epoch. # 当前 epoch。
                - iters (int): The current iteration. # 当前迭代次数。
                - num_iters_per_epoch (int): The number of iterations per epoch. # 每个 epoch 的迭代次数。

        Returns: # 返回：
            BlipOutput: A BlipOutput object containing loss and intermediate output. See ``lavis.models.blip_models.blip_outputs.BlipOutput`` for more details. # 一个 BlipOutput 对象，包含损失和中间输出。更多详情请参阅 ``lavis.models.blip_models.blip_outputs.BlipOutput``。

        Examples: # 示例：
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("blip_pretrain", "base")
            >>> images = torch.randn(4, 3, 224, 224)
            >>> text_input = ["caption of image 1", "another caption of image 1", "caption of image 2", "caption of image 3"]
            >>> samples = {"image": images, "text_input": text_input, "epoch": 0, "iters": 0, "num_iters_per_epoch": 100}
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['sims', 'intermediate_output', 'loss', 'loss_itc', 'loss_itm', 'loss_lm'])

            >>> output.intermediate_output.keys()
            odict_keys(['image_embeds', 'text_embeds', 'image_embeds_m', 'text_embeds_m', 'encoder_output', 'encoder_output_neg', 'itm_logits', 'itm_labels', 'decoder_output', 'decoder_labels'])
            >>> output.intermediate_output.image_embeds.shape
            >>> # shape: (batch_size, num_patches, embed_dim)
            torch.Size([4, 197, 768])
            >>> output.intermediate_output.text_embeds.shape
            >>> # shape: (batch_size, max_txt_len, embed_dim)
            torch.Size([4, 30, 768])
            >>> output.intermediate_output.image_embeds_m.shape
            >>> # shape: (batch_size, num_patches, embed_dim)
            torch.Size([4, 197, 768])
            >>> output.intermediate_output.text_embeds_m.shape
            >>> # shape: (batch_size, max_txt_len, embed_dim)
            torch.Size([4, 30, 768])
            >>> output.intermediate_output.itm_logits.shape
            >>> # shape: (batch_size * 3, 2)
            torch.Size([12, 2])
            >>> output.intermediate_output.itm_labels.shape
            >>> # shape: (batch_size * 3,)
            torch.Size([12])
            >>> output.intermediate_output.encoder_output.last_hidden_state.shape
            >>> # shape: (batch_size, max_txt_len, embed_dim)
            torch.Size([4, 30, 768])
            >>> output.intermediate_output.encoder_output_m.last_hidden_state.shape
            >>> # shape: (batch_size, max_txt_len, embed_dim)
            torch.Size([4, 30, 768])
            >>> output.intermediate_output.decoder_output.logits.shape
            >>> # shape: (batch_size, max_txt_len, vocab_size)
            torch.Size([4, 30, 30524])
            >>> output.intermediate_output.decoder_labels.shape
            >>> # shape: (batch_size, max_txt_len)
            torch.Size([4, 30])
        """

        # 从 samples 字典中获取图像和文本输入
        image = samples["image"]
        caption = samples["text_input"]

        # 计算当前迭代的 alpha 值，根据 ramp-up 因子调整
        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        # 在 no_grad 模式下，将温度参数 temp 限制在 [0.001, 0.5] 之间
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # 图像嵌入和特征
        # 通过视觉编码器获取图像嵌入
        image_embeds = self.visual_encoder.forward_features(image)
        # 创建图像注意力掩码，所有位置都设置为 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # 从图像嵌入中提取 [CLS] token 的特征，通过视觉投影层，并进行 L2 归一化
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # 使用分词器处理文本输入
        text = self.tokenizer(
            caption, # 文本列表
            padding="max_length", # 填充到最大长度
            truncation=True, # 截断超过最大长度的文本
            max_length=self.max_txt_len, # 最大长度
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(image.device) # 将张量移动到图像所在的设备

        # 文本嵌入和特征
        # 通过文本编码器获取文本输出
        text_output = self.text_encoder.forward_text(text)
        # 提取文本嵌入 (last_hidden_state)
        text_embeds = text_output.last_hidden_state
        # 从文本嵌入中提取 [CLS] token 的特征，通过文本投影层，并进行 L2 归一化
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # 获取动量模型的特征
        with torch.no_grad(): # 在 no_grad 模式下进行动量更新和特征提取
            # 执行动量更新
            self._momentum_update()
            # 通过动量视觉编码器获取图像嵌入
            image_embeds_m = self.visual_encoder_m(image)
            # 从动量图像嵌入中提取 [CLS] token 的特征，通过动量视觉投影层，并进行 L2 归一化
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
            )
            # 将动量图像特征与图像队列中的特征拼接
            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            # 通过动量文本编码器获取文本输出
            text_output_m = self.text_encoder_m.forward_text(text)
            # 提取动量文本嵌入
            text_embeds_m = text_output_m.last_hidden_state
            # 从动量文本嵌入中提取 [CLS] token 的特征，通过动量文本投影层，并进行 L2 归一化
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            # 将动量文本特征与文本队列中的特征拼接
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            # 计算动量模型下的图像到文本相似度
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            # 计算动量模型下的文本到图像相似度
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            # 创建相似度目标矩阵，对角线为 1，其余为 0
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            # 计算图像到文本的软目标，结合了动量模型的预测和硬目标
            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            # 计算文本到图像的软目标
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        # 计算当前模型下的图像到文本相似度 (与动量模型类似，但不使用动量特征)
        sim_i2t = image_feat @ text_feat_all / self.temp
        # 计算当前模型下的文本到图像相似度
        sim_t2i = text_feat @ image_feat_all / self.temp

        # 计算图像到文本的对比损失 (ITC loss)
        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        # 计算文本到图像的对比损失 (ITC loss)
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        # 计算总的 ITC 损失
        loss_itc = (loss_i2t + loss_t2i) / 2

        # 将当前批次的动量特征入队，并移除队列中最旧的特征
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        # Image-text Matching (ITM) 任务
        # 复制文本输入的 input_ids
        encoder_input_ids = text.input_ids.clone()
        # 将第一个 token (通常是 [CLS]) 替换为编码器 token [ENC]
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # 前向传播正样本图像-文本对
        bs = image.size(0) # 获取批次大小
        output_pos = self.text_encoder(
            encoder_input_ids, # 编码器输入 ids
            attention_mask=text.attention_mask, # 注意力掩码
            encoder_hidden_states=image_embeds, # 编码器隐藏状态 (图像嵌入)
            encoder_attention_mask=image_atts, # 编码器注意力掩码 (图像注意力)
            return_dict=True, # 返回字典格式的输出
        )

        with torch.no_grad(): # 在 no_grad 模式下计算负样本权重
            # 计算文本到图像的权重，用于选择负样本图像
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4 # 仅考虑当前批次的图像，加一个小的 epsilon 避免零概率
            weights_t2i.fill_diagonal_(0) # 将对角线设置为 0，避免选择正样本作为负样本
            # 计算图像到文本的权重，用于选择负样本文本
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4 # 仅考虑当前批次的文本
            weights_i2t.fill_diagonal_(0) # 将对角线设置为 0

        # 为每个文本选择一个负样本图像
        image_embeds_neg = [] # 存储负样本图像嵌入的列表
        for b in range(bs):
            # 根据 weights_t2i 的概率分布，为第 b 个文本随机选择一个负样本图像的索引
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            # 将选定的负样本图像嵌入添加到列表中
            image_embeds_neg.append(image_embeds[neg_idx])
        # 将负样本图像嵌入列表堆叠成一个张量
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # 为每个图像选择一个负样本文本
        text_ids_neg = [] # 存储负样本文本 ids 的列表
        text_atts_neg = [] # 存储负样本文本注意力掩码的列表
        for b in range(bs):
            # 根据 weights_i2t 的概率分布，为第 b 个图像随机选择一个负样本文本的索引
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            # 将选定的负样本文本 ids 添加到列表中
            text_ids_neg.append(encoder_input_ids[neg_idx])
            # 将选定的负样本文本注意力掩码添加到列表中
            text_atts_neg.append(text.attention_mask[neg_idx])

        # 将负样本文本 ids 和注意力掩码列表堆叠成张量
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # 拼接正样本和负样本的文本 ids 和注意力掩码
        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        # 拼接负样本图像嵌入和正样本图像嵌入 (注意顺序，负样本在前)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        # 拼接图像注意力掩码 (正样本和负样本的注意力掩码相同)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        # 前向传播负样本图像-文本对 (包括图像负样本和文本负样本)
        output_neg = self.text_encoder(
            text_ids_all, # 所有文本 ids (正样本和负样本)
            attention_mask=text_atts_all, # 所有文本注意力掩码
            encoder_hidden_states=image_embeds_all, # 所有图像嵌入 (负样本和正样本)
            encoder_attention_mask=image_atts_all, # 所有图像注意力掩码
            return_dict=True, # 返回字典格式的输出
        )

        # 拼接正样本和负样本的视觉-语言 (VL) 嵌入 ([CLS] token 的输出)
        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :], # 正样本的 [CLS] 嵌入
                output_neg.last_hidden_state[:, 0, :], # 负样本的 [CLS] 嵌入
            ],
            dim=0,
        )
        # 通过 ITM 头部计算 ITM 预测 logits
        itm_logits = self.itm_head(vl_embeddings)

        # 创建 ITM 任务的标签
        # 前 bs 个是正样本 (标签为 1)，后 2*bs 个是负样本 (标签为 0)
        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device) # 将标签移动到图像所在的设备
        # 计算 ITM 损失 (交叉熵损失)
        loss_itm = F.cross_entropy(itm_logits, itm_labels)

        # Language Modeling (LM) 任务
        # 复制文本输入的 input_ids
        decoder_input_ids = text.input_ids.clone()
        # 将第一个 token (通常是 [CLS]) 替换为解码器 token [DEC] (bos_token)
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # 创建解码器目标标签，将填充 token (-100) 忽略计算损失
        decoder_targets = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        # 通过文本解码器进行前向传播
        decoder_output = self.text_decoder(
            decoder_input_ids, # 解码器输入 ids
            attention_mask=text.attention_mask, # 注意力掩码
            encoder_hidden_states=image_embeds, # 编码器隐藏状态 (图像嵌入)
            encoder_attention_mask=image_atts, # 编码器注意力掩码 (图像注意力)
            labels=decoder_targets, # 目标标签
            return_dict=True, # 返回字典格式的输出
        )

        # 获取 LM 损失 (由解码器内部计算)
        loss_lm = decoder_output.loss

        # 返回 BlipOutput 对象，包含总损失、各项损失、相似度信息和中间输出
        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm, # 总损失
            loss_itc=loss_itc, # ITC 损失
            loss_itm=loss_itm, # ITM 损失
            loss_lm=loss_lm, # LM 损失
            sims=BlipSimilarity( # 相似度信息
                sim_i2t=sim_i2t, # 当前模型图像到文本相似度
                sim_t2i=sim_t2i, # 当前模型文本到图像相似度
                sim_i2t_m=sim_i2t_m, # 动量模型图像到文本相似度
                sim_t2i_m=sim_t2i_m, # 动量模型文本到图像相似度
                sim_i2t_targets=sim_i2t_targets, # 图像到文本软目标
                sim_t2i_targets=sim_t2i_targets, # 文本到图像软目标
            ),
            intermediate_output=BlipIntermediateOutput( # 中间输出
                image_embeds=image_embeds, # 图像嵌入
                text_embeds=text_embeds, # 文本嵌入
                image_embeds_m=image_embeds_m, # 动量图像嵌入
                text_embeds_m=text_embeds_m, # 动量文本嵌入
                encoder_output=output_pos, # 正样本编码器输出
                encoder_output_neg=output_neg, # 负样本编码器输出
                itm_logits=itm_logits, # ITM logits
                itm_labels=itm_labels, # ITM 标签
                decoder_output=decoder_output, # 解码器输出
                decoder_labels=decoder_targets, # 解码器标签
            ),
        )

    # 重置队列指针的方法
    def reset_queue_ptr(self):
        # 将队列指针设置为零张量
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    # 类方法，从配置字典创建模型实例
    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased' # 设置 from_pretrained=True 以加载 'bert-base-uncased' 的权重
        # 从配置中创建视觉编码器，并加载预训练权重
        image_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=True)
        # 从配置中创建文本编码器，并加载预训练权重
        text_encoder = XBertEncoder.from_config(cfg, from_pretrained=True)
        # 从配置中创建文本解码器，并加载预训练权重
        text_decoder = XBertLMHeadDecoder.from_config(cfg, from_pretrained=True)

        # 从配置中获取模型参数，如果不存在则使用默认值
        embed_dim = cfg.get("embed_dim", 256) # 嵌入维度
        momentum = cfg.get("momentum", 0.995) # 动量
        alpha = cfg.get("alpha", 0.4) # alpha
        max_txt_len = cfg.get("max_txt_len", 30) # 最大文本长度
        queue_size = cfg.get("queue_size", 57600) # 队列大小

        # 创建模型实例
        model = cls(
            image_encoder=image_encoder, # 图像编码器
            text_encoder=text_encoder, # 文本编码器
            text_decoder=text_decoder, # 文本解码器
            embed_dim=embed_dim, # 嵌入维度
            queue_size=queue_size, # 队列大小
            momentum=momentum, # 动量
            alpha=alpha, # alpha
            tie_enc_dec_weights=True, # 绑定编码器和解码器权重
            max_txt_len=max_txt_len, # 最大文本长度
        )

        # [IMPORTANT] to reset queue pointer to 0. # [重要] 将队列指针重置为 0。
        # Otherwise when updating last batch in the queue, the batch size and remaining queue length may be un-equal. # 否则，在更新队列中的最后一个批次时，批次大小和剩余队列长度可能不相等。
        # 重置队列指针
        model.reset_queue_ptr()

        # 返回创建的模型实例
        return model
