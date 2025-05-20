"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可证声明

import torch # 导入 PyTorch 深度学习库
import torch.nn.functional as F # 导入 PyTorch 的函数式接口，通常用于激活函数、损失函数等
from lavis.common.registry import registry # 从 lavis.common.registry 导入 registry，用于注册模型
from lavis.models.blip_models.blip import BlipBase # 从 lavis.models.blip_models.blip 导入 BlipBase 类，作为 BLIP 模型的基础类
from torch import nn # 导入 PyTorch 的神经网络模块
from lavis.models.med import XBertEncoder # 从 lavis.models.med 导入 XBertEncoder，用于文本编码器

from lavis.models.vit import VisionTransformerEncoder # 从 lavis.models.vit 导入 VisionTransformerEncoder，用于视觉编码器

# 使用 registry 注册模型，名称为 "blip_image_text_matching"
@registry.register_model("blip_image_text_matching")
# 定义 BlipITM 类，继承自 BlipBase
class BlipITM(BlipBase):
    """
    BLIP Image-Text Matching (ITM) model. # BLIP 图像-文本匹配 (ITM) 模型。

    Supported model types: # 支持的模型类型：
        - base: fine-tuned BLIP retrieval weights on COCO dataset (Karpathy split). # base: 在 COCO 数据集 (Karpathy 分割) 上微调的 BLIP 检索权重。
        - large: fine-tuned BLIP retrieval weights on COCO dataset (Karpathy split). # large: 在 COCO 数据集 (Karpathy 分割) 上微调的 BLIP 检索权重。

    Usage: # 用法：
        >>> from lavis.models import load_model
        >>> model = load_model("blip_image_text_matching", "base")
        >>> model = load_model("blip_image_text_matching", "large")
    """

    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_itm_base.yaml", # base 模型对应的配置文件路径
        "large": "configs/models/blip_itm_large.yaml", # large 模型对应的配置文件路径
    }

    # 类的初始化方法
    def __init__(self, image_encoder, text_encoder, embed_dim=256, max_txt_len=35):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化分词器
        self.tokenizer = self.init_tokenizer()

        # 初始化文本编码器
        self.text_encoder = text_encoder

        # 初始化视觉编码器
        self.visual_encoder = image_encoder

        # 存储最大文本长度
        self.max_txt_len = max_txt_len

        # creating projection layers for ITC # 创建用于 ITC (Image-Text Contrastive) 任务的投影层
        text_width = text_encoder.config.hidden_size # 获取文本编码器的隐藏层大小
        vision_width = image_encoder.vision_width # 获取图像编码器的视觉宽度

        self.vision_proj = nn.Linear(vision_width, embed_dim) # 视觉投影层
        self.text_proj = nn.Linear(text_width, embed_dim) # 文本投影层

        # 创建用于 ITM (Image-Text Matching) 任务的头部
        self.itm_head = nn.Linear(text_width, 2) # ITM 头部，输出维度为 2 (匹配/不匹配)

    # 模型的前向传播方法
    def forward(self, samples, match_head="itm"):
        # 从 samples 字典中获取图像和文本输入
        image = samples["image"]
        caption = samples["text_input"]

        # 通过视觉编码器获取图像嵌入
        image_embeds = self.visual_encoder.forward_features(image)
        # 创建图像注意力掩码，所有位置都设置为 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 使用分词器处理文本输入
        text = self.tokenizer(
            caption, # 文本列表
            padding="longest", # 填充到最长文本的长度
            truncation=True, # 截断超过最大长度的文本
            max_length=self.max_txt_len, # 最大长度
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(image.device) # 将张量移动到图像所在的设备

        # 如果 match_head 是 "itm"，执行 ITM 任务的前向传播
        if match_head == "itm":
            # 复制文本输入的 input_ids
            encoder_input_ids = text.input_ids.clone()
            # 将第一个 token (通常是 [CLS]) 替换为编码器 token [ENC]
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id  # extra code # 额外代码
            # 通过文本编码器进行前向传播，将图像嵌入作为 encoder_hidden_states
            output = self.text_encoder(
                encoder_input_ids, # 编码器输入 ids
                attention_mask=text.attention_mask, # 注意力掩码
                encoder_hidden_states=image_embeds, # 编码器隐藏状态 (图像嵌入)
                encoder_attention_mask=image_atts, # 编码器注意力掩码 (图像注意力)
                return_dict=True, # 返回字典格式的输出
            )
            # 从文本编码器输出的 [CLS] token 嵌入中，通过 ITM 头部获取 ITM 预测 logits
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            # 返回 ITM 预测 logits
            return itm_output

        # 如果 match_head 是 "itc"，执行 ITC 任务的前向传播
        elif match_head == "itc":
            # 通过文本编码器获取文本输出 (仅文本模式)
            text_output = self.text_encoder(
                text.input_ids, # 文本输入 ids
                attention_mask=text.attention_mask, # 注意力掩码
                return_dict=True, # 返回字典格式的输出
                mode="text", # 指定为文本模式
            )
            # 从图像嵌入中提取 [CLS] token 的特征，通过视觉投影层，并进行 L2 归一化
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            # 从文本编码器输出的 [CLS] token 嵌入中，通过文本投影层，并进行 L2 归一化
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            # 计算图像特征和文本特征之间的相似度矩阵
            sim = image_feat @ text_feat.t()
            # 返回相似度矩阵
            return sim

    # ITM 排序方法
    def itm_rank(self, image_embeds, image_atts, encoder_input_ids, match_head='itm'):
        # breakpoint() # 调试断点 (注释掉)
        # 复制编码器输入 ids
        encoder_input_ids = encoder_input_ids.clone()
        # 移除前 3 个 token (通常是 [CLS], [SEP], [PAD] 或其他特殊 token)
        encoder_input_ids = encoder_input_ids[:, 3:]
        # 根据填充 token 创建文本注意力掩码
        text_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id).long()

        # 如果 match_head 是 'itm'
        if match_head == 'itm':
            # encoder_input_ids = encoder_input_ids.clone() # 复制 (重复代码，注释掉)
            # 将第一个 token 替换为编码器 token [ENC]
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
            # 通过文本编码器进行前向传播，将图像嵌入作为 encoder_hidden_states
            output = self.text_encoder(encoder_input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            # print(output.last_hidden_state.shape) # 打印形状 (调试代码，注释掉)
            # 从文本编码器输出的 [CLS] token 嵌入中，通过 ITM 头部获取 ITM 预测 logits
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            # 对 logits 进行 softmax，并取匹配 (标签为 1) 的概率
            itm_output = F.softmax(itm_output, dim=1)[:,1]
            # 返回匹配概率
            return itm_output #, mask, token_length # 返回匹配概率 (注释掉返回 mask 和 token_length)

        # 如果 match_head 是 'itc'
        elif match_head == 'itc':
            # 将第一个 token 替换为 [CLS] token
            encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
            # 通过文本编码器获取文本输出 (文本模式)
            text_output = self.text_encoder(encoder_input_ids, attention_mask=text_attention_mask,
                                            return_dict=True, mode='text')
            # 从图像嵌入中提取 [CLS] token 的特征，通过视觉投影层，并进行 L2 归一化
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            # 从文本编码器输出的 [CLS] token 嵌入中，通过文本投影层，并进行 L2 归一化
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

            # 计算图像特征和文本特征之间的相似度矩阵
            sim = image_feat @ text_feat.t()
            # 返回相似度矩阵
            return sim

    # 类方法，从配置字典创建模型实例
    @classmethod
    def from_config(cls, cfg=None):
        # 从配置中创建视觉编码器
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        # 从配置中创建文本编码器
        text_encoder = XBertEncoder.from_config(cfg)

        # 从配置中获取模型参数，如果不存在则使用默认值
        embed_dim = cfg.get("embed_dim", 256) # 嵌入维度
        max_txt_len = cfg.get("max_txt_len", 35) # 最大文本长度

        # 创建模型实例
        model = cls(
            image_encoder=image_encoder, # 图像编码器
            text_encoder=text_encoder, # 文本编码器
            embed_dim=embed_dim, # 嵌入维度
            max_txt_len=max_txt_len, # 最大文本长度
        )

        # 从配置中加载 checkpoint
        model.load_checkpoint_from_config(cfg)

        # 返回创建的模型实例
        return model


# 计算 Grad-CAM 的函数
def compute_gradcam(model, visual_input, text_input, tokenized_text, block_num=6):
    # 设置文本编码器中指定 block 的 crossattention 的 save_attention 为 True，以便获取注意力权重
    model.text_encoder.base_model.base_model.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True

    # 通过模型进行前向传播，获取 ITM 输出
    output = model({"image": visual_input, "text_input": text_input}, match_head="itm")
    # 计算 ITM 匹配概率的总和作为损失 (用于反向传播)
    loss = output[:, 1].sum()

    # 清零模型的梯度
    model.zero_grad()
    # 对损失进行反向传播，计算梯度
    loss.backward()
    with torch.no_grad(): # 在 no_grad 模式下处理梯度和注意力图
        # 调整 tokenized_text 的注意力掩码形状，以便后续广播计算
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1) # (批次大小, 1, token 长度, 1, 1)
        # 计算每个文本的有效 token 长度 (排除特殊 token)
        token_length = tokenized_text.attention_mask.sum(dim=-1) - 2
        # 将 token 长度移动到 CPU
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch] # 梯度和注意力图 [批次大小, 头数, 序列长度, 图像 patch 数]
        # 获取注意力梯度
        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attn_gradients()
        # 获取注意力权重 (注意力图)
        cams = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attention_map()

        # assume using vit with 576 num image patch # 假设使用具有 576 个图像 patch 的 ViT
        # 调整注意力图的形状，并应用掩码
        cams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24) * mask
        # 调整梯度的形状，取 ReLU (clamp(0))，并应用掩码
        grads = (
            grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 24, 24)
            * mask
        )

        # 计算 Grad-CAM (注意力图 * 梯度)
        gradcams = cams * grads
        gradcam_list = [] # 存储每个样本的 Grad-CAM 列表

        # 遍历每个样本
        for ind in range(visual_input.size(0)):
            token_length_ = token_length[ind] # 获取当前样本的有效 token 长度
            # 计算当前样本的 Grad-CAM，在所有注意力头上取平均，并移动到 CPU
            gradcam = gradcams[ind].mean(0).cpu().detach()
            # [enc token gradcam, average gradcam across token, gradcam for individual token] # [enc token 的 Grad-CAM, 跨 token 的平均 Grad-CAM, 单个 token 的 Grad-CAM]
            # 拼接不同类型的 Grad-CAM
            gradcam = torch.cat(
                (
                    gradcam[0:1, :], # [ENC] token 的 Grad-CAM
                    gradcam[1 : token_length_ + 1, :].sum(dim=0, keepdim=True)
                    / token_length_, # 跨有效 token 的平均 Grad-CAM
                    gradcam[1:, :], # 所有 token (包括填充 token) 的 Grad-CAM
                )
            )
            # 将计算出的 Grad-CAM 添加到列表中
            gradcam_list.append(gradcam)

    # 返回 Grad-CAM 列表和 ITM 输出
    return gradcam_list, output
