"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可证声明

from dataclasses import dataclass # 从 dataclasses 模块导入 dataclass 装饰器，用于创建数据类
from typing import Optional # 从 typing 模块导入 Optional，表示字段可以是指定类型或 None

import torch # 导入 PyTorch 深度学习库
from transformers.modeling_outputs import ( # 从 transformers.modeling_outputs 导入模型输出相关的类
    ModelOutput, # 基础模型输出类
    BaseModelOutputWithPoolingAndCrossAttentions, # 包含 pooling 和交叉注意力的基础模型输出类
    CausalLMOutputWithCrossAttentions, # 包含交叉注意力的因果语言模型输出类
)


# 使用 dataclass 装饰器定义 BlipSimilarity 数据类，继承自 ModelOutput
@dataclass
class BlipSimilarity(ModelOutput):
    # 图像到文本的相似度分数
    sim_i2t: torch.FloatTensor = None
    # 文本到图像的相似度分数
    sim_t2i: torch.FloatTensor = None

    # 来自动量模型的图像到文本的相似度分数 (可选)
    sim_i2t_m: Optional[torch.FloatTensor] = None
    # 来自动量模型的文本到图像的相似度分数 (可选)
    sim_t2i_m: Optional[torch.FloatTensor] = None

    # 图像到文本的相似度目标 (可选)
    sim_i2t_targets: Optional[torch.FloatTensor] = None
    # 文本到图像的相似度目标 (可选)
    sim_t2i_targets: Optional[torch.FloatTensor] = None


# 使用 dataclass 装饰器定义 BlipIntermediateOutput 数据类，继承自 ModelOutput
@dataclass
class BlipIntermediateOutput(ModelOutput):
    """
    Data class for intermediate outputs of BLIP models. # BLIP 模型中间输出的数据类。

    image_embeds (torch.FloatTensor): Image embeddings, shape (batch_size, num_patches, embed_dim). # 图像嵌入，形状为 (batch_size, num_patches, embed_dim)。
    text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim). # 文本嵌入，形状为 (batch_size, seq_len, embed_dim)。

    image_embeds_m (torch.FloatTensor): Image embeddings from momentum visual encoder, shape (batch_size, num_patches, embed_dim). # 来自动量视觉编码器的图像嵌入，形状为 (batch_size, num_patches, embed_dim)。
    text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder, shape (batch_size, seq_len, embed_dim). # 来自动量文本编码器的文本嵌入，形状为 (batch_size, seq_len, embed_dim)。

    encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder. # 来自图像-文本编码器的输出。
    encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder for negative pairs. # 来自图像-文本编码器用于负样本对的输出。

    decoder_output (CausalLMOutputWithCrossAttentions): output from the image-grounded text decoder. # 来自图像-文本解码器的输出。
    decoder_labels (torch.LongTensor): labels for the captioning loss. # 用于字幕生成损失的标签。

    itm_logits (torch.FloatTensor): logits for the image-text matching loss, shape (batch_size * 3, 2). # 图像-文本匹配损失的 logits，形状为 (batch_size * 3, 2)。
    itm_labels (torch.LongTensor): labels for the image-text matching loss, shape (batch_size * 3,) # 图像-文本匹配损失的标签，形状为 (batch_size * 3,)

    """

    # uni-modal features # 单模态特征
    # 图像嵌入 (可选)
    image_embeds: torch.FloatTensor = None
    # 文本嵌入 (可选)
    text_embeds: Optional[torch.FloatTensor] = None

    # 来自动量模型的图像嵌入 (可选)
    image_embeds_m: Optional[torch.FloatTensor] = None
    # 来自动量模型的文本嵌入 (可选)
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder # 多模态编码器的中间输出
    # 编码器输出 (可选)
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    # 负样本对的编码器输出 (可选)
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    # 图像-文本匹配任务的 logits (可选)
    itm_logits: Optional[torch.FloatTensor] = None
    # 图像-文本匹配任务的标签 (可选)
    itm_labels: Optional[torch.LongTensor] = None

    # intermediate outputs of multimodal decoder # 多模态解码器的中间输出
    # 解码器输出 (可选)
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    # 解码器标签 (可选)
    decoder_labels: Optional[torch.LongTensor] = None


# 使用 dataclass 装饰器定义 BlipOutput 数据类，继承自 ModelOutput
@dataclass
class BlipOutput(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional. # 一些微调模型 (例如 BlipVQA) 不计算相似度，因此是可选的。
    # 相似度输出 (可选)
    sims: Optional[BlipSimilarity] = None

    # 中间输出
    intermediate_output: BlipIntermediateOutput = None

    # 总损失 (可选)
    loss: Optional[torch.FloatTensor] = None

    # 图像-文本对比损失 (可选)
    loss_itc: Optional[torch.FloatTensor] = None

    # 图像-文本匹配损失 (可选)
    loss_itm: Optional[torch.FloatTensor] = None

    # 语言模型损失 (可选)
    loss_lm: Optional[torch.FloatTensor] = None


# 使用 dataclass 装饰器定义 BlipOutputWithLogits 数据类，继承自 BlipOutput
@dataclass
class BlipOutputWithLogits(BlipOutput):
    # 预测 logits
    logits: torch.FloatTensor = None
    # 来自动量模型的预测 logits
    logits_m: torch.FloatTensor = None


# 使用 dataclass 装饰器定义 BlipOutputFeatures 数据类，继承自 ModelOutput
@dataclass
class BlipOutputFeatures(ModelOutput):
    """
    Data class of features from BlipFeatureExtractor. # BlipFeatureExtractor 提取的特征数据类。

    Args: # 参数：
        image_embeds: (torch.FloatTensor) of shape (batch_size, num_patches+1, embed_dim), optional # 图像嵌入，形状为 (batch_size, num_patches+1, embed_dim)，可选
        image_features: (torch.FloatTensor) of shape (batch_size, num_patches+1, feature_dim), optional # 图像特征，形状为 (batch_size, num_patches+1, feature_dim)，可选
        text_embeds: (torch.FloatTensor) of shape (batch_size, sequence_length+1, embed_dim), optional # 文本嵌入，形状为 (batch_size, sequence_length+1, embed_dim)，可选
        text_features: (torch.FloatTensor) of shape (batch_size, sequence_length+1, feature_dim), optional # 文本特征，形状为 (batch_size, sequence_length+1, feature_dim)，可选

        The first embedding or feature is for the [CLS] token. # 第一个嵌入或特征对应于 [CLS] token。

        Features are obtained by projecting the corresponding embedding into a normalized low-dimensional space. # 特征是通过将相应的嵌入投影到归一化的低维空间中获得的。
    """

    # 图像嵌入 (可选)
    image_embeds: Optional[torch.FloatTensor] = None
    # 投影后的图像嵌入 (可选)
    image_embeds_proj: Optional[torch.FloatTensor] = None

    # 文本嵌入 (可选)
    text_embeds: Optional[torch.FloatTensor] = None
    # 投影后的文本嵌入 (可选)
    text_embeds_proj: Optional[torch.FloatTensor] = None

    # 多模态嵌入 (可选)
    multimodal_embeds: Optional[torch.FloatTensor] = None
