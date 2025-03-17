# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence

import torch
from transformers import DataCollatorForSeq2Seq


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    将带有索引的注意力掩码从(batch_size, seq_len)扩展到(batch_size, 1, seq_len, seq_len)，
    处理打包序列并转换为下三角形式以防止未来信息泄露
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min  # 获取当前数据类型的最小值
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)  # 扩展为4D
    
    # 创建填充掩码（非零位置为1，零位置保持0）
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    
    # 生成块对角注意力掩码（相同索引的位置为1）
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    
    # 应用下三角掩码（保留左下角，右上角置零）
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    
    # 反转掩码：有效位置设为0，填充位置设为最小值
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.
    支持多模态大语言模型的数据整理器

    Features should contain input_ids, attention_mask, labels and images.
    特征应包含input_ids, attention_mask, labels和images
    """
    template: Optional["Template"] = None  # 模板对象，用于处理多模态输入
    processor: Optional["ProcessorMixin"] = None  # 处理器对象，用于处理图像/视频

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids = [], [], [], [], []
        # 遍历特征提取多媒体数据
        for feature in features:
            images = feature.pop("images", None) or []  # 弹出图像数据
            videos = feature.pop("videos", None) or []  # 弹出视频数据
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))  # 记录每个样本的图像数量
            batch_vidlens.append(len(videos))  # 记录每个样本的视频数量
            batch_input_ids.append(feature["input_ids"])  # 收集输入ID

        # 通过模板插件获取多模态输入
        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids, self.processor
        )
        # 处理token类型ID（如果存在）
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        # 调用父类处理文本特征
        features: Dict[str, "torch.Tensor"] = super().__call__(features)
        features.update(mm_inputs)  # 合并多模态输入
        
        # 处理特殊格式的像素值（如pixtral模型）
        if isinstance(features.get("pixel_values"), list):  # for pixtral inputs
            features = features.data  # 使用默认的collate函数代替BatchEncoding.to()

        return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for 4d attention mask.
    生成4D注意力掩码的数据整理器
    """
    block_diag_attn: bool = False  # 是否使用块对角注意力
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"  # 注意力实现方式
    compute_dtype: "torch.dtype" = torch.float32  # 计算数据类型

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)  # 先处理基础特征
        # 当需要块对角注意力且不使用flash attention时，生成4D掩码
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    处理成对数据的数据整理器
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.
        将批次数据填充到批次中最长的序列长度

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        生成2*n个样本，前n个是优选样本，后n个是次选样本
        """
        concatenated_features = []
        # 遍历每个特征，分别处理chosen和rejected样本
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)  # 调用父类处理拼接后的特征


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for KTO data.
    处理KTO（Kahneman-Tversky Optimization）数据的数据整理器
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        target_features = []  # 目标特征列表
        kl_features = []  # KL散度特征列表
        kto_tags = []  # KTO标签列表
        
        # 分离目标特征和KL特征
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature["images"],
                "videos": feature["videos"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])  # 收集KTO标签

        # 分别处理目标特征和KL特征
        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        
        # 合并结果
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)  # 转换为张量
        return batch
