"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可证声明

import logging # 导入日志模块
import os # 导入操作系统模块

import torch # 导入 PyTorch 深度学习库
from lavis.common.dist_utils import download_cached_file # 从 lavis.common.dist_utils 导入 download_cached_file 函数，用于下载缓存文件
from lavis.common.utils import is_url # 从 lavis.common.utils 导入 is_url 函数，用于判断是否是 URL
from lavis.models.base_model import BaseModel # 从 lavis.models.base_model 导入 BaseModel 类，作为模型的基类
from lavis.models.vit import interpolate_pos_embed # 从 lavis.models.vit 导入 interpolate_pos_embed 函数，用于插值位置嵌入
from transformers import BertTokenizer # 从 transformers 库导入 BertTokenizer，用于处理文本

# 定义 BlipBase 类，继承自 BaseModel
class BlipBase(BaseModel):
    # 类方法，用于初始化分词器
    @classmethod
    def init_tokenizer(cls):
        # 从预训练的 "bert-base-uncased" 模型加载 Bert 分词器
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # 添加特殊 token "[DEC]" 作为 bos_token (beginning of sequence token)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # 添加额外的特殊 token "[ENC]"
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        # 将 "[ENC]" token 的 ID 赋值给 tokenizer 的 enc_token_id 属性
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        # 返回初始化后的分词器
        return tokenizer

    # 从预训练模型加载权重的方法
    def load_from_pretrained(self, url_or_filename):
        # 判断输入的 url_or_filename 是否是 URL
        if is_url(url_or_filename):
            # 如果是 URL，下载缓存文件
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            # 加载 checkpoint 文件到 CPU
            checkpoint = torch.load(cached_file, map_location="cpu")
        # 判断输入的 url_or_filename 是否是文件路径
        elif os.path.isfile(url_or_filename):
            # 如果是文件路径，直接加载 checkpoint 文件到 CPU
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        # 如果既不是 URL 也不是文件路径，则抛出运行时错误
        else:
            raise RuntimeError("checkpoint url or path is invalid") # checkpoint url 或路径无效

        # 从 checkpoint 中提取模型的 state_dict
        state_dict = checkpoint["model"]

        # 对视觉编码器的位置嵌入进行插值，以适应当前模型的尺寸
        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        # 检查模型是否包含 visual_encoder_m (momentum visual encoder) 的位置嵌入
        if "visual_encoder_m.pos_embed" in self.state_dict().keys():
            # 如果存在，也对其位置嵌入进行插值
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
            )

        # 遍历当前模型的 state_dict 中的所有键
        for key in self.state_dict().keys():
            # 如果 checkpoint 的 state_dict 中也包含这个键
            if key in state_dict.keys():
                # 检查对应 tensor 的形状是否一致
                if state_dict[key].shape != self.state_dict()[key].shape:
                    # 如果形状不一致，删除 checkpoint 中的这个键，避免加载时出错
                    del state_dict[key]

        # 加载 state_dict 到当前模型，strict=False 表示允许部分加载，忽略不匹配的键
        msg = self.load_state_dict(state_dict, strict=False)

        # 记录加载时缺失的键
        logging.info("Missing keys {}".format(msg.missing_keys))
        # 记录从哪个路径或 URL 加载了 checkpoint
        logging.info("load checkpoint from %s" % url_or_filename)

        # 返回加载结果信息
        return msg
