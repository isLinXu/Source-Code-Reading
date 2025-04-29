# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# ------------------------------------------------------------------------
import os  
# 导入 os 模块，用于与操作系统进行交互（文件路径、环境变量等）
import copy  
# 导入 copy 模块，用于执行对象的浅拷贝和深拷贝
import random  
# 导入 random 模块，用于生成随机数
from dataclasses import dataclass, field  
# 从 dataclasses 模块导入 dataclass 和 field，用于简化类的数据字段定义
import json  
# 导入 json 模块，用于 JSON 序列化和反序列化
import logging  
# 导入 logging 模块，用于记录日志
import pathlib  
# 导入 pathlib 模块，用于面向对象的文件系统路径操作
from typing import Dict, Optional, Sequence, List  
# 导入类型提示：Dict, Optional, Sequence, List

import torch  
# 导入 PyTorch 库，用于张量计算和深度学习
import numpy as np  
# 导入 NumPy 库，并简写为 np，用于数值计算

import transformers  
# 导入 HuggingFace transformers 库
import tokenizers  
# 导入 HuggingFace tokenizers 库

from mgm.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,  
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)  
# 从 mgm.constants 模块导入常量：IGNORE_INDEX（忽略标签标记）、IMAGE_TOKEN_INDEX（图像 token 索引）、  
# DEFAULT_IMAGE_TOKEN（默认图像占位符）、DEFAULT_IM_START_TOKEN（图像开始标记）、DEFAULT_IM_END_TOKEN（图像结束标记）
from torch.utils.data import Dataset  
# 从 torch.utils.data 导入 Dataset 基类
from mgm.train.llava_trainer import LLaVATrainer  
# 从自定义模块 mgm.train.llava_trainer 中导入 LLaVATrainer 训练器类

from mgm import conversation as conversation_lib  
# 导入 mgm.conversation 并重命名为 conversation_lib，用于对话构造
from mgm.model import *  
# 导入 mgm.model 包中的所有内容（模型定义等）
from mgm.mm_utils import tokenizer_image_token  
# 从 mgm.mm_utils 导入 tokenizer_image_token，用于处理带图像的 tokenization

from PIL import Image  
# 从 Pillow 库导入 Image，用于图像处理
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  
# 从 transformers.models.mixtral.modeling_mixtral 导入 MixtralSparseMoeBlock 混合专家模块

local_rank = None  
# 定义 local_rank 变量，用于分布式训练的本地进程编号

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
# 定义 rank0_print 函数，仅在 local_rank==0 时打印，用于在多卡训练中只在主卡输出

from packaging import version  
# 导入 packaging.version 模块，用于版本比较
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')  
# 比较当前 tokenizers 版本是否 >= 0.14，保存布尔值

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    # 模型名称或路径，默认 "facebook/opt-125m"
    version: Optional[str] = field(default="v0")
    # 模型版本标识，默认 "v0"
    freeze_backbone: bool = field(default=False)
    # 是否冻结主干网络参数
    tune_mm_mlp_adapter: bool = field(default=False)
    # 是否仅微调多模态 MLP adapter
    vision_tower: Optional[str] = field(default=None)
    # 主视觉塔模型路径或名称
    vision_tower_aux: Optional[str] = field(default=None) # auxiliary vision tower
    # 英文注释保留：auxiliary vision tower
    # 中文：辅助视觉塔模型路径或名称
    optimize_vision_tower: bool = field(default=False) # whether to optimize vision tower
    # 英文注释保留：whether to optimize vision tower
    # 中文：是否优化主视觉塔
    optimize_vision_tower_aux: bool = field(default=False) # whether to optimize auxiliary vision tower
    # 英文注释保留：whether to optimize auxiliary vision tower
    # 中文：是否优化辅助视觉塔
    drop_path: Optional[bool] = field(default=True) # whether to use drop path in auxiliary tower
    # 英文注释保留：whether to use drop path in auxiliary tower
    # 中文：在辅助塔中是否使用 drop path 技术
    image_processor: Optional[str] = field(default=None)
    # 图像预处理器名称或路径
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    # 英文注释保留：default to the last layer
    # 中文：默认选择最后一层视觉特征
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # 预训练多模态 MLP adapter 路径
    mm_projector_type: Optional[str] = field(default='linear')
    # 多模态投影器类型，默认 'linear'
    mm_use_im_start_end: bool = field(default=False)
    # 是否在图像 token 前后添加特殊起止标记
    mm_use_im_patch_token: bool = field(default=True)
    # 是否将图像切片作为单独 token
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # 多模态视觉特征选择方式，默认为 "patch"

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    # 训练数据文件路径
    lazy_preprocess: bool = False
    # 是否延迟预处理
    is_multimodal: bool = False
    # 是否多模态任务
    image_folder: Optional[str] = field(default=None)
    # 图像文件夹路径
    image_aspect_ratio: str = 'square'
    # 图像宽高比，默认 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    # 图像网格定位点配置
    image_size_aux: Optional[int] = field(default=320)
    # 辅助图像大小
    image_grid: Optional[int] = field(default=1)
    # 图像网格大小
    image_global: Optional[bool] = field(default=False)
    # 是否使用全局图像特征

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # 缓存目录
    optim: str = field(default="adamw_torch")
    # 优化器类型
    remove_unused_columns: bool = field(default=False)
    # 是否移除数据集中未使用的列
    freeze_mm_mlp_adapter: bool = field(default=False)
    # 是否冻结多模态 MLP adapter
    mpt_attn_impl: Optional[str] = field(default="triton")
    # MPT 注意力实现方式
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # 英文注释保留：Maximum sequence length...
    # 中文：最大序列长度，序列将进行右侧填充或截断
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    # 英文注释保留：Compress the quantization statistics...
    # 中文：是否使用双重量化压缩统计信息
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    # 英文注释保留：Quantization data type to use...
    # 中文：量化数据类型，可选 `fp4` 或 `nf4`
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    # 英文注释保留：How many bits to use.
    # 中文：使用多少位量化
    lora_enable: bool = False
    # 是否启用 LoRA
    lora_r: int = 64
    # LoRA 低秩矩阵秩 r
    lora_alpha: int = 16
    # LoRA 学习率缩放系数 α
    lora_dropout: float = 0.05
    # LoRA dropout 比例
    lora_weight_path: str = ""
    # LoRA 权重加载路径
    lora_bias: str = "none"
    # LoRA 偏置选项：none/all/lora_only
    mm_projector_lr: Optional[float] = None
    # 多模态投影器学习率
    group_by_modality_length: bool = field(default=False)
    # 是否按模态长度分组采样
    lr_multi: Optional[str] = field(default=None)
    # 多学习率配置字符串

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    # 导入 deepspeed.zero 和 ZeroParamStatus，用于分布式零冗余优化
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param
# maybe_zero_3：在 ZeRO 优化下安全获取参数到 CPU

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    # 英文注释保留：Borrowed from peft.utils.get_peft_model_state_dict
    # 中文：借鉴自 peft.utils.get_peft_model_state_dict
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return
# get_peft_state_maybe_zero_3：根据 bias 策略提取 PEFT 状态并搬运到 CPU

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
# get_peft_state_non_lora_maybe_zero_3：提取非 LoRA 参数状态并搬运到 CPU

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
# get_mm_adapter_state_maybe_zero_3：提取多模态 adapter 参数状态并搬运到 CPU

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_uni']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        # 英文注释保留：needed for 16-bit
        # 中文：16 位场景下需要剔除 lm_head
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
# find_all_linear_names：查找模型中所有可应用 LoRA 的 Linear 层名称

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    # 英文注释保留：Collects the state dict and dump to disk.
    # 中文：收集模型状态字典并保存到磁盘
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        # 英文注释保留：Only save Adapter
        # 中文：仅保存多模态 MLP adapter 权重
        keys_to_match = ['mm_projector', 'vision_resampler', 'vlm_uni']
        # add vision tower
        # 英文注释保留：add vision tower
        # 中文：添加 vision tower 关键字
        keys_to_match.extend(['vision_tower'])
        # add vision tower aux
        # 英文注释保留：add vision tower aux
        # 中文：添加辅助视觉塔关键字
        keys_to_match.extend(['vision_fpn', 'vision_stages'])
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
# safe_save_model_for_hf_trainer：根据配置安全地保存模型或 Adapter 权重

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # 英文注释保留：Resize tokenizer and embedding.
    # 中文：调整 tokenizer 词表并重设嵌入矩阵大小
    # 英文注释保留：Note: This is the unoptimized version...
    # 中文：注意：此版本未经优化，可能导致嵌入维度不可被64整除
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 向 tokenizer 添加特殊 tokens，返回新添加的数量
    model.resize_token_embeddings(len(tokenizer))
    # 根据新词表长度调整模型嵌入矩阵大小
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        # 计算旧 embeddings 的均值，作为新 token 的初始值
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # 英文注释保留：Tokenize a list of strings.
    # 中文：对字符串列表进行分词处理
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    # 对每个字符串调用 tokenizer，返回 padded/truncated 的张量
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    # 提取 input_ids 作为输入和标签
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    # 计算每条序列的真实长度（非 pad token 数量）
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    # 对 prompt 部分全部 mask 掉
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len
# _mask_targets：根据说话者信息将生成目标中对应位置 mask 为 IGNORE_INDEX

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    # 英文注释保留：Add speaker and start/end signal on each round.
    # 中文：在每轮对话中添加说话者标识和起止信号
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation
# _add_speaker_and_signal：为对话轮次添加前缀"### 说话者: "及换行，构建完整对话字符串

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    # 如果不是多模态，直接返回原始 sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN,
                        '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>'
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources
# preprocess_multimodal：对多模态输入做 IMAGE_TOKEN 插入和特殊标记替换

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    # 复制默认对话模板
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # 定义人类和 GPT 的角色映射

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # 构建每条对话的完整 prompt

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([
            tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
            for prompt in conversations
        ], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # 分词并返回 input_ids 张量

    targets = input_ids.clone()
    # 复制作为训练目标

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2
    # 确保使用 LLAMA_2 分隔风格

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # 计算非 pad token 总长度
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )
# preprocess_llama_2：为 LLAMA-2 风格对话构建 prompt、分词并 mask 掉非回答部分，返回 input_ids 和 labels

def preprocess_v1(  # 定义函数 preprocess_v1，用于 v1 风格对话的预处理，输入 sources、tokenizer、has_image，返回 Dict
    sources,  # sources: 原始对话数据列表，每个元素为若干轮对话消息构成的列表
    tokenizer: transformers.PreTrainedTokenizer,  # tokenizer: HuggingFace 预训练分词器实例
    has_image: bool = False  # has_image: 是否包含图像标记，默认 False
) -> Dict:  # 返回一个字典，包含 input_ids 和 labels
    conv = conversation_lib.default_conversation.copy()  
    # 复制默认对话模板对象，用于后续拼接消息构建 prompt
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  
    # 定义角色映射：human 对应第一个角色，gpt 对应第二个角色

    # Apply prompt templates  
    # 英文注释保留：Apply prompt templates  
    # 中文：应用 prompt 模板，构建对话字符串
    conversations = []  
    # 初始化列表，用于存储格式化后的对话 prompt
    for i, source in enumerate(sources):  
        # 遍历所有源对话，i 为索引，source 为一条对话的消息列表
        if roles[source[0]["from"]] != conv.roles[0]:  
            # 如果第一条消息不是 human 发起，则跳过该消息
            # Skip the first one if it is not from human  
            # 中文：若首条消息非 human，则丢弃
            source = source[1:]  
            # 丢弃首条消息

        conv.messages = []  
        # 重置 conv.messages，以便重新添加本条对话的消息
        for j, sentence in enumerate(source):  
            # 遍历本条对话中每条消息，j 为轮次索引，sentence 为消息字典
            role = roles[sentence["from"]]  
            # 根据消息的 "from" 字段获取对应角色
            assert role == conv.roles[j % 2], f"{i}"  
            # 断言角色与模板预期顺序一致，否则报错并输出对话索引
            conv.append_message(role, sentence["value"])  
            # 将该条消息（角色 + 文本）添加到 conv 对象
        conversations.append(conv.get_prompt())  
        # 调用 conv.get_prompt() 构建完整 prompt，并追加到 conversations 列表

    # Tokenize conversations  
    # 英文注释保留：Tokenize conversations  
    # 中文：对构建好的 prompt 列表进行分词编码

    if has_image:  
        # 如果包含图像，则使用自定义 tokenizer_image_token 处理
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
             for prompt in conversations],
            dim=0
        )  
        # 对每个 prompt 调用 tokenizer_image_token，并在第 0 维堆叠成 batch 张量
    else:  
        # 否则使用标准 tokenizer 批量分词
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids  
        # 返回 input_ids 张量，已按最长序列 padding/truncate

    targets = input_ids.clone()  
    # 克隆一份 input_ids 作为训练目标 labels
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO  
    # 确保当前对话分隔风格为 TWO

    # Mask targets  
    # 英文注释保留：Mask targets  
    # 中文：对 labels 进行 mask，只保留模型回复部分参与计算
    sep = conv.sep + conv.roles[1] + ": "  
    # 定义本轮对话中指令与回复的分隔符，例如 "[INST]GPT: "
    for conversation, target in zip(conversations, targets):  
        # 遍历每条 prompt 文本及对应的 target 张量
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  
        # 计算非 pad token 的总长度

        rounds = conversation.split(conv.sep2)  
        # 按 conv.sep2（轮次分隔符）拆分对话为多轮
        cur_len = 1  
        # 初始化当前位置 cur_len=1，保留起始 token（<bos>）的位置
        target[:cur_len] = IGNORE_INDEX  
        # 将第一个 token 的 label 置为 IGNORE_INDEX

        for i, rou in enumerate(rounds):  
            # 遍历每一轮的完整文本 rou
            if rou == "":  
                break  # 若文本为空，退出循环

            parts = rou.split(sep)  
            # 按 sep 将本轮文本分为"指令"和"回复"两部分
            if len(parts) != 2:  
                print(f"WARNING: parts!=: {parts}")  
                # 打印警告：分割失败
                break  # 退出处理

            parts[0] += sep  
            # 恢复指令部分末尾的 sep

            if has_image:  
                # 含图像模式下，使用 tokenizer_image_token 计算长度
                round_len = len(tokenizer_image_token(rou, tokenizer))  
                # 本轮总 token 个数
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2  
                # 指令部分 token 数，减去起止标记
            else:  
                # 纯文本模式下
                round_len = len(tokenizer(rou).input_ids)  
                # 本轮总 token 数
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2  
                # 指令部分 token 数

            if i != 0 and not getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                # 针对第 2 轮及以后新版本 tokenizer，长度需减 1 以对齐
                round_len -= 1  
                instruction_len -= 1  

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  
            # 将当前轮指令部分的 labels 全部置为 IGNORE_INDEX

            cur_len += round_len  
            # 更新 cur_len，移动到下一轮文本起点

        target[cur_len:] = IGNORE_INDEX  
        # 将剩余部分（包括模型回复后可能的填充）全部 mask

        if cur_len < tokenizer.model_max_length:  
            # 若实际处理长度小于最大长度，则校验 token 总数
            if cur_len != total_len:  
                # 如果计算位置与统计长度不一致
                target[:] = IGNORE_INDEX  
                # 将整个序列 labels 全部 mask
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )  
                # 输出 tokenization mismatch 警告

    return dict(
        input_ids=input_ids,  # 输入 token ids 张量
        labels=targets,       # 已 mask 的 labels 张量
    )  # 返回处理结果

def preprocess_llama_3(  # 定义函数 preprocess_llama_3：适用于 LLAMA-3 风格对话的预处理
    sources,  # sources: 原始对话数据，每条数据是一系列消息的列表
    tokenizer: transformers.PreTrainedTokenizer,  # tokenizer: HuggingFace 预训练分词器实例
    has_image: bool = False  # has_image: 是否包含图像标记，默认 False
) -> Dict:  # 返回一个字典，包含 input_ids 和 labels
    conv = conversation_lib.default_conversation.copy()  
    # 复制默认对话模板对象，用于构建 prompt
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  
    # 定义角色映射：human 对应第一个角色，gpt 对应第二个角色

    # Apply prompt templates  
    # 英文注释保留：Apply prompt templates  
    # 中文：应用 prompt 模板，构建对话字符串
    conversations = []  
    # 初始化空列表，用来存储格式化后的对话 prompt
    for i, source in enumerate(sources):  
        # 遍历所有源对话，i 为索引，source 为一条对话的消息列表
        if roles[source[0]["from"]] != conv.roles[0]:  
            # Skip the first one if it is not from human  
            # 中文：若首条消息非 human，则跳过该消息
            source = source[1:]  
            # 丢弃列表中的第一条消息

        conv.messages = []  
        # 重置 conv.messages，准备重新添加本条对话的消息
        for j, sentence in enumerate(source):  
            # 遍历本条对话的每条消息，j 为轮次索引，sentence 为消息字典
            role = roles[sentence["from"]]  
            # 根据消息的 "from" 字段映射到角色
            assert role == conv.roles[j % 2], f"{i}"  
            # 断言角色和模板中预期的轮次顺序一致，若不一致则报错并输出索引
            conv.append_message(role, sentence["value"])  
            # 将该条消息（角色 + 文本）添加到 conv 对象
        conversations.append(conv.get_prompt())  
        # 调用 get_prompt() 构建完整 prompt，并追加到列表

    # Tokenize conversations  
    # 英文注释保留：Tokenize conversations  
    # 中文：对构建好的 prompt 列表进行分词编码
    if has_image:  
        # 包含图像时，使用自定义的 tokenizer_image_token 处理
        input_ids = torch.stack([  # 堆叠所有样本的 input_ids
            tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
            for prompt in conversations
        ], dim=0)
    else:  
        # 纯文本模式时，调用标准 tokenizer
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids  # 返回张量形式的 input_ids

    targets = input_ids.clone()  
    # 克隆一份 input_ids，作为训练目标 labels
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3  
    # 确保当前对话模板风格为 LLAMA_3

    # Mask targets  
    # 英文注释保留：Mask targets  
    # 中文：对 labels 进行 mask，只保留模型回复部分计算损失
    sep = conv.sep + conv.roles[1]  
    # 定义分隔符 sep，如 "[INST]GPT"
    for conversation, target in zip(conversations, targets):  
        # 遍历每条 prompt 文本及对应的 target 张量
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  
        # 计算非 pad token 的总长度

        rounds = conversation.split(conv.sep)  
        # 按 conv.sep 将对话拆分为若干"轮"
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt  
        # 英文注释保留：system + user + gpt  
        # 中文：重组前三段：system、user、gpt
        for conv_idx in range(3, len(rounds), 2):  
            # 从第四段开始，每两段组成一轮用户-模型对话
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt  
            # 英文注释保留：user + gpt  
            # 中文：将后续每对 user 与 gpt 拼为一轮

        # include <bos> for all rounds  
        # 英文注释保留：include <bos> for all rounds  
        # 中文：为所有轮次保留起始 <bos> token
        cur_len = 1  
        # 起始位置为第一个 token
        target[:cur_len] = IGNORE_INDEX  
        # 将首个 token 的 label 置为 IGNORE_INDEX

        for i, rou in enumerate(re_rounds):  
            # 遍历每个重组后的轮次文本
            if rou == "":  
                break  # 若轮次文本为空，退出循环

            parts = rou.split(sep)  
            # 按 sep 分割为"指令"和"响应"两部分
            if len(parts) != 2:  
                print(f"WARNING: parts!=: {parts}")  
                # 分割失败时打印警告
                break

            parts[0] += sep  
            # 恢复指令部分末尾的 sep

            # include <bos> for all rounds  
            # 英文注释保留：include <bos> for all rounds  
            # 中文：为所有轮次保留开始 token
            if has_image:  
                # 图像模式下，计算不含首 <bos> 的 token 数
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2  
            else:  
                # 纯文本模式下
                round_len = len(tokenizer(rou).input_ids) - 1  
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2  

            # include <|eot_id|> for all rounds  
            # 英文注释保留：include <|eot_id|> for all rounds  
            # 中文：为所有轮次包含结束 token
            round_len += 1  
            instruction_len += 1  

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  
            # 将当前轮指令部分的 labels 全部置为 IGNORE_INDEX
            cur_len += round_len  
            # 更新 cur_len，跳过本轮总长度

        target[cur_len:] = IGNORE_INDEX  
        # 将当前位置之后的所有 labels 全部 mask

        if cur_len < tokenizer.model_max_length:  
            # 若实际使用长度小于模型最大长度，则校验长度一致性
            if cur_len != total_len:  
                target[:] = IGNORE_INDEX  
                # 不一致时，整体置为 IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. "
                    f"(ignored)"
                )
                # 打印 tokenization mismatch 警告

    return dict(
        input_ids=input_ids,  # 返回输入 id 张量
        labels=targets,       # 返回已 mask 的 labels 张量
    )

def preprocess_gemma(  # 定义函数 preprocess_gemma：适用于 GEMMA 风格对话的预处理
    sources,  # sources: 原始对话数据，每条为若干消息的列表
    tokenizer: transformers.PreTrainedTokenizer,  # tokenizer: HuggingFace 预训练分词器实例
    has_image: bool = False  # has_image: 是否包含图像标记，默认 False
) -> Dict:  # 返回一个字典，包含 input_ids 和 labels
    conv = conversation_lib.default_conversation.copy()  
    # 复制默认对话模板对象，用于后续拼接消息构建 prompt
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  
    # 定义角色映射：human 对应模板的第一个角色，gpt 对应第二个角色

    # Apply prompt templates  
    # 英文注释保留：Apply prompt templates  
    # 中文：应用 prompt 模板，构建对话字符串
    conversations = []  
    # 初始化列表，用于存储格式化后的对话 prompt
    for i, source in enumerate(sources):  
        # 遍历所有源对话，i 为索引，source 为一条对话的消息列表
        if roles[source[0]["from"]] != conv.roles[0]:  
            # Skip the first one if it is not from human  
            # 中文：若首条消息非 human，则丢弃该消息
            source = source[1:]  
            # 丢弃首条消息

        conv.messages = []  
        # 重置 conv.messages，以便重新添加本条对话的消息
        for j, sentence in enumerate(source):  
            # 遍历本条对话中每条消息，j 为轮次索引，sentence 为消息字典
            role = roles[sentence["from"]]  
            # 根据消息的 "from" 字段获取对应角色
            assert role == conv.roles[j % 2], f"{i}"  
            # 断言角色与模板预期顺序一致，否则抛出索引信息
            conv.append_message(role, sentence["value"])  
            # 将该条消息（角色 + 文本）添加到 conv 对象
        conversations.append(conv.get_prompt())  
        # 调用 get_prompt() 构建完整 prompt，并追加到 conversations 列表

    # Tokenize conversations  
    # 英文注释保留：Tokenize conversations  
    # 中文：对构建好的 prompt 列表进行分词编码
    if has_image:  
        # 如果包含图像，则使用自定义 tokenizer_image_token 处理
        input_ids = torch.stack(  
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
             for prompt in conversations],  
            dim=0  
        )  
        # 将每个示例的张量堆叠成 batch
    else:  
        # 否则使用标准 tokenizer 批量分词
        input_ids = tokenizer(  
            conversations,  
            return_tensors="pt",  
            padding="longest",  
            max_length=tokenizer.model_max_length,  
            truncation=True,  
        ).input_ids  
        # 返回 input_ids 张量，已按最长序列 padding/truncate

    targets = input_ids.clone()  
    # 克隆一份 input_ids 作为训练目标 labels
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA  
    # 确保当前对话分隔风格为 GEMMA

    # Mask targets  
    # 英文注释保留：Mask targets  
    # 中文：对 labels 进行 mask，只保留模型回复部分参与计算
    sep = "<start_of_turn>" + conv.sep + conv.roles[1] + "\n"  
    # 定义 GEMMA 风格的轮次开始分隔符，例如 "<start_of_turn>[INST]GPT\n"
    for conversation, target in zip(conversations, targets):  
        # 遍历每条 prompt 文本及对应的 target 张量
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  
        # 计算非 pad token 的总长度

        rounds = conversation.split(conv.sep2)  
        # 按 conv.sep2（多轮分隔符）拆分对话为若干轮
        cur_len = 1  
        # 初始化当前位置 cur_len=1，保留起始 <bos> token
        target[:cur_len] = IGNORE_INDEX  
        # 将第一个 token 的 label 置为 IGNORE_INDEX

        for i, rou in enumerate(rounds):  
            # 遍历每一轮的完整文本 rou
            if rou == "":  
                break  # 若轮次文本为空，退出循环

            parts = rou.split(sep)  
            # 按 sep 分割为"指令"和"回复"两部分
            if len(parts) != 2:  
                print(f"WARNING: parts!=: {parts}")  
                # 分割失败时打印警告
                break  # 退出当前对话处理

            parts[0] += sep  
            # 恢复指令部分末尾的 sep

            if has_image:  
                # 含图像模式下，计算本轮总 token 数
                round_len = len(tokenizer_image_token(rou, tokenizer))  
                # 计算指令部分 token 数时，排除首个 <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  
                # exclude <bos>
            else:  
                # 纯文本模式下
                round_len = len(tokenizer(rou).input_ids)  
                # 计算本轮总 token 数
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  
                # exclude <bos>，指令部分 token 数

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  
            # 将当前轮指令部分的 labels 全部置为 IGNORE_INDEX
            cur_len += round_len  
            # 更新 cur_len，跳过本轮总 token 长度

        target[cur_len:] = IGNORE_INDEX  
        # 将当前位置之后的所有 labels 全部 mask

        if cur_len < tokenizer.model_max_length:  
            # 若实际处理长度小于模型最大长度，则校验长度一致性
            if cur_len != total_len:  
                target[:] = IGNORE_INDEX  
                # 不一致时，将整个序列 labels 全部置为 IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )  
                # 输出 tokenization mismatch 警告

    return dict(
        input_ids=input_ids,  # 返回输入 ids 张量
        labels=targets,       # 返回已 mask 的 labels 张量
    )

def preprocess_mpt(  # 定义函数 preprocess_mpt：适用于 MPT 风格对话的预处理，输入 sources、tokenizer 和是否含图像标志，返回 Dict
    sources,  # sources: 原始对话数据列表，每条为若干消息字典构成的列表
    tokenizer: transformers.PreTrainedTokenizer,  # tokenizer: HuggingFace 预训练分词器实例
    has_image: bool = False  # has_image: 是否包含图像标记，默认 False
) -> Dict:  # 返回字典，包含 input_ids 和 labels
    conv = conversation_lib.default_conversation.copy()  
    # 复制默认对话模板，用于拼接消息生成 prompt
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  
    # 定义角色映射：human 对应第一个角色，gpt 对应第二个角色

    # Apply prompt templates  
    # 英文注释保留：Apply prompt templates  
    # 中文：应用 prompt 模板，构建对话字符串
    conversations = []  
    # 初始化列表，用于存储格式化后的对话 prompt
    for i, source in enumerate(sources):  
        # 遍历所有对话源，i 为索引，source 为该对话中若干消息的列表
        if roles[source[0]["from"]] != conv.roles[0]:  
            # Skip the first one if it is not from human  
            # 中文：若首条消息非 human，则跳过
            source = source[1:]  
            # 丢弃首条消息

        conv.messages = []  
        # 重置 conv.messages，准备重新添加本条对话
        for j, sentence in enumerate(source):  
            # 遍历本条对话的每条消息
            role = roles[sentence["from"]]  
            # 根据消息的 "from" 字段映射角色
            assert role == conv.roles[j % 2], f"{i}"  
            # 断言消息角色与模板角色顺序匹配，否则报错并输出对话索引
            conv.append_message(role, sentence["value"])  
            # 将角色 + 文本添加到 conv
        conversations.append(conv.get_prompt())  
        # 构建该对话的完整 prompt 并加入列表

    # Tokenize conversations  
    # 英文注释保留：Tokenize conversations  
    # 中文：对所有 prompt 列表进行分词编码
    if has_image:  
        # 若包含图像，使用自定义 tokenizer_image_token 处理
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
             for prompt in conversations],
            dim=0
        )  
        # 将各样本张量沿第 0 维堆叠
    else:  
        # 纯文本模式，调用标准 tokenizer
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids  
        # 返回已 padding/truncate 的 input_ids 张量

    targets = input_ids.clone()  
    # 克隆一份 input_ids，作为 labels
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT  
    # 确保分隔风格为 MPT

    # Mask targets  
    # 英文注释保留：Mask targets  
    # 中文：对 labels 进行 mask，仅保留模型回复部分计算损失
    sep = conv.sep + conv.roles[1]  
    # 定义本轮回复分隔符，如 "[INST]GPT"
    for conversation, target in zip(conversations, targets):  
        # 遍历每条 prompt 及其对应的 label 张量
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  
        # 统计非 pad token 的总长度

        rounds = conversation.split(conv.sep)  
        # 使用 conv.sep 将对话拆分为轮次
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt  
        # 英文注释保留：system + user + gpt  
        # 中文：重组前三段：系统指令、用户、模型
        for conv_idx in range(3, len(rounds), 2):  
            # 从第 4 段开始，每 2 段组成一轮 user + gpt
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt  
            # 英文注释保留：user + gpt

        cur_len = 0  
        # 初始化当前位置为 0（MPT 风格可能不包含开头 <bos>）
        target[:cur_len] = IGNORE_INDEX  
        # 将这一区间置为 IGNORE_INDEX（此处为空操作）

        for i, rou in enumerate(re_rounds):  
            # 遍历每个重组后的轮次文本
            if rou == "":  
                break  # 若为空则退出该对话处理

            parts = rou.split(sep)  
            # 按 sep 分割为"指令"和"回复"两部分
            if len(parts) != 2:  
                break  # 分割失败则退出

            parts[0] += sep  
            # 恢复指令部分末尾的 sep
            # not included <|im_end|>  
            # 英文注释保留：not included <|im_end|>  
            # 中文：指令部分不含图像结束标记

            if has_image:  
                # 图像模式下
                round_len = len(tokenizer_image_token(rou, tokenizer))  
                # 本轮总 token 数
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  
                # 指令部分 token 数，减去 <bos>
            else:  
                # 纯文本模式下
                round_len = len(tokenizer(rou).input_ids)  
                # 本轮总 token 数
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  
                # 指令部分 token 数

            # include <|im_end|> for all rounds  
            # 英文注释保留：include <|im_end|> for all rounds  
            # 中文：若 legacy tokenizer 且新版且含图像，包含图像结束标记
            if getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:  
                round_len += 1  
                instruction_len += 1  

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  
            # 将当前轮指令部分的 labels 全部置为 IGNORE_INDEX

            cur_len += round_len  
            # 更新 cur_len，移动到下一轮起始位置

        target[cur_len:] = IGNORE_INDEX  
        # 将当前位置之后的所有 labels 全部 mask

        if cur_len < tokenizer.model_max_length:  
            # 若实际长度小于模型最大长度，则校验一致性
            if cur_len != total_len:  
                target[:] = IGNORE_INDEX  
                # 不匹配时，整体置为 IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )  
                # 输出警告

    return dict(
        input_ids=input_ids,  # 返回输入 id 张量
        labels=targets,       # 返回已 mask 的 labels 张量
    )


def preprocess_plain(
    sources: Sequence[str],  # sources: 序列，每个元素包含一条对话的两部分（通常是图像占位符和文本）
    tokenizer: transformers.PreTrainedTokenizer,  # tokenizer: 用于分词的预训练分词器
) -> Dict:
    # add end signal and concatenate together
    # 英文注释保留：add end signal and concatenate together
    # 中文：添加结束信号并将对话各部分拼接为单条字符串
    conversations = []  
    # 初始化列表，用于存储拼接后的对话字符串
    for source in sources:  
        # 遍历每条原始对话 source，source 应为长度为 2 的列表
        assert len(source) == 2  
        # 确保 source 列表中恰好有两项
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']  
        # 确保 source[0]['value'] 中包含图像占位符 DEFAULT_IMAGE_TOKEN
        source[0]['value'] = DEFAULT_IMAGE_TOKEN  
        # 将 source[0]['value'] 整体替换为 DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep  
        # 拼接图像占位符、用户文本和默认对话模板中的分隔符
        conversations.append(conversation)  
        # 将拼接结果加入 conversations 列表

    # tokenize conversations
    # 英文注释保留：tokenize conversations
    # 中文：对拼接后的对话字符串列表进行分词编码
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
        for prompt in conversations
    ]
    # 使用 tokenizer_image_token 对每个拼接字符串进行编码，返回张量形式

    targets = copy.deepcopy(input_ids)  
    # 深拷贝 input_ids 作为目标 labels

    for target, source in zip(targets, sources):  
        # 遍历每个目标张量和对应的原始 source
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))  
        # 计算图像占位符部分单独分词后的长度
        target[:tokenized_len] = IGNORE_INDEX  
        # 将该长度范围内的 labels 全部置为 IGNORE_INDEX（不计算损失）

    return dict(input_ids=input_ids, labels=targets)  
    # 返回字典，包含 input_ids 和已 mask 的 labels

def preprocess(
    sources: Sequence[str],  # sources: 序列，每个元素是一条对话（消息列表）
    tokenizer: transformers.PreTrainedTokenizer,  # tokenizer: 预训练分词器
    has_image: bool = False,  # has_image: 是否包含图像，默认 False
    prompt: str = None,  # prompt: 自定义提示，当前未使用
    refine_prompt: bool = False,  # refine_prompt: 是否对 prompt 进行微调，当前未使用
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        # 如果对话分隔风格为 PLAIN，则调用 preprocess_plain
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        # 如果风格为 LLAMA_2，则调用 preprocess_llama_2
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        # 如果风格为 LLAMA_3，则调用 preprocess_llama_3
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        # 如果模板版本以 "v1" 开头，则调用 preprocess_v1
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version.startswith("gemma"):
        # 如果是 gemma 系列模板，则调用 preprocess_gemma
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        # 如果模板版本为 mpt，则调用 preprocess_mpt
        return preprocess_mpt(sources, tokenizer, has_image=has_image)

    # add end signal and concatenate together
    # 英文注释保留：add end signal and concatenate together
    # 中文：在每条对话中添加结束信号并拼接
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        # 构造对话头部，包括系统提示和两个换行
        conversation = _add_speaker_and_signal(header, source)
        # 调用辅助函数，为每条消息添加说话者标记和结束信号
        conversations.append(conversation)
        # 将结果加入列表

    # tokenize conversations
    # 英文注释保留：tokenize conversations
    # 中文：对拼接后的对话字符串列表进行分词编码
    def get_tokenize_len(prompts):
        # 定义局部函数：计算每个 prompt 的分词后长度
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        # 包含图像时，调用 tokenizer_image_token 并返回张量列表
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
            for prompt in conversations
        ]
    else:
        # 纯文本时，调用标准分词函数
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]
        # 获取 input_ids 张量

    targets = copy.deepcopy(input_ids)
    # 深拷贝 input_ids 作为 labels
    for target, source in zip(targets, sources):
        # 遍历每个目标张量和对应的原始 source
        if has_image:
            # 如果包含图像，先构造 prompts 列表以获取各部分长度
            tokenized_lens = get_tokenize_len(
                [header] + [s["value"] for s in source]
            )
        else:
            # 纯文本时，使用 _tokenize_fn 获取每部分长度列表
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        # 获取每条消息的说话者列表
        _mask_targets(target, tokenized_lens, speakers)
        # 调用辅助函数，对 labels 进行 mask：将 human 部分置为 IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)
    # 返回包含 input_ids 和已 mask labels 的字典


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    # 英文注释保留：Dataset for supervised fine-tuning.
    # 中文：用于监督微调的惰性加载数据集（仅在获取数据时处理）

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))  # 从 JSON 文件加载原始数据

        rank0_print("Formatting inputs...Skip in lazy mode")  # 主进程打印提示信息
        self.tokenizer = tokenizer  # 保存分词器实例
        self.list_data_dict = list_data_dict  # 原始数据列表
        self.data_args = data_args  # 数据配置参数

    def __len__(self):
        return len(self.list_data_dict)  # 返回数据集总样本数

    @property
    def lengths(self):
        # 计算每个样本的 token 长度（近似值）
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0  # 图像 token 的估算长度
            # 累加对话中每个语句的分词后长度（按空格分割估算）
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        # 计算模态相关长度（正数表示含图像，负数表示纯文本）
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample) else -cur_len  # 含图像取正值，否则取负
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 带重试机制的数据获取方法
        attempt, max_attempt = 0, 10  # 最多重试 10 次
        while attempt < max_attempt:
            try:
                data_dict = self._sample_item(i)  # 尝试获取数据
                break
            except:
                attempt += 1
                print(f"Error in loading {i}, retrying...")
                i = random.randint(0, len(self.list_data_dict)-1)  # 失败时随机选择新索引
        return data_dict

    def _sample_item(self, i) -> Dict[str, torch.Tensor]:
        # 实际处理单个数据项的核心方法
        image = None
        sources = self.list_data_dict[i]  # 获取原始数据项
        suffix = None
        if isinstance(i, int):
            sources = [sources]  # 统一转换为列表形式处理
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # 图像处理分支
        if 'image' in sources[0]:
            image_files = self.list_data_dict[i]['image']  # 获取图像文件名/列表
            image_folder = self.data_args.image_folder  # 图像存储根目录
            processor = self.data_args.image_processor  # 图像预处理器
            
            image_total = []  # 存储处理后的图像张量
            if not isinstance(image_files, list):
                image_files = [image_files]  # 统一转换为列表处理
            
            for image_file in image_files:
                # 处理特殊数据集的图像路径
                if 'ocr' in image_file:  # OCR VQA 数据集格式转换
                    if not os.path.exists(os.path.join(image_folder, image_file)):
                        image_file = image_file.replace(".jpg", ".png")  # jpg 转 png

                elif 'VG_100K' in image_file:  # Visual Genome 数据集路径修正
                    image_file = image_file.replace('VG_100K_2', 'images')
                    image_file = image_file.replace('VG_100K', 'images')

                # 图像加载与预处理
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':  # 填充为正方形处理
                    def expand2square(pil_img, background_color):
                        # 将图像扩展为正方形，背景填充 processor.image_mean 颜色
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    # 转换颜色通道并填充
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:  # 普通预处理
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                
                image_total.append(image)  # 添加到图像列表
            
            # 多图像堆叠处理
            if len(image_total) > 1:
                image = torch.stack(image_total, dim=0)  # 沿第0维堆叠成张量
            else:
                image = image_total[0]  # 单图像直接取用
            
            # 多模态数据预处理
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:  # 纯文本分支
            sources = copy.deepcopy([e["conversations"] for e in sources])
                
        has_image = ('image' in self.list_data_dict[i])  # 判断是否含图像
        data_dict = preprocess(  # 调用预处理函数
            sources,
            self.tokenizer,
            has_image=has_image)

        # 维度调整（单样本情况）
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        # 生成辅助图像（raw + aux）
        if hasattr(self.data_args, 'image_size_raw') and (image is not None): 
            data_dict['image_aux'] = image.clone()  # 克隆原始图像作为辅助图像
            raw_shape = [  # 计算原始图像目标尺寸
                self.data_args.image_size_raw['height'] * self.data_args.image_grid,
                self.data_args.image_size_raw['width'] * self.data_args.image_grid]
            
            if 'image' in self.list_data_dict[i]:  # 仅处理真实图像输入
                if len(image.shape) == 3:  # 单图像插值
                    image = torch.nn.functional.interpolate(image[None], 
                                                            size=raw_shape, 
                                                            mode='bilinear', 
                                                            align_corners=False)[0]
                else:  # 多图像插值
                    image = torch.nn.functional.interpolate(image, 
                                                            size=raw_shape, 
                                                            mode='bilinear', 
                                                            align_corners=False)
        
        # 图像数据注入
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image  # 注入处理后的图像
        elif self.data_args.is_multimodal:  # 多模态模型但无图像时填充零张量
            crop_size = self.data_args.image_processor.crop_size
            if hasattr(self.data_args, 'image_size_raw'):  # 含原始尺寸配置时
                data_dict['image'] = torch.zeros(3, 
                                                 self.data_args.image_size_raw['height'] * self.data_args.image_grid, 
                                                 self.data_args.image_size_raw['width'] * self.data_args.image_grid)
                data_dict['image_aux'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            else:  # 普通情况
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        # 多网格图像处理（当 image_grid >= 2 时）
        if 'image' in data_dict and self.data_args.image_grid >= 2:
            # 重构原始图像维度 [C, H*G, W*G] -> [G, G, C, H, W]
            raw_image = data_dict['image'].reshape(3, 
                                                   self.data_args.image_grid,
                                                   self.data_args.image_size_raw['height'],
                                                   self.data_args.image_grid,
                                                   self.data_args.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)  # 调整维度顺序
            raw_image = raw_image.reshape(-1, 3,  # 合并网格维度 [G*G, C, H, W]
                                          self.data_args.image_size_raw['height'],
                                          self.data_args.image_size_raw['width'])
            
            if self.data_args.image_global:  # 需要全局图像
                global_image = data_dict['image']  # 获取原始全局图像
                if len(global_image.shape) == 3:
                    global_image = global_image[None]  # 添加批次维度
                # 下采样到原始尺寸
                global_image = torch.nn.functional.interpolate(global_image, 
                                                        size=[self.data_args.image_size_raw['height'],
                                                              self.data_args.image_size_raw['width']], 
                                                        mode='bilinear', 
                                                        align_corners=False)
                # 拼接局部图像块和全局图像 [N+1, C, H, W]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            data_dict['image'] = raw_image.contiguous()  # 确保内存连续
        
        return data_dict  # 返回最终数据字典


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    # 英文注释保留：Collate examples for supervised fine-tuning.
    # 中文：用于监督微调的数据整理器，将多个样本整理为 batch

    tokenizer: transformers.PreTrainedTokenizer  # 分词器实例，用于处理 pad token

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 核心调用方法：将多个样本整理成 batch
        input_ids, labels = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels"))  
        # 分别提取所有样本的 input_ids 和 labels

        # 对 input_ids 进行填充对齐
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,  # 输出形状为 [batch_size, seq_len]
            padding_value=self.tokenizer.pad_token_id)  # 使用 pad_token_id 填充
        
        # 对 labels 进行填充对齐（用 IGNORE_INDEX 填充）
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        # 截断到模型最大长度限制
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        # 构建基础 batch 字典
        batch = dict(
            input_ids=input_ids,  # 填充后的输入 token ids
            labels=labels,  # 填充后的 labels
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),  # 生成注意力掩码（非 pad 位置为 True）
        )

        # 处理主图像数据
        if 'image' in instances[0]:  # 检查是否存在图像数据
            images = [instance['image'] for instance in instances]  # 提取所有图像张量
            
            # 特殊处理非配对图像（不进行堆叠）
            # 条件：所有图像形状相同 且 不是二维张量 且 样本数>1
            if all(x is not None and x.shape == images[0].shape and len(x)!=2 for x in images) and len(images) > 1:
                batch['images'] = torch.stack(images)  # 堆叠为 [B, C, H, W]
            else:
                batch['images'] = images  # 保持列表形式
            
        # 处理辅助图像数据
        if 'image_aux' in instances[0]:  # 检查是否存在辅助图像
            images = [instance['image_aux'] for instance in instances]
            # 当所有图像形状一致且数量>1时堆叠
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
                batch['images_aux'] = torch.stack(images)
            else:
                batch['images_aux'] = images  # 保持列表形式

        return batch  # 返回最终整理好的 batch 数据


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # 英文注释保留：Make dataset and collator for supervised fine-tuning.
    # 中文：构建监督微调所需的数据模块（数据集 + 整理器）
    
    # 实例化惰性加载数据集
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,  # 数据文件路径
        data_args=data_args)  # 数据配置参数
    
    # 实例化数据整理器
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # 返回供 Trainer 使用的数据模块
    return dict(
        train_dataset=train_dataset,  # 训练数据集
        eval_dataset=None,  # 无验证集
        data_collator=data_collator  # 使用的数据整理器
    )


def train(attn_implementation=None):  # 主训练函数，attn_implementation 用于指定注意力实现方式
    global local_rank  # 声明全局变量 local_rank（用于分布式训练）

    # 解析命令行参数到三个数据类
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank  # 获取当前进程的本地 rank
    
    # 根据训练精度设置计算 dtype
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 初始化量化配置参数
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:  # 4bit 或 8bit 量化配置
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},  # 指定加载设备
            load_in_4bit=training_args.bits == 4,  # 4bit 加载标志
            load_in_8bit=training_args.bits == 8,  # 8bit 加载标志
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],  # 跳过量化 mm_projector 模块
                llm_int8_threshold=6.0,  # 量化阈值
                llm_int8_has_fp16_weight=False,  # 不使用 fp16 权重
                bnb_4bit_compute_dtype=compute_dtype,  # 4bit 计算 dtype
                bnb_4bit_use_double_quant=training_args.double_quant,  # 是否使用双重量化
                bnb_4bit_quant_type=training_args.quant_type # 量化类型 {'fp4', 'nf4'}
            )
        ))

    # 根据模型名称选择不同的模型类进行加载
    if model_args.vision_tower is not None:  # 如果配置了视觉 tower
        # 根据模型名称选择对应的 MGM 模型类
        if "mistral" in  model_args.model_name_or_path.lower():
            model = MGMMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,  # 模型缓存目录
                attn_implementation=attn_implementation,  # 注意力实现方式
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),  # 模型 dtype
                **bnb_model_from_pretrained_args  # 量化相关参数
            )
        elif "mixtral" in  model_args.model_name_or_path.lower():
            model = MGMMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            # 为 Mixtral 的稀疏 MoE 块设置 ZeRO-3 优化
            from deepspeed.utils import set_z3_leaf_modules
            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "gemma" in  model_args.model_name_or_path.lower():
            model = MGMGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:  # 默认使用 LLaMA 模型
            model = MGMLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),   
                **bnb_model_from_pretrained_args
            )
    else:  # 无视觉 tower 时加载普通 LLaMA 模型
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False  # 禁用缓存以支持梯度检查点

    # 冻结主干网络参数
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # 准备低精度训练模型
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = compute_dtype  # 设置模型 dtype
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing  # 是否启用梯度检查点
        )

    # 梯度检查点相关设置
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  # 确保输入需要梯度
        else:
            # 通过前向钩子强制输出需要梯度
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # LoRA 适配器配置
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,  # LoRA 秩
            lora_alpha=training_args.lora_alpha,  # 缩放系数
            target_modules=find_all_linear_names(model),  # 目标模块列表
            lora_dropout=training_args.lora_dropout,  # dropout 率
            bias=training_args.lora_bias,  # 是否训练偏置项
            task_type="CAUSAL_LM",  # 任务类型
        )
        # 精度转换
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)  # 应用 LoRA

    # 分词器初始化（根据不同模型类型调整参数）
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"  # 填充方向
        )
    elif "gemma" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:  # 其他模型使用非快速分词器（避免特殊 token 问题）
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,  # 禁用快速分词
        )

    # 特殊 token 处理与对话模板设置
    if model_args.version == "v0":
        if tokenizer.pad_token is None:  # 添加 pad token
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token  # 使用 unk 作为 pad
    elif "gemma" in model_args.version:        
        # 设置 Gemma 对话模板
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["gemma"]
    elif "llama_3" in model_args.version:
        # 设置 Llama3 特殊 token 和对话模板
        if tokenizer.unk_token is None:
            tokenizer.unk_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3"]
    else:  # 默认使用 Vicuna v1 模板
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 视觉模块初始化（当配置了 vision_tower 时）
    if model_args.vision_tower is not None:
        # 初始化视觉模块
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp  # 是否使用完全分片数据并行
        )
        
        # 获取视觉 tower 并设置精度和设备
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # 配置数据参数中的处理器
        data_args.image_processor = copy.deepcopy(vision_tower.image_processor)
        data_args.video_processor = copy.deepcopy(vision_tower.image_processor)  # 视频处理复用图像处理器
        data_args.is_multimodal = True  # 标记为多模态数据

        # 将数据参数注入模型配置
        model.config.image_grid = data_args.image_grid  # 图像网格参数
        model.config.image_global = data_args.image_global  # 是否使用全局图像
        model.config.image_aspect_ratio = data_args.image_aspect_ratio  # 图像宽高比处理方式
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints  # 网格定位点
        model.config.tokenizer_padding_side = tokenizer.padding_side  # 分词器填充方向
        model.config.tokenizer_model_max_length = tokenizer.model_max_length  # 分词器最大长度

        # 配置是否微调 mm_projector
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:  # 仅微调 mm_projector
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        # 配置是否冻结 mm_projector
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # 量化模式下设置 mm_projector 的 dtype
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        # 优化视觉 tower 的部分层
        if model_args.optimize_vision_tower:
            print('Optimize last 1/2 layers in vision tower')
            total_num = len(vision_tower.vision_tower.vision_model.encoder.layers)
            for _idx in range(total_num//2, total_num):  # 仅优化后半部分层
                vision_tower.vision_tower.vision_model.encoder.layers[_idx].requires_grad_(True)

        # 配置图像特殊 token 的使用
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr  # 设置投影层学习率
        training_args.use_im_start_end = model_args.mm_use_im_start_end  # 传递参数到训练配置
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token  # 是否使用图像 patch token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)  # 初始化视觉 tokenizer

    # 辅助视觉 tower 处理
    if model_args.vision_tower_aux is not None:
        vision_tower_aux = model.get_vision_tower_aux()  # 获取辅助视觉 tower
        vision_tower_aux.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # 校验图像处理器参数一致性
        assert data_args.image_processor.image_mean == vision_tower_aux.config['preprocess_cfg']['mean'] \
                and data_args.image_processor.image_std == vision_tower_aux.config['preprocess_cfg']['std'], \
                'image processor should be the same'
        
        # 优化辅助视觉 tower 的最后一层
        if model_args.optimize_vision_tower_aux:
            print('Optimize last layer of each block in vision tower aux')
            for _idx in range(len(vision_tower_aux.vision_stages)):
                vision_tower_aux.vision_stages[_idx].blocks[-1].requires_grad_(True)
        
        # 配置原始图像尺寸和辅助尺寸
        data_args.image_size_raw = data_args.image_processor.crop_size.copy()  # 保存原始裁剪尺寸
        model_args.image_size_aux = data_args.image_size_aux  # 辅助尺寸参数
        # 更新图像处理器参数为辅助尺寸
        data_args.image_processor.crop_size['height'] = data_args.image_size_aux
        data_args.image_processor.crop_size['width'] = data_args.image_size_aux
        data_args.image_processor.size['shortest_edge'] = data_args.image_size_aux

        model.get_model().initialize_uni_modules(model_args)  # 初始化统一模块

    # 处理量化模型中的特定层精度
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):  # LoRA 层精度转换
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:  # 归一化层保持 float32
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:  # 词嵌入层精度处理
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # 构建数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    # 初始化训练器
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module  # 注入数据集和整理器
    )

    # 断点续训逻辑
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)  # 从检查点恢复训练
    else:
        trainer.train()  # 开始训练
    trainer.save_state()  # 保存训练状态

    model.config.use_cache = True  # 训练完成后启用缓存

    # LoRA 模型保存处理
    if training_args.lora_enable:
        # 获取 LoRA 状态字典
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        # 获取非 LoRA 参数
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:  # 主进程保存
            model.config.save_pretrained(training_args.output_dir)  # 保存配置
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)  # 保存模型
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))  # 保存非 LoRA 参数
    else:  # 普通模型保存
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()  # 脚本入口
