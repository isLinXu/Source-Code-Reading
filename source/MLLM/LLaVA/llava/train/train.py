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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")                  # 模型的名称或路径。默认为"facebook/opt-125m"。
    version: Optional[str] = field(default="v0")                                            # 模型的版本。默认为"v0"。
    freeze_backbone: bool = field(default=False)                                            # 是否冻结模型的主干部分。默认为False。
    tune_mm_mlp_adapter: bool = field(default=False)                                        # 是否调整多模态MLP适配器。默认为False。
    vision_tower: Optional[str] = field(default=None)                                       # 视觉塔的名称或路径。默认为None。
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer # 选择的多模态视觉层。默认为-1，表示最后一层。
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)                            # 预训练的多模态MLP适配器的路径。默认为None。
    mm_projector_type: Optional[str] = field(default='linear')                              # 多模态投影器的类型。默认为'linear'。
    mm_use_im_start_end: bool = field(default=False)                                        # 是否使用图像的开始和结束标记。默认为False。
    mm_use_im_patch_token: bool = field(default=True)                                       # 是否使用图像的分片标记。默认为True。
    mm_patch_merge_type: Optional[str] = field(default='flat')                              # 图像分片合并的方式。默认为'flat'。
    mm_vision_select_feature: Optional[str] = field(default="patch")                        # 选择的视觉特征类型。默认为"patch"。


@dataclass
class DataArguments:
    # # 定义数据路径，用于指向训练数据的文件路
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # 是否采用惰性预处理，惰性预处理意味着在数据加载时延迟预处理步骤
    lazy_preprocess: bool = False
    # 是否为多模态数据，多模态数据通常包含文本和图像等多种类型
    is_multimodal: bool = False
    # 图像文件夹路径，当数据为多模态时指定图像文件夹的位置
    image_folder: Optional[str] = field(default=None)
    # 图像的纵横比，默认为'square'，即正方形
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    继承自transformers.TrainingArguments的TrainingArguments类，
    用于扩展训练时的参数配置。新增了与多模态模型相关的优化器、
    列移除、冻结MLP适配器、MPT注意力实现、模型最大长度、
    量化相关设置、LoRA相关参数以及按模态长度分组等配置项。
    """
    # 缓存目录的路径，用于存储缓存数据。
    cache_dir: Optional[str] = field(default=None)
    # 优化器类型，默认使用adamw_torch。
    optim: str = field(default="adamw_torch")
    # 是否移除未使用的列，默认不移除。
    remove_unused_columns: bool = field(default=False)
    # 是否冻结多模态MLP适配器参数，默认不冻结。
    freeze_mm_mlp_adapter: bool = field(default=False)
    # MPT注意力实现方式，默认使用triton实现。
    mpt_attn_impl: Optional[str] = field(default="triton")
    # 模型的最大序列长度。序列将在右侧填充（并可能截断）以达到此长度
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # 是否通过二次量化压缩量化统计信息，默认启用。
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    # 量化的数据类型，默认使用nf4。可选`fp4`或`nf4`
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    # 使用的位数，默认为16位。
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    # 是否启用LoRA微调，默认不启用
    lora_enable: bool = False
    # LoRA的中间维度，默认为64
    lora_r: int = 64
    # LoRA的Alpha参数，默认为16
    lora_alpha: int = 16
    # LoRA的Dropout比例，默认为0.05
    lora_dropout: float = 0.05
    # LoRA权重的保存路径，默认为空
    lora_weight_path: str = ""
    # LoRA的偏置类型，默认不使用偏置
    lora_bias: str = "none"
    # 多模态投影器的学习率，默认为None
    mm_projector_lr: Optional[float] = None
    # 是否按模态长度分组进行训练，默认不分组
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    """
    根据参数是否使用了DeepSpeed的Zero优化，以不同方式处理参数。

    当参数具有`ds_id`属性时，表明它是一个通过Zero优化的参数。在这种情况下，
    函数会检查参数的状态是否为`NOT_AVAILABLE`，如果不是且`ignore_status`为`False`，
    则会记录警告信息。然后，使用`zero.GatheredParameters`上下文管理器收集参数，
    并将参数的数据从GPU移动到CPU并创建一个副本。如果参数没有使用Zero优化，
    则直接将参数的数据从GPU移动到CPU并创建一个副本。

    参数:
        param: 需要处理的参数，可以是一个PyTorch张量或一个包含`ds_id`属性的Zero参数对象。
        ignore_status: 是否忽略参数的状态，即使状态为`NOT_AVAILABLE`也不记录警告，默认为`False`。
        name: 参数的名称，用于在记录警告时提供更多信息，默认为`None`。

    返回:
        处理后的参数副本，始终是位于CPU上的PyTorch张量。
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        # 如果参数状态为NOT_AVAILABLE且不忽略状态，则记录警告
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        # 使用GatheredParameters收集参数，并操作参数数据
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        # 如果参数没有使用Zero优化，则直接操作参数数据
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """
    根据指定的bias参数，从模型参数中筛选出相应的参数集，并对这些参数应用maybe_zero_3处理。

    Args:
        named_params (iterable): 模型的命名参数，即参数带有名称的列表。
        bias (str): 控制是否包含bias参数的策略，可选值有'none'、'all'和'lora_only'。

    Returns:
        dict: 处理后的参数字典，键为参数名称，值为应用了maybe_zero_3处理的参数。

    Raises:
        NotImplementedError: 如果bias参数的值不属于'none'、'all'或'lora_only'，则抛出此异常。
    """
    # 根据bias参数的值，决定如何筛选和处理参数
    if bias == "none":
        # 当bias为'none'时，只选择包含'lora_'的参数
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        # 当bias为'all'时，选择包含'lora_'或'bias'的参数
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        # 当bias为'lora_only'时，选择包含'lora_'的参数，并根据特定逻辑选择bias参数
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                # 通过'lora_'关键字提取可能关联的bias名称，并收集到集合中
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        # 从maybe_lora_bias中选择与lora_bias_names集合中的名称匹配的bias参数
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        # 如果bias的值不属于已知选项，抛出异常
        raise NotImplementedError
    # 对筛选出的参数应用maybe_zero_3处理，忽略其状态
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    获取非LoRA参数的状态，可选地仅获取需要梯度的参数，并执行maybe_zero_3操作。

    Args:
        named_params (iterable): 模型的命名参数，通常由模型的named_parameters方法获得。
        require_grad_only (bool, optional): 是否仅获取需要梯度的参数。 Defaults to True。

    Returns:
        dict: 处理后的非LoRA参数状态字典。
    """
    # 过滤掉名称中包含'LoRA'的参数，仅保留其他参数
    to_return = {k: t for k, t in named_params if "lora_" not in k}

    # 如果设置了require_grad_only，进一步过滤以仅保留需要梯度的参数
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}

    # 对保留的参数执行maybe_zero_3操作，并转换到CPU上
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """
    获取与指定关键字匹配的参数，并将这些参数转换为特定状态。

    此函数旨在从一个包含命名参数的字典中，筛选出键名中包含指定关键字的参数，并对这些参数应用特定的状态转换函数`maybe_zero_3`。这个转换函数可能会将参数的状态置为零，而且在转换过程中忽略某些状态检查。最终，所有经过筛选和转换的参数将被移至CPU上。

    参数:
    - named_params: 一个包含命名参数的字典，通常来自模型的`named_parameters()`方法。
    - keys_to_match: 一个字符串列表，定义了要与参数键名匹配的关键字。这是用于过滤哪些参数应该被转换的标准。

    返回值:
    - 一个字典，包含经过筛选和状态转换的参数。这些参数被移至CPU上，以便于后续处理或保存。
    """
    # 筛选匹配关键字的参数
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}

    # 对筛选出的参数应用特定的状态转换，忽略状态检查，并移至CPU
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    """
    遍历模型的所有模块，寻找并返回所有线性层（torch.nn.Linear）的名称。

    此函数用于识别模型中所有线性层的名称，这些名称通常用于在训练或微调过程中
    应用特定的技术（如LoRA）来减少内存使用或加速训练过程。函数还会特别排除多模态
    关键词相关的模块，以确保只获取与纯视觉或语言模型相关的线性层名称。

    参数:
    - model: torch.nn.Module类型的对象，代表一个经过初始化的神经网络模型。

    返回值:
    - 一个字符串列表，包含模型中所有线性层的名称。
    """
    # 定义线性层的类
    cls = torch.nn.Linear
    # 初始化一个用于存储线性层名称的集合
    lora_module_names = set()
    # 定义多模态关键词，用于排除多模态相关的模块
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果模块名称包含多模态关键词，则跳过该模块
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        # 如果模块是线性层实例，则处理其名称
        if isinstance(module, cls):
            # 将模块名称按'.'分割，处理成列表
            names = name.split('.')
            # 如果列表只有一个元素，直接添加到集合中，否则添加最后一个元素
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # 如果集合中包含'lm_head'，则移除它（在16-bit模式下需要）
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    # 返回线性层名称的列表
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    """
    安全地保存HuggingFace Trainer模型。

    根据Trainer的参数决定如何保存模型，包括处理Adapter权重、DeepSpeed以及标准模型保存。

    参数:
    - trainer: 包含模型及其参数的HuggingFace Trainer实例。
    - output_dir: 保存模型权重的输出目录。
    """
    # 如果设置了使用Tune-MM-MLP-Adapter参数，则仅保存Adapter权重。
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        # 如果使用im_start_end，则需要保存额外的权重。
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        # 收集要保存的权重。
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        # 保存模型配置。
        trainer.model.config.save_pretrained(output_dir)
        # 确定当前文件夹名称和父目录。
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        # 在特定条件下创建目录并保存权重
        # 当local_rank为0或-1时，执行以下操作
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            # 如果current_folder以'checkpoint-'开头，表明这是一个检查点文件夹
            if current_folder.startswith('checkpoint-'):
                # 构建mm_projector文件夹的路径
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                # 确保mm_projector文件夹存在，如果不存在则创建
                os.makedirs(mm_projector_folder, exist_ok=True)
                # 将权重保存到mm_projector文件夹中，文件名基于current_folder
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                # 如果current_folder不符合checkpoint格式，则直接将权重保存到output_dir中
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        # 函数执行完毕后返回
        return

    # 如果使用了deepspeed，则需要同步所有GPU上的操作
    if trainer.deepspeed:
        torch.cuda.synchronize()
        # 保存模型到指定目录
        trainer.save_model(output_dir)
        # 完成保存后直接返回
        return

    # 获取模型的当前状态字典
    state_dict = trainer.model.state_dict()
    # 如果根据配置需要保存模型配置，则保存配置
    if trainer.args.should_save:
        # # 将状态字典中的所有张量移动到CPU上以释放GPU内存
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        # 删除原始状态字典以进一步释放GPU内存
        del state_dict
        # 调用内部保存函数以保存模型状态到指定目录
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    """
    动态调整词 tokenizer 和嵌入层的大小。

    本函数旨在添加特殊 tokens 并调整模型的嵌入层大小，以适应新的 tokenizer。
    注意：这是一个未优化的版本，可能会导致你的嵌入大小无法被 64 整除。

    参数:
    - special_tokens_dict: 包含需要添加的特殊 tokens 的字典。
    - tokenizer: 预训练的 tokenizer 对象。
    - model: 预训练的模型对象。

    返回:
    无返回值，但会原地修改 tokenizer 和 model。
    """
    # 添加特殊 tokens 到 tokenizer，并获取新添加的 token 数量
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    # 调整模型的嵌入层大小，以适应 tokenizer 的新大小
    model.resize_token_embeddings(len(tokenizer))

    # 如果有新的 tokens 被添加，计算输入和输出嵌入的平均值，并将新 tokens 的嵌入初始化为这些
    if num_new_tokens > 0:
        # 获取输入和输出嵌入
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 计算输入和输出嵌入的平均值，排除新添加的 tokens
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        # 将新 tokens 的嵌入初始化为平均值
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    """
    对一系列字符串进行分词处理。

    参数:
        strings: 需要分词的字符串序列。
        tokenizer: 用于分词的预训练分词器。

    返回:
        包含分词后的输入ID、标签及其长度的字典。
    """
    # 使用tokenizer对每个字符串进行分词，并设置相关参数
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    # 提取每个分词结果的第一个input_ids作为模型输入
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    # 计算每个input_ids的有效长度（去除padding）
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    # 将分词结果组织成字典形式并返回
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    """
    将目标序列中特定部分的元素设置为IGNORE_INDEX，以掩盖这些部分在训练过程中的影响。

    :param target: 待处理的目标序列，通常是一个包含序列标签的列表或数组。
    :param tokenized_lens: 一个列表，包含每个说话者发言的标记化后长度。
    :param speakers: 一个列表，包含每个发言的说话者类型（"human"或非"human"）。
    """
    # cur_idx = 0
    # 初始化当前索引为对话中第一个说话者的发言结束位置
    cur_idx = tokenized_lens[0]
    # 更新tokenized_lens，去掉已经处理的第一个元素
    tokenized_lens = tokenized_lens[1:]
    # 将目标序列中对话的第一个说话者发言部分设置为IGNORE_INDEX
    target[:cur_idx] = IGNORE_INDEX
    # 遍历剩余的每个说话者发言长度和说话者类型
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        # 如果当前说话者是人类，则将其发言部分在目标序列中设置为IGNORE_INDEX
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        # 更新当前索引到下一个说话者发言的开始位置
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    """
    在每轮对话中添加说话者和开始/结束信号。

    此函数处理对话数据，在每轮对话中添加说话者信息和对话开始/结束信号。
    主要用于将对话数据格式化成模型训练或分析所需的特定格式。

    参数:
    - header: 对话的初始内容或前缀，用于初始化对话上下文。
    - source: 包含每轮对话消息的列表或可迭代对象，每条消息通常包括说话者信息和对话内容。
    - get_conversation: 是否返回完整的对话内容。如果为 False，则仅返回处理后的对话信号序列。

    返回:
    - conversation: 处理并格式化的完整对话内容，包括初始化内容和所有对话信号。
    """
    # 定义每轮对话的开始和结束信号
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    # 初始化对话内容
    conversation = header
    # 遍历源对话中的每一句话，处理每轮对话
    for sentence in source:
        # 获取当前句子的说话者，并根据情况处理说话者角色
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        # 添加说话者和对话开始/结束信号到当前句子
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        # 如果需要返回完整的对话内容，则将当前句子追加到对话中
        if get_conversation:
            conversation += sentence["value"]

    # 在对话末尾添加开始信号，为下一轮对话处理做准备
    conversation += BEGIN_SIGNAL
    # 返回处理后的对话内容
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    """
    对多模态数据进行预处理。

    如果数据不是多模态的，则直接返回原始数据。
    对于多模态数据，移除默认图片标记，添加适当的开始和结束标记，并处理特定的对话版本。

    参数:
        sources: Sequence[str] - 原始数据源序列。
        data_args: DataArguments - 数据处理参数，包括是否使用多模态和使用的具体配置。

    返回:
        Dict - 预处理后的数据。
    """
    # 判断是否为多模态数据
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        # 如果不是多模态数据，直接返回原始数据源
        return sources

    # 对每个数据源进行处理
    for source in sources:
        # 对每个句子进行处理
        for sentence in source:
            # 如果句子中包含默认图片标记，则进行处理，将其替换为默认图片标记
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                # 移除默认图片标记并整理文本格式
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                # 对特定版本的对话进行特殊处理
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            # 设置需要替换的标记
            replace_token = DEFAULT_IMAGE_TOKEN
            # 如果配置中使用了图片开始和结束标记，则进行替换
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            # 替换默认图片标记为新的标记
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    # 返回预处理后的数据源
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    对LLaMA 2模型的数据进行预处理。

    该函数处理源数据，对文本进行分词，并准备输入和目标序列以供模型训练。

    参数:
    - sources: 会话数据源列表。
    - tokenizer: 预训练的分词器，用于将文本转换为token序列。
    - has_image: 输入是否包含图像。

    返回:
    - 包含输入和目标序列的字典。
    """
    # 初始化默认的对话模板
    conv = conversation_lib.default_conversation.copy()
    # 定义对话参与者的角色
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    # 应用提示模板
    conversations = []
    for i, source in enumerate(sources):
        # 如果第一个不是来自人类，则跳过
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # 确保对话按角色交替
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # 对话进行分词，并返回对应的input_ids
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # 复制输入id作为目标
    targets = input_ids.clone()

    # 确保对话分隔符样式为LLaMA 2
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    # 对目标进行掩码
    sep = "[/INST] "
    # 遍历对话和目标序列，处理每个对话回合
    for conversation, target in zip(conversations, targets):
        # 计算目标序列中非填充token的总数
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # 根据分隔符分割对话为多个回合
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        # 将目标序列的前cur_len个token设置为IGNORE_INDEX
        target[:cur_len] = IGNORE_INDEX
        # 遍历每个对话回合
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            # 根据指令和回复的分隔符分割对话回合为两部分
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # 根据是否有图片，计算每个回合和指令的长度
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # 将当前回合的指令部分在目标序列中设置为IGNORE_INDEX
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            # 更新当前长度
            cur_len += round_len
        # 将当前长度之后的目标序列设置为IGNORE_INDEX
        target[cur_len:] = IGNORE_INDEX

        # 检查分词不匹配的情况
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 如果长度不匹配，将目标数组中的所有值设置为忽略索引，并打印警告信息
                # 这是为了处理在文本 tokenization 过程中可能出现的不一致情况
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # 将准备好的数据作为字典返回
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    对输入的对话源进行预处理，使其适合模型输入。

    参数:
        sources: 一个对话源列表，每个源是一个字典列表，包含 'from' 和 'value' 键。
        tokenizer: 用于分词的分词器。
        has_image: 对话源中是否包含图片，默认为 False。

    返回:
        包含预处理后的对话提示的字典。
    """
    # 初始化默认的对话模板
    conv = conversation_lib.default_conversation.copy()
    # 定义人类和 GPT 的角色，对应对话模板中的角色
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    # 应用提示模板
    conversations = []
    for i, source in enumerate(sources):
        # 如果第一条消息不是来自人类，则跳过
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        # 清除对话中的现有消息
        conv.messages = []
        for j, sentence in enumerate(source):
            # 将每条消息的发送者映射到对话模板中的相应角色
            role = roles[sentence["from"]]
            # 确保消息发送者的顺序交替出现
            assert role == conv.roles[j % 2], f"{i}"
            # 将消息添加到对话中
            conv.append_message(role, sentence["value"])
        # 将编译后的对话提示添加到列表中
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # 根据是否包含图片，准备输入的ID序列
    if has_image:
        # 如果包含图片，使用tokenizer_image_token函数处理每个对话，并用torch.stack整合
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        # 如果不包含图片，直接使用tokenizer处理对话，进行填充、截断和返回输入ID
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # 复制输入ID作为目标
    targets = input_ids.clone()

    # 确认对话的分隔风格为指定的样式
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    # 掩码目标
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        # 计算对话中非填充部分的总长度
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 按照分隔符分割对话为多个回合
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        # 对每个回合进行处理
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            # 分割每个回合为指令和响应
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            # 计算每个回合和指令的长度
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # 根据tokenizer版本和设置调整长度
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            # 掩码指令部分的目标
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        # 对当前长度之后的部分进行掩码
        target[cur_len:] = IGNORE_INDEX

        # 检查处理后的长度是否符合预期
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    # 返回处理后的输入ID和目标ID
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    对输入的数据进行预处理以适应MPT模型训练。

    参数:
        sources: 包含对话历史记录的列表。
        tokenizer: 用于文本分词的预训练分词器实例。
        has_image: 对话是否包含图像，默认为False。

    返回:
        包含分词后的输入ID和对应的标签目标的字典。

    本函数主要完成三个任务：应用提示模板、对话分词以及目标掩码。
    """
    # 初始化默认的对话模板
    conv = conversation_lib.default_conversation.copy()
    # 定义对话双方的角色，以便后续识别不同角色的句子
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    # 应用提示模板
    conversations = []
    for i, source in enumerate(sources):
        # 如果第一条消息不是来自人类，则跳过
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # 确保当前句子的角色与对话模板定义的角色一致
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # 对话分词
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # 复制输入ID作为目标
    targets = input_ids.clone()
    # 确认对话分隔符样式为MPT
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    # 设置分隔符，用于后续分割对话
    sep = conv.sep + conv.roles[1]
    # 遍历每一对对话和目标序列
    for conversation, target in zip(conversations, targets):
        # 计算目标序列中有效token的总数
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # 将对话按分隔符分割成多轮对话
        rounds = conversation.split(conv.sep)
        # 初始化重新组合的对话，从系统消息、用户消息和GPT回复开始
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        # 继续组合后续的对话轮次，每轮包括用户消息和GPT回复
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2])) # user + gpt

        # 初始化当前长度计数
        cur_len = 0
        # 将目标序列的前cur_len个token设置为IGNORE_INDEX
        target[:cur_len] = IGNORE_INDEX
        # 遍历重新组合的每轮对话
        for i, rou in enumerate(re_rounds):
            # 如果对话为空，停止处理
            if rou == "":
                break

            # 分割每轮对话成两部分：用户消息和GPT回复
            parts = rou.split(sep)
            # 如果分割结果不等于2，停止处理
            if len(parts) != 2:
                break
            parts[0] += sep

            # 计算每轮对话和指令的长度
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # 对于非首轮对话，如果tokenizer是legacy模式且版本大于0.14，则长度加1
            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            # 将目标序列中指令部分设置为IGNORE_INDEX
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            # 更新当前长度计数
            cur_len += round_len

        # 将剩余的目标序列设置为IGNORE_INDEX
        target[cur_len:] = IGNORE_INDEX

        # 如果当前长度小于tokenizer的最大长度，且与总长度不一致，则打印警告并忽略该样本
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # 返回处理后的数据
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    给定一个源列表，每个源是一个对话列表。此转换执行以下操作：
    1. 在每个句子的开头添加信号 '### '，并在结尾添加信号 '\n'；
    2. 将对话连接在一起；
    3. 对连接后的对话进行分词；
    4. 深拷贝作为目标，并用 IGNORE_INDEX 掩码人类词语。

    参数:
    - sources (Sequence[str]): 包含多个对话的列表。
    - tokenizer (transformers.PreTrainedTokenizer): 用于分词的预训练分词器。
    - has_image (bool, optional): 是否包含图像，默认为 False。

    返回:
    - Dict: 包含分词后的输入 ID 和标签的字典。
    """
    # 根据默认对话的分隔符样式和版本确定预处理方法
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)

    # add end signal and concatenate together
    # 添加结束信号并连接对话
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    # 将对话进行分词
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    # 根据是否有图像内容，选择不同的方式对对话进行token化处理
    if has_image:
        # 如果有图像内容，则使用tokenizer_image_token函数对每个prompt进行token化处理，并返回pt格式的tensor
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        # 如果没有图像内容，则使用_tokenize_fn函数对整个对话进行token化处理
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        # 提取token化后的输入ID
        input_ids = conversations_tokenized["input_ids"]

    # 深拷贝input_ids列表，以避免修改原始数据
    targets = copy.deepcopy(input_ids)
    # 遍历每个目标序列和其对应的源序列
    for target, source in zip(targets, sources):
        # 根据是否有图像内容，选择不同的方式计算token化后的长度
        if has_image:
            # 如果有图像内容，则使用get_tokenize_len函数计算长度
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            # 如果没有图像内容，则使用_tokenize_fn函数计算长度
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        # 提取每个句子的说话者信息
        speakers = [sentence["from"] for sentence in source]
        # 使用计算出的长度和说话者信息，对目标序列进行掩码处理
        _mask_targets(target, tokenized_lens, speakers)

    # 返回处理后的输入ID和掩码后的目标序列
    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    """用于监督微调的数据集。

    该类继承自 Dataset，用于创建监督微调的数据集。
    它懒加载和处理数据，适用于无法一次性全部加载到内存中的大型数据集。

    属性:
        tokenizer (transformers.PreTrainedTokenizer): 文本处理的分词器。
        list_data_dict (list): 包含数据字典的列表。
        data_args (DataArguments): 数据相关的参数。
    """
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        """初始化 LazySupervisedDataset 实例。

        参数:
            data_path (str): 数据文件路径。
            tokenizer (transformers.PreTrainedTokenizer): 文本处理的分词器。
            data_args (DataArguments): 数据相关的参数。
        """
        super(LazySupervisedDataset, self).__init__()
        # 从文件中加载数据，并将其转换为列表形式
        list_data_dict = json.load(open(data_path, "r"))
        # 打印输入格式化消息
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        """返回数据集的大小。

        返回:
            int: 数据集中样本的数量。
        """
        return len(self.list_data_dict)

    @property
    def lengths(self):
        """计算数据集中所有样本的长度。

        该属性计算数据集中每个样本的长度，包括文本和图像数据。
        如果样本包含图像数据，则添加一个固定长度值。

        返回:
            list: 包含每个样本长度的列表。
        """
        length_list = []
        for sample in self.list_data_dict:
            # 计算图像数据的 token 长度，如果存在则固定为 128
            img_tokens = 128 if 'image' in sample else 0
            # 计算文本数据的总长度
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        """
        计算数据集中每个样本的对话文本长度。
        对于包含图像的样本，直接返回对话文本的长度。
        对于不包含图像的样本，将对话文本的长度取负值。
        """
        length_list = []
        for sample in self.list_data_dict:
            # 计算样本中所有对话文本的总长度
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # 根据样本是否包含图像调整长度的符号
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        获取指定索引处样本的处理后的数据字典。
        如果样本包含图像，根据图像处理设置进行处理。
        如果模型是多模态的，对话文本将进行预处理以输入多模态模型。
        """
        # 获取列表中的数据字典
        sources = self.list_data_dict[i]
        # 如果索引是整数，则将sources包装成一个列表
        if isinstance(i, int):
            sources = [sources]
        # 断言sources列表长度为1，否则提示出错
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # 如果数据字典中包含'image'键，则处理图像数据
        if 'image' in sources[0]:
            # 获取图像文件名、图像文件夹路径和图像处理器
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            # 打开图像并转换为RGB模式
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

            # 如果图像处理设置为'pad'，则将图像扩展为方形并填充到指定大小
            if self.data_args.image_aspect_ratio == 'pad':
                # 将图像扩展为方形并填充到指定大小
                def expand2square(pil_img, background_color):
                    """
                    将给定的PIL图像扩展为正方形，通过在图像的边缘填充背景色来实现。

                    参数:
                    pil_img (PIL.Image): 需要扩展的图像。
                    background_color (tuple): 用于填充背景的颜色值，格式为RGB。

                    返回:
                    PIL.Image: 扩展后的正方形图像。
                    """
                    width, height = pil_img.size  # 获取图像的宽度和高度
                    if width == height: # 如果图像已经是正方形，则直接返回原图像
                        return pil_img
                    elif width > height:    # 如果宽度大于高度
                        # 创建一个新的正方形图像，大小为宽度，背景色为指定颜色
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        # 将原图像粘贴到新图像的中心位置
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        # 创建一个新的正方形图像，大小为高度，背景色为指定颜色
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        # 将原图像粘贴到新图像的中心位置
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # 对多模态数据进行预处理
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            # 如果没有图像数据，直接复制对话内容
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # 对数据字典进行预处理
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))

        # 如果索引是整数，则提取第一个输入ID和标签
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        # 如果数据中包含图像，则添加图像数据到数据字典
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # 如果数据中没有图像，但模型是多模态的，则添加全零的图像张量到数据字典
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        # 返回处理后的数据字典
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    """
    用于监督微调的数据收集器。

    该类负责将输入数据实例收集成适合监督学习的批量格式。
    它主要处理输入ID和标签的填充，并创建注意力掩码。

    属性:
        tokenizer: transformers.PreTrainedTokenizer 的实例，用于文本的编码和解码。
    """
    # 用于处理文本数据的 Tokenizer 实例
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        调用方法，用于将实例收集成一个批量。

        该方法接收一个包含 'input_ids' 和 'labels' 的实例字典序列，
        并将其收集成带有填充的批量格式，适用于模型输入。

        参数:
            instances: 包含 'input_ids' 和 'labels' 的字典序列。

        返回:
            一个字典，包含批量的 'input_ids'、'labels' 和 'attention_mask'。
            如果实例中包含图像数据，也会包含 'images'。
        """
        # 从每个实例中提取 input_ids 和 labels
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        # 对 input_ids 和 labels 进行填充，使其在每个批量内长度一致
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        # 将 input_ids 和 labels 截断到模型的最大长度
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        # 创建包含 input_ids、labels 和 attention_mask 的批量字典
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 处理实例中的图像数据
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # 如果所有图像都存在且形状相同，则将图像堆叠成一个张量
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    """
    创建用于监督微调的数据集和数据整理器。

    该函数创建一个适用于监督微调任务的数据集和数据整理器。它使用提供的分词器处理数据集，
    确保数据格式正确以供训练。函数返回一个字典，包含训练数据集、评估数据集（此处为None，
    因为没有提供评估数据集）和数据整理器。

    参数:
    - tokenizer (transformers.PreTrainedTokenizer): 用于处理文本数据的分词器。
      它应与用于微调的模型兼容。
    - data_args: 数据集的参数或配置。这包括数据路径和需要应用的任何预处理步骤的信息。

    返回:
    - Dict: 包含以下键的字典：
        * 'train_dataset': 处理后的训练数据集。
        * 'eval_dataset': 评估数据集，此处为None。
        * 'data_collator': 负责批量和填充数据集的数据整理器。
    """
    # 使用提供的分词器和数据参数创建训练数据集
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)

    # 创建数据整理器，用于批量和填充数据集
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # 返回组件作为字典
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    """
    训练模型的主要函数。

    Args:
        attn_implementation (Optional): 注意力实现的类型。
    """
    global local_rank
    # 解析命令行参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    # 解析命令行参数并将其转换为相应的数据类实例
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 获取排名，用于分布式训练
    local_rank = training_args.local_rank

    # 根据训练参数中的fp16和bf16设置，确定模型计算使用的数据类型
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 初始化BitsAndBytes相关的模型加载参数字典
    bnb_model_from_pretrained_args = {}

    # 如果训练参数中指定的位数为4位或8位，则进行相应的量化配置
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        # 更新模型加载参数，包括设备映射、量化加载配置以及量化配置的详细参数
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # 如果设置了视觉塔（vision_tower），表明模型可能需要处理视觉信息
    if model_args.vision_tower is not None:
        # 如果模型名称或路径中包含'mpt'，则使用MptForCausalLM模型
        if 'mpt' in model_args.model_name_or_path:
            # 从预训练模型中加载配置，信任远程代码以支持可能的自定义实现
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            # 根据训练参数设置注意力实现方式
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            # 使用配置好的参数从预训练模型中加载MptForCausalLM模型
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            # 如果模型名称或路径中不包含'mpt'，则使用LlamaForCausalLM模型
            # 从预训练模型中加载LlamaForCausalLM模型，根据训练参数配置注意力方式
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        # 如果没有设置视觉塔，使用标准的LlamaForCausalLM模型
        # 从预训练模型中加载LlamaForCausalLM模型，同样根据训练参数配置注意力实现方式和数据类型
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    # 禁用缓存的使用，以防止在训练过程中出现意外的行为
    model.config.use_cache = False

    # 如果设置了冻结模型主干的参数，则禁止模型主干的梯度计算，以冻结其参数
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # 如果训练参数中的位数为4或8位，则导入准备模型进行k位训练的函数，并配置模型
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        # 准备模型进行k位训练，根据训练参数配置模型
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 如果训练参数中设置了梯度检查点，则根据模型属性启用输入梯度计算
    if training_args.gradient_checkpointing:
        # 如果模型有启用输入梯度计算的方法，则调用该方法
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # 如果模型没有该方法，则定义一个前向钩子函数，使输入梯度计算生效，并注册到模型的输入嵌入层
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 如果启用LoRA，则添加LoRA适配器
    # 检查是否启用了LoRA（低秩适应）进行训练
    if training_args.lora_enable:
        # 导入LoRA所需的模块
        from peft import LoraConfig, get_peft_model

        # 使用指定的参数初始化LoRA配置，
        # 其中r是LoRA的秩，alpha是LoRA的学习率，dropout是LoRA的dropout率，bias是LoRA是否使用偏置
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        # 检查模型训练的位数，并将模型转换为适当的精度
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        # 打印消息，表示开始添加LoRA适配器
        rank0_print("Adding LoRA adapters...")
        # 将LoRA应用于模型，并返回新的模型
        model = get_peft_model(model, lora_config)

    # 根据模型名称加载合适的分词器
    if 'mpt' in model_args.model_name_or_path:
        # 如果模型类型为'mpt'，则使用指定参数加载分词器
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        # 对于其他模型，使用额外的参数加载分词器
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    # 根据模型版本调整tokenizer的pad_token和模型配置
    if model_args.version == "v0":
        # 如果tokenizer没有pad_token，则通过smart_tokenizer_and_embedding_resize函数添加
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        # v0.5版本将tokenizer的pad_token设置为unk_token
        tokenizer.pad_token = tokenizer.unk_token
    else:
        # 其他版本将tokenizer的pad_token设置为unk_token，并根据版本设置对话模板
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 如果模型有视觉塔（vision_tower），则进行初始化和配置
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        # 将视觉塔的数据类型和设备设置为训练所需的配置
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        # 根据数据参数调整模型配置
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # 配置和调整多模态学习的参数
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # 如果使用4位或8位训练，则调整mm_projector的数据类型和设备
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        # 配置多模态学习的其他参数
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # 创建数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    # 创建训练器
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # 检查是否有检查点文件，如果有则从检查点恢复训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # 保存训练状态
    trainer.save_state()

    # 启用模型的缓存功能，以加速推理或减少计算资源消耗
    model.config.use_cache = True

    # 判断是否启用LoRA（Low-Rank Adaptation）训练
    if training_args.lora_enable:
        # 获取LoRA可训练参数状态字典，可能包括偏置项
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        # 获取非LoRA可训练参数状态字典
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        # 当local_rank为0或-1时，保存模型配置、LoRA可训练参数和非LoRA可训练参数
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # 如果未启用LoRA，使用安全方式保存模型，适用于分布式训练
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
