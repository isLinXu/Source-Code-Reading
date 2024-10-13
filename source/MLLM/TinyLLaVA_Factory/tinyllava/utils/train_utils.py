import logging
import os

import torch
from peft.tuners.lora import LoraLayer
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def make_inputs_require_grad(module, input, output):
    """
    为指定的模型模块设置输入输出的梯度要求
    :param module:  模型中的一个模块
    :param input:   输入数据
    :param output:  模块的输出数据
    :return:
    """
    output.requires_grad_(True) # 设置输出数据的requires_grad属性为True，使其可以计算梯度


def lora_kbit_setting(model, training_args):
    """
    根据训练参数设置LoRa模型的K比特配置
    :param model:         需要配置的模型
    :param training_args: 训练参数对象，包含是否使用bf16等配置
    :return:
    """
    for name, module in model.named_modules():                                  # 遍历模型中的所有模块
        if isinstance(module, LoraLayer):                                       # 如果模块是LoRaLayer类型
            if training_args.bf16:                                              # 如果训练参数中指定了使用bf16
                module = module.to(torch.bfloat16)                              # 将模块转换为bfloat16类型
        if 'norm' in name:                                                      # 如果模块名称中包含'norm'
            module = module.to(torch.float32)                                   # 将模块转换为float32类型
        if 'lm_head' in name or 'embed_tokens' in name:                         # 如果模块名称中包含'lm_head'或'embed_tokens'
            if hasattr(module, 'weight'):                                       # 如果模块有'weight'属性
                if training_args.bf16 and module.weight.dtype == torch.float32: # 如果训练参数中指定了使用bf16且权重数据类型为float32
                    module = module.to(torch.bfloat16)                          # 将模块转换为bfloat16类型
    
        
        
def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """
    根据bias参数的值，从named_params中提取特定的参数。

    :param named_params: 一个包含参数名称和值的字典
    :param bias: 一个字符串，用于决定如何处理bias参数，可选值为"none"、"all"、"lora_only"
    :return: 一个字典，包含根据bias参数选择的named_params中的项
    """
    # 如果bias为"none"，则只返回包含"lora_"的参数
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    # 如果bias为"all"，则返回包含"lora_"或"bias"的参数
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    # 如果bias为"lora_only"，则只返回包含"lora_"的参数，并且如果存在对应的bias参数，则一并返回
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
    # 对返回的字典中的每个值应用maybe_zero_3函数
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    获取非lora参数的状态，并根据需要返回可能需要置零的状态。

    :param named_params: 包含参数名称和参数张量的字典。
    :param require_grad_only: 布尔值，如果为True，则只返回需要梯度的参数。
    :return: 一个字典，包含筛选后的参数名称和对应的张量。
    """
    # 筛选出名称中不包含'lora_'的参数
    to_return = {k: t for k, t in named_params if "lora_" not in k}

    # 如果require_grad_only为True，则进一步筛选出需要梯度的参数
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}

    # 对筛选出的参数应用maybe_zero_3函数，并将结果转移到CPU上
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_state_maybe_zero_3(named_params, keys_to_match=[''], require_grad_only=True):
    """
    根据提供的键匹配条件，从命名参数中筛选出符合条件的参数状态。

    :param named_params: 包含命名参数的字典，键为参数名，值为参数张量。
    :param keys_to_match: 用于匹配参数名的键列表。
    :param require_grad_only: 如果为True，则只返回需要梯度的参数状态。
    :return: 符合条件的参数状态字典。
    """
    # 筛选出包含任意匹配键的参数
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}

    # 如果require_grad_only为True，则进一步筛选出需要梯度的参数
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}

    # 对筛选出的参数应用maybe_zero_3函数，并将结果转移到CPU上
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, skip_keywords=['connector', 'vision_tower']):
    """
    查找模型中所有线性层的名称，忽略包含特定关键字的层。

    :param model: 要检查的模型对象
    :param skip_keywords: 要忽略的层名称关键字列表，默认为['connector', 'vision_tower']
    :return: 包含所有线性层名称的列表
    """
    cls = torch.nn.Linear                           # 定义线性层类
    lora_module_names = set()                       # 创建一个集合用于存储线性层的名称
    skip_keywords = skip_keywords                   # 重定义跳过关键字的变量（虽然这行代码多余，但保留了原代码的结构）

    # 遍历模型中的所有模块
    for name, module in model.named_modules():
        # 如果模块名称包含跳过关键字，或者名称中包含'lm_head', 'output_layer', 'head'，则跳过该模块
        if any(skip_keyword in name for skip_keyword in skip_keywords) or 'lm_head' in name or 'output_layer' in name or 'head' in name:
            continue
        # 如果当前模块是线性层，则将其名称添加到集合中
        if isinstance(module, cls):
            names = name.split('.')             # 将模块名称按'.'分割成列表
            #lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)         # 将完整的模块名称添加到集合中
    # if 'lm_head' in lora_module_names:
    #    lora_module_names.remove('lm_head')
    return list(lora_module_names)              # 将集合转换为列表并返回
