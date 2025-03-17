# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Set, Tuple, Union

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from ..data import get_dataset, get_template_and_fix_tokenizer
from ..extras.misc import get_current_device
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer


if TYPE_CHECKING:
    from datasets import Dataset
    from peft import LoraModel
    from transformers import PreTrainedModel


def compare_model(model_a: "torch.nn.Module", model_b: "torch.nn.Module", diff_keys: Sequence[str] = []) -> None:
    """
    比较两个模型参数的差异
    diff_keys: 预期存在差异的参数名列表
    """
    state_dict_a = model_a.state_dict()  # 获取模型A的参数状态字典
    state_dict_b = model_b.state_dict()  # 获取模型B的参数状态字典
    assert set(state_dict_a.keys()) == set(state_dict_b.keys())  # 确保参数结构相同
    
    for name in state_dict_a.keys():
        if any(key in name for key in diff_keys):  # 检查是否为预期差异参数
            # 断言参数存在显著差异
            assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5) is False
        else:
            # 断言参数基本一致
            assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5) is True


def check_lora_model(model: "LoraModel") -> Tuple[Set[str], Set[str]]:
    """
    检查LoRA模型参数配置
    返回: (线性层模块集合, 额外可训练模块集合)
    """
    linear_modules, extra_modules = set(), set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):  # LoRA特定参数
            module_name = name.split(".lora_", maxsplit=1)[0].split(".")[-1]  # 提取基础层名称
            linear_modules.add(module_name)
            assert param.requires_grad is True  # 确保可训练
            assert param.dtype == torch.float32  # 确保精度为float32
        elif "modules_to_save" in name:  # 需要保存的额外模块
            module_name = name.split(".modules_to_save", maxsplit=1)[0].split(".")[-1]
            extra_modules.add(module_name)
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:  # 基础模型参数
            assert param.requires_grad is False  # 确保冻结
            assert param.dtype == torch.float16  # 确保半精度

    return linear_modules, extra_modules


def load_train_model(add_valuehead: bool = False, **kwargs) -> "PreTrainedModel":
    """加载可训练模型（支持添加value head）"""
    model_args, _, _, finetuning_args, _ = get_train_args(kwargs)  # 获取训练参数
    tokenizer = load_tokenizer(model_args)["tokenizer"]  # 加载分词器
    return load_model(
        tokenizer, 
        model_args, 
        finetuning_args, 
        is_trainable=True,  # 设置为可训练模式
        add_valuehead=add_valuehead  # 是否添加value head
    )


def load_infer_model(add_valuehead: bool = False, **kwargs) -> "PreTrainedModel":
    """加载推理模型（支持添加value head）"""
    model_args, _, finetuning_args, _ = get_infer_args(kwargs)  # 获取推理参数
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    return load_model(
        tokenizer,
        model_args,
        finetuning_args,
        is_trainable=False,  # 设置为不可训练模式
        add_valuehead=add_valuehead
    )


def load_reference_model(
    model_path: str,
    lora_path: Optional[str] = None,
    use_lora: bool = False,
    use_pissa: bool = False,
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> Union["PreTrainedModel", "LoraModel"]:
    """加载参考模型（支持LoRA/PiSSA）"""
    current_device = get_current_device()  # 获取当前设备
    
    if add_valuehead:  # 处理带value head的情况
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 半精度加载
            device_map=current_device
        )
        if not is_trainable:
            model.v_head = model.v_head.to(torch.float16)  # 推理模式保持半精度
        return model

    # 基础模型加载
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=current_device
    )
    
    if use_lora or use_pissa:  # 处理LoRA/PiSSA适配器
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            subfolder="pissa_init" if use_pissa else None,  # PiSSA特殊子目录
            is_trainable=is_trainable
        )
        # 可训练参数转为float32
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def load_train_dataset(**kwargs) -> "Dataset":
    """加载训练数据集"""
    model_args, data_args, training_args, _, _ = get_train_args(kwargs)  # 获取训练参数
    tokenizer_module = load_tokenizer(model_args)  # 加载分词器
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)  # 处理模板
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        kwargs["stage"],  # 训练阶段
        **tokenizer_module
    )
    return dataset_module["train_dataset"]  # 返回训练集


def patch_valuehead_model() -> None:
    """修补value head模型的后初始化方法"""
    def post_init(self: "AutoModelForCausalLMWithValueHead", state_dict: Dict[str, "torch.Tensor"]) -> None:
        # 调整state_dict键名以匹配v_head结构
        state_dict = {k[7:]: state_dict[k] for k in state_dict.keys() if k.startswith("v_head.")}
        self.v_head.load_state_dict(state_dict, strict=False)  # 非严格加载
        del state_dict  # 释放内存

    AutoModelForCausalLMWithValueHead.post_init = post_init  # 覆盖原始方法
