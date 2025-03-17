# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/commands/env.py
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

# 导入系统和依赖库
import platform  # 导入platform模块，用于获取系统平台信息
import accelerate  # 导入accelerate库，用于分布式训练加速
import datasets  # 导入datasets库，用于数据集处理
import peft  # 导入peft库，用于参数高效微调
import torch  # 导入PyTorch深度学习框架
import transformers  # 导入transformers库，用于处理预训练模型
import trl  # 导入trl库，用于强化学习训练
from transformers.utils import is_torch_cuda_available, is_torch_npu_available  # 导入GPU和NPU可用性检查函数

# 定义软件版本号
VERSION = "0.9.1"  # LLaMAFactory当前版本号


def print_env() -> None:  # 定义打印环境信息的函数
    info = {  # 创建包含环境信息的字典
        "`llamafactory` version": VERSION,  # LLaMAFactory版本信息
        "Platform": platform.platform(),  # 系统平台信息
        "Python version": platform.python_version(),  # Python版本信息
        "PyTorch version": torch.__version__,  # PyTorch版本信息
        "Transformers version": transformers.__version__,  # Transformers库版本信息
        "Datasets version": datasets.__version__,  # Datasets库版本信息
        "Accelerate version": accelerate.__version__,  # Accelerate库版本信息
        "PEFT version": peft.__version__,  # PEFT库版本信息
        "TRL version": trl.__version__,  # TRL库版本信息
    }

    if is_torch_cuda_available():  # 如果CUDA（GPU）可用
        info["PyTorch version"] += " (GPU)"  # 在PyTorch版本信息后添加GPU标识
        info["GPU type"] = torch.cuda.get_device_name()  # 获取GPU设备名称

    if is_torch_npu_available():  # 如果NPU（昇腾AI处理器）可用
        info["PyTorch version"] += " (NPU)"  # 在PyTorch版本信息后添加NPU标识
        info["NPU type"] = torch.npu.get_device_name()  # 获取NPU设备名称
        info["CANN version"] = torch.version.cann  # 获取CANN（昇腾计算架构）版本信息

    try:  # 尝试导入并获取DeepSpeed版本信息
        import deepspeed  # type: ignore  # 导入DeepSpeed库
        info["DeepSpeed version"] = deepspeed.__version__  # 添加DeepSpeed版本信息
    except Exception:  # 如果导入失败，则跳过
        pass

    try:  # 尝试导入并获取bitsandbytes版本信息
        import bitsandbytes  # 导入bitsandbytes库（用于模型量化）
        info["Bitsandbytes version"] = bitsandbytes.__version__  # 添加bitsandbytes版本信息
    except Exception:  # 如果导入失败，则跳过
        pass

    try:  # 尝试导入并获取vLLM版本信息
        import vllm  # 导入vLLM库（用于大语言模型推理优化）
        info["vLLM version"] = vllm.__version__  # 添加vLLM版本信息
    except Exception:  # 如果导入失败，则跳过
        pass

    # 打印所有环境信息，每条信息占一行，前面加上"- "标记
    print("\n" + "\n".join([f"- {key}: {value}" for key, value in info.items()]) + "\n")
