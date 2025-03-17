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

import os
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Dict, Optional

from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME  # 安全适配器权重文件名
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME  # 适配器权重文件名
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME  # 导入权重相关的常量


CHECKPOINT_NAMES = {  # 检查点文件名集合
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}

CHOICES = ["A", "B", "C", "D"]  # 多选题的选项列表

DATA_CONFIG = "dataset_info.json"  # 数据集配置文件名

DEFAULT_TEMPLATE = defaultdict(str)  # 默认模板，对于未指定模型使用空字符串作为默认值

FILEEXT2TYPE = {  # 文件扩展名到类型的映射字典
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

IGNORE_INDEX = -100  # 忽略索引值，通常用于损失计算中忽略某些标记

IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<image>")  # 图像占位符，可通过环境变量覆盖

LAYERNORM_NAMES = {"norm", "ln"}  # 层归一化名称集合

LLAMABOARD_CONFIG = "llamaboard_config.yaml"  # LLaMABoard配置文件名

METHODS = ["full", "freeze", "lora"]  # 训练方法列表

MOD_SUPPORTED_MODELS = {"bloom", "falcon", "gemma", "llama", "mistral", "mixtral", "phi", "starcoder2"}  # 支持MoD的模型集合

PEFT_METHODS = {"lora"}  # 参数高效微调方法集合

RUNNING_LOG = "running_log.txt"  # 运行日志文件名

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]  # 学科类别列表

SUPPORTED_MODELS = OrderedDict()  # 支持的模型有序字典

TRAINER_LOG = "trainer_log.jsonl"  # 训练器日志文件名

TRAINING_ARGS = "training_args.yaml"  # 训练参数文件名

TRAINING_STAGES = {  # 训练阶段映射字典
    "Supervised Fine-Tuning": "sft",  # 监督微调
    "Reward Modeling": "rm",  # 奖励建模
    "PPO": "ppo",  # 近端策略优化
    "DPO": "dpo",  # 直接偏好优化
    "KTO": "kto",  # KTO优化
    "Pre-Training": "pt",  # 预训练
}

STAGES_USE_PAIR_DATA = {"rm", "dpo"}  # 使用成对数据的训练阶段集合

SUPPORTED_CLASS_FOR_BLOCK_DIAG_ATTN = {  # 支持块对角注意力的模型类别集合
    "cohere",
    "falcon",
    "gemma",
    "gemma2",
    "llama",
    "mistral",
    "phi",
    "phi3",
    "qwen2",
    "starcoder2",
}

SUPPORTED_CLASS_FOR_S2ATTN = {"llama"}  # 支持S2Attention的模型类别集合

VIDEO_PLACEHOLDER = os.environ.get("VIDEO_PLACEHOLDER", "<video>")  # 视频占位符，可通过环境变量覆盖

V_HEAD_WEIGHTS_NAME = "value_head.bin"  # 值头部权重文件名

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"  # 值头部安全权重文件名

VISION_MODELS = set()  # 视觉模型集合


class DownloadSource(str, Enum):  # 下载源枚举类
    DEFAULT = "hf"      # 默认源 (HuggingFace)
    MODELSCOPE = "ms"   # ModelScope源
    OPENMIND = "om"     # OpenMind源


def register_model_group(
    models: Dict[str, Dict[DownloadSource, str]],  # 模型名称到下载源路径的映射
    template: Optional[str] = None,  # 可选的模板名称
    vision: bool = False,  # 是否为视觉模型
) -> None:
    """注册一组模型到支持的模型列表中"""
    for name, path in models.items():
        SUPPORTED_MODELS[name] = path  # 将模型添加到支持的模型字典中
        if template is not None and any(suffix in name for suffix in ("-Chat", "-Instruct")):
            DEFAULT_TEMPLATE[name] = template  # 为对话或指令模型设置默认模板
        if vision:
            VISION_MODELS.add(name)  # 将视觉模型添加到视觉模型集合中


register_model_group(
    models={  # 注册Aya系列模型
        "Aya-23-8B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/aya-23-8B",
        },
        "Aya-23-35B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/aya-23-35B",
        },
    },
    template="cohere",  # 使用cohere模板
)


register_model_group(
    models={  # 注册百川-7B和13B系列模型
        "Baichuan-7B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-7B",
            DownloadSource.MODELSCOPE: "baichuan-inc/baichuan-7B",
        },
        "Baichuan-13B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-13B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan-13B-Base",
        },
        "Baichuan-13B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-13B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan-13B-Chat",
        },
    },
    template="baichuan",  # 使用baichuan模板
)


register_model_group(
    models={  # 注册百川2-7B和13B系列模型
        "Baichuan2-7B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-7B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-7B-Base",
        },
        "Baichuan2-13B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-13B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-13B-Base",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_13b_base_pt",
        },
        "Baichuan2-7B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-7B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-7B-Chat",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_7b_chat_pt",
        },
        "Baichuan2-13B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-13B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-13B-Chat",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_13b_chat_pt",
        },
    },
    template="baichuan2",  # 使用baichuan2模板
)


register_model_group(
    models={  # 注册BLOOM系列模型
        "BLOOM-560M": {
            DownloadSource.DEFAULT: "bigscience/bloom-560m",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-560m",
        },
        "BLOOM-3B": {
            DownloadSource.DEFAULT: "bigscience/bloom-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-3b",
        },
        "BLOOM-7B1": {
            DownloadSource.DEFAULT: "bigscience/bloom-7b1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-7b1",
        },
    },
)  # 没有指定模板，因为BLOOM是基础模型


register_model_group(
    models={  # 注册BLOOMZ系列模型（多语言指令微调版BLOOM）
        "BLOOMZ-560M": {
            DownloadSource.DEFAULT: "bigscience/bloomz-560m",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-560m",
        },
        "BLOOMZ-3B": {
            DownloadSource.DEFAULT: "bigscience/bloomz-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-3b",
        },
        "BLOOMZ-7B1-mt": {
            DownloadSource.DEFAULT: "bigscience/bloomz-7b1-mt",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-7b1-mt",
        },
    },
)  # 没有指定模板，因为BLOOMZ是基础模型


register_model_group(
    models={  # 注册BlueLM系列模型
        "BlueLM-7B-Base": {
            DownloadSource.DEFAULT: "vivo-ai/BlueLM-7B-Base",
            DownloadSource.MODELSCOPE: "vivo-ai/BlueLM-7B-Base",
        },
        "BlueLM-7B-Chat": {
            DownloadSource.DEFAULT: "vivo-ai/BlueLM-7B-Chat",
            DownloadSource.MODELSCOPE: "vivo-ai/BlueLM-7B-Chat",
        },
    },
    template="bluelm",  # 使用bluelm模板
)


register_model_group(
    models={  # 注册Breeze系列模型
        "Breeze-7B": {
            DownloadSource.DEFAULT: "MediaTek-Research/Breeze-7B-Base-v1_0",
        },
        "Breeze-7B-Instruct": {
            DownloadSource.DEFAULT: "MediaTek-Research/Breeze-7B-Instruct-v1_0",
        },
    },
    template="breeze",  # 使用breeze模板
)


register_model_group(
    models={  # 注册ChatGLM2系列模型
        "ChatGLM2-6B-Chat": {
            DownloadSource.DEFAULT: "THUDM/chatglm2-6b",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm2-6b",
        }
    },
    template="chatglm2",  # 使用chatglm2模板
)


register_model_group(
    models={  # 注册ChatGLM3系列模型
        "ChatGLM3-6B-Base": {
            DownloadSource.DEFAULT: "THUDM/chatglm3-6b-base",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm3-6b-base",
        },
        "ChatGLM3-6B-Chat": {
            DownloadSource.DEFAULT: "THUDM/chatglm3-6b",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm3-6b",
        },
    },
    template="chatglm3",  # 使用chatglm3模板
)


register_model_group(
    models={  # 注册中文Llama-2和中文Alpaca-2系列模型
        "Chinese-Llama-2-1.3B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-1.3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-1.3b",
        },
        "Chinese-Llama-2-7B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-7b",
        },
        "Chinese-Llama-2-13B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-13b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-13b",
        },
        "Chinese-Alpaca-2-1.3B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-1.3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-1.3b",
        },
        "Chinese-Alpaca-2-7B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-7b",
        },
        "Chinese-Alpaca-2-13B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-13b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-13b",
        },
    },
    template="llama2_zh",  # 使用中文版llama2模板
)


register_model_group(
    models={  # 注册CodeGeeX4代码生成模型
        "CodeGeeX4-9B-Chat": {
            DownloadSource.DEFAULT: "THUDM/codegeex4-all-9b",
            DownloadSource.MODELSCOPE: "ZhipuAI/codegeex4-all-9b",
        },
    },
    template="codegeex4",  # 使用codegeex4模板
)


register_model_group(
    models={  # 注册CodeGemma代码生成模型系列
        "CodeGemma-7B": {
            DownloadSource.DEFAULT: "google/codegemma-7b",
        },
        "CodeGemma-7B-Instruct": {
            DownloadSource.DEFAULT: "google/codegemma-7b-it",
            DownloadSource.MODELSCOPE: "AI-ModelScope/codegemma-7b-it",
        },
        "CodeGemma-1.1-2B": {
            DownloadSource.DEFAULT: "google/codegemma-1.1-2b",
        },
        "CodeGemma-1.1-7B-Instruct": {
            DownloadSource.DEFAULT: "google/codegemma-1.1-7b-it",
        },
    },
    template="gemma",  # 使用gemma模板
)


register_model_group(
    models={  # 注册Codestral代码模型
        "Codestral-22B-v0.1-Chat": {
            DownloadSource.DEFAULT: "mistralai/Codestral-22B-v0.1",  # HuggingFace默认下载源
        },
    },
    template="mistral",  # 使用mistral模板格式
)


register_model_group(
    models={  # 注册CommandR系列模型
        "CommandR-35B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-v01",
            DownloadSource.MODELSCOPE: "AI-ModelScope/c4ai-command-r-v01",
        },
        "CommandR-Plus-104B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-plus",
            DownloadSource.MODELSCOPE: "AI-ModelScope/c4ai-command-r-plus",
        },
        "CommandR-35B-4bit-Chat": {  # 4bit表示4位量化版本
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-v01-4bit",
            DownloadSource.MODELSCOPE: "mirror013/c4ai-command-r-v01-4bit",
        },
        "CommandR-Plus-104B-4bit-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-plus-4bit",
        },
    },
    template="cohere",  # 使用cohere模板格式
)


register_model_group(
    models={  # 注册DBRX系列模型（Databricks的大规模语言模型）
        "DBRX-132B-Base": {
            DownloadSource.DEFAULT: "databricks/dbrx-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/dbrx-base",
        },
        "DBRX-132B-Instruct": {
            DownloadSource.DEFAULT: "databricks/dbrx-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/dbrx-instruct",
        },
    },
    template="dbrx",  # 使用dbrx模板格式
)


register_model_group(
    models={  # 注册DeepSeek系列模型
        "DeepSeek-LLM-7B-Base": {  # 通用基础模型，7B参数量
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-7b-base",
        },
        "DeepSeek-LLM-67B-Base": {  # 通用基础模型，67B参数量
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-67b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-67b-base",
        },
        "DeepSeek-LLM-7B-Chat": {  # 通用对话模型，7B参数量
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-7b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-7b-chat",
        },
        "DeepSeek-LLM-67B-Chat": {  # 通用对话模型，67B参数量
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-67b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-67b-chat",
        },
        "DeepSeek-Math-7B-Base": {  # 数学专用基础模型
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-math-7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-math-7b-base",
        },
        "DeepSeek-Math-7B-Instruct": {  # 数学专用指令模型
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-math-7b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-math-7b-instruct",
        },
        "DeepSeek-MoE-16B-Base": {  # 混合专家基础模型
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-moe-16b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-moe-16b-base",
        },
        "DeepSeek-MoE-16B-Chat": {  # 混合专家对话模型
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-moe-16b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-moe-16b-chat",
        },
        "DeepSeek-V2-16B-Base": {  # DeepSeek第二代轻量版基础模型
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Lite",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Lite",
        },
        "DeepSeek-V2-236B-Base": {  # DeepSeek第二代全量版基础模型
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2",
        },
        "DeepSeek-V2-16B-Chat": {  # DeepSeek第二代轻量版对话模型
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Lite-Chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Lite-Chat",
        },
        "DeepSeek-V2-236B-Chat": {  # DeepSeek第二代全量版对话模型
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Chat",
        },
        "DeepSeek-Coder-V2-16B-Base": {  # DeepSeek第二代代码轻量版基础模型
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        },
        "DeepSeek-Coder-V2-236B-Base": {  # DeepSeek第二代代码全量版基础模型
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Base",
        },
        "DeepSeek-Coder-V2-16B-Instruct": {  # DeepSeek第二代代码轻量版指令模型
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        },
        "DeepSeek-Coder-V2-236B-Instruct": {  # DeepSeek第二代代码全量版指令模型
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        },
    },
    template="deepseek",  # 使用deepseek模板格式
)


register_model_group(
    models={  # 注册DeepSeek Coder系列模型（代码特化模型）
        "DeepSeek-Coder-6.7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-6.7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-base",
        },
        "DeepSeek-Coder-7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-7b-base-v1.5",
        },
        "DeepSeek-Coder-33B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-33b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-base",
        },
        "DeepSeek-Coder-6.7B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-6.7b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-instruct",
        },
        "DeepSeek-Coder-7B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        },
        "DeepSeek-Coder-33B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-33b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-instruct",
        },
    },
    template="deepseekcoder",  # 使用deepseekcoder模板格式（与通用deepseek模板不同）
)


register_model_group(
    models={  # 注册EXAONE-3.0系列模型
        "EXAONE-3.0-7.8B-Instruct": {
            DownloadSource.DEFAULT: "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        },
    },
    template="exaone",  # 使用exaone模板格式
)


register_model_group(
    models={  # 注册Falcon系列模型
        "Falcon-7B": {  # 基础模型，7B参数量
            DownloadSource.DEFAULT: "tiiuae/falcon-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-7b",
        },
        "Falcon-11B": {  # 基础模型，11B参数量
            DownloadSource.DEFAULT: "tiiuae/falcon-11B",
        },
        "Falcon-40B": {  # 基础模型，40B参数量
            DownloadSource.DEFAULT: "tiiuae/falcon-40b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-40b",
        },
        "Falcon-180B": {  # 基础模型，180B参数量
            DownloadSource.DEFAULT: "tiiuae/falcon-180b",
            DownloadSource.MODELSCOPE: "modelscope/falcon-180B",
        },
        "Falcon-7B-Instruct": {  # 指令微调模型，7B参数量
            DownloadSource.DEFAULT: "tiiuae/falcon-7b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-7b-instruct",
        },
        "Falcon-40B-Instruct": {  # 指令微调模型，40B参数量
            DownloadSource.DEFAULT: "tiiuae/falcon-40b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-40b-instruct",
        },
        "Falcon-180B-Chat": {  # 对话模型，180B参数量
            DownloadSource.DEFAULT: "tiiuae/falcon-180b-chat",
            DownloadSource.MODELSCOPE: "modelscope/falcon-180B-chat",
        },
    },
    template="falcon",  # 使用falcon模板格式
)


register_model_group(
    models={  # 注册Gemma系列模型
        "Gemma-2B": {  # 基础版Gemma 2B参数模型
            DownloadSource.DEFAULT: "google/gemma-2b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gemma-2b",
        },
        "Gemma-7B": {  # 基础版Gemma 7B参数模型
            DownloadSource.DEFAULT: "google/gemma-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gemma-2b-it",
        },
        "Gemma-2B-Instruct": {  # 指令微调版Gemma 2B参数模型
            DownloadSource.DEFAULT: "google/gemma-2b-it",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gemma-7b",
        },
        "Gemma-7B-Instruct": {  # 指令微调版Gemma 7B参数模型
            DownloadSource.DEFAULT: "google/gemma-7b-it",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gemma-7b-it",
        },
        "Gemma-1.1-2B-Instruct": {  # Gemma 1.1版本的2B指令模型
            DownloadSource.DEFAULT: "google/gemma-1.1-2b-it",
        },
        "Gemma-1.1-7B-Instruct": {  # Gemma 1.1版本的7B指令模型
            DownloadSource.DEFAULT: "google/gemma-1.1-7b-it",
        },
        "Gemma-2-2B": {  # Gemma 第二代2B基础模型
            DownloadSource.DEFAULT: "google/gemma-2-2b",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-2b",
        },
        "Gemma-2-9B": {  # Gemma 第二代9B基础模型
            DownloadSource.DEFAULT: "google/gemma-2-9b",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-9b",
        },
        "Gemma-2-27B": {  # Gemma 第二代27B基础模型
            DownloadSource.DEFAULT: "google/gemma-2-27b",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-27b",
        },
        "Gemma-2-2B-Instruct": {  # Gemma 第二代2B指令模型
            DownloadSource.DEFAULT: "google/gemma-2-2b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-2b-it",
            DownloadSource.OPENMIND: "LlamaFactory/gemma-2-2b-it",
        },
        "Gemma-2-9B-Instruct": {  # Gemma 第二代9B指令模型
            DownloadSource.DEFAULT: "google/gemma-2-9b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-9b-it",
            DownloadSource.OPENMIND: "LlamaFactory/gemma-2-9b-it",
        },
        "Gemma-2-27B-Instruct": {  # Gemma 第二代27B指令模型
            DownloadSource.DEFAULT: "google/gemma-2-27b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-27b-it",
        },
    },
    template="gemma",  # 使用gemma模板格式
)


register_model_group(
    models={  # 注册GLM-4系列模型
        "GLM-4-9B": {  # GLM-4基础模型，9B参数量
            DownloadSource.DEFAULT: "THUDM/glm-4-9b",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b",
        },
        "GLM-4-9B-Chat": {  # GLM-4对话模型，9B参数量
            DownloadSource.DEFAULT: "THUDM/glm-4-9b-chat",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b-chat",
            DownloadSource.OPENMIND: "LlamaFactory/glm-4-9b-chat",
        },
        "GLM-4-9B-1M-Chat": {  # GLM-4对话模型，9B参数量，支持100万tokens上下文窗口
            DownloadSource.DEFAULT: "THUDM/glm-4-9b-chat-1m",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b-chat-1m",
        },
    },
    template="glm4",  # 使用glm4模板格式
)


register_model_group(
    models={  # 注册Index系列模型
        "Index-1.9B-Chat": {  # Index对话模型，1.9B参数量
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B-Chat",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B-Chat",
        },
        "Index-1.9B-Character-Chat": {  # Index角色扮演对话模型，1.9B参数量
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B-Character",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B-Character",
        },
        "Index-1.9B-Base": {  # Index基础模型，1.9B参数量
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B",
        },
        "Index-1.9B-Base-Pure": {  # Index纯净基础模型，1.9B参数量
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B-Pure",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B-Pure",
        },
        "Index-1.9B-Chat-32K": {  # Index对话模型，1.9B参数量，支持32K上下文窗口
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B-32K",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B-32K",
        },
    },
    template="index",  # 使用index模板格式
)


register_model_group(
    models={  # 注册InternLM第一代系列模型
        "InternLM-7B": {  # 基础模型，7B参数量
            DownloadSource.DEFAULT: "internlm/internlm-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-7b",
        },
        "InternLM-20B": {  # 基础模型，20B参数量
            DownloadSource.DEFAULT: "internlm/internlm-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-20b",
        },
        "InternLM-7B-Chat": {  # 对话模型，7B参数量
            DownloadSource.DEFAULT: "internlm/internlm-chat-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-chat-7b",
        },
        "InternLM-20B-Chat": {  # 对话模型，20B参数量
            DownloadSource.DEFAULT: "internlm/internlm-chat-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-chat-20b",
        },
    },
    template="intern",  # 使用intern模板格式
)


register_model_group(
    models={  # 注册InternLM2和InternLM2.5系列模型
        "InternLM2-7B": {
            DownloadSource.DEFAULT: "internlm/internlm2-7b",  # HuggingFace默认下载源
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2-7b",  # ModelScope下载源
        },
        "InternLM2-20B": {
            DownloadSource.DEFAULT: "internlm/internlm2-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2-20b",
        },
        "InternLM2-7B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2-chat-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2-chat-7b",
        },
        "InternLM2-20B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2-chat-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2-chat-20b",
        },
        "InternLM2.5-1.8B": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-1_8b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-1_8b",
            DownloadSource.OPENMIND: "Intern/internlm2_5-1_8b",  # OpenMind下载源
        },
        "InternLM2.5-7B": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-7b",
        },
        "InternLM2.5-20B": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-20b",
            DownloadSource.OPENMIND: "Intern/internlm2_5-20b",
        },
        "InternLM2.5-1.8B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-1_8b-chat",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-1_8b-chat",
            DownloadSource.OPENMIND: "Intern/internlm2_5-1_8b-chat",
        },
        "InternLM2.5-7B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-7b-chat",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-7b-chat",
            DownloadSource.OPENMIND: "Intern/internlm2_5-7b-chat",
        },
        "InternLM2.5-7B-1M-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-7b-chat-1m",  # 1M表示训练了100万上下文长度
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m",
            DownloadSource.OPENMIND: "Intern/internlm2_5-7b-chat-1m",
        },
    },
)  # 没有指定模板，可能使用默认模板或内部处理


register_model_group(
    models={  # 注册Jamba模型
        "Jamba-v0.1": {  # Jamba v0.1版本模型
            DownloadSource.DEFAULT: "ai21labs/Jamba-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Jamba-v0.1",
        }
    },
)  # 没有指定模板，可能使用默认模板或内部处理


register_model_group(
    models={  # 注册LingoWhale模型
        "LingoWhale-8B": {  # LingoWhale 8B参数模型
            DownloadSource.DEFAULT: "deeplang-ai/LingoWhale-8B",
            DownloadSource.MODELSCOPE: "DeepLang/LingoWhale-8B",
        }
    },
)  # 没有指定模板，可能使用默认模板或内部处理


register_model_group(
    models={  # 注册Llama第一代系列模型
        "Llama-7B": {  # 基础模型，7B参数量
            DownloadSource.DEFAULT: "huggyllama/llama-7b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-7b",
        },
        "Llama-13B": {  # 基础模型，13B参数量
            DownloadSource.DEFAULT: "huggyllama/llama-13b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-13b",
        },
        "Llama-30B": {  # 基础模型，30B参数量
            DownloadSource.DEFAULT: "huggyllama/llama-30b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-30b",
        },
        "Llama-65B": {  # 基础模型，65B参数量
            DownloadSource.DEFAULT: "huggyllama/llama-65b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-65b",
        },
    }
)  # 没有指定模板，因为这是基础模型，不是对话模型


register_model_group(
    models={  # 注册Llama-2系列模型
        "Llama-2-7B": {  # Llama-2基础模型，7B参数量
            DownloadSource.DEFAULT: "meta-llama/Llama-2-7b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-ms",
        },
        "Llama-2-13B": {  # Llama-2基础模型，13B参数量
            DownloadSource.DEFAULT: "meta-llama/Llama-2-13b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-ms",
        },
        "Llama-2-70B": {  # Llama-2基础模型，70B参数量
            DownloadSource.DEFAULT: "meta-llama/Llama-2-70b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-ms",
        },
        "Llama-2-7B-Chat": {  # Llama-2对话模型，7B参数量
            DownloadSource.DEFAULT: "meta-llama/Llama-2-7b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-chat-ms",
        },
        "Llama-2-13B-Chat": {  # Llama-2对话模型，13B参数量
            DownloadSource.DEFAULT: "meta-llama/Llama-2-13b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-chat-ms",
        },
        "Llama-2-70B-Chat": {  # Llama-2对话模型，70B参数量
            DownloadSource.DEFAULT: "meta-llama/Llama-2-70b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-chat-ms",
        },
    },
    template="llama2",  # 使用llama2模板格式
)


register_model_group(
    models={  # 注册Llama-3系列模型
        "Llama-3-8B": {  # Llama-3基础模型，8B参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B",
        },
        "Llama-3-70B": {  # Llama-3基础模型，70B参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-70B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B",
        },
        "Llama-3-8B-Instruct": {  # Llama-3指令模型，8B参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B-Instruct",
        },
        "Llama-3-70B-Instruct": {  # Llama-3指令模型，70B参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B-Instruct",
        },
        "Llama-3-8B-Chinese-Chat": {  # Llama-3中文优化对话模型，8B参数量
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama3-8B-Chinese-Chat",
            DownloadSource.OPENMIND: "LlamaFactory/Llama3-Chinese-8B-Instruct",
        },
        "Llama-3-70B-Chinese-Chat": {  # Llama-3中文优化对话模型，70B参数量
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-70B-Chinese-Chat",
        },
        "Llama-3.1-8B": {  # Llama-3.1基础模型，8B参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B",
        },
        "Llama-3.1-70B": {  # Llama-3.1基础模型，70B参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-70B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B",
        },
        "Llama-3.1-405B": {  # Llama-3.1基础模型，405B超大参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-405B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B",
        },
        "Llama-3.1-8B-Instruct": {  # Llama-3.1指令模型，8B参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        },
        "Llama-3.1-70B-Instruct": {  # Llama-3.1指令模型，70B参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B-Instruct",
        },
        "Llama-3.1-405B-Instruct": {  # Llama-3.1指令模型，405B超大参数量
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-405B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B-Instruct",
        },
        "Llama-3.1-8B-Chinese-Chat": {  # Llama-3.1中文优化对话模型，8B参数量
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3.1-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-8B-Chinese-Chat",
        },
        "Llama-3.1-70B-Chinese-Chat": {  # Llama-3.1中文优化对话模型，70B参数量
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3.1-70B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-70B-Chinese-Chat",
        },
        "Llama-3.2-1B": {  # Llama-3.2基础模型，1B参数量（超轻量版）
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-1B",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B",
        },
        "Llama-3.2-3B": {  # Llama-3.2基础模型，3B参数量（轻量版）
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-3B",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B",
        },
        "Llama-3.2-1B-Instruct": {  # Llama-3.2指令模型，1B参数量（超轻量版）
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-1B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B-Instruct",
        },
        "Llama-3.2-3B-Instruct": {  # Llama-3.2指令模型，3B参数量（轻量版）
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-3B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B-Instruct",
        },
    },
    template="llama3",  # 使用llama3模板格式
)


register_model_group(
    models={  # 注册Llama-3.2视觉模型
        "Llama-3.2-11B-Vision-Instruct": {  # 视觉指令模型，11B参数量
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-11B-Vision-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-11B-Vision-Instruct",
        },
        "Llama-3.2-90B-Vision-Instruct": {  # 视觉指令模型，90B参数量
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-90B-Vision-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-90B-Vision-Instruct",
        },
    },
    template="mllama",  # 使用mllama多模态模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册LLaVA-1.5系列视觉-语言模型
        "LLaVA-1.5-7B-Chat": {  # 7B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llava-1.5-7b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-1.5-7b-hf",
        },
        "LLaVA-1.5-13B-Chat": {  # 13B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llava-1.5-13b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-1.5-13b-hf",
        },
    },
    template="llava",  # 使用llava模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册LLaVA-NeXT系列视觉模型
        "LLaVA-NeXT-7B-Chat": {  # 基于Vicuna的7B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llava-v1.6-vicuna-7b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-v1.6-vicuna-7b-hf",
        },
        "LLaVA-NeXT-13B-Chat": {  # 基于Vicuna的13B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llava-v1.6-vicuna-13b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-v1.6-vicuna-13b-hf",
        },
    },
    template="llava_next",  # 使用llava_next模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册基于Mistral的LLaVA-NeXT视觉模型
        "LLaVA-NeXT-Mistral-7B-Chat": {  # 基于Mistral的7B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llava-v1.6-mistral-7b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-v1.6-mistral-7b-hf",
        },
    },
    template="llava_next_mistral",  # 使用llava_next_mistral模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册基于Llama3的LLaVA-NeXT视觉模型
        "LLaVA-NeXT-Llama3-8B-Chat": {  # 基于Llama3的8B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llama3-llava-next-8b-hf",
            DownloadSource.MODELSCOPE: "swift/llama3-llava-next-8b-hf",
        },
    },
    template="llava_next_llama3",  # 使用llava_next_llama3模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册基于Yi的LLaVA-NeXT视觉模型
        "LLaVA-NeXT-34B-Chat": {  # 基于Yi的34B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llava-v1.6-34b-hf",
            DownloadSource.MODELSCOPE: "LLM-Research/llava-v1.6-34b-hf",
        },
    },
    template="llava_next_yi",  # 使用llava_next_yi模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册基于Qwen的LLaVA-NeXT大规模视觉模型
        "LLaVA-NeXT-72B-Chat": {  # 72B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llava-next-72b-hf",
            DownloadSource.MODELSCOPE: "AI-ModelScope/llava-next-72b-hf",
        },
        "LLaVA-NeXT-110B-Chat": {  # 110B参数量视觉对话模型
            DownloadSource.DEFAULT: "llava-hf/llava-next-110b-hf",
            DownloadSource.MODELSCOPE: "AI-ModelScope/llava-next-110b-hf",
        },
    },
    template="llava_next_qwen",  # 使用llava_next_qwen模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册LLaVA-NeXT视频模型
        "LLaVA-NeXT-Video-7B-Chat": {  # 7B参数量视频对话模型
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-7B-hf",
            DownloadSource.MODELSCOPE: "swift/LLaVA-NeXT-Video-7B-hf",
        },
        "LLaVA-NeXT-Video-7B-DPO-Chat": {  # 经过DPO优化的7B参数量视频对话模型
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf",
            DownloadSource.MODELSCOPE: "swift/LLaVA-NeXT-Video-7B-DPO-hf",
        },
    },
    template="llava_next_video",  # 使用llava_next_video模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册LLaVA-NeXT长上下文视频模型
        "LLaVA-NeXT-Video-7B-32k-Chat": {  # 支持32k上下文长度的7B参数量视频对话模型
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
            DownloadSource.MODELSCOPE: "swift/LLaVA-NeXT-Video-7B-32K-hf",
        },
    },
    template="llava_next_video_mistral",  # 使用llava_next_video_mistral模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={  # 注册基于Yi的LLaVA-NeXT视频模型
        "LLaVA-NeXT-Video-34B-Chat": {  # 基于Yi的34B参数量视频对话模型
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-34B-hf",
            DownloadSource.MODELSCOPE: "swift/LLaVA-NeXT-Video-34B-hf",
        },
        "LLaVA-NeXT-Video-34B-DPO-Chat": {  # 经过DPO优化的基于Yi的34B参数量视频对话模型
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-34B-DPO-hf",
        },
    },
    template="llava_next_video_yi",  # 使用llava_next_video_yi模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={
        "MiniCPM-2B-SFT-Chat": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM-2B-sft-bf16",
            DownloadSource.MODELSCOPE: "OpenBMB/miniCPM-bf16",
        },
        "MiniCPM-2B-DPO-Chat": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM-2B-dpo-bf16",
            DownloadSource.MODELSCOPE: "OpenBMB/MiniCPM-2B-dpo-bf16",
        },
    },
    template="cpm",
)


register_model_group(
    models={
        "MiniCPM3-4B-Chat": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM3-4B",
            DownloadSource.MODELSCOPE: "OpenBMB/MiniCPM3-4B",
            DownloadSource.OPENMIND: "LlamaFactory/MiniCPM3-4B",
        },
    },
    template="cpm3",
)


register_model_group(
    models={
        "Mistral-7B-v0.1": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-v0.1",
        },
        "Mistral-7B-Instruct-v0.1": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-Instruct-v0.1",
        },
        "Mistral-7B-v0.2": {
            DownloadSource.DEFAULT: "alpindale/Mistral-7B-v0.2-hf",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-v0.2-hf",
        },
        "Mistral-7B-Instruct-v0.2": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-Instruct-v0.2",
        },
        "Mistral-7B-v0.3": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-v0.3",
        },
        "Mistral-7B-Instruct-v0.3": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.3",
            DownloadSource.MODELSCOPE: "LLM-Research/Mistral-7B-Instruct-v0.3",
        },
        "Mistral-Nemo-Instruct-2407": {
            DownloadSource.DEFAULT: "mistralai/Mistral-Nemo-Instruct-2407",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-Nemo-Instruct-2407",
        },
    },
    template="mistral",
)


register_model_group(
    models={
        "Mixtral-8x7B-v0.1": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x7B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x7B-v0.1",
        },
        "Mixtral-8x7B-v0.1-Instruct": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1",
        },
        "Mixtral-8x22B-v0.1": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x22B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x22B-v0.1",
        },
        "Mixtral-8x22B-v0.1-Instruct": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x22B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x22B-Instruct-v0.1",
        },
    },
    template="mistral",
)


register_model_group(
    models={
        "OLMo-1B": {
            DownloadSource.DEFAULT: "allenai/OLMo-1B-hf",
        },
        "OLMo-7B": {
            DownloadSource.DEFAULT: "allenai/OLMo-7B-hf",
        },
        "OLMo-7B-Chat": {
            DownloadSource.DEFAULT: "ssec-uw/OLMo-7B-Instruct-hf",
        },
        "OLMo-1.7-7B": {
            DownloadSource.DEFAULT: "allenai/OLMo-1.7-7B-hf",
        },
    },
)


register_model_group(
    models={
        "OpenChat3.5-7B-Chat": {
            DownloadSource.DEFAULT: "openchat/openchat-3.5-0106",
            DownloadSource.MODELSCOPE: "xcwzxcwz/openchat-3.5-0106",
        }
    },
    template="openchat",
)


register_model_group(
    models={
        "OpenChat3.6-8B-Chat": {
            DownloadSource.DEFAULT: "openchat/openchat-3.6-8b-20240522",
        }
    },
    template="openchat-3.6",
)


register_model_group(
    models={
        "OpenCoder-1.5B-Base": {
            DownloadSource.DEFAULT: "infly/OpenCoder-1.5B-Base",
            DownloadSource.MODELSCOPE: "infly/OpenCoder-1.5B-Base",
        },
        "OpenCoder-8B-Base": {
            DownloadSource.DEFAULT: "infly/OpenCoder-8B-Base",
            DownloadSource.MODELSCOPE: "infly/OpenCoder-8B-Base",
        },
        "OpenCoder-1.5B-Instruct": {
            DownloadSource.DEFAULT: "infly/OpenCoder-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "infly/OpenCoder-1.5B-Instruct",
        },
        "OpenCoder-8B-Instruct": {
            DownloadSource.DEFAULT: "infly/OpenCoder-8B-Instruct",
            DownloadSource.MODELSCOPE: "infly/OpenCoder-8B-Instruct",
        },
    },
    template="opencoder",
)


register_model_group(
    models={
        "Orion-14B-Base": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-Base",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-Base",
        },
        "Orion-14B-Chat": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-Chat",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-Chat",
        },
        "Orion-14B-Long-Chat": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-LongChat",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-LongChat",
        },
        "Orion-14B-RAG-Chat": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-Chat-RAG",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-Chat-RAG",
        },
        "Orion-14B-Plugin-Chat": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-Chat-Plugin",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-Chat-Plugin",
        },
    },
    template="orion",
)


register_model_group(
    models={
        "PaliGemma-3B-pt-224-Chat": {  # PaliGemma 3B参数预训练224像素分辨率模型
            DownloadSource.DEFAULT: "google/paligemma-3b-pt-224",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-pt-224",
        },
        "PaliGemma-3B-pt-448-Chat": {  # PaliGemma 3B参数预训练448像素分辨率模型
            DownloadSource.DEFAULT: "google/paligemma-3b-pt-448",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-pt-448",
        },
        "PaliGemma-3B-pt-896-Chat": {  # PaliGemma 3B参数预训练896像素分辨率模型
            DownloadSource.DEFAULT: "google/paligemma-3b-pt-896",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-pt-896",
        },
        "PaliGemma-3B-mix-224-Chat": {  # PaliGemma 3B参数混合训练224像素分辨率模型
            DownloadSource.DEFAULT: "google/paligemma-3b-mix-224",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-mix-224",
        },
        "PaliGemma-3B-mix-448-Chat": {  # PaliGemma 3B参数混合训练448像素分辨率模型
            DownloadSource.DEFAULT: "google/paligemma-3b-mix-448",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-mix-448",
        },
    },
    template="paligemma",  # 使用paligemma模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={
        "Phi-1.5-1.3B": {  # Phi-1.5版本1.3B参数模型
            DownloadSource.DEFAULT: "microsoft/phi-1_5",
            DownloadSource.MODELSCOPE: "allspace/PHI_1-5",
        },
        "Phi-2-2.7B": {  # Phi-2版本2.7B参数模型
            DownloadSource.DEFAULT: "microsoft/phi-2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/phi-2",
        },
    }
)  # 未指定模板，可能使用默认模板


register_model_group(
    models={
        "Phi-3-4B-4k-Instruct": {  # Phi-3 4B参数支持4k上下文的指令模型
            DownloadSource.DEFAULT: "microsoft/Phi-3-mini-4k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-mini-4k-instruct",
        },
        "Phi-3-4B-128k-Instruct": {  # Phi-3 4B参数支持128k长上下文的指令模型
            DownloadSource.DEFAULT: "microsoft/Phi-3-mini-128k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-mini-128k-instruct",
        },
        "Phi-3-14B-8k-Instruct": {  # Phi-3 14B参数支持8k上下文的指令模型
            DownloadSource.DEFAULT: "microsoft/Phi-3-medium-4k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-medium-4k-instruct",
        },
        "Phi-3-14B-128k-Instruct": {  # Phi-3 14B参数支持128k长上下文的指令模型
            DownloadSource.DEFAULT: "microsoft/Phi-3-medium-128k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-medium-128k-instruct",
        },
    },
    template="phi",  # 使用phi模板格式
)


register_model_group(
    models={
        "Phi-3-7B-8k-Instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3-small-8k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-small-8k-instruct",
        },
        "Phi-3-7B-128k-Instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3-small-128k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-small-128k-instruct",
        },
    },
    template="phi_small",  # 使用phi_small专用模板格式
)


register_model_group(
    models={
        "Pixtral-12B-Chat": {
            DownloadSource.DEFAULT: "mistral-community/pixtral-12b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/pixtral-12b",
        }
    },
    template="pixtral",  # 使用pixtral模板格式
    vision=True,  # 支持视觉输入
)


register_model_group(
    models={
        "Qwen-1.8B": {  # Qwen 1.8B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B",
            DownloadSource.MODELSCOPE: "qwen/Qwen-1_8B",
        },
        "Qwen-7B": {  # Qwen 7B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen-7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen-7B",
        },
        "Qwen-14B": {  # Qwen 14B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen-14B",
            DownloadSource.MODELSCOPE: "qwen/Qwen-14B",
        },
        "Qwen-72B": {  # Qwen 72B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen-72B",
            DownloadSource.MODELSCOPE: "qwen/Qwen-72B",
        },
        "Qwen-1.8B-Chat": {  # Qwen 1.8B参数对话模型
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen-1_8B-Chat",
        },
        "Qwen-7B-Chat": {  # Qwen 7B参数对话模型
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen-7B-Chat",
        },
        "Qwen-14B-Chat": {  # Qwen 14B参数对话模型
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen-14B-Chat",
        },
        "Qwen-72B-Chat": {  # Qwen 72B参数对话模型
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen-72B-Chat",
        },
        "Qwen-1.8B-Chat-Int8": {  # Qwen 1.8B参数对话模型的8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen-1_8B-Chat-Int8",
        },
        "Qwen-1.8B-Chat-Int4": {  # Qwen 1.8B参数对话模型的4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen-1_8B-Chat-Int4",
        },
        "Qwen-7B-Chat-Int8": {  # Qwen 7B参数对话模型的8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen-7B-Chat-Int8",
        },
        "Qwen-7B-Chat-Int4": {  # Qwen 7B参数对话模型的4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen-7B-Chat-Int4",
        },
        "Qwen-14B-Chat-Int8": {  # Qwen 14B参数对话模型的8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen-14B-Chat-Int8",
        },
        "Qwen-14B-Chat-Int4": {  # Qwen 14B参数对话模型的4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen-14B-Chat-Int4",
        },
        "Qwen-72B-Chat-Int8": {  # Qwen 72B参数对话模型的8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen-72B-Chat-Int8",
        },
        "Qwen-72B-Chat-Int4": {  # Qwen 72B参数对话模型的4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen-72B-Chat-Int4",
        },
    },
    template="qwen",  # 使用qwen模板格式进行对话
)


register_model_group(
    models={  # 注册Qwen1.5系列模型
        "Qwen1.5-0.5B": {  # Qwen1.5基础模型，0.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-0.5B",
        },
        "Qwen1.5-1.8B": {  # Qwen1.5基础模型，1.8B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-1.8B",
        },
        "Qwen1.5-4B": {  # Qwen1.5基础模型，4B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-4B",
        },
        "Qwen1.5-7B": {  # Qwen1.5基础模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-7B",
        },
        "Qwen1.5-14B": {  # Qwen1.5基础模型，14B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-14B",
        },
        "Qwen1.5-32B": {  # Qwen1.5基础模型，32B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-32B",
        },
        "Qwen1.5-72B": {  # Qwen1.5基础模型，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-72B",
        },
        "Qwen1.5-110B": {  # Qwen1.5基础模型，110B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-110B",
        },
        "Qwen1.5-MoE-A2.7B": {  # MoE表示Mixture of Experts混合专家模型，A2.7B表示激活参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-MoE-A2.7B",
        },
        "Qwen1.5-0.5B-Chat": {  # Qwen1.5对话模型，0.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-0.5B-Chat",
        },
        "Qwen1.5-1.8B-Chat": {  # Qwen1.5对话模型，1.8B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-1.8B-Chat",
        },
        "Qwen1.5-4B-Chat": {  # Qwen1.5对话模型，4B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-4B-Chat",
        },
        "Qwen1.5-7B-Chat": {  # Qwen1.5对话模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-7B-Chat",
        },
        "Qwen1.5-14B-Chat": {  # Qwen1.5对话模型，14B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-14B-Chat",
        },
        "Qwen1.5-32B-Chat": {  # Qwen1.5对话模型，32B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-32B-Chat",
        },
        "Qwen1.5-72B-Chat": {  # Qwen1.5对话模型，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-72B-Chat",
        },
        "Qwen1.5-110B-Chat": {  # Qwen1.5对话模型，110B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-110B-Chat",
        },
        "Qwen1.5-MoE-A2.7B-Chat": {  # Qwen1.5混合专家对话模型，激活参数量为2.7B
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-MoE-A2.7B-Chat",
        },
        "Qwen1.5-0.5B-Chat-GPTQ-Int8": {  # GPTQ-Int8是8位量化版本，减小模型体积和推理资源消耗
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-0.5B-Chat-AWQ": {  # AWQ是另一种量化方法，平衡性能和效率
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-0.5B-Chat-AWQ",
        },
        "Qwen1.5-1.8B-Chat-GPTQ-Int8": {  # 1.8B参数对话模型的GPTQ 8位量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-1.8B-Chat-AWQ": {  # 1.8B参数对话模型的AWQ量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-1.8B-Chat-AWQ",
        },
        "Qwen1.5-4B-Chat-GPTQ-Int8": {  # 4B参数对话模型的GPTQ 8位量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-4B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-4B-Chat-AWQ": {  # 4B参数对话模型的AWQ量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-4B-Chat-AWQ",
        },
        "Qwen1.5-7B-Chat-GPTQ-Int8": {  # 7B参数对话模型的GPTQ 8位量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-7B-Chat-AWQ": {  # 7B参数对话模型的AWQ量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-7B-Chat-AWQ",
        },
        "Qwen1.5-14B-Chat-GPTQ-Int8": {  # 14B参数对话模型的GPTQ 8位量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-14B-Chat-AWQ": {  # 14B参数对话模型的AWQ量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-14B-Chat-AWQ",
        },
        "Qwen1.5-32B-Chat-AWQ": {  # 32B参数对话模型的AWQ量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-32B-Chat-AWQ",
        },
        "Qwen1.5-72B-Chat-GPTQ-Int8": {  # 72B参数对话模型的GPTQ 8位量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-72B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-72B-Chat-AWQ": {  # 72B参数对话模型的AWQ量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-72B-Chat-AWQ",
        },
        "Qwen1.5-110B-Chat-AWQ": {  # 110B参数对话模型的AWQ量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-110B-Chat-AWQ",
        },
        "Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4": {  # 混合专家对话模型的GPTQ 4位量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        },
        "CodeQwen1.5-7B": {  # Qwen1.5代码专用基础模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B",
            DownloadSource.MODELSCOPE: "qwen/CodeQwen1.5-7B",
        },
        "CodeQwen1.5-7B-Chat": {  # Qwen1.5代码专用对话模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B-Chat",
            DownloadSource.MODELSCOPE: "qwen/CodeQwen1.5-7B-Chat",
        },
        "CodeQwen1.5-7B-Chat-AWQ": {  # Qwen1.5代码专用对话模型的AWQ量化版本
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "qwen/CodeQwen1.5-7B-Chat-AWQ",
        },
    },
    template="qwen",  # 使用qwen模板格式进行对话
)


register_model_group(  # 注册Qwen2系列模型组
    models={  # 定义模型字典，包含所有Qwen2模型及其下载地址
        "Qwen2-0.5B": {  # Qwen2 0.5B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B",  # HuggingFace默认下载源
            DownloadSource.MODELSCOPE: "qwen/Qwen2-0.5B",  # ModelScope下载源
        },
        "Qwen2-1.5B": {  # Qwen2 1.5B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-1.5B",
        },
        "Qwen2-7B": {  # Qwen2 7B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-7B",
        },
        "Qwen2-72B": {  # Qwen2 72B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-72B",
        },
        "Qwen2-MoE-57B-A14B": {  # Qwen2 混合专家模型(MoE)，总参数57B，激活参数14B的基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-57B-A14B",
        },
        "Qwen2-0.5B-Instruct": {  # Qwen2 0.5B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-0.5B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-0.5B-Instruct",  # OpenMind平台下载源
        },
        "Qwen2-1.5B-Instruct": {  # Qwen2 1.5B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-1.5B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-1.5B-Instruct",
        },
        "Qwen2-7B-Instruct": {  # Qwen2 7B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-7B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-7B-Instruct",
        },
        "Qwen2-72B-Instruct": {  # Qwen2 72B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-72B-Instruct",
        },
        "Qwen2-MoE-57B-A14B-Instruct": {  # Qwen2 混合专家模型，总参数57B，激活参数14B的指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-57B-A14B-Instruct",
        },
        "Qwen2-0.5B-Instruct-GPTQ-Int8": {  # Qwen2 0.5B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2-0.5B-Instruct-GPTQ-Int4": {  # Qwen2 0.5B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2-0.5B-Instruct-AWQ": {  # Qwen2 0.5B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-0.5B-Instruct-AWQ",
        },
        "Qwen2-1.5B-Instruct-GPTQ-Int8": {  # Qwen2 1.5B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2-1.5B-Instruct-GPTQ-Int4": {  # Qwen2 1.5B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2-1.5B-Instruct-AWQ": {  # Qwen2 1.5B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-1.5B-Instruct-AWQ",
        },
        "Qwen2-7B-Instruct-GPTQ-Int8": {  # Qwen2 7B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2-7B-Instruct-GPTQ-Int4": {  # Qwen2 7B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2-7B-Instruct-AWQ": {  # Qwen2 7B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-7B-Instruct-AWQ",
        },
        "Qwen2-72B-Instruct-GPTQ-Int8": {  # Qwen2 72B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2-72B-Instruct-GPTQ-Int4": {  # Qwen2 72B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2-72B-Instruct-AWQ": {  # Qwen2 72B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-72B-Instruct-AWQ",
        },
        "Qwen2-57B-A14B-Instruct-GPTQ-Int4": {  # Qwen2 混合专家模型(总参数57B，激活14B)指令版本的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
        },
        "Qwen2-Math-1.5B": {  # Qwen2 数学专用基础模型，1.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-1.5B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-Math-1.5B",
        },
        "Qwen2-Math-7B": {  # Qwen2 数学专用基础模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-Math-7B",
        },
        "Qwen2-Math-72B": {  # Qwen2 数学专用基础模型，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-72B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-Math-72B",
        },
        "Qwen2-Math-1.5B-Instruct": {  # Qwen2 数学专用指令模型，1.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-Math-1.5B-Instruct",
        },
        "Qwen2-Math-7B-Instruct": {  # Qwen2 数学专用指令模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-7B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-Math-7B-Instruct",
        },
        "Qwen2-Math-72B-Instruct": {  # Qwen2 数学专用指令模型，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-72B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-Math-72B-Instruct",
        },
    },
    template="qwen",  # 使用qwen对话模板格式进行交互
)


register_model_group(  # 注册Qwen2.5系列模型组
    models={  # 定义模型字典，包含所有Qwen2.5模型及其下载地址
        "Qwen2.5-0.5B": {  # Qwen2.5 0.5B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B",  # HuggingFace默认下载源
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-0.5B",  # ModelScope下载源
        },
        "Qwen2.5-1.5B": {  # Qwen2.5 1.5B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-1.5B",
        },
        "Qwen2.5-3B": {  # Qwen2.5 3B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-3B",
        },
        "Qwen2.5-7B": {  # Qwen2.5 7B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-7B",
        },
        "Qwen2.5-14B": {  # Qwen2.5 14B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-14B",
        },
        "Qwen2.5-32B": {  # Qwen2.5 32B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-32B",
        },
        "Qwen2.5-72B": {  # Qwen2.5 72B参数基础模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-72B",
        },
        "Qwen2.5-0.5B-Instruct": {  # Qwen2.5 0.5B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-0.5B-Instruct",
        },
        "Qwen2.5-1.5B-Instruct": {  # Qwen2.5 1.5B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-1.5B-Instruct",
        },
        "Qwen2.5-3B-Instruct": {  # Qwen2.5 3B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-3B-Instruct",
        },
        "Qwen2.5-7B-Instruct": {  # Qwen2.5 7B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-7B-Instruct",
        },
        "Qwen2.5-14B-Instruct": {  # Qwen2.5 14B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-14B-Instruct",
        },
        "Qwen2.5-32B-Instruct": {  # Qwen2.5 32B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-32B-Instruct",
        },
        "Qwen2.5-72B-Instruct": {  # Qwen2.5 72B参数指令微调模型
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-72B-Instruct",
        },
        "Qwen2.5-0.5B-Instruct-GPTQ-Int8": {  # Qwen2.5 0.5B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-0.5B-Instruct-GPTQ-Int4": {  # Qwen2.5 0.5B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-0.5B-Instruct-AWQ": {  # Qwen2.5 0.5B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-0.5B-Instruct-AWQ",
        },
        "Qwen2.5-1.5B-Instruct-GPTQ-Int8": {  # Qwen2.5 1.5B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-1.5B-Instruct-GPTQ-Int4": {  # Qwen2.5 1.5B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-1.5B-Instruct-AWQ": {  # Qwen2.5 1.5B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-1.5B-Instruct-AWQ",
        },
        "Qwen2.5-3B-Instruct-GPTQ-Int8": {  # Qwen2.5 3B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-3B-Instruct-GPTQ-Int4": {  # Qwen2.5 3B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-3B-Instruct-AWQ": {  # Qwen2.5 3B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-3B-Instruct-AWQ",
        },
        "Qwen2.5-7B-Instruct-GPTQ-Int8": {  # Qwen2.5 7B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-7B-Instruct-GPTQ-Int4": {  # Qwen2.5 7B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-7B-Instruct-AWQ": {  # Qwen2.5 7B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-7B-Instruct-AWQ",
        },
        "Qwen2.5-14B-Instruct-GPTQ-Int8": {  # Qwen2.5 14B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-14B-Instruct-GPTQ-Int4": {  # Qwen2.5 14B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-14B-Instruct-AWQ": {  # Qwen2.5 14B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-14B-Instruct-AWQ",
        },
        "Qwen2.5-32B-Instruct-GPTQ-Int8": {  # Qwen2.5 32B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-32B-Instruct-GPTQ-Int4": {  # Qwen2.5 32B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-32B-Instruct-AWQ": {  # Qwen2.5 32B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-32B-Instruct-AWQ",
        },
        "Qwen2.5-72B-Instruct-GPTQ-Int8": {  # Qwen2.5 72B参数指令模型的GPTQ 8位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-72B-Instruct-GPTQ-Int4": {  # Qwen2.5 72B参数指令模型的GPTQ 4位整数量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-72B-Instruct-AWQ": {  # Qwen2.5 72B参数指令模型的AWQ激活感知量化版本
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-72B-Instruct-AWQ",
        },
        "Qwen2.5-Coder-0.5B": {  # Qwen2.5代码专用模型，0.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-0.5B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-0.5B",
        },
        "Qwen2.5-Coder-1.5B": {  # Qwen2.5代码专用基础模型，1.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-1.5B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-1.5B",
        },
        "Qwen2.5-Coder-3B": {  # Qwen2.5代码专用基础模型，3B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-3B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-3B",
        },
        "Qwen2.5-Coder-7B": {  # Qwen2.5代码专用基础模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-7B",
        },
        "Qwen2.5-Coder-14B": {  # Qwen2.5代码专用基础模型，14B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-14B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-14B",
        },
        "Qwen2.5-Coder-32B": {  # Qwen2.5代码专用基础模型，32B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-32B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-32B",
        },
        "Qwen2.5-Coder-0.5B-Instruct": {  # Qwen2.5代码专用指令微调模型，0.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-0.5B-Instruct",
        },
        "Qwen2.5-Coder-1.5B-Instruct": {  # Qwen2.5代码专用指令微调模型，1.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-1.5B-Instruct",
        },
        "Qwen2.5-Coder-3B-Instruct": {  # Qwen2.5代码专用指令微调模型，3B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-3B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-3B-Instruct",
        },
        "Qwen2.5-Coder-7B-Instruct": {  # Qwen2.5代码专用指令微调模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-7B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "Qwen2.5-Coder-14B-Instruct": {  # Qwen2.5代码专用指令微调模型，14B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-14B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-14B-Instruct",
        },
        "Qwen2.5-Coder-32B-Instruct": {  # Qwen2.5代码专用指令微调模型，32B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-32B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-32B-Instruct",
        },
        "Qwen2.5-Math-1.5B": {  # Qwen2.5数学专用基础模型，1.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-1.5B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Math-1.5B",
        },
        "Qwen2.5-Math-7B": {  # Qwen2.5数学专用基础模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-7B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Math-7B",
        },
        "Qwen2.5-Math-72B": {  # Qwen2.5数学专用基础模型，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-72B",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Math-72B",
        },
        "Qwen2.5-Math-1.5B-Instruct": {  # Qwen2.5数学专用指令微调模型，1.5B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-1.5B-Instruct",  # 注意：此处ModelScope路径可能有误，指向了Coder模型
        },
        "Qwen2.5-Math-7B-Instruct": {  # Qwen2.5数学专用指令微调模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-7B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-7B-Instruct",  # 注意：此处ModelScope路径可能有误，指向了Coder模型
        },
        "Qwen2.5-Math-72B-Instruct": {  # Qwen2.5数学专用指令微调模型，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-72B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2.5-Coder-72B-Instruct",  # 注意：此处ModelScope路径可能有误，指向了Coder模型
        },
    },
    template="qwen",  # 使用qwen对话模板格式进行交互
)


register_model_group(  # 注册Qwen2视觉语言(Vision-Language)多模态模型组
    models={  # 定义模型字典，包含所有Qwen2-VL模型及其下载地址
        "Qwen2-VL-2B-Instruct": {  # Qwen2视觉语言指令模型，2B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct",  # HuggingFace默认下载源
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-2B-Instruct",  # ModelScope下载源
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-VL-2B-Instruct",  # OpenMind平台下载源
        },
        "Qwen2-VL-7B-Instruct": {  # Qwen2视觉语言指令模型，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-7B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-VL-7B-Instruct",
        },
        "Qwen2-VL-72B-Instruct": {  # Qwen2视觉语言指令模型，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-72B-Instruct",
        },
        "Qwen2-VL-2B-Instruct-GPTQ-Int8": {  # Qwen2视觉语言指令模型GPTQ 8位整数量化版本，2B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-2B-Instruct-GPTQ-Int4": {  # Qwen2视觉语言指令模型GPTQ 4位整数量化版本，2B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-2B-Instruct-AWQ": {  # Qwen2视觉语言指令模型AWQ激活感知量化版本，2B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-2B-Instruct-AWQ",
        },
        "Qwen2-VL-7B-Instruct-GPTQ-Int8": {  # Qwen2视觉语言指令模型GPTQ 8位整数量化版本，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-7B-Instruct-GPTQ-Int4": {  # Qwen2视觉语言指令模型GPTQ 4位整数量化版本，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-7B-Instruct-AWQ": {  # Qwen2视觉语言指令模型AWQ激活感知量化版本，7B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-7B-Instruct-AWQ",
        },
        "Qwen2-VL-72B-Instruct-GPTQ-Int8": {  # Qwen2视觉语言指令模型GPTQ 8位整数量化版本，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-72B-Instruct-GPTQ-Int4": {  # Qwen2视觉语言指令模型GPTQ 4位整数量化版本，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-72B-Instruct-AWQ": {  # Qwen2视觉语言指令模型AWQ激活感知量化版本，72B参数量
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-VL-72B-Instruct-AWQ",
        },
    },
    template="qwen2_vl",  # 使用qwen2_vl专用模板格式进行对话
    vision=True,  # 启用视觉输入功能
)


register_model_group(  # 注册SOLAR模型组
    models={  # 定义模型字典
        "SOLAR-10.7B-v1.0": {  # SOLAR基础模型，10.7B参数量
            DownloadSource.DEFAULT: "upstage/SOLAR-10.7B-v1.0",  # HuggingFace默认下载源
        },
        "SOLAR-10.7B-Instruct-v1.0": {  # SOLAR指令微调模型，10.7B参数量
            DownloadSource.DEFAULT: "upstage/SOLAR-10.7B-Instruct-v1.0",
            DownloadSource.MODELSCOPE: "AI-ModelScope/SOLAR-10.7B-Instruct-v1.0",  # ModelScope下载源
        },
    },
    template="solar",  # 使用solar模板格式进行对话
)


register_model_group(  # 注册Skywork模型组
    models={
        "Skywork-13B-Base": {  # Skywork基础模型，13B参数量
            DownloadSource.DEFAULT: "Skywork/Skywork-13B-base",
            DownloadSource.MODELSCOPE: "skywork/Skywork-13B-base",
        }
    }
)


register_model_group(  # 注册StarCoder2代码生成模型组
    models={
        "StarCoder2-3B": {  # StarCoder2代码模型，3B参数量
            DownloadSource.DEFAULT: "bigcode/starcoder2-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/starcoder2-3b",
        },
        "StarCoder2-7B": {  # StarCoder2代码模型，7B参数量
            DownloadSource.DEFAULT: "bigcode/starcoder2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/starcoder2-7b",
        },
        "StarCoder2-15B": {  # StarCoder2代码模型，15B参数量
            DownloadSource.DEFAULT: "bigcode/starcoder2-15b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/starcoder2-15b",
        },
    }
)


register_model_group(  # 注册TeleChat电信行业聊天模型组
    models={
        "TeleChat-1B-Chat": {  # TeleChat聊天模型，1B参数量
            DownloadSource.DEFAULT: "Tele-AI/TeleChat-1B",
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat-1B",
        },
        "TeleChat-7B-Chat": {  # TeleChat聊天模型，7B参数量
            DownloadSource.DEFAULT: "Tele-AI/telechat-7B",
            DownloadSource.MODELSCOPE: "TeleAI/telechat-7B",
            DownloadSource.OPENMIND: "TeleAI/TeleChat-7B-pt",  # OpenMind平台下载源，pt后缀表示预训练版本
        },
        "TeleChat-12B-Chat": {  # TeleChat聊天模型，12B参数量
            DownloadSource.DEFAULT: "Tele-AI/TeleChat-12B",
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat-12B",
            DownloadSource.OPENMIND: "TeleAI/TeleChat-12B-pt",
        },
        "TeleChat-12B-v2-Chat": {  # TeleChat聊天模型第2版，12B参数量
            DownloadSource.DEFAULT: "Tele-AI/TeleChat-12B-v2",
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat-12B-v2",
        },
    },
    template="telechat",  # 使用telechat模板格式进行对话
)
