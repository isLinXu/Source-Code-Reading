# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/import_utils.py
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

# 导入必要的库
import importlib.metadata  # 导入元数据处理模块
import importlib.util  # 导入模块工具
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import TYPE_CHECKING  # 导入类型检查工具

from packaging import version  # 导入版本处理工具


if TYPE_CHECKING:  # 类型检查时导入
    from packaging.version import Version  # 导入Version类型


def _is_package_available(name: str) -> bool:
    """
    检查指定名称的包是否可用
    参数:
        name: 包名
    返回:
        bool: 包是否可用
    """
    return importlib.util.find_spec(name) is not None  # 通过查找包的规范来判断包是否已安装


def _get_package_version(name: str) -> "Version":
    """
    获取指定包的版本号
    参数:
        name: 包名
    返回:
        Version: 包的版本号，如果获取失败则返回0.0.0
    """
    try:
        return version.parse(importlib.metadata.version(name))  # 尝试获取并解析包的版本号
    except Exception:
        return version.parse("0.0.0")  # 如果失败则返回0.0.0版本


def is_pyav_available():
    """检查PyAV（音视频处理库）是否可用"""
    return _is_package_available("av")


def is_fastapi_available():
    """检查FastAPI（Web框架）是否可用"""
    return _is_package_available("fastapi")


def is_galore_available():
    """检查Galore（优化器库）是否可用"""
    return _is_package_available("galore_torch")


def is_gradio_available():
    """检查Gradio（Web界面库）是否可用"""
    return _is_package_available("gradio")


def is_matplotlib_available():
    """检查Matplotlib（绘图库）是否可用"""
    return _is_package_available("matplotlib")


def is_pillow_available():
    """检查Pillow（图像处理库）是否可用"""
    return _is_package_available("PIL")


def is_requests_available():
    """检查Requests（HTTP库）是否可用"""
    return _is_package_available("requests")


def is_rouge_available():
    """检查Rouge（中文文本评估工具）是否可用"""
    return _is_package_available("rouge_chinese")


def is_starlette_available():
    """检查SSE-Starlette（服务器发送事件库）是否可用"""
    return _is_package_available("sse_starlette")


@lru_cache  # 使用LRU缓存装饰器避免重复检查
def is_transformers_version_greater_than(content: str):
    """
    检查Transformers库版本是否大于指定版本
    参数:
        content: 版本号字符串
    返回:
        bool: 是否大于指定版本
    """
    return _get_package_version("transformers") >= version.parse(content)


@lru_cache  # 使用LRU缓存装饰器避免重复检查
def is_transformers_version_equal_to_4_46():
    """检查Transformers库版本是否为4.46.x版本"""
    return version.parse("4.46.0") <= _get_package_version("transformers") <= version.parse("4.46.1")


def is_uvicorn_available():
    """检查Uvicorn（ASGI服务器）是否可用"""
    return _is_package_available("uvicorn")


def is_vllm_available():
    """检查vLLM（大语言模型推理优化库）是否可用"""
    return _is_package_available("vllm")
