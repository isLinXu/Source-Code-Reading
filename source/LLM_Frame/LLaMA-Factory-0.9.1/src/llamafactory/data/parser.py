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

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from transformers.utils import cached_file

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope, use_openmind


# 数据集属性配置类
@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    数据集属性配置类，用于定义数据集的加载方式和结构
    """
    # 基础配置
    load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file"]  # 数据来源：HuggingFace/ModelScope/OpenMind/脚本/文件
    dataset_name: str  # 数据集名称或路径
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"  # 数据格式：Alpaca格式或ShareGPT对话格式
    ranking: bool = False  # 是否为排序任务数据
    
    # 扩展配置
    subset: Optional[str] = None  # 子集名称（用于多子集数据集）
    split: str = "train"  # 数据集划分（train/validation/test）
    folder: Optional[str] = None  # 自定义数据文件夹路径
    num_samples: Optional[int] = None  # 最大采样数量
    
    # 通用字段映射
    system: Optional[str] = None  # 系统提示字段名
    tools: Optional[str] = None  # 工具调用字段名
    images: Optional[str] = None  # 图像字段名
    videos: Optional[str] = None  # 视频字段名
    
    # RLHF相关字段
    chosen: Optional[str] = None  # 优选回答字段名
    rejected: Optional[str] = None  # 拒绝回答字段名
    kto_tag: Optional[str] = None  # KTO标签字段名
    
    # Alpaca格式字段映射
    prompt: Optional[str] = "instruction"  # 指令字段名
    query: Optional[str] = "input"  # 输入字段名
    response: Optional[str] = "output"  # 输出字段名
    history: Optional[str] = None  # 历史对话字段名
    
    # ShareGPT格式字段映射
    messages: Optional[str] = "conversations"  # 对话列表字段名
    
    # ShareGPT标签配置
    role_tag: Optional[str] = "from"  # 角色标签字段名
    content_tag: Optional[str] = "value"  # 内容标签字段名
    user_tag: Optional[str] = "human"  # 用户角色标识
    assistant_tag: Optional[str] = "gpt"  # 助手角色标识
    observation_tag: Optional[str] = "observation"  # 观察结果标识
    function_tag: Optional[str] = "function_call"  # 函数调用标识
    system_tag: Optional[str] = "system"  # 系统消息标识

    def __repr__(self) -> str:
        return self.dataset_name  # 实例表示时返回数据集名称

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        """动态设置属性值，用于从配置字典加载参数"""
        setattr(self, key, obj.get(key, default))  # 从字典获取值或使用默认值


def get_dataset_list(dataset_names: Optional[Sequence[str]], dataset_dir: str) -> List["DatasetAttr"]:
    r"""
    Gets the attributes of the datasets.
    根据数据集名称和目录获取数据集属性配置列表
    """
    dataset_list: List["DatasetAttr"] = []
    if dataset_names is None:
        return dataset_list

    # 处理在线数据集配置
    if dataset_dir == "ONLINE":
        dataset_info = None
    else:
        # 处理远程数据集配置
        if dataset_dir.startswith("REMOTE:"):
            config_path = cached_file(
                path_or_repo_id=dataset_dir[7:],  # 去除REMOTE:前缀
                filename=DATA_CONFIG,  # 使用预定义配置文件名
                repo_type="dataset"
            )
        else:
            config_path = os.path.join(dataset_dir, DATA_CONFIG)  # 本地配置文件路径

        # 加载配置文件
        try:
            with open(config_path) as f:
                dataset_info = json.load(f)  # 解析JSON配置
        except Exception as err:
            if dataset_names:
                raise ValueError(f"Cannot open {config_path} due to {str(err)}.")
            dataset_info = None

    # 遍历数据集名称构建配置
    for name in dataset_names:
        if dataset_info is None:  # 在线数据集
            # 根据平台选择数据源
            if use_modelscope():
                load_from = "ms_hub"
            elif use_openmind():
                load_from = "om_hub"
            else:
                load_from = "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")

        # 确定数据来源优先级
        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]
        has_om_url = "om_hub_url" in dataset_info[name]

        # 根据平台优先级选择数据源
        if has_hf_url or has_ms_url or has_om_url:
            if has_ms_url and (use_modelscope() or not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"])
            elif has_om_url and (use_openmind() or not has_hf_url):
                dataset_attr = DatasetAttr("om_hub", dataset_name=dataset_info[name]["om_hub_url"])
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        elif "script_url" in dataset_info[name]:  # 脚本加载方式
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        else:  # 本地文件加载方式
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        # 设置基础属性
        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("split", dataset_info[name], default="train")
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("num_samples", dataset_info[name])

        # 设置列映射关系
        if "columns" in dataset_info[name]:
            column_names = ["system", "tools", "images", "videos", "chosen", "rejected", "kto_tag"]
            # 根据格式添加不同字段
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history"])
            else:
                column_names.append("messages")

            # 遍历设置列属性
            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        # 设置ShareGPT格式的标签
        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
            tag_names = (
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            )
            for tag in tag_names:
                dataset_attr.set_attr(tag, dataset_info[name]["tags"])

        dataset_list.append(dataset_attr)

    return dataset_list
