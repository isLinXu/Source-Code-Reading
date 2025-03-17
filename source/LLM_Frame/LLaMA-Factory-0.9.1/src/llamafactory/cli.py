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
import random
import subprocess
import sys
from enum import Enum, unique

from . import launcher
from .api.app import run_api
from .chat.chat_model import run_chat
from .eval.evaluator import run_eval
from .extras import logging
from .extras.env import VERSION, print_env
from .extras.misc import get_device_count
from .train.tuner import export_model, run_exp
from .webui.interface import run_web_demo, run_web_ui


# 定义命令行使用说明
USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli eval -h: evaluate models                        |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "-" * 70
)

# 欢迎信息模板
WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to LLaMA Factory, version {VERSION}"
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
    + "-" * 58
)

logger = logging.get_logger(__name__)


# 定义命令枚举类型
@unique
class Command(str, Enum):
    API = "api"  # 启动API服务
    CHAT = "chat"  # 启动CLI聊天
    ENV = "env"  # 显示环境信息
    EVAL = "eval"  # 模型评估
    EXPORT = "export"  # 导出模型
    TRAIN = "train"  # 训练模型
    WEBDEMO = "webchat"  # Web聊天界面
    WEBUI = "webui"  # 启动LlamaBoard
    VER = "version"  # 显示版本
    HELP = "help"  # 显示帮助


def main():
    # 获取用户输入的命令（默认显示帮助）
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    
    # 命令分发逻辑
    if command == Command.API:
        run_api()  # 启动API服务
    elif command == Command.CHAT:
        run_chat()  # 启动CLI聊天
    elif command == Command.ENV:
        print_env()  # 打印环境信息
    elif command == Command.EVAL:
        run_eval()  # 执行模型评估
    elif command == Command.EXPORT:
        export_model()  # 导出合并后的模型
    elif command == Command.TRAIN:
        # 分布式训练处理逻辑
        force_torchrun = os.getenv("FORCE_TORCHRUN", "0").lower() in ["true", "1"]  # 强制使用torchrun
        if force_torchrun or get_device_count() > 1:  # 多卡训练条件判断
            # 设置分布式参数
            master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")  # 主节点地址
            master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))  # 随机端口
            logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")
            
            # 构建torchrun命令
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=os.getenv("NNODES", "1"),  # 节点数
                    node_rank=os.getenv("NODE_RANK", "0"),  # 当前节点rank
                    nproc_per_node=os.getenv("NPROC_PER_NODE", str(get_device_count())),  # 每卡进程数
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,  # 启动脚本路径
                    args=" ".join(sys.argv[1:]),  # 传递剩余参数
                )
                .split()
            )
            sys.exit(process.returncode)  # 退出并返回子进程状态码
        else:
            run_exp()  # 单卡训练
    elif command == Command.WEBDEMO:
        run_web_demo()  # 启动Web聊天
    elif command == Command.WEBUI:
        run_web_ui()  # 启动LlamaBoard
    elif command == Command.VER:
        print(WELCOME)  # 显示版本信息
    elif command == Command.HELP:
        print(USAGE)  # 显示帮助信息
    else:
        raise NotImplementedError(f"Unknown command: {command}.")  # 未知命令处理
