# Copyright 2024 Optuna, HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/logging.py
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

import logging  # 导入Python标准库中的日志模块
import os  # 导入操作系统接口模块
import sys  # 导入系统特定参数和函数模块
import threading  # 导入线程相关模块
from concurrent.futures import ThreadPoolExecutor  # 导入线程池执行器
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import Optional  # 导入类型提示模块

from .constants import RUNNING_LOG  # 从常量模块导入运行日志文件名


_thread_lock = threading.RLock()  # 创建可重入线程锁
_default_handler: Optional["logging.Handler"] = None  # 默认日志处理器
_default_log_level: "logging._Level" = logging.INFO  # 默认日志级别


class LoggerHandler(logging.Handler):
    r"""
    Redirects the logging output to the logging file for LLaMA Board.
    将日志输出重定向到LLaMA Board的日志文件中
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self._formatter = logging.Formatter(  # 创建日志格式化器
            fmt="[%(levelname)s|%(asctime)s] %(filename)s:%(lineno)s >> %(message)s",  # 日志格式：[级别|时间] 文件名:行号 >> 消息
            datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式
        )
        self.setLevel(logging.INFO)  # 设置日志级别为INFO
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果不存在）
        self.running_log = os.path.join(output_dir, RUNNING_LOG)  # 构建日志文件完整路径
        if os.path.exists(self.running_log):  # 如果日志文件已存在
            os.remove(self.running_log)  # 删除已存在的日志文件

        self.thread_pool = ThreadPoolExecutor(max_workers=1)  # 创建单线程的线程池

    def _write_log(self, log_entry: str) -> None:  # 写入日志条目的私有方法
        with open(self.running_log, "a", encoding="utf-8") as f:  # 以追加模式打开日志文件
            f.write(log_entry + "\n\n")  # 写入日志条目并添加空行

    def emit(self, record) -> None:  # 发送日志记录的方法
        if record.name == "httpx":  # 如果是httpx模块的日志
            return  # 忽略httpx模块的日志

        log_entry = self._formatter.format(record)  # 格式化日志记录
        self.thread_pool.submit(self._write_log, log_entry)  # 提交写日志任务到线程池

    def close(self) -> None:  # 关闭日志处理器
        self.thread_pool.shutdown(wait=True)  # 关闭线程池，等待所有任务完成
        return super().close()  # 调用父类的close方法


class _Logger(logging.Logger):
    r"""
    A logger that supports info_rank0 and warning_once.
    支持info_rank0和warning_once方法的日志记录器
    """

    def info_rank0(self, *args, **kwargs) -> None:  # 仅在rank0进程输出info日志
        self.info(*args, **kwargs)

    def warning_rank0(self, *args, **kwargs) -> None:  # 仅在rank0进程输出warning日志
        self.warning(*args, **kwargs)

    def warning_once(self, *args, **kwargs) -> None:  # 仅输出一次warning日志
        self.warning(*args, **kwargs)


def _get_default_logging_level() -> "logging._Level":
    r"""
    Returns the default logging level.
    返回默认的日志级别
    """
    env_level_str = os.environ.get("LLAMAFACTORY_VERBOSITY", None)  # 从环境变量获取日志级别
    if env_level_str:  # 如果设置了环境变量
        if env_level_str.upper() in logging._nameToLevel:  # 如果是有效的日志级别名称
            return logging._nameToLevel[env_level_str.upper()]  # 返回对应的日志级别
        else:
            raise ValueError(f"Unknown logging level: {env_level_str}.")  # 抛出错误：未知的日志级别

    return _default_log_level  # 返回默认日志级别


def _get_library_name() -> str:  # 获取库名称
    return __name__.split(".")[0]  # 返回模块名的第一部分


def _get_library_root_logger() -> "_Logger":  # 获取库的根日志记录器
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    r"""
    Configures root logger using a stdout stream handler with an explicit format.
    使用标准输出流处理器和显式格式配置根日志记录器
    """
    global _default_handler  # 声明使用全局变量

    with _thread_lock:  # 使用线程锁确保线程安全
        if _default_handler:  # 如果已经配置过
            return  # 直接返回

        formatter = logging.Formatter(  # 创建格式化器
            fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _default_handler = logging.StreamHandler(sys.stdout)  # 创建标准输出流处理器
        _default_handler.setFormatter(formatter)  # 设置格式化器
        library_root_logger = _get_library_root_logger()  # 获取根日志记录器
        library_root_logger.addHandler(_default_handler)  # 添加处理器
        library_root_logger.setLevel(_get_default_logging_level())  # 设置日志级别
        library_root_logger.propagate = False  # 禁止日志传播到更高级别的处理器


def get_logger(name: Optional[str] = None) -> "_Logger":
    r"""
    Returns a logger with the specified name. It it not supposed to be accessed externally.
    返回指定名称的日志记录器。不建议在外部直接访问。
    """
    if name is None:  # 如果没有指定名称
        name = _get_library_name()  # 使用库名作为日志记录器名称

    _configure_library_root_logger()  # 确保根日志记录器已配置
    return logging.getLogger(name)  # 返回指定名称的日志记录器


def add_handler(handler: "logging.Handler") -> None:
    r"""
    Adds a handler to the root logger.
    向根日志记录器添加处理器
    """
    _configure_library_root_logger()  # 确保根日志记录器已配置
    _get_library_root_logger().addHandler(handler)  # 添加处理器


def remove_handler(handler: logging.Handler) -> None:
    r"""
    Removes a handler to the root logger.
    从根日志记录器移除处理器
    """
    _configure_library_root_logger()  # 确保根日志记录器已配置
    _get_library_root_logger().removeHandler(handler)  # 移除处理器


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:  # 如果是rank0进程（主进程）
        self.info(*args, **kwargs)  # 输出info级别日志


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:  # 如果是rank0进程（主进程）
        self.warning(*args, **kwargs)  # 输出warning级别日志


@lru_cache(None)  # 使用LRU缓存装饰器，确保相同的警告只输出一次
def warning_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:  # 如果是rank0进程（主进程）
        self.warning(*args, **kwargs)  # 输出warning级别日志


# 将新方法添加到Logger类
logging.Logger.info_rank0 = info_rank0  # 添加info_rank0方法
logging.Logger.warning_rank0 = warning_rank0  # 添加warning_rank0方法
logging.Logger.warning_once = warning_once  # 添加warning_once方法
