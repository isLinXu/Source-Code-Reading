import datetime
import logging
import logging.handlers
import os
import sys

import requests

from mgm.constants import LOGDIR

# 定义错误消息常量
server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"  # 服务器高流量错误提示
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."  # 内容审核未通过提示

handler = None  # 全局日志处理器


def build_logger(logger_name, logger_filename):
    """构建日志记录器
    Args:
        logger_name: str - 日志记录器名称
        logger_filename: str - 日志文件名
    Returns:
        logging.Logger - 配置好的日志记录器
    """
    global handler

    # 设置日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",  # 日志格式：时间 | 级别 | 名称 | 消息
        datefmt="%Y-%m-%d %H:%M:%S",  # 日期格式
    )

    # 设置根日志记录器的格式
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)  # 如果根日志记录器没有处理器，则进行基本配置
    logging.getLogger().handlers[0].setFormatter(formatter)  # 设置第一个处理器的格式

    # 将标准输出和标准错误重定向到日志记录器
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # 获取指定名称的日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 为所有日志记录器添加文件处理器
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)  # 创建日志目录
        filename = os.path.join(LOGDIR, logger_filename)  # 日志文件路径
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')  # 每天轮换日志文件
        handler.setFormatter(formatter)

        # 为所有日志记录器添加处理器
        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    将流输出重定向到日志记录器的伪文件对象
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout  # 保存原始标准输出
        self.logger = logger  # 目标日志记录器
        self.log_level = log_level  # 日志级别
        self.linebuf = ''  # 行缓冲区

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)  # 转发未定义属性到原始标准输出

    def write(self, buf):
        """写入数据到日志记录器"""
        temp_linebuf = self.linebuf + buf  # 合并缓冲区和新数据
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):  # 按行处理
            # 从io.TextIOWrapper文档：
            #   在输出时，如果newline为None，任何写入的'\n'字符
            #   都会被转换为系统默认的行分隔符。
            # 默认情况下sys.stdout.write()期望'\n'换行符，
            # 然后进行转换，因此这仍然是跨平台的。
            if line[-1] == '\n':  # 如果行以换行符结尾
                self.logger.log(self.log_level, line.rstrip())  # 记录日志
            else:
                self.linebuf += line  # 将不完整的行存入缓冲区

    def flush(self):
        """刷新缓冲区"""
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())  # 记录缓冲区内容
        self.linebuf = ''


def disable_torch_init():
    """
    禁用torch的默认初始化以加速模型创建
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)  # 禁用Linear层的初始化
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)  # 禁用LayerNorm层的初始化


def violates_moderation(text):
    """
    检查文本是否违反OpenAI内容审核API
    Args:
        text: str - 待检查的文本
    Returns:
        bool - 是否违反内容审核
    """
    url = "https://api.openai.com/v1/moderations"  # OpenAI审核API地址
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}  # 请求头
    text = text.replace("\n", "")  # 去除换行符
    data = "{" + '"input": ' + f'"{text}"' + "}"  # 构造请求数据
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)  # 发送请求
        flagged = ret.json()["results"][0]["flagged"]  # 获取审核结果
    except requests.exceptions.RequestException as e:  # 处理请求异常
        flagged = False
    except KeyError as e:  # 处理键错误
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    """美化打印信号量对象
    Args:
        semaphore: threading.Semaphore - 信号量对象
    Returns:
        str - 格式化后的信号量信息
    """
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"  # 返回信号量状态信息
