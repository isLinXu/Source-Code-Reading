import datetime
import logging
import logging.handlers
import os
import sys

import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    """
    构建日志记录器

    参数:
        logger_name(str): 日志记录器的名称
        logger_filename(str): 日志文件的名称

    返回:
        logger: 配置好的日志记录器实例
    """
    global handler
    # 配置日志消息格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    # 设置根处理器的格式
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    # 重定向标准输出和错误输出到日志记录器
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    # 获取日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    # 为所有日志记录器添加文件处理器
    if handler is None:
        # 确保日志目录存在
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)
        # 为所有现有日志记录器添加文件处理器
        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    StreamToLogger类提供了一个假的文件流对象，将写入操作重定向到一个日志记录器实例。
    这个类的主要用途是捕获标准输出或标准错误，并将它们重定向到日志，以便于日志管理
    """
    def __init__(self, logger, log_level=logging.INFO):
        """
        初始化StreamToLogger对象。

        :param logger: 日志记录器实例，用于输出日志。
        :param log_level: 日志记录的级别，默认为INFO级别。
        """
        self.terminal = sys.stdout  # 保存原始的标准输出流
        self.logger = logger  # 用于记录日志的logger对象
        self.log_level = log_level  # 设置日志记录级别
        self.linebuf = ''  # 缓冲区，用于存储待记录的日志行

    def __getattr__(self, attr):
        """
        代理到终端对象的属性获取。

        :param attr: 查询的属性名称。
        :return: 终端对象的属性值。
        """
        return getattr(self.terminal, attr)

    def write(self, buf):
        """
        写入缓冲区，并根据内容进行换行处理和日志记录。

        :param buf: 要写入的字符串。
        """
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            # 从io.TextIOWrapper文档得知：
            # 在输出时，如果newline为None，则写入的任何'\n'字符都会转换为系统默认的行分隔符。
            # 默认情况下，sys.stdout.write()期望'\n'换行符并进行转换，因此这仍然是跨平台的。
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        """
        刷新缓冲区，将剩余的内容记录到日志中。
        """
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    禁用冗余的 PyTorch 默认初始化以加速模型创建。
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    检查文本是否违反 OpenAI 的审核 API。

    Parameters:
    - text: 需要检查的文本。

    Returns:
    布尔值，表示文本是否违规。
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    """
    以友好的方式打印信号量对象的状态。

    Parameters:
    - semaphore: 要打印的信号量对象，可以为 None。

    Returns:
    描述信号量状态的字符串。
    """
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
