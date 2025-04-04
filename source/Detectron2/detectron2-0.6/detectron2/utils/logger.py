# Copyright (c) Facebook, Inc. and its affiliates. # 版权声明：属于Facebook, Inc.及其附属公司
import atexit  # 导入atexit模块，用于注册程序退出时执行的函数
import functools  # 导入functools模块，提供高阶函数和可调用对象的操作
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，用于与操作系统交互
import sys  # 导入sys模块，提供对解释器使用或维护的变量和函数的访问
import time  # 导入time模块，用于时间相关功能
from collections import Counter  # 从collections导入Counter类，用于计数
import torch  # 导入PyTorch库
from tabulate import tabulate  # 导入tabulate函数，用于创建格式化表格
from termcolor import colored  # 导入colored函数，用于在终端中输出彩色文本

from detectron2.utils.file_io import PathManager  # 从detectron2.utils.file_io导入PathManager

__all__ = ["setup_logger", "log_first_n", "log_every_n", "log_every_n_seconds"]  # 定义模块的公开API


class _ColorfulFormatter(logging.Formatter):  # 定义彩色日志格式化器类，继承自logging.Formatter
    def __init__(self, *args, **kwargs):  # 初始化方法
        self._root_name = kwargs.pop("root_name") + "."  # 从kwargs获取并移除root_name参数，添加点号
        self._abbrev_name = kwargs.pop("abbrev_name", "")  # 从kwargs获取并移除abbrev_name参数，默认为空字符串
        if len(self._abbrev_name):  # 如果缩写名称非空
            self._abbrev_name = self._abbrev_name + "."  # 添加点号
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)  # 调用父类初始化方法

    def formatMessage(self, record):  # 重写formatMessage方法
        record.name = record.name.replace(self._root_name, self._abbrev_name)  # 将记录中的根名称替换为缩写名称
        log = super(_ColorfulFormatter, self).formatMessage(record)  # 调用父类方法格式化消息
        if record.levelno == logging.WARNING:  # 如果是警告级别日志
            prefix = colored("WARNING", "red", attrs=["blink"])  # 创建红色闪烁的WARNING前缀
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:  # 如果是错误或严重级别日志
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])  # 创建红色闪烁下划线的ERROR前缀
        else:  # 其他级别日志
            return log  # 直接返回原始日志
        return prefix + " " + log  # 返回带前缀的日志


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers  # 使用LRU缓存装饰器，这样多次调用setup_logger不会添加多个处理器
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="detectron2", abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.

    Returns:
        logging.Logger: a logger
    """  # 初始化detectron2日志记录器并将其详细级别设置为"DEBUG"。
        # 参数:
        # output (str): 保存日志的文件名或目录。如果为None，将不保存日志文件。
        #     如果以".txt"或".log"结尾，则视为文件名。
        #     否则，日志将保存到`output/log.txt`。
        # name (str): 此日志记录器的根模块名称
        # abbrev_name (str): 模块的缩写，避免日志中出现长名称。
        #     设置为""以不在日志中记录根模块。
        #     默认情况下，将"detectron2"缩写为"d2"，保持其他模块不变。
        # 返回:
        # logging.Logger: 一个日志记录器

    logger = logging.getLogger(name)  # 获取指定名称的日志记录器
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG
    logger.propagate = False  # 禁止日志传播到父级记录器

    if abbrev_name is None:  # 如果未提供缩写名称
        abbrev_name = "d2" if name == "detectron2" else name  # 如果名称是"detectron2"则使用"d2"，否则使用原名

    plain_formatter = logging.Formatter(  # 创建一个普通的日志格式化器
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"  # 定义日志格式和日期格式
    )
    # stdout logging: master only  # 标准输出日志记录：仅主进程
    if distributed_rank == 0:  # 如果是主进程（分布式排名为0）
        ch = logging.StreamHandler(stream=sys.stdout)  # 创建一个输出到标准输出的流处理器
        ch.setLevel(logging.DEBUG)  # 设置处理器的日志级别为DEBUG
        if color:  # 如果启用颜色
            formatter = _ColorfulFormatter(  # 创建一个彩色格式化器
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",  # 定义彩色日志格式
                datefmt="%m/%d %H:%M:%S",  # 定义日期格式
                root_name=name,  # 设置根名称
                abbrev_name=str(abbrev_name),  # 设置缩写名称
            )
        else:  # 如果不启用颜色
            formatter = plain_formatter  # 使用普通格式化器
        ch.setFormatter(formatter)  # 设置处理器的格式化器
        logger.addHandler(ch)  # 将处理器添加到日志记录器

    # file logging: all workers  # 文件日志记录：所有工作进程
    if output is not None:  # 如果提供了输出路径
        if output.endswith(".txt") or output.endswith(".log"):  # 如果输出路径以.txt或.log结尾
            filename = output  # 直接使用输出路径作为文件名
        else:  # 否则
            filename = os.path.join(output, "log.txt")  # 将输出路径和log.txt拼接作为文件名
        if distributed_rank > 0:  # 如果是非主进程（分布式排名大于0）
            filename = filename + ".rank{}".format(distributed_rank)  # 在文件名后添加排名后缀
        PathManager.mkdirs(os.path.dirname(filename))  # 创建文件所在目录

        fh = logging.StreamHandler(_cached_log_stream(filename))  # 创建一个输出到缓存日志流的流处理器
        fh.setLevel(logging.DEBUG)  # 设置处理器的日志级别为DEBUG
        fh.setFormatter(plain_formatter)  # 设置处理器的格式化器为普通格式化器
        logger.addHandler(fh)  # 将处理器添加到日志记录器

    return logger  # 返回配置好的日志记录器


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.  # 缓存已打开的文件对象，使得不同的setup_logger调用（使用相同的文件名）可以安全地写入同一个文件
@functools.lru_cache(maxsize=None)  # 使用LRU缓存装饰器，不限制缓存大小
def _cached_log_stream(filename):  # 缓存日志流函数
    # use 1K buffer if writing to cloud storage  # 如果写入云存储，使用1K缓冲区
    io = PathManager.open(filename, "a", buffering=1024 if "://" in filename else -1)  # 以追加模式打开文件，根据是否为云存储设置缓冲区大小
    atexit.register(io.close)  # 注册程序退出时关闭文件的函数
    return io  # 返回文件对象


"""
Below are some other convenient logging methods.
They are mainly adopted from
https://github.com/abseil/abseil-py/blob/master/absl/logging/__init__.py
"""  # 以下是一些其他方便的日志记录方法。
    # 它们主要来自
    # https://github.com/abseil/abseil-py/blob/master/absl/logging/__init__.py


def _find_caller():  # 查找调用者函数
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """  # 返回:
        # str: 调用者的模块名
        # tuple: 用于识别不同调用者的可哈希键

    frame = sys._getframe(2)  # 获取调用堆栈中的第三帧（跳过当前函数和直接调用者）
    while frame:  # 遍历堆栈帧
        code = frame.f_code  # 获取帧的代码对象
        if os.path.join("utils", "logger.") not in code.co_filename:  # 如果代码文件名不包含"utils/logger."
            mod_name = frame.f_globals["__name__"]  # 获取模块名
            if mod_name == "__main__":  # 如果模块名是__main__
                mod_name = "detectron2"  # 将模块名替换为detectron2
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)  # 返回模块名和可哈希键（文件名、行号、代码名）
        frame = frame.f_back  # 获取前一帧


_LOG_COUNTER = Counter()  # 创建日志计数器
_LOG_TIMER = {}  # 创建日志计时器字典


def log_first_n(lvl, msg, n=1, *, name=None, key="caller"):  # 仅记录前n次日志的函数
    """
    Log only for the first n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
        key (str or tuple[str]): the string(s) can be one of "caller" or
            "message", which defines how to identify duplicated logs.
            For example, if called with `n=1, key="caller"`, this function
            will only log the first call from the same caller, regardless of
            the message content.
            If called with `n=1, key="message"`, this function will log the
            same content only once, even if they are called from different places.
            If called with `n=1, key=("caller", "message")`, this function
            will not log only if the same caller has logged the same message before.
    """  # 仅记录前n次日志。
        # 参数:
        # lvl (int): 日志级别
        # msg (str): 日志消息
        # n (int): 记录次数
        # name (str): 要使用的日志记录器名称。默认使用调用者的模块。
        # key (str或tuple[str]): 字符串可以是"caller"或"message"之一，
        #     用于定义如何识别重复的日志。
        #     例如，如果使用`n=1, key="caller"`调用，该函数
        #     将只记录同一调用者的第一次调用，无论消息内容如何。
        #     如果使用`n=1, key="message"`调用，该函数将记录
        #     相同内容仅一次，即使它们来自不同的地方。
        #     如果使用`n=1, key=("caller", "message")`调用，该函数
        #     将仅在同一调用者之前已记录相同消息时不记录。

    if isinstance(key, str):  # 如果key是字符串
        key = (key,)  # 将其转换为元组
    assert len(key) > 0  # 断言key长度大于0

    caller_module, caller_key = _find_caller()  # 获取调用者模块和调用者键
    hash_key = ()  # 初始化哈希键为空元组
    if "caller" in key:  # 如果key包含"caller"
        hash_key = hash_key + caller_key  # 将调用者键添加到哈希键
    if "message" in key:  # 如果key包含"message"
        hash_key = hash_key + (msg,)  # 将消息添加到哈希键

    _LOG_COUNTER[hash_key] += 1  # 增加哈希键对应的计数
    if _LOG_COUNTER[hash_key] <= n:  # 如果计数小于等于n
        logging.getLogger(name or caller_module).log(lvl, msg)  # 记录日志


def log_every_n(lvl, msg, n=1, *, name=None):  # 每n次记录一次日志的函数
    """
    Log once per n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """  # 每n次记录一次日志。
        # 参数:
        # lvl (int): 日志级别
        # msg (str): 日志消息
        # n (int): 间隔次数
        # name (str): 要使用的日志记录器名称。默认使用调用者的模块。

    caller_module, key = _find_caller()  # 获取调用者模块和键
    _LOG_COUNTER[key] += 1  # 增加键对应的计数
    if n == 1 or _LOG_COUNTER[key] % n == 1:  # 如果n为1或计数对n取模为1
        logging.getLogger(name or caller_module).log(lvl, msg)  # 记录日志


def log_every_n_seconds(lvl, msg, n=1, *, name=None):  # 每n秒记录一次日志的函数
    """
    Log no more than once per n seconds.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """  # 每n秒最多记录一次日志。
        # 参数:
        # lvl (int): 日志级别
        # msg (str): 日志消息
        # n (int): 间隔秒数
        # name (str): 要使用的日志记录器名称。默认使用调用者的模块。

    caller_module, key = _find_caller()  # 获取调用者模块和键
    last_logged = _LOG_TIMER.get(key, None)  # 获取上次记录时间，如果不存在返回None
    current_time = time.time()  # 获取当前时间
    if last_logged is None or current_time - last_logged >= n:  # 如果未记录过或距离上次记录已超过n秒
        logging.getLogger(name or caller_module).log(lvl, msg)  # 记录日志
        _LOG_TIMER[key] = current_time  # 更新上次记录时间


def create_small_table(small_dict):  # 创建小表格的函数
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """  # 使用small_dict的键作为表头创建小表格。这仅适用于小型字典。
        # 参数:
        # small_dict (dict): 只有几个项目的结果字典。
        # 返回:
        # str: 表格字符串。

    keys, values = tuple(zip(*small_dict.items()))  # 将字典的键和值分别提取为元组
    table = tabulate(  # 使用tabulate创建表格
        [values],  # 值列表
        headers=keys,  # 表头为键
        tablefmt="pipe",  # 表格格式为pipe
        floatfmt=".3f",  # 浮点数格式为保留3位小数
        stralign="center",  # 字符串对齐方式为居中
        numalign="center",  # 数字对齐方式为居中
    )
    return table  # 返回表格字符串


def _log_api_usage(identifier: str):  # 记录API使用情况的内部函数
    """
    Internal function used to log the usage of different detectron2 components
    inside facebook's infra.
    """  # 内部函数，用于记录在Facebook基础设施内部使用的不同detectron2组件的使用情况。

    torch._C._log_api_usage_once("detectron2." + identifier)  # 使用PyTorch的API使用记录功能记录detectron2组件的使用
