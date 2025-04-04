# Copyright (c) Facebook, Inc. and its affiliates.

import logging  # 导入日志模块，用于记录日志信息
from contextlib import contextmanager  # 导入上下文管理器装饰器，用于创建上下文管理器
from functools import wraps  # 导入wraps装饰器，用于保留被装饰函数的元信息
import torch  # 导入PyTorch库

__all__ = ["retry_if_cuda_oom"]  # 定义模块的公开API，只暴露retry_if_cuda_oom函数


@contextmanager  # 使用上下文管理器装饰器
def _ignore_torch_cuda_oom():  # 定义一个忽略PyTorch CUDA内存不足异常的上下文管理器
    """
    A context which ignores CUDA OOM exception from pytorch.
    """  # 一个忽略PyTorch的CUDA内存不足(OOM)异常的上下文
    try:  # 尝试执行上下文中的代码
        yield  # 暂停执行，将控制权交给上下文中的代码块
    except RuntimeError as e:  # 捕获运行时错误
        # NOTE: the string may change?  # 注意：错误字符串可能会变化
        if "CUDA out of memory. " in str(e):  # 如果错误信息中包含CUDA内存不足的字符串
            pass  # 忽略该错误
        else:  # 如果是其他运行时错误
            raise  # 重新抛出异常


def retry_if_cuda_oom(func):  # 定义一个装饰器函数，在CUDA内存不足时重试被装饰的函数
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """  # 使函数在遇到PyTorch的CUDA内存不足(OOM)错误后重试自身。
        # 它首先会在调用`torch.cuda.empty_cache()`后重试。
        # 
        # 如果仍然失败，它会尝试将输入转换到CPU上再重试。
        # 在这种情况下，它期望函数能够转交给CPU实现。
        # 返回值也可能变成CPU张量，用户有责任在需要时将其转回CUDA张量。
        # 
        # 参数：
        #     func: 一个无状态的可调用对象，接受类张量对象作为参数
        # 
        # 返回：
        #     一个可调用对象，在遇到OOM时重试`func`。
        # 
        # 示例：
        # ::
        #     output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        #     # 即使输入在GPU上，输出也可能在CPU上
        # 
        # 注意：
        #     1. 在将输入转换到CPU时，它只会查看每个参数并检查
        #        它是否有`.device`和`.to`用于转换。不支持嵌套的张量结构。
        # 
        #     2. 由于函数可能被调用多次，它必须是无状态的。

    def maybe_to_cpu(x):  # 定义一个辅助函数，尝试将输入转换到CPU
        try:  # 尝试检查对象是否可以转移到CPU
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")  # 检查对象是否是CUDA张量且有to方法
        except AttributeError:  # 捕获属性错误（如对象没有device属性）
            like_gpu_tensor = False  # 设置为False，表示不是类似GPU张量的对象
        if like_gpu_tensor:  # 如果是类似GPU张量的对象
            return x.to(device="cpu")  # 将其转移到CPU并返回
        else:  # 否则
            return x  # 原样返回

    @wraps(func)  # 使用wraps装饰器保留被装饰函数的元信息
    def wrapped(*args, **kwargs):  # 定义包装函数，接受任意位置参数和关键字参数
        with _ignore_torch_cuda_oom():  # 使用忽略CUDA OOM错误的上下文
            return func(*args, **kwargs)  # 尝试执行原函数

        # Clear cache and retry  # 清空缓存并重试
        torch.cuda.empty_cache()  # 清空CUDA缓存
        with _ignore_torch_cuda_oom():  # 再次使用忽略CUDA OOM错误的上下文
            return func(*args, **kwargs)  # 重试执行原函数

        # Try on CPU. This slows down the code significantly, therefore print a notice.  # 尝试在CPU上执行。这会显著降低代码速度，因此打印一个通知。
        logger = logging.getLogger(__name__)  # 获取日志记录器
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))  # 记录尝试将输入复制到CPU的信息
        new_args = (maybe_to_cpu(x) for x in args)  # 将所有位置参数转换到CPU
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}  # 将所有关键字参数转换到CPU
        return func(*new_args, **new_kwargs)  # 使用转换后的参数执行原函数

    return wrapped  # 返回包装后的函数
