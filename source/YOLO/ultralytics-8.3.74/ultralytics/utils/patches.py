# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Monkey patches to update/extend functionality of existing functions."""
# 猴子补丁，用于更新/扩展现有功能。

import time
from pathlib import Path

import cv2
import numpy as np
import torch

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
# OpenCV 多语言友好的函数
_imshow = cv2.imshow  # copy to avoid recursion errors
# 复制 cv2.imshow 以避免递归错误


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    """
    Read an image from a file.
    从文件中读取图像。

    Args:
        filename (str): Path to the file to read.
        filename（字符串）：要读取的文件路径。
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.
        flags（整数，可选）：可以取 cv2.IMREAD_* 的值的标志。默认为 cv2.IMREAD_COLOR。

    Returns:
        (np.ndarray): The read image.
        (np.ndarray)：读取的图像。
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)
    # 从文件中读取图像，并解码为指定的格式


def imwrite(filename: str, img: np.ndarray, params=None):
    """
    Write an image to a file.
    将图像写入文件。

    Args:
        filename (str): Path to the file to write.
        filename（字符串）：要写入的文件路径。
        img (np.ndarray): Image to write.
        img（np.ndarray）：要写入的图像。
        params (list of ints, optional): Additional parameters. See OpenCV documentation.
        params（整数列表，可选）：附加参数。请参见 OpenCV 文档。

    Returns:
        (bool): True if the file was written, False otherwise.
        (布尔值)：如果文件写入成功，则返回 True，否则返回 False。
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        # 使用 cv2.imencode 将图像编码为指定格式并写入文件
        return True
    except Exception:
        return False
        # 如果发生异常，返回 False


def imshow(winname: str, mat: np.ndarray):
    """
    Displays an image in the specified window.
    在指定窗口中显示图像。

    Args:
        winname (str): Name of the window.
        winname（字符串）：窗口的名称。
        mat (np.ndarray): Image to be shown.
        mat（np.ndarray）：要显示的图像。
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)
    # 使用 OpenCV 显示图像，处理窗口名称的编码


# PyTorch functions ----------------------------------------------------------------------------------------------------
# PyTorch 函数
_torch_load = torch.load  # copy to avoid recursion errors
_torch_save = torch.save


def torch_load(*args, **kwargs):
    """
    Load a PyTorch model with updated arguments to avoid warnings.
    加载 PyTorch 模型，并更新参数以避免警告。

    This function wraps torch.load and adds the 'weights_only' argument for PyTorch 1.13.0+ to prevent warnings.
    此函数包装 torch.load，并为 PyTorch 1.13.0+ 添加 'weights_only' 参数以防止警告。

    Args:
        *args (Any): Variable length argument list to pass to torch.load.
        *args（任何类型）：可变长度的参数列表，传递给 torch.load。
        **kwargs (Any): Arbitrary keyword arguments to pass to torch.load.
        **kwargs（任何类型）：传递给 torch.load 的任意关键字参数。

    Returns:
        (Any): The loaded PyTorch object.
        (任何类型)：加载的 PyTorch 对象。

    Note:
        For PyTorch versions 2.0 and above, this function automatically sets 'weights_only=False'
        if the argument is not provided, to avoid deprecation warnings.
        注意：对于 PyTorch 2.0 及以上版本，如果未提供该参数，则此函数会自动将 'weights_only' 设置为 False，以避免弃用警告。
    """
    from ultralytics.utils.torch_utils import TORCH_1_13
    # 从 ultralytics.utils.torch_utils 导入 TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False
        # 如果是 PyTorch 1.13，并且未提供 'weights_only' 参数，则将其设置为 False

    return _torch_load(*args, **kwargs)
    # 调用原始的 torch.load 函数


def torch_save(*args, **kwargs):
    """
    Optionally use dill to serialize lambda functions where pickle does not, adding robustness with 3 retries and
    exponential standoff in case of save failure.
    可选地使用 dill 序列化 lambda 函数（在 pickle 无法序列化的情况下），在保存失败时增加 3 次重试和指数延迟的健壮性。

    Args:
        *args (tuple): Positional arguments to pass to torch.save.
        *args（元组）：传递给 torch.save 的位置参数。
        **kwargs (Any): Keyword arguments to pass to torch.save.
        **kwargs（任何类型）：传递给 torch.save 的关键字参数。
    """
    for i in range(4):  # 3 retries
        # 尝试 3 次重试
        try:
            return _torch_save(*args, **kwargs)
            # 调用原始的 torch.save 函数
        except RuntimeError as e:  # unable to save, possibly waiting for device to flush or antivirus scan
            # 如果无法保存，可能是等待设备刷新或防病毒扫描
            if i == 3:
                raise e
                # 如果是最后一次重试，则抛出异常
            time.sleep((2**i) / 2)  # exponential standoff: 0.5s, 1.0s, 2.0s
            # 指数延迟：0.5秒、1.0秒、2.0秒