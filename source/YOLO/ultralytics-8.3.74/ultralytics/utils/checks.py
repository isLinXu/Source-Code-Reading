# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import torch

from ultralytics.utils import (
    ARM64,
    ASSETS,
    AUTOINSTALL,
    IS_COLAB,
    IS_GIT_DIR,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PYTHON_VERSION,
    RKNN_CHIPS,
    ROOT,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    SimpleNamespace,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    emojis,
    is_github_action_running,
    url2file,
)

def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.
    解析 requirements.txt 文件，忽略以 '#' 开头的行和 '#' 后的任何文本。

    Args:
        file_path (Path): Path to the requirements.txt file.  文件路径：requirements.txt 的路径。
        package (str, optional): Python package to use instead of requirements.txt file, i.e. package='ultralytics'.  
        package（str，可选）：要使用的 Python 包，替代 requirements.txt 文件，例如 package='ultralytics'。

    Returns:
        (List[Dict[str, str]]): List of parsed requirements as dictionaries with `name` and `specifier` keys.
        返回：解析后的需求列表，作为字典，包含 `name` 和 `specifier` 键。

    Example:
        ```python
        from ultralytics.utils.checks import parse_requirements

        parse_requirements(package="ultralytics")
        ```
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.split("#")[0].strip()  # ignore inline comments
            # 忽略行内注释
            if match := re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line):
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.
    将版本字符串转换为整数元组，忽略附加的任何非数字字符串。此函数替代已弃用的 'pkg_resources.parse_version(v)'。

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'  版本字符串，例如 '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
        返回：表示版本数字部分的整数元组和额外字符串，例如 (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.
    检查字符串是否仅由 ASCII 字符组成。

    Args:
        s (str): String to be checked.  要检查的字符串。

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
        返回：如果字符串仅由 ASCII 字符组成，则为 True，否则为 False。
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.
    验证图像大小是否是每个维度给定步幅的倍数。如果图像大小不是步幅的倍数，则将其更新为大于或等于给定下限值的最近步幅倍数。

    Args:
        imgsz (int | cList[int]): Image size.  图像大小。
        stride (int): Stride value.  步幅值。
        min_dim (int): Minimum number of dimensions.  最小维度数。
        max_dim (int): Maximum number of dimensions.  最大维度数。
        floor (int): Minimum allowed value for image size.  图像大小的最小允许值。

    Returns:
        (List[int] | int): Updated image size.  返回：更新后的图像大小。
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # i.e. '640' or '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.
    检查当前版本与所需版本或范围。

    Args:
        current (str): Current version or package name to get version from.  当前版本或用于获取版本的包名称。
        required (str): Required version or range (in pip-style format).  所需版本或范围（以 pip 风格格式）。
        name (str, optional): Name to be used in warning message.  用于警告消息的名称。
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.  如果为 True，则在未满足要求时引发 AssertionError。
        verbose (bool, optional): If True, print warning message if requirement is not met.  如果为 True，则在未满足要求时打印警告消息。
        msg (str, optional): Extra message to display if verbose.  如果 verbose，则显示的额外消息。

    Returns:
        (bool): True if requirement is met, False otherwise.  返回：如果满足要求，则为 True，否则为 False。

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current="22.04", required="==22.04")

        # Check if current version is greater than or equal to 22.04
        check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current="22.04", required="<=22.04")

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current="21.10", required=">20.04,<22.04")
        ```
    """
    if not current:  # if current is '' or None
        LOGGER.warning(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f"WARNING ⚠️ {current} package is required but not installed")) from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))  # assert version requirements met
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ultralytics"):
    """
    Returns the latest version of a PyPI package without downloading or installing it.
    返回 PyPI 包的最新版本，而不下载或安装它。

    Args:
        package_name (str): The name of the package to find the latest version for.  包名称：要查找最新版本的包的名称。

    Returns:
        (str): The latest version of the package.  返回：包的最新版本。
    """
    try:
        requests.packages.urllib3.disable_warnings()  # Disable the InsecureRequestWarning
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except Exception:
        return None


from ultralytics.utils import (
    ARM64,
    ASSETS,
    AUTOINSTALL,
    IS_COLAB,
    IS_GIT_DIR,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PYTHON_VERSION,
    RKNN_CHIPS,
    ROOT,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    SimpleNamespace,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    emojis,
    is_github_action_running,
    url2file,
)

def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.
    解析 requirements.txt 文件，忽略以 '#' 开头的行和 '#' 后的任何文本。

    Args:
        file_path (Path): Path to the requirements.txt file.  文件路径：requirements.txt 的路径。
        package (str, optional): Python package to use instead of requirements.txt file, i.e. package='ultralytics'.  
        package（str，可选）：要使用的 Python 包，替代 requirements.txt 文件，例如 package='ultralytics'。

    Returns:
        (List[Dict[str, str]]): List of parsed requirements as dictionaries with `name` and `specifier` keys.
        返回：解析后的需求列表，作为字典，包含 `name` 和 `specifier` 键。

    Example:
        ```python
        from ultralytics.utils.checks import parse_requirements

        parse_requirements(package="ultralytics")
        ```
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.split("#")[0].strip()  # ignore inline comments
            # 忽略行内注释
            if match := re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line):
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.
    将版本字符串转换为整数元组，忽略附加的任何非数字字符串。此函数替代已弃用的 'pkg_resources.parse_version(v)'。

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'  版本字符串，例如 '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
        返回：表示版本数字部分的整数元组和额外字符串，例如 (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.
    检查字符串是否仅由 ASCII 字符组成。

    Args:
        s (str): String to be checked.  要检查的字符串。

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
        返回：如果字符串仅由 ASCII 字符组成，则为 True，否则为 False。
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.
    验证图像大小是否是每个维度给定步幅的倍数。如果图像大小不是步幅的倍数，则将其更新为大于或等于给定下限值的最近步幅倍数。

    Args:
        imgsz (int | cList[int]): Image size.  图像大小。
        stride (int): Stride value.  步幅值。
        min_dim (int): Minimum number of dimensions.  最小维度数。
        max_dim (int): Maximum number of dimensions.  最大维度数。
        floor (int): Minimum allowed value for image size.  图像大小的最小允许值。

    Returns:
        (List[int] | int): Updated image size.  返回：更新后的图像大小。
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # i.e. '640' or '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.
    检查当前版本与所需版本或范围。

    Args:
        current (str): Current version or package name to get version from.  当前版本或用于获取版本的包名称。
        required (str): Required version or range (in pip-style format).  所需版本或范围（以 pip 风格格式）。
        name (str, optional): Name to be used in warning message.  用于警告消息的名称。
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.  如果为 True，则在未满足要求时引发 AssertionError。
        verbose (bool, optional): If True, print warning message if requirement is not met.  如果为 True，则在未满足要求时打印警告消息。
        msg (str, optional): Extra message to display if verbose.  如果 verbose，则显示的额外消息。

    Returns:
        (bool): True if requirement is met, False otherwise.  返回：如果满足要求，则为 True，否则为 False。

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current="22.04", required="==22.04")

        # Check if current version is greater than or equal to 22.04
        check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current="22.04", required="<=22.04")

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current="21.10", required=">20.04,<22.04")
        ```
    """
    if not current:  # if current is '' or None
        LOGGER.warning(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f"WARNING ⚠️ {current} package is required but not installed")) from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))  # assert version requirements met
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ultralytics"):
    """
    Returns the latest version of a PyPI package without downloading or installing it.
    返回 PyPI 包的最新版本，而不下载或安装它。

    Args:
        package_name (str): The name of the package to find the latest version for.  包名称：要查找最新版本的包的名称。

    Returns:
        (str): The latest version of the package.  返回：包的最新版本。
    """
    try:
        requests.packages.urllib3.disable_warnings()  # Disable the InsecureRequestWarning
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except Exception:
        return None

def check_amp(model):
    """
    Checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLO11 model. If the checks fail, it means
    there are anomalies with AMP on the system that may cause NaN losses or zero-mAP results, so AMP will be disabled
    during training.
    检查 YOLO11 模型的 PyTorch 自动混合精度（AMP）功能。如果检查失败，则表示系统的 AMP 存在异常，可能导致 NaN 损失或零 mAP 结果，因此在训练期间将禁用 AMP。

    Args:
        model (nn.Module): A YOLO11 model instance.  模型 (nn.Module): YOLO11 模型实例。

    Example:
        ```python
        from ultralytics import YOLO
        from ultralytics.utils.checks import check_amp

        model = YOLO("yolo11n.pt").model.cuda()
        check_amp(model)
        ```

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLO11 model, else False.
        返回：如果 AMP 功能与 YOLO11 模型正常工作，则返回 True，否则返回 False。
    """
    from ultralytics.utils.torch_utils import autocast  # 导入自动混合精度上下文管理器

    device = next(model.parameters()).device  # 获取模型的设备
    prefix = colorstr("AMP: ")  # 设置前缀颜色
    if device.type in {"cpu", "mps"}:  # 如果设备是 CPU 或 MPS
        return False  # AMP 仅在 CUDA 设备上使用
    else:
        # GPUs that have issues with AMP
        # 存在 AMP 问题的 GPU
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)  # 获取 GPU 名称
        if bool(pattern.search(gpu)):  # 如果 GPU 名称匹配模式
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False  # 返回 False，表示 AMP 检查失败

    def amp_allclose(m, im):
        """All close FP32 vs AMP results. FP32 与 AMP 结果的比较。"""
        batch = [im] * 8  # 创建一个包含 8 个图像的批次
        imgsz = max(256, int(model.stride.max() * 4))  # 最大步幅 P5-32 和 P6-64
        a = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # FP32 推理
        with autocast(enabled=True):  # 使用 AMP 上下文管理器
            b = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # AMP 推理
        del m  # 删除模型以释放内存
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # 检查结果是否接近，绝对公差为 0.5

    im = ASSETS / "bus.jpg"  # 要检查的图像
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks...")  # 日志记录 AMP 检查开始
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."  # 警告信息
    try:
        from ultralytics import YOLO  # 导入 YOLO 模型

        assert amp_allclose(YOLO("yolo11n.pt"), im)  # 检查 AMP 是否正常工作
        LOGGER.info(f"{prefix}checks passed ✅")  # 日志记录检查通过
    except ConnectionError:
        LOGGER.warning(
            f"{prefix}checks skipped ⚠️. Offline and unable to download YOLO11n for AMP checks. {warning_msg}"
        )  # 日志记录检查跳过
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped ⚠️. "
            f"Unable to load YOLO11n for AMP checks due to possible Ultralytics package modifications. {warning_msg}"
        )  # 日志记录无法加载模型的警告
    except AssertionError:
        LOGGER.warning(
            f"{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to "
            f"NaN losses or zero-mAP results, so AMP will be disabled during training."
        )  # 日志记录检查失败
        return False  # 返回 False，表示 AMP 检查失败
    return True  # 返回 True，表示 AMP 检查通过


def git_describe(path=ROOT):  # path must be a directory
    """Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe."""
    try:
        return subprocess.check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]  # 返回 git 描述
    except Exception:
        return ""  # 返回空字符串


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Print function arguments (optional args dict). 打印函数参数（可选参数字典）。"""

    def strip_auth(v):
        """Clean longer Ultralytics HUB URLs by stripping potential authentication information. 清理较长的 Ultralytics HUB URL，去除潜在的身份验证信息。"""
        return clean_url(v) if (isinstance(v, str) and v.startswith("http") and len(v) > 100) else v

    x = inspect.currentframe().f_back  # previous frame  # 获取上一个帧
    file, _, func, _, _ = inspect.getframeinfo(x)  # 获取帧信息
    if args is None:  # get args automatically  # 自动获取参数
        args, _, _, frm = inspect.getargvalues(x)  # 获取参数值
        args = {k: v for k, v in frm.items() if k in args}  # 过滤参数
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")  # 解析文件路径
    except ValueError:
        file = Path(file).stem  # 获取文件名
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")  # 设置日志字符串
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in args.items()))  # 打印参数


def cuda_device_count() -> int:
    """
    Get the number of NVIDIA GPUs available in the environment.
    获取环境中可用的 NVIDIA GPU 数量。

    Returns:
        (int): The number of NVIDIA GPUs available. 返回：可用的 NVIDIA GPU 数量。
    """
    try:
        # Run the nvidia-smi command and capture its output
        # 运行 nvidia-smi 命令并捕获其输出
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
        )

        # Take the first line and strip any leading/trailing white space
        # 获取第一行并去除前后的空白字符
        first_line = output.strip().split("\n")[0]

        return int(first_line)  # 返回 GPU 数量
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # If the command fails, nvidia-smi is not found, or output is not an integer, assume no GPUs are available
        # 如果命令失败，nvidia-smi 未找到，或输出不是整数，则假设没有可用的 GPU
        return 0


def cuda_is_available() -> bool:
    """
    Check if CUDA is available in the environment.
    检查环境中是否可用 CUDA。

    Returns:
        (bool): True if one or more NVIDIA GPUs are available, False otherwise.
        返回：如果有一个或多个 NVIDIA GPU 可用，则为 True，否则为 False。
    """
    return cuda_device_count() > 0  # 返回 GPU 数量是否大于 0


def is_rockchip():
    """Check if the current environment is running on a Rockchip SoC. 检查当前环境是否在 Rockchip SoC 上运行。"""
    if LINUX and ARM64:  # 如果是 Linux 且是 ARM64
        try:
            with open("/proc/device-tree/compatible") as f:  # 打开设备树文件
                dev_str = f.read()  # 读取设备字符串
                *_, soc = dev_str.split(",")  # 获取 SoC 名称
                if soc.replace("\x00", "") in RKNN_CHIPS:  # 检查 SoC 是否在 RKNN_CHIPS 中
                    return True  # 返回 True，表示是 Rockchip
        except OSError:
            return False  # 返回 False，表示不是 Rockchip
    else:
        return False  # 返回 False，表示不是 Rockchip


def is_sudo_available() -> bool:
    """
    Check if the sudo command is available in the environment.
    检查环境中是否可用 sudo 命令。

    Returns:
        (bool): True if the sudo command is available, False otherwise.
        返回：如果 sudo 命令可用，则为 True，否则为 False。
    """
    if WINDOWS:  # 如果是 Windows
        return False  # 返回 False，表示不支持 sudo
    cmd = "sudo --version"  # 检查 sudo 版本
    return subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0  # 返回 sudo 是否可用


# Run checks and define constants
check_python("3.8", hard=False, verbose=True)  # 检查 Python 版本
check_torchvision()  # 检查 torch 和 torchvision 的兼容性

# Define constants
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)  # 检查 Python 是否至少为 3.10
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")  # 检查 Python 版本是否为 3.12