# 导入必要的库
import re
import torch


def parse_device_spec(device_spec: str | torch.device) -> torch.device:
    """
    Convert a string or torch.device into a valid torch.device. Allowed strings: 'auto', 'cpu',
    'cuda', 'cuda:N' (e.g. 'cuda:0'), or 'mps'. This function raises ValueError if the input
    is unrecognized or the GPU index is out of range.
    将字符串或torch.device转换为有效的torch.device。允许的字符串：'auto'、'cpu'、
    'cuda'、'cuda:N'（例如'cuda:0'）或'mps'。如果输入无法识别或GPU索引超出范围，
    此函数会抛出ValueError。

    Args:
        device_spec (str | torch.device): A specification for the device. This can be a valid
        torch.device object or one of the recognized strings described above.
        设备的规范。可以是有效的torch.device对象或上述识别的字符串之一。

    Returns:
        torch.device: The corresponding torch.device object.
        对应的torch.device对象。

    Raises:
        ValueError: If the device specification is unrecognized or the provided GPU index
        exceeds the available devices.
        如果设备规范无法识别或提供的GPU索引超出可用设备，则抛出ValueError。
    """
    # 如果输入已经是torch.device对象，直接返回
    if isinstance(device_spec, torch.device):
        return device_spec

    # 将设备字符串转换为小写
    device_str = device_spec.lower()
    # 处理'auto'情况，自动选择可用设备
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    # 处理'cpu'情况
    elif device_str == "cpu":
        return torch.device("cpu")
    # 处理'cuda'情况
    elif device_str == "cuda":
        return torch.device("cuda")
    # 处理'mps'情况
    elif device_str == "mps":
        return torch.device("mps")
    # 处理'cuda:N'情况
    else:
        # 使用正则表达式匹配'cuda:N'格式
        match = re.match(r"^cuda:(\d+)$", device_str)
        if match:
            # 提取GPU索引
            index = int(match.group(1))
            # 检查索引是否为非负数
            if index < 0:
                raise ValueError(f"GPU index must be non-negative, got {index}.")
            # 检查索引是否超出可用GPU数量
            if index >= torch.cuda.device_count():
                raise ValueError(f"Requested cuda:{index} but only {torch.cuda.device_count()} GPU(s) are available.")
            return torch.device(f"cuda:{index}")

        # 如果设备规范无法识别，抛出ValueError
        raise ValueError(f"Unrecognized device spec: {device_spec}")


def device_is_available(device: torch.device) -> bool:
    """
    Check whether a given torch.device is available on the current system.
    检查给定的torch.device在当前系统上是否可用。

    Args:
        device (torch.device): The device to verify.
        要验证的设备。

    Returns:
        bool: True if the device is available, False otherwise.
        如果设备可用则返回True，否则返回False。
    """
    # 检查CUDA设备是否可用
    if device.type == "cuda":
        return torch.cuda.is_available()
    # 检查MPS设备是否可用
    elif device.type == "mps":
        return torch.backends.mps.is_available()
    # CPU设备始终可用
    elif device.type == "cpu":
        return True
    # 其他设备类型默认不可用
    return False