# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-

# import torch
# from torch import nn
# from torch.hub import load_state_dict_from_url

# __all__ = [
#     "create_yolox_model",
#     "yolox_nano",
#     "yolox_tiny",
#     "yolox_s",
#     "yolox_m",
#     "yolox_l",
#     "yolox_x",
#     "yolov3",
#     "yolox_custom"
# ]

# _CKPT_ROOT_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download"
# _CKPT_FULL_PATH = {
#     "yolox-nano": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_nano.pth",
#     "yolox-tiny": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_tiny.pth",
#     "yolox-s": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_s.pth",
#     "yolox-m": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_m.pth",
#     "yolox-l": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_l.pth",
#     "yolox-x": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_x.pth",
#     "yolov3": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_darknet.pth",
# }


# def create_yolox_model(name: str, pretrained: bool = True, num_classes: int = 80, device=None,
#                        exp_path: str = None, ckpt_path: str = None) -> nn.Module:
#     """creates and loads a YOLOX model

#     Args:
#         name (str): name of model. for example, "yolox-s", "yolox-tiny" or "yolox_custom"
#         if you want to load your own model.
#         pretrained (bool): load pretrained weights into the model. Default to True.
#         device (str): default device to for model. Default to None.
#         num_classes (int): number of model classes. Default to 80.
#         exp_path (str): path to your own experiment file. Required if name="yolox_custom"
#         ckpt_path (str): path to your own ckpt. Required if name="yolox_custom" and you want to
#             load a pretrained model


#     Returns:
#         YOLOX model (nn.Module)
#     """
#     from yolox.exp import get_exp, Exp

#     if device is None:
#         device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     device = torch.device(device)

#     assert name in _CKPT_FULL_PATH or name == "yolox_custom", \
#         f"user should use one of value in {_CKPT_FULL_PATH.keys()} or \"yolox_custom\""
#     if name in _CKPT_FULL_PATH:
#         exp: Exp = get_exp(exp_name=name)
#         exp.num_classes = num_classes
#         yolox_model = exp.get_model()
#         if pretrained and num_classes == 80:
#             weights_url = _CKPT_FULL_PATH[name]
#             ckpt = load_state_dict_from_url(weights_url, map_location="cpu")
#             if "model" in ckpt:
#                 ckpt = ckpt["model"]
#             yolox_model.load_state_dict(ckpt)
#     else:
#         assert exp_path is not None, "for a \"yolox_custom\" model exp_path must be provided"
#         exp: Exp = get_exp(exp_file=exp_path)
#         yolox_model = exp.get_model()
#         if ckpt_path:
#             ckpt = torch.load(ckpt_path, map_location="cpu")
#             if "model" in ckpt:
#                 ckpt = ckpt["model"]
#             yolox_model.load_state_dict(ckpt)

#     yolox_model.to(device)
#     return yolox_model


# def yolox_nano(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
#     return create_yolox_model("yolox-nano", pretrained, num_classes, device)


# def yolox_tiny(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
#     return create_yolox_model("yolox-tiny", pretrained, num_classes, device)


# def yolox_s(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
#     return create_yolox_model("yolox-s", pretrained, num_classes, device)


# def yolox_m(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
#     return create_yolox_model("yolox-m", pretrained, num_classes, device)


# def yolox_l(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
#     return create_yolox_model("yolox-l", pretrained, num_classes, device)


# def yolox_x(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
#     return create_yolox_model("yolox-x", pretrained, num_classes, device)


# def yolov3(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
#     return create_yolox_model("yolov3", pretrained, num_classes, device)


# def yolox_custom(ckpt_path: str = None, exp_path: str = None, device: str = None) -> nn.Module:
#     return create_yolox_model("yolox_custom", ckpt_path=ckpt_path, exp_path=exp_path, device=device)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.hub import load_state_dict_from_url  # 从URL加载状态字典的功能

__all__ = [  # 定义模块的公共接口
    "create_yolox_model",
    "yolox_nano",
    "yolox_tiny",
    "yolox_s",
    "yolox_m",
    "yolox_l",
    "yolox_x",
    "yolov3",
    "yolox_custom"
]

_CKPT_ROOT_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download"  # 检查点根URL
_CKPT_FULL_PATH = {  # 定义不同YOLOX模型的检查点路径
    "yolox-nano": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_nano.pth",
    "yolox-tiny": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_tiny.pth",
    "yolox-s": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_s.pth",
    "yolox-m": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_m.pth",
    "yolox-l": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_l.pth",
    "yolox-x": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_x.pth",
    "yolov3": f"{_CKPT_ROOT_URL}/0.1.1rc0/yolox_darknet.pth",
}


def create_yolox_model(name: str, pretrained: bool = True, num_classes: int = 80, device=None,
                       exp_path: str = None, ckpt_path: str = None) -> nn.Module:
    """creates and loads a YOLOX model
    创建并加载一个YOLOX模型

    Args:
        name (str): name of model. for example, "yolox-s", "yolox-tiny" or "yolox_custom"
        name (str): 模型名称，例如 "yolox-s", "yolox-tiny" 或 "yolox_custom"
        if you want to load your own model.
        pretrained (bool): load pretrained weights into the model. Default to True.
        pretrained (bool): 是否加载预训练权重到模型中，默认为True。
        device (str): default device to for model. Default to None.
        device (str): 模型的默认设备，默认为None。
        num_classes (int): number of model classes. Default to 80.
        num_classes (int): 模型类别的数量，默认为80。
        exp_path (str): path to your own experiment file. Required if name="yolox_custom"
        exp_path (str): 自定义实验文件的路径。如果名称为 "yolox_custom"，则该参数为必需。
        ckpt_path (str): path to your own ckpt. Required if name="yolox_custom" and you want to
            load a pretrained model
        ckpt_path (str): 自定义检查点的路径。如果名称为 "yolox_custom" 且希望加载预训练模型，则该参数为必需。

    Returns:
        YOLOX model (nn.Module)
        返回YOLOX模型 (nn.Module)
    """
    from yolox.exp import get_exp, Exp  # 从yolox.exp模块导入get_exp和Exp

    if device is None:  # 如果未指定设备
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 根据CUDA可用性选择设备
    device = torch.device(device)  # 将设备转换为torch.device对象

    assert name in _CKPT_FULL_PATH or name == "yolox_custom", \  # 确保名称在检查点路径中
        f"user should use one of value in {_CKPT_FULL_PATH.keys()} or \"yolox_custom\""
    if name in _CKPT_FULL_PATH:  # 如果名称在检查点路径中
        exp: Exp = get_exp(exp_name=name)  # 获取实验配置
        exp.num_classes = num_classes  # 设置类别数量
        yolox_model = exp.get_model()  # 获取YOLOX模型
        if pretrained and num_classes == 80:  # 如果需要加载预训练权重并且类别数量为80
            weights_url = _CKPT_FULL_PATH[name]  # 获取权重URL
            ckpt = load_state_dict_from_url(weights_url, map_location="cpu")  # 从URL加载检查点
            if "model" in ckpt:  # 如果检查点中包含模型
                ckpt = ckpt["model"]  # 获取模型部分
            yolox_model.load_state_dict(ckpt)  # 加载状态字典到模型中
    else:  # 如果是自定义模型
        assert exp_path is not None, "for a \"yolox_custom\" model exp_path must be provided"  # 确保提供实验路径
        exp: Exp = get_exp(exp_file=exp_path)  # 获取自定义实验配置
        yolox_model = exp.get_model()  # 获取YOLOX模型
        if ckpt_path:  # 如果提供了检查点路径
            ckpt = torch.load(ckpt_path, map_location="cpu")  # 加载检查点
            if "model" in ckpt:  # 如果检查点中包含模型
                ckpt = ckpt["model"]  # 获取模型部分
            yolox_model.load_state_dict(ckpt)  # 加载状态字典到模型中

    yolox_model.to(device)  # 将模型移动到指定设备
    return yolox_model  # 返回模型


def yolox_nano(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
    return create_yolox_model("yolox-nano", pretrained, num_classes, device)  # 创建yolox-nano模型


def yolox_tiny(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
    return create_yolox_model("yolox-tiny", pretrained, num_classes, device)  # 创建yolox-tiny模型


def yolox_s(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
    return create_yolox_model("yolox-s", pretrained, num_classes, device)  # 创建yolox-s模型


def yolox_m(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
    return create_yolox_model("yolox-m", pretrained, num_classes, device)  # 创建yolox-m模型


def yolox_l(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
    return create_yolox_model("yolox-l", pretrained, num_classes, device)  # 创建yolox-l模型


def yolox_x(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
    return create_yolox_model("yolox-x", pretrained, num_classes, device)  # 创建yolox-x模型


def yolov3(pretrained: bool = True, num_classes: int = 80, device: str = None) -> nn.Module:
    return create_yolox_model("yolov3", pretrained, num_classes, device)  # 创建yolov3模型


def yolox_custom(ckpt_path: str = None, exp_path: str = None, device: str = None) -> nn.Module:
    return create_yolox_model("yolox_custom", ckpt_path=ckpt_path, exp_path=exp_path, device=device)  # 创建自定义YOLOX模型
