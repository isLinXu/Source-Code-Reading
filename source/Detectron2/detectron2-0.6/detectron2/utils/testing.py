# Copyright (c) Facebook, Inc. and its affiliates.
import io  # 导入io模块，用于处理输入输出流
import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库

from detectron2 import model_zoo  # 从detectron2导入model_zoo模块，用于获取预训练模型
from detectron2.data import DatasetCatalog  # 从detectron2.data导入DatasetCatalog，用于访问数据集
from detectron2.data.detection_utils import read_image  # 导入read_image函数，用于读取图像
from detectron2.modeling import build_model  # 导入build_model函数，用于构建模型
from detectron2.structures import Boxes, Instances, ROIMasks  # 导入结构类，用于表示检测结果
from detectron2.utils.file_io import PathManager  # 导入PathManager，用于文件路径管理


"""
Internal utilities for tests. Don't use except for writing tests.
"""  # 测试用的内部工具函数。除了编写测试外，不要使用这些函数。


def get_model_no_weights(config_path):  # 定义函数，获取不加载任何权重的模型
    """
    Like model_zoo.get, but do not load any weights (even pretrained)
    """  # 类似于model_zoo.get，但不加载任何权重（甚至预训练权重）
    cfg = model_zoo.get_config(config_path)  # 获取指定路径的配置
    if not torch.cuda.is_available():  # 如果没有可用的CUDA设备
        cfg.MODEL.DEVICE = "cpu"  # 将模型设备设置为CPU
    return build_model(cfg)  # 根据配置构建并返回模型


def random_boxes(num_boxes, max_coord=100, device="cpu"):  # 定义函数，生成随机边界框
    """
    Create a random Nx4 boxes tensor, with coordinates < max_coord.
    """  # 创建一个随机的Nx4边界框张量，坐标值小于max_coord
    boxes = torch.rand(num_boxes, 4, device=device) * (max_coord * 0.5)  # 生成随机坐标值
    boxes.clamp_(min=1.0)  # tiny boxes cause numerical instability in box regression  # 限制最小值为1.0，避免小框导致框回归的数值不稳定
    # Note: the implementation of this function in torchvision is:  # 注意：torchvision中这个函数的实现是：
    # boxes[:, 2:] += torch.rand(N, 2) * 100  # boxes[:, 2:] += torch.rand(N, 2) * 100
    # but it does not guarantee non-negative widths/heights constraints:  # 但它不保证非负的宽度/高度约束：
    # boxes[:, 2] >= boxes[:, 0] and boxes[:, 3] >= boxes[:, 1]:  # boxes[:, 2] >= boxes[:, 0] 且 boxes[:, 3] >= boxes[:, 1]:
    boxes[:, 2:] += boxes[:, :2]  # 确保右下角坐标大于左上角坐标
    return boxes  # 返回生成的边界框


def get_sample_coco_image(tensor=True):  # 定义函数，获取COCO样本图像
    """
    Args:
        tensor (bool): if True, returns 3xHxW tensor.
            else, returns a HxWx3 numpy array.

    Returns:
        an image, in BGR color.
    """  # 参数：
        # tensor (bool)：如果为True，返回3xHxW张量。
        #     否则，返回HxWx3的numpy数组。
        # 返回：
        #     一张BGR格式的图像。
    
    try:  # 尝试获取COCO验证集中的第一张图片
        file_name = DatasetCatalog.get("coco_2017_val_100")[0]["file_name"]  # 获取COCO 2017验证集中第一张图片的文件名
        if not PathManager.exists(file_name):  # 如果文件不存在
            raise FileNotFoundError()  # 抛出文件未找到错误
    except IOError:  # 捕获IO错误
        # for public CI to run  # 为公共CI运行
        file_name = "http://images.cocodataset.org/train2017/000000000009.jpg"  # 使用网络上的COCO图片
    ret = read_image(file_name, format="BGR")  # 以BGR格式读取图像
    if tensor:  # 如果需要返回张量
        ret = torch.from_numpy(np.ascontiguousarray(ret.transpose(2, 0, 1)))  # 将HxWx3的numpy数组转换为3xHxW的张量
    return ret  # 返回图像


def convert_scripted_instances(instances):  # 定义函数，转换脚本化的Instances对象
    """
    Convert a scripted Instances object to a regular :class:`Instances` object
    """  # 将脚本化的Instances对象转换为常规的Instances对象
    ret = Instances(instances.image_size)  # 创建一个新的Instances对象，使用相同的图像大小
    for name in instances._field_names:  # 遍历所有字段名
        val = getattr(instances, "_" + name, None)  # 获取字段值
        if val is not None:  # 如果值不为None
            ret.set(name, val)  # 设置新Instances对象的相应字段
    return ret  # 返回转换后的Instances对象


def assert_instances_allclose(input, other, *, rtol=1e-5, msg="", size_as_tensor=False):  # 定义函数，断言两个Instances对象相近
    """
    Args:
        input, other (Instances):
        size_as_tensor: compare image_size of the Instances as tensors (instead of tuples).
             Useful for comparing outputs of tracing.
    """  # 参数：
        # input, other (Instances)：要比较的两个Instances对象
        # size_as_tensor：将Instances的image_size作为张量比较（而不是元组）。
        #     对比较追踪输出很有用。
    
    if not isinstance(input, Instances):  # 如果input不是Instances类型
        input = convert_scripted_instances(input)  # 将其转换为Instances对象
    if not isinstance(other, Instances):  # 如果other不是Instances类型
        other = convert_scripted_instances(other)  # 将其转换为Instances对象

    if not msg:  # 如果没有提供消息
        msg = "Two Instances are different! "  # 设置默认错误消息
    else:  # 如果提供了消息
        msg = msg.rstrip() + " "  # 移除尾部空格并添加一个空格

    size_error_msg = msg + f"image_size is {input.image_size} vs. {other.image_size}!"  # 构建图像大小不匹配的错误消息
    if size_as_tensor:  # 如果以张量形式比较图像大小
        assert torch.equal(
            torch.tensor(input.image_size), torch.tensor(other.image_size)
        ), size_error_msg  # 断言两个图像大小张量相等
    else:  # 否则
        assert input.image_size == other.image_size, size_error_msg  # 断言两个图像大小元组相等
    fields = sorted(input.get_fields().keys())  # 获取并排序input的所有字段名
    fields_other = sorted(other.get_fields().keys())  # 获取并排序other的所有字段名
    assert fields == fields_other, msg + f"Fields are {fields} vs {fields_other}!"  # 断言两个对象有相同的字段

    for f in fields:  # 遍历所有字段
        val1, val2 = input.get(f), other.get(f)  # 获取两个对象中相应字段的值
        if isinstance(val1, (Boxes, ROIMasks)):  # 如果值是Boxes或ROIMasks类型
            # boxes in the range of O(100) and can have a larger tolerance  # 框的范围在O(100)级别，可以有更大的容差
            assert torch.allclose(val1.tensor, val2.tensor, atol=100 * rtol), (
                msg + f"Field {f} differs too much!"
            )  # 断言两个张量足够接近
        elif isinstance(val1, torch.Tensor):  # 如果值是张量
            if val1.dtype.is_floating_point:  # 如果是浮点张量
                mag = torch.abs(val1).max().cpu().item()  # 计算值的最大绝对值
                assert torch.allclose(val1, val2, atol=mag * rtol), (
                    msg + f"Field {f} differs too much!"
                )  # 断言两个张量足够接近，容差与值的量级相关
            else:  # 如果是整数张量
                assert torch.equal(val1, val2), msg + f"Field {f} is different!"  # 断言两个张量完全相等
        else:  # 如果是其他类型
            raise ValueError(f"Don't know how to compare type {type(val1)}")  # 抛出不支持的类型错误


def reload_script_model(module):  # 定义函数，重新加载脚本模型
    """
    Save a jit module and load it back.
    Similar to the `getExportImportCopy` function in torch/testing/
    """  # 保存一个JIT模块并重新加载它。
        # 类似于torch/testing/中的`getExportImportCopy`函数。
    
    buffer = io.BytesIO()  # 创建一个字节流缓冲区
    torch.jit.save(module, buffer)  # 将模块保存到缓冲区
    buffer.seek(0)  # 将缓冲区指针重置到开始位置
    return torch.jit.load(buffer)  # 从缓冲区加载模块并返回
