# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Functions for estimating the best YOLO batch size to use a fraction of the available CUDA memory in PyTorch.  # 用于估算在 PyTorch 中使用可用 CUDA 内存的一部分的最佳 YOLO 批量大小的函数。"""

import os  # 导入 os 模块
from copy import deepcopy  # 从 copy 导入 deepcopy

import numpy as np  # 导入 numpy 作为 np
import torch  # 导入 torch 模块

from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr  # 从 ultralytics.utils 导入 DEFAULT_CFG、LOGGER 和 colorstr
from ultralytics.utils.torch_utils import autocast, profile  # 从 ultralytics.utils.torch_utils 导入 autocast 和 profile


def check_train_batch_size(model, imgsz=640, amp=True, batch=-1, max_num_obj=1):
    """
    Compute optimal YOLO training batch size using the autobatch() function.  # 使用 autobatch() 函数计算最佳 YOLO 训练批量大小。

    Args:  # 参数：
        model (torch.nn.Module): YOLO model to check batch size for.  # model (torch.nn.Module): 用于检查批量大小的 YOLO 模型。
        imgsz (int, optional): Image size used for training.  # imgsz (int, optional): 用于训练的图像大小。
        amp (bool, optional): Use automatic mixed precision if True.  # amp (bool, optional): 如果为 True，则使用自动混合精度。
        batch (float, optional): Fraction of GPU memory to use. If -1, use default.  # batch (float, optional): 要使用的 GPU 内存的比例。如果为 -1，则使用默认值。
        max_num_obj (int, optional): The maximum number of objects from dataset.  # max_num_obj (int, optional): 数据集中最大对象数量。

    Returns:  # 返回：
        (int): Optimal batch size computed using the autobatch() function.  # (int): 使用 autobatch() 函数计算的最佳批量大小。

    Note:  # 注意：
        If 0.0 < batch < 1.0, it's used as the fraction of GPU memory to use.  # 如果 0.0 < batch < 1.0，则将其用作要使用的 GPU 内存的比例。
        Otherwise, a default fraction of 0.6 is used.  # 否则，使用默认比例 0.6。
    """
    with autocast(enabled=amp):  # 使用自动混合精度上下文
        return autobatch(  # 返回调用 autobatch 函数
            deepcopy(model).train(), imgsz, fraction=batch if 0.0 < batch < 1.0 else 0.6, max_num_obj=max_num_obj
        )


def autobatch(model, imgsz=640, fraction=0.60, batch_size=DEFAULT_CFG.batch, max_num_obj=1):
    """
    Automatically estimate the best YOLO batch size to use a fraction of the available CUDA memory.  # 自动估算最佳 YOLO 批量大小，以使用可用 CUDA 内存的一部分。

    Args:  # 参数：
        model (torch.nn.module): YOLO model to compute batch size for.  # model (torch.nn.module): 用于计算批量大小的 YOLO 模型。
        imgsz (int, optional): The image size used as input for the YOLO model. Defaults to 640.  # imgsz (int, optional): 用作 YOLO 模型输入的图像大小。默认为 640。
        fraction (float, optional): The fraction of available CUDA memory to use. Defaults to 0.60.  # fraction (float, optional): 要使用的可用 CUDA 内存的比例。默认为 0.60。
        batch_size (int, optional): The default batch size to use if an error is detected. Defaults to 16.  # batch_size (int, optional): 如果检测到错误，则使用的默认批量大小。默认为 16。
        max_num_obj (int, optional): The maximum number of objects from dataset.  # max_num_obj (int, optional): 数据集中最大对象数量。

    Returns:  # 返回：
        (int): The optimal batch size.  # (int): 最佳批量大小。
    """
    # Check device  # 检查设备
    prefix = colorstr("AutoBatch: ")  # 设置前缀为 "AutoBatch: "
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% CUDA memory utilization.")  # 记录计算最佳批量大小的信息
    device = next(model.parameters()).device  # get model device  # 获取模型设备
    if device.type in {"cpu", "mps"}:  # 如果设备类型为 CPU 或 MPS
        LOGGER.info(f"{prefix} ⚠️ intended for CUDA devices, using default batch-size {batch_size}")  # 记录警告信息
        return batch_size  # 返回默认批量大小
    if torch.backends.cudnn.benchmark:  # 如果启用了 cudnn.benchmark
        LOGGER.info(f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")  # 记录警告信息
        return batch_size  # 返回默认批量大小

    # Inspect CUDA memory  # 检查 CUDA 内存
    gb = 1 << 30  # bytes to GiB (1024 ** 3)  # 字节转换为 GiB（1024 ** 3）
    d = f"CUDA:{os.getenv('CUDA_VISIBLE_DEVICES', '0').strip()[0]}"  # 'CUDA:0'  # 获取 CUDA 设备
    properties = torch.cuda.get_device_properties(device)  # device properties  # 获取设备属性
    t = properties.total_memory / gb  # GiB total  # GiB 总内存
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved  # GiB 保留内存
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated  # GiB 分配内存
    f = t - (r + a)  # GiB free  # GiB 可用内存
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")  # 记录内存信息

    # Profile batch sizes  # 记录批量大小
    batch_sizes = [1, 2, 4, 8, 16] if t < 16 else [1, 2, 4, 8, 16, 32, 64]  # 根据总内存选择批量大小
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]  # 创建输入图像
        results = profile(img, model, n=1, device=device, max_num_obj=max_num_obj)  # 记录性能

        # Fit a solution  # 拟合解决方案
        xy = [  # 创建有效结果列表
            [x, y[2]]
            for i, (x, y) in enumerate(zip(batch_sizes, results))
            if y  # valid result  # 有效结果
            and isinstance(y[2], (int, float))  # is numeric  # 是数字
            and 0 < y[2] < t  # between 0 and GPU limit  # 在 0 和 GPU 限制之间
            and (i == 0 or not results[i - 1] or y[2] > results[i - 1][2])  # first item or increasing memory  # 第一个项或增加内存
        ]
        fit_x, fit_y = zip(*xy) if xy else ([], [])  # 拆分有效结果
        p = np.polyfit(np.log(fit_x), np.log(fit_y), deg=1)  # first-degree polynomial fit in log space  # 在对数空间中进行一次多项式拟合
        b = int(round(np.exp((np.log(f * fraction) - p[1]) / p[0])))  # y intercept (optimal batch size)  # y 截距（最佳批量大小）
        if None in results:  # some sizes failed  # 一些大小失败
            i = results.index(None)  # first fail index  # 第一个失败索引
            if b >= batch_sizes[i]:  # y intercept above failure point  # y 截距在失败点之上
                b = batch_sizes[max(i - 1, 0)]  # select prior safe point  # 选择之前的安全点
        if b < 1 or b > 1024:  # b outside of safe range  # b 超出安全范围
            LOGGER.info(f"{prefix}WARNING ⚠️ batch={b} outside safe range, using default batch-size {batch_size}.")  # 记录警告信息
            b = batch_size  # 使用默认批量大小

        fraction = (np.exp(np.polyval(p, np.log(b))) + r + a) / t  # predicted fraction  # 预测的比例
        LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")  # 记录使用的批量大小
        return b  # 返回最佳批量大小
    except Exception as e:  # 捕获异常
        LOGGER.warning(f"{prefix}WARNING ⚠️ error detected: {e},  using default batch-size {batch_size}.")  # 记录警告信息
        return batch_size  # 返回默认批量大小
    finally:
        torch.cuda.empty_cache()  # 清空 CUDA 缓存