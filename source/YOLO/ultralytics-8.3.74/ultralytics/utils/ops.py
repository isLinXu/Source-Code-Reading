# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou
class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class. Use as a decorator with @Profile() or as a context manager with 'with Profile():'.
    YOLOv8 配置文件类。可以作为装饰器使用 @Profile() 或作为上下文管理器使用 'with Profile():'。

    Example:
        ```python
        from ultralytics.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            t（浮点数）：初始时间。默认为 0.0。
            device (torch.device): Devices used for model inference. Defaults to None (cpu).
            device（torch.device）：用于模型推理的设备。默认为 None（cpu）。
        """
        self.t = t  # 初始化时间
        self.device = device  # 设备
        self.cuda = bool(device and str(device).startswith("cuda"))  # 判断是否使用 CUDA

    def __enter__(self):
        """Start timing."""
        self.start = self.time()  # 记录开始时间
        return self  # 返回当前实例

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # 累加时间

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"  # 返回格式化的耗时字符串

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize(self.device)  # 同步 CUDA 设备
        return time.time()  # 返回当前时间


def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (torch.Tensor): the segment label
        segment（torch.Tensor）：分割标签
        width (int): the width of the image. Defaults to 640
        width（整数）：图像的宽度。默认为 640
        height (int): The height of the image. Defaults to 640
        height（整数）：图像的高度。默认为 640

    Returns:
        (np.ndarray): the minimum and maximum x and y values of the segment.
        (np.ndarray)：分割的最小和最大 x 和 y 值。
    """
    x, y = segment.T  # segment xy
    # any 3 out of 4 sides are outside the image, clip coordinates first, https://github.com/ultralytics/ultralytics/pull/18294
    if np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum() >= 3:
        x = x.clip(0, width)  # 限制 x 坐标在图像宽度范围内
        y = y.clip(0, height)  # 限制 y 坐标在图像高度范围内
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)  # 检查坐标是否在图像内部
    x = x[inside]  # 仅保留在图像内部的 x 坐标
    y = y[inside]  # 仅保留在图像内部的 y 坐标
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        img1_shape（元组）：bounding boxes 所在图像的形状，格式为 (高度, 宽度)。
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        boxes（torch.Tensor）：图像中物体的边界框，格式为 (x1, y1, x2, y2)。
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        img0_shape（元组）：目标图像的形状，格式为 (高度, 宽度)。
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        ratio_pad（元组）：用于缩放框的 (ratio, pad) 元组。如果未提供，则根据两幅图像的大小差异计算比率和填充。
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        padding（布尔值）：如果为 True，假设框是基于 YOLO 风格增强的图像。如果为 False，则进行常规缩放。
        xywh (bool): The box format is xywh or not, default=False.
        xywh（布尔值）：框格式是否为 xywh，默认为 False。

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        boxes（torch.Tensor）：缩放后的边界框，格式为 (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),  # 计算宽度填充
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),  # 计算高度填充
        )
    else:
        gain = ratio_pad[0][0]  # 从提供的比率中获取缩放比
        pad = ratio_pad[1]  # 获取填充值

    if padding:
        boxes[..., 0] -= pad[0]  # x 填充
        boxes[..., 1] -= pad[1]  # y 填充
        if not xywh:
            boxes[..., 2] -= pad[0]  # x 填充
            boxes[..., 3] -= pad[1]  # y 填充
    boxes[..., :4] /= gain  # 根据缩放比调整边界框
    return clip_boxes(boxes, img0_shape)  # 返回裁剪后的边界框

def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.
    返回最接近的可被给定除数整除的数字。

    Args:
        x (int): The number to make divisible.
        x（整数）：要使其可被整除的数字。
        divisor (int | torch.Tensor): The divisor.
        divisor（整数 | torch.Tensor）：除数。

    Returns:
        (int): The nearest number divisible by the divisor.
        (int)：可被除数整除的最接近的数字。
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor  # 返回最接近的可被除数整除的数字


def nms_rotated(boxes, scores, threshold=0.45, use_triu=True):
    """
    NMS for oriented bounding boxes using probiou and fast-nms.
    使用 probiou 和 fast-nms 的定向边界框 NMS。

    Args:
        boxes (torch.Tensor): Rotated bounding boxes, shape (N, 5), format xywhr.
        boxes（torch.Tensor）：旋转的边界框，形状为 (N, 5)，格式为 xywhr。
        scores (torch.Tensor): Confidence scores, shape (N,).
        scores（torch.Tensor）：置信度分数，形状为 (N,)。
        threshold (float, optional): IoU threshold. Defaults to 0.45.
        threshold（浮点数，可选）：IoU 阈值。默认为 0.45。
        use_triu (bool, optional): Whether to use `torch.triu` operator. It'd be useful for disable it
            when exporting obb models to some formats that do not support `torch.triu`.
        use_triu（布尔值，可选）：是否使用 `torch.triu` 操作符。在将 obb 模型导出到不支持 `torch.triu` 的某些格式时禁用它会很有用。

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
        (torch.Tensor)：在 NMS 后要保留的框的索引。
    """
    sorted_idx = torch.argsort(scores, descending=True)  # 按置信度分数降序排序索引
    boxes = boxes[sorted_idx]  # 根据排序索引重新排列边界框
    ious = batch_probiou(boxes, boxes)  # 计算 IoU
    if use_triu:
        ious = ious.triu_(diagonal=1)  # 只保留上三角矩阵部分
        pick = torch.nonzero((ious >= threshold).sum(0) <= 0).squeeze_(-1)  # 选择 IoU 小于阈值的框
    else:
        n = boxes.shape[0]  # 框的数量
        row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)  # 行索引
        col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)  # 列索引
        upper_mask = row_idx < col_idx  # 上三角掩码
        ious = ious * upper_mask  # 应用掩码
        scores[~((ious >= threshold).sum(0) <= 0)] = 0  # 将不满足 IoU 条件的分数置为 0
        pick = torch.topk(scores, scores.shape[0]).indices  # 返回前 N 个框的索引
    return sorted_idx[pick]  # 返回保留框的排序索引


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.
    对一组框执行非最大抑制 (NMS)，支持每个框的掩码和多个标签。

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        prediction（torch.Tensor）：形状为 (batch_size, num_classes + 4 + num_masks, num_boxes) 的张量，包含预测的框、类别和掩码。张量应为模型输出的格式，例如 YOLO。
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        conf_thres（浮点数）：低于此置信度阈值的框将被过滤。有效值范围为 0.0 到 1.0。
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        iou_thres（浮点数）：在 NMS 期间低于此 IoU 阈值的框将被过滤。有效值范围为 0.0 到 1.0。
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        classes（列表[int]）：要考虑的类别索引列表。如果为 None，则考虑所有类别。
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        agnostic（布尔值）：如果为 True，模型对类别数量不敏感，所有类别将被视为一个。
        multi_label (bool): If True, each box may have multiple labels.
        multi_label（布尔值）：如果为 True，每个框可能有多个标签。
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        labels（列表[列表[联合[整数，浮点数，torch.Tensor]]]）：一个列表的列表，每个内部列表包含给定图像的先验标签。列表应为数据加载器输出的格式，每个标签为 (class_index, x1, y1, x2, y2) 的元组。
        max_det (int): The maximum number of boxes to keep after NMS.
        max_det（整数）：在 NMS 后要保留的最大框数量。
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        nc（整数，可选）：模型输出的类别数量。此后任何索引将被视为掩码。
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_time_img（浮点数）：处理一幅图像的最大时间（秒）。
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_nms（整数）：传入 torchvision.ops.nms() 的最大框数量。
        max_wh (int): The maximum box width and height in pixels.
        max_wh（整数）：框的最大宽度和高度（以像素为单位）。
        in_place (bool): If True, the input prediction tensor will be modified in place.
        in_place（布尔值）：如果为 True，输入预测张量将被就地修改。
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.
        rotated（布尔值）：如果传递的是定向边界框 (OBB) 进行 NMS。
        end2end (bool): If the model doesn't require NMS.
        end2end（布尔值）：如果模型不需要 NMS。

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        (List[torch.Tensor])：长度为 batch_size 的列表，其中每个元素是形状为 (num_boxes, 6 + num_masks) 的张量，包含保留的框，列为 (x1, y1, x2, y2, 置信度, 类别, mask1, mask2, ...)。
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"  # 检查置信度阈值
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"  # 检查 IoU 阈值
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output  # 选择仅推理输出
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)  # 将类别转换为张量

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]  # 过滤置信度低于阈值的框
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]  # 过滤特定类别的框
        return output  # 返回输出

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)  # 批量大小
    nc = nc or (prediction.shape[1] - 4)  # number of classes  # 类别数量
    nm = prediction.shape[1] - nc - 4  # number of masks  # 掩码数量
    mi = 4 + nc  # mask start index  # 掩码起始索引
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates  # 候选框

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height  # 最小框宽度和高度
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after  # 超过时间限制后退出的秒数
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)  # 每个框多个标签

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)  # 转置张量形状
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy  # 将 xywh 转换为 xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy  # 将 xywh 转换为 xyxy

    t = time.time()  # 记录开始时间
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs  # 初始化输出
    for xi, x in enumerate(prediction):  # image index, image inference  # 图像索引，图像推理
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height  # 应用约束
        x = x[xc[xi]]  # confidence  # 过滤置信度低于阈值的框

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:  # 如果存在标签且未旋转
            lb = labels[xi]  # 获取标签
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)  # 初始化标签张量
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box  # 将标签转换为 xyxy
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls  # 设置类别
            x = torch.cat((x, v), 0)  # 合并标签和预测

        # If none remain process next image
        if not x.shape[0]:  # 如果没有框，处理下一幅图像
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)  # 将预测分割为框、类别和掩码

        if multi_label:  # 如果每个框有多个标签
            i, j = torch.where(cls > conf_thres)  # 获取置信度高于阈值的索引
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)  # 合并框和标签
        else:  # best class only  # 仅选择最佳类别
            conf, j = cls.max(1, keepdim=True)  # 获取最大置信度和对应类别
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]  # 合并框和置信度

        # Filter by class
        if classes is not None:  # 如果指定了类别
            x = x[(x[:, 5:6] == classes).any(1)]  # 过滤特定类别的框

        # Check shape
        n = x.shape[0]  # number of boxes  # 框的数量
        if not n:  # no boxes  # 如果没有框
            continue
        if n > max_nms:  # excess boxes  # 如果框的数量超过最大值
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes  # 按置信度排序并移除多余框

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes  # 类别
        scores = x[:, 4]  # scores  # 置信度
        if rotated:  # 如果是旋转框
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr  # 合并框
            i = nms_rotated(boxes, scores, iou_thres)  # 进行 NMS
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)  # 框（按类别偏移）
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS  # 进行 NMS
        i = i[:max_det]  # limit detections  # 限制检测框数量

        output[xi] = x[i]  # 保存结果
        if (time.time() - t) > time_limit:  # 超过时间限制
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")  # 记录警告
            break  # time limit exceeded  # 超过时间限制，退出循环

    return output  # 返回输出


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.
    接受边界框列表和形状（高度，宽度），并将边界框裁剪到该形状。

    Args:
        boxes (torch.Tensor): The bounding boxes to clip.
        boxes（torch.Tensor）：要裁剪的边界框。
        shape (tuple): The shape of the image.
        shape（元组）：图像的形状。

    Returns:
        (torch.Tensor | numpy.ndarray): The clipped boxes.
        (torch.Tensor | numpy.ndarray)：裁剪后的边界框。
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1  # 限制 x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1  # 限制 y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2  # 限制 x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2  # 限制 y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2  # 限制 x1 和 x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2  # 限制 y1 和 y2
    return boxes  # 返回裁剪后的边界框


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.
    将线坐标裁剪到图像边界。

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        coords（torch.Tensor | numpy.ndarray）：线坐标列表。
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).
        shape（元组）：表示图像大小的整数元组，格式为 (高度, 宽度)。

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped coordinates
        (torch.Tensor | numpy.ndarray)：裁剪后的坐标。
    """
    if isinstance(coords, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x  # 限制 x 坐标
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y  # 限制 y 坐标
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x  # 限制 x 坐标
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y  # 限制 y 坐标
    return coords  # 返回裁剪后的坐标


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.
    接受一个掩码，并将其调整为原始图像大小。

    Args:
        masks (np.ndarray): Resized and padded masks/images, [h, w, num]/[h, w, 3].
        masks（np.ndarray）：调整大小和填充的掩码/图像，形状为 [h, w, num]/[h, w, 3]。
        im0_shape (tuple): The original image shape.
        im0_shape（元组）：原始图像的形状。
        ratio_pad (tuple): The ratio of the padding to the original image.
        ratio_pad（元组）：填充与原始图像的比率。

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
        masks（np.ndarray）：返回的掩码，形状为 [h, w, num]。
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape  # 获取掩码的形状
    if im1_shape[:2] == im0_shape[:2]:  # 如果形状相同，直接返回
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]  # 获取填充值
    top, left = int(pad[1]), int(pad[0])  # y, x  # 填充的顶部和左侧
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])  # 填充的底部和右侧

    if len(masks.shape) < 2:  # 如果掩码的维度小于 2，抛出错误
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]  # 裁剪掩码
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))  # 调整掩码大小
    if len(masks.shape) == 2:  # 如果掩码是二维的
        masks = masks[:, :, None]  # 添加一个维度

    return masks  # 返回调整后的掩码

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.
    将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, width, height) 格式，其中 (x1, y1) 是左上角，(x2, y2) 是右下角。

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        x（np.ndarray | torch.Tensor）：输入的边界框坐标，格式为 (x1, y1, x2, y2)。

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
        y（np.ndarray | torch.Tensor）：边界框坐标，格式为 (x, y, width, height)。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    # 确保输入的最后一个维度为4，表示四个坐标
    y = empty_like(x)  # faster than clone/copy
    # 创建一个与 x 形状相同的空数组 y，速度比克隆/复制快
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    # 计算 x 中心坐标
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    # 计算 y 中心坐标
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    # 计算宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    # 计算高度
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.
    将边界框坐标从 (x, y, width, height) 格式转换为 (x1, y1, x2, y2) 格式，其中 (x1, y1) 是左上角，(x2, y2) 是右下角。注意：每两个通道的操作比每个通道的操作更快。

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
        x（np.ndarray | torch.Tensor）：输入的边界框坐标，格式为 (x, y, width, height)。

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        y（np.ndarray | torch.Tensor）：边界框坐标，格式为 (x1, y1, x2, y2)。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    # 确保输入的最后一个维度为4，表示四个坐标
    y = empty_like(x)  # faster than clone/copy
    # 创建一个与 x 形状相同的空数组 y，速度比克隆/复制快
    xy = x[..., :2]  # centers
    # 获取中心坐标
    wh = x[..., 2:] / 2  # half width-height
    # 计算宽度和高度的一半
    y[..., :2] = xy - wh  # top left xy
    # 计算左上角坐标
    y[..., 2:] = xy + wh  # bottom right xy
    # 计算右下角坐标
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.
    将归一化的边界框坐标转换为像素坐标。

    Args:
        x (np.ndarray | torch.Tensor): The bounding box coordinates.
        x（np.ndarray | torch.Tensor）：边界框坐标。
        w (int): Width of the image. Defaults to 640
        w（整数）：图像的宽度，默认为640
        h (int): Height of the image. Defaults to 640
        h（整数）：图像的高度，默认为640
        padw (int): Padding width. Defaults to 0
        padw（整数）：填充宽度，默认为0
        padh (int): Padding height. Defaults to 0
        padh（整数）：填充高度，默认为0

    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
        y（np.ndarray | torch.Tensor）：边界框坐标，格式为 [x1, y1, x2, y2]，其中 x1,y1 是左上角，x2,y2 是右下角。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    # 确保输入的最后一个维度为4，表示四个坐标
    y = empty_like(x)  # faster than clone/copy
    # 创建一个与 x 形状相同的空数组 y，速度比克隆/复制快
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    # 计算左上角 x 坐标
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    # 计算左上角 y 坐标
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    # 计算右下角 x 坐标
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    # 计算右下角 y 坐标
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.
    将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, width, height, normalized) 格式。x, y, 宽度和高度被归一化到图像尺寸。

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        x（np.ndarray | torch.Tensor）：输入的边界框坐标，格式为 (x1, y1, x2, y2)。
        w (int): The width of the image. Defaults to 640
        w（整数）：图像的宽度，默认为640
        h (int): The height of the image. Defaults to 640
        h（整数）：图像的高度，默认为640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        clip（布尔值）：如果为 True，框将被裁剪到图像边界。默认为 False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0
        eps（浮点数）：框的宽度和高度的最小值。默认为0.0

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
        y（np.ndarray | torch.Tensor）：边界框坐标，格式为 (x, y, width, height, normalized)。
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
        # 如果需要裁剪，调用裁剪函数
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    # 确保输入的最后一个维度为4，表示四个坐标
    y = empty_like(x)  # faster than clone/copy
    # 创建一个与 x 形状相同的空数组 y，速度比克隆/复制快
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    # 计算 x 中心坐标并归一化
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    # 计算 y 中心坐标并归一化
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    # 计算宽度并归一化
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    # 计算高度并归一化
    return y


def xywh2ltwh(x):
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.
    将边界框格式从 [x, y, w, h] 转换为 [x1, y1, w, h]，其中 x1, y1 是左上角坐标。

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding box coordinates in the xywh format
        x（np.ndarray | torch.Tensor）：输入的张量，格式为 xywh 的边界框坐标。

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format
        y（np.ndarray | torch.Tensor）：边界框坐标，格式为 xyltwh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 如果是 PyTorch 张量则克隆，否则使用 numpy 复制
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    # 计算左上角 x 坐标
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    # 计算左上角 y 坐标
    return y


def xyxy2ltwh(x):
    """
    Convert nx4 bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h], where xy1=top-left, xy2=bottom-right.
    将 nx4 边界框从 [x1, y1, x2, y2] 转换为 [x1, y1, w, h]，其中 xy1 是左上角，xy2 是右下角。

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding boxes coordinates in the xyxy format
        x（np.ndarray | torch.Tensor）：输入的张量，格式为 xyxy 的边界框坐标。

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format.
        y（np.ndarray | torch.Tensor）：边界框坐标，格式为 xyltwh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 如果是 PyTorch 张量则克隆，否则使用 numpy 复制
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    # 计算宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    # 计算高度
    return y


def ltwh2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.
    将 nx4 框从 [x1, y1, w, h] 转换为 [x, y, w, h]，其中 xy1 是左上角，xy 是中心。

    Args:
        x (torch.Tensor): the input tensor
        x（torch.Tensor）：输入的张量。

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xywh format.
        y（np.ndarray | torch.Tensor）：边界框坐标，格式为 xywh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 如果是 PyTorch 张量则克隆，否则使用 numpy 复制
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    # 计算中心 x 坐标
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    # 计算中心 y 坐标
    return y


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    returned in radians from 0 to pi/2.
    将批量定向边界框 (OBB) 从 [xy1, xy2, xy3, xy4] 转换为 [xywh, rotation]。旋转值以弧度形式返回，从 0 到 pi/2。

    Args:
        x (numpy.ndarray | torch.Tensor): Input box corners [xy1, xy2, xy3, xy4] of shape (n, 8).
        x（numpy.ndarray | torch.Tensor）：输入框角 [xy1, xy2, xy3, xy4]，形状为 (n, 8)。

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
        (numpy.ndarray | torch.Tensor)：转换后的数据，格式为 [cx, cy, w, h, rotation]，形状为 (n, 5)。
    """
    is_torch = isinstance(x, torch.Tensor)
    # 判断 x 是否为 PyTorch 张量
    points = x.cpu().numpy() if is_torch else x
    # 如果是 PyTorch 张量，将其转换为 numpy 数组
    points = points.reshape(len(x), -1, 2)
    # 将点的形状调整为 (n, -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        # 注意：使用 cv2.minAreaRect 获取准确的 xywhr，特别是在数据加载器中某些对象被增强裁剪时。
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        # 使用 cv2.minAreaRect 计算中心坐标、宽度和高度
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
        # 将结果添加到 rboxes 列表中，角度转换为弧度
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)
    # 如果是 PyTorch 张量，则返回 PyTorch 张量，否则返回 numpy 数组


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.
    将批量定向边界框 (OBB) 从 [xywh, rotation] 转换为 [xy1, xy2, xy3, xy4]。旋转值应以弧度形式表示，从 0 到 pi/2。

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).
        x（numpy.ndarray | torch.Tensor）：格式为 [cx, cy, w, h, rotation] 的框，形状为 (n, 5) 或 (b, n, 5)。

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
        (numpy.ndarray | torch.Tensor)：转换后的角点，形状为 (n, 4, 2) 或 (b, n, 4, 2)。
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )
    # 根据 x 的类型选择相应的 numpy 或 PyTorch 函数

    ctr = x[..., :2]
    # 获取中心坐标
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    # 获取宽度、高度和角度
    cos_value, sin_value = cos(angle), sin(angle)
    # 计算 cos 和 sin 值
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    # 计算第一个向量
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    # 计算第二个向量
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    # 合并向量
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    # 计算四个角点
    return stack([pt1, pt2, pt3, pt4], -2)
    # 将四个角点堆叠在一起并返回


def ltwh2xyxy(x):
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.
    将边界框从 [x1, y1, w, h] 转换为 [x1, y1, x2, y2]，其中 xy1 是左上角，xy2 是右下角。

    Args:
        x (np.ndarray | torch.Tensor): the input image
        x（np.ndarray | torch.Tensor）：输入的图像。

    Returns:
        y (np.ndarray | torch.Tensor): the xyxy coordinates of the bounding boxes.
        y（np.ndarray | torch.Tensor）：边界框的 xyxy 坐标。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # 如果是 PyTorch 张量则克隆，否则使用 numpy 复制
    y[..., 2] = x[..., 2] + x[..., 0]  # width
    # 计算右下角 x 坐标
    y[..., 3] = x[..., 3] + x[..., 1]  # height
    # 计算右下角 y 坐标
    return y


def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh).
    将段标签转换为框标签，即 (cls, xy1, xy2, ...) 转换为 (cls, xywh)。

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates
        segments（列表）：段的列表，每个段是一个点的列表，每个点是 x 和 y 坐标的列表。

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
        (np.ndarray)：边界框的 xywh 坐标。
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        # 获取段的 x 和 y 坐标
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
        # 将最小和最大值作为边界框添加到 boxes 列表中
    return xyxy2xywh(np.array(boxes))  # cls, xywh
    # 将 boxes 转换为 xywh 格式并返回


def resample_segments(segments, n=1000):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.
    输入一组段 (n,2) 的列表，并返回每个段上采样到 n 个点的段列表。

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        segments（列表）：一组 (n,2) 数组的列表，其中 n 是段中的点数。
        n (int): number of points to resample the segment to. Defaults to 1000
        n（整数）：将段重新采样到的点数。默认为1000

    Returns:
        segments (list): the resampled segments.
        segments（列表）：重新采样后的段。
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        # 如果段的长度已经是 n，则继续
        s = np.concatenate((s, s[0:1, :]), axis=0)
        # 将段的第一个点添加到末尾以闭合段
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        # 创建等间隔的点
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        # 如果段的长度小于 n，则在适当位置插入原始点
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
        # 对每个坐标进行插值并重新调整形状
    return segments


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.
    它接受一个掩码和一个边界框，并返回裁剪到边界框的掩码。

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        masks（torch.Tensor）：形状为 [n, h, w] 的掩码张量。
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form
        boxes（torch.Tensor）：形状为 [n, 4] 的边界框坐标张量，表示相对坐标。

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
        (torch.Tensor)：裁剪到边界框的掩码。
    """
    _, h, w = masks.shape
    # 获取掩码的高度和宽度
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    # 将 boxes 拆分为 x1, y1, x2, y2
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    # 创建行索引
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
    # 创建列索引

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    # 根据边界框裁剪掩码

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.
    使用掩码头的输出将掩码应用于边界框。

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        protos（torch.Tensor）：形状为 [mask_dim, mask_h, mask_w] 的张量。
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        masks_in（torch.Tensor）：形状为 [n, mask_dim] 的张量，其中 n 是 NMS 后的掩码数量。
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        bboxes（torch.Tensor）：形状为 [n, 4] 的张量，其中 n 是 NMS 后的掩码数量。
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        shape（元组）：表示输入图像大小的整数元组，格式为 (h, w)。
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.
        upsample（布尔值）：指示是否将掩码上采样到原始图像大小的标志。默认为 False。

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
        (torch.Tensor)：形状为 [n, h, w] 的二进制掩码张量，其中 n 是 NMS 后的掩码数量，h 和 w 是输入图像的高度和宽度。掩码应用于边界框。
    """
    c, mh, mw = protos.shape  # CHW
    # 获取 protos 的通道数、高度和宽度
    ih, iw = shape
    # 获取输入图像的高度和宽度
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    # 计算掩码，使用掩码输入和原型的矩阵乘法
    width_ratio = mw / iw
    # 计算宽度比例
    height_ratio = mh / ih
    # 计算高度比例

    downsampled_bboxes = bboxes.clone()
    # 克隆边界框以进行下采样
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio
    # 根据比例调整边界框坐标

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    # 根据下采样的边界框裁剪掩码
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
        # 如果需要上采样，则使用双线性插值将掩码调整到原始图像大小
    return masks.gt_(0.0)
    # 返回二进制掩码，值大于0的部分为True


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.
    它接受掩码头的输出，并在上采样后裁剪到边界框。

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        protos（torch.Tensor）：形状为 [mask_dim, mask_h, mask_w] 的张量。
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms.
        masks_in（torch.Tensor）：形状为 [n, mask_dim] 的张量，n 是 NMS 后的掩码数量。
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms.
        bboxes（torch.Tensor）：形状为 [n, 4] 的张量，n 是 NMS 后的掩码数量。
        shape (tuple): The size of the input image (h,w).
        shape（元组）：输入图像的大小 (h,w)。

    Returns:
        masks (torch.Tensor): The returned masks with dimensions [h, w, n].
        masks（torch.Tensor）：返回的掩码，维度为 [h, w, n]。
    """
    c, mh, mw = protos.shape  # CHW
    # 获取 protos 的通道数、高度和宽度
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    # 计算掩码，使用掩码输入和原型的矩阵乘法
    masks = scale_masks(masks[None], shape)[0]  # CHW
    # 将掩码上采样到原始图像大小
    masks = crop_mask(masks, bboxes)  # CHW
    # 根据边界框裁剪掩码
    return masks.gt_(0.0)
    # 返回二进制掩码，值大于0的部分为True


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.
    将段掩码调整为指定形状。

    Args:
        masks (torch.Tensor): (N, C, H, W).
        masks（torch.Tensor）：形状为 (N, C, H, W) 的张量。
        shape (tuple): Height and width.
        shape（元组）：高度和宽度。
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        padding（布尔值）：如果为 True，假设边界框基于 YOLO 风格的增强图像。如果为 False，则进行常规缩放。
    """
    mh, mw = masks.shape[2:]
    # 获取掩码的高度和宽度
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    # 计算缩放比例
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    # 计算宽度和高度的填充
    if padding:
        pad[0] /= 2
        pad[1] /= 2
        # 如果需要填充，则将填充值减半
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    # 根据填充情况设置上和左的填充
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    # 计算下和右的填充
    masks = masks[..., top:bottom, left:right]
    # 根据填充裁剪掩码

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    # 将掩码上采样到指定形状
    return masks


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.
    将段坐标 (xy) 从 img1_shape 缩放到 img0_shape。

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        img1_shape（元组）：坐标来源的图像形状。
        coords (torch.Tensor): the coords to be scaled of shape n,2.
        coords（torch.Tensor）：要缩放的坐标，形状为 n,2。
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        img0_shape（元组）：应用分割的图像形状。
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        ratio_pad（元组）：图像大小与填充图像大小的比率。
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        normalize（布尔值）：如果为 True，坐标将被归一化到 [0, 1] 范围内。默认为 False。
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        padding（布尔值）：如果为 True，假设边界框基于 YOLO 风格的增强图像。如果为 False，则进行常规缩放。

    Returns:
        coords (torch.Tensor): The scaled coordinates.
        coords（torch.Tensor）：缩放后的坐标。
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # 计算缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        # 计算宽度和高度的填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        # 应用 x 方向的填充
        coords[..., 1] -= pad[1]  # y padding
        # 应用 y 方向的填充
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    # 根据缩放比例调整坐标
    coords = clip_coords(coords, img0_shape)
    # 裁剪坐标以确保在图像边界内
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        # 将 x 坐标归一化到图像宽度
        coords[..., 1] /= img0_shape[0]  # height
        # 将 y 坐标归一化到图像高度
    return coords


def regularize_rboxes(rboxes):
    """
    Regularize rotated boxes in range [0, pi/2].
    将旋转框规范化到范围 [0, pi/2]。

    Args:
        rboxes (torch.Tensor): Input boxes of shape(N, 5) in xywhr format.
        rboxes（torch.Tensor）：形状为 (N, 5) 的输入框，格式为 xywhr。

    Returns:
        (torch.Tensor): The regularized boxes.
        (torch.Tensor)：规范化后的框。
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # 将 rboxes 拆分为 x, y, w, h, t
    # Swap edge and angle if h >= w
    w_ = torch.where(w > h, w, h)
    # 计算宽度，选择较大的值
    h_ = torch.where(w > h, h, w)
    # 计算高度，选择较大的值
    t = torch.where(w > h, t, t + math.pi / 2) % math.pi
    # 如果高度大于宽度，则调整角度
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes
    # 将规范化后的框堆叠并返回

def masks2segments(masks, strategy="all"):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy).
    它接受一个形状为 (n,h,w) 的掩码列表，并返回一个形状为 (n,xy) 的段列表。

    Args:
        masks (torch.Tensor): the output of the model, which is a tensor of shape (batch_size, 160, 160)
        masks（torch.Tensor）：模型的输出，形状为 (batch_size, 160, 160) 的张量。
        strategy (str): 'all' or 'largest'. Defaults to all
        strategy（字符串）：'all' 或 'largest'。默认为 'all'。

    Returns:
        segments (List): list of segment masks
        segments（列表）：段掩码的列表。
    """
    from ultralytics.data.converter import merge_multi_segment
    # 从 ultralytics.data.converter 导入 merge_multi_segment 函数

    segments = []
    # 初始化一个空列表，用于存储段
    for x in masks.int().cpu().numpy().astype("uint8"):
        # 将掩码转换为整数，移到 CPU 上，转为 numpy 数组，并转换为无符号8位整数
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # 使用 OpenCV 查找轮廓，提取外部轮廓
        if c:
            if strategy == "all":  # merge and concatenate all segments
                # 如果策略是 'all'，则合并所有段
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    # 将每个轮廓重塑为 (n, 2) 的形状并合并
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                    # 如果只有一个轮廓，则直接使用
                )
            elif strategy == "largest":  # select largest segment
                # 如果策略是 'largest'，则选择最大的段
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
                # 找到长度最大的轮廓并重塑为 (n, 2) 的形状
        else:
            c = np.zeros((0, 2))  # no segments found
            # 如果未找到任何段，则创建一个形状为 (0, 2) 的空数组
        segments.append(c.astype("float32"))
        # 将段添加到 segments 列表，并转换为 float32 类型
    return segments
    # 返回段列表


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of FP32 torch tensors (0.0-1.0) to a NumPy uint8 array (0-255), changing from BCHW to BHWC layout.
    将一批 FP32 的 PyTorch 张量 (0.0-1.0) 转换为 NumPy uint8 数组 (0-255)，并将布局从 BCHW 改为 BHWC。

    Args:
        batch (torch.Tensor): Input tensor batch of shape (Batch, Channels, Height, Width) and dtype torch.float32.
        batch（torch.Tensor）：形状为 (Batch, Channels, Height, Width) 且数据类型为 torch.float32 的输入张量。

    Returns:
        (np.ndarray): Output NumPy array batch of shape (Batch, Height, Width, Channels) and dtype uint8.
        (np.ndarray)：形状为 (Batch, Height, Width, Channels) 且数据类型为 uint8 的输出 NumPy 数组。
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    # 将张量的维度调整为 BHWC，乘以 255 转换为 uint8 类型，并限制在 0 到 255 之间


def clean_str(s):
    """
    Cleans a string by replacing special characters with '_' character.
    通过用 '_' 字符替换特殊字符来清理字符串。

    Args:
        s (str): a string needing special characters replaced
        s（字符串）：需要替换特殊字符的字符串。

    Returns:
        (str): a string with special characters replaced by an underscore _
        (str)：用下划线替换特殊字符后的字符串。
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
    # 使用正则表达式替换特殊字符


def empty_like(x):
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    # 创建一个与输入形状相同且数据类型为 float32 的空 torch.Tensor 或 np.ndarray。
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )