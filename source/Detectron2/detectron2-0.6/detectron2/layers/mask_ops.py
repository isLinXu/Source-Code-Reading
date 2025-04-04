# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Tuple
import torch
from PIL import Image
from torch.nn import functional as F

from detectron2.structures import Boxes

__all__ = ["paste_masks_in_image"]  # 指定公开的API


BYTES_PER_FLOAT = 4  # 每个浮点数占用的字节数
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
# TODO: 这个内存限制可能太多或太少。最好根据可用资源确定。
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit
                           # 1 GB内存限制


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        # masks: N个掩码，维度为N, 1, H, W
        boxes: N, 4
        # boxes: N个边界框，每个框4个坐标
        img_h, img_w (int):
        # img_h, img_w (int): 图像高度和宽度
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.
        # skip_empty (bool): 仅粘贴紧密包围所有框的区域内的掩码，
        # 并仅返回该区域的结果。这是CPU上的重要优化。

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        # 如果skip_empty == False，返回形状为(N, img_h, img_w)的掩码
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
        # 如果skip_empty == True，返回形状为(N, h', w')的掩码，
        # 以及对应区域的切片对象。
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    # 在GPU上，通过使用整个图像来采样掩码，将所有掩码一起粘贴（根据块大小）
    # 与一个一个粘贴相比，这种方法虽然操作更多，但在COCO规模的数据集上更快。
    device = masks.device  # 获取设备类型

    if skip_empty and not torch.jit.is_scripting():
        # 如果skip_empty为True且不是在JIT脚本模式下，计算掩码区域的边界
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )  # 计算左上角坐标并确保不小于0
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)  # 右下角x坐标
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)  # 右下角y坐标
    else:
        # 如果不跳过空区域，使用整个图像
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1
                                                   # 将边界框分解为单独的坐标，每个都是Nx1

    N = masks.shape[0]  # 掩码数量

    # 创建图像采样坐标网格
    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5  # y坐标
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5  # x坐标
    # 将坐标归一化到[-1,1]范围，用于grid_sample
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # img_x, img_y的形状分别为(N, w)和(N, h)

    # 扩展坐标网格
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))  # 扩展x坐标
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))  # 扩展y坐标
    grid = torch.stack([gx, gy], dim=3)  # 组合x和y坐标形成网格

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()  # 确保掩码为浮点型
    # 使用grid_sample对掩码进行重采样
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        # 如果跳过空区域，返回掩码和区域切片
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()  # 否则只返回掩码


def paste_masks_in_image(
    masks: torch.Tensor, boxes: Boxes, image_shape: Tuple[int, int], threshold: float = 0.5
):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.
    将一组固定分辨率的掩码（例如28 x 28）粘贴到图像中。
    每个掩码的粘贴位置、高度和宽度由其对应的边界框确定。

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.
    注意：
        这是一个复杂但更准确的实现。在实际部署中，通常使用更快但精度较低的实现就足够了。
        关于替代实现，请参见本文件中的:func:`paste_mask_in_image_old`函数。

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        # masks (tensor): 形状为(Bimg, Hmask, Wmask)的张量，其中Bimg是图像中检测到的
        # 对象实例数，Hmask和Wmask是预测掩码的宽度和高度（例如，Hmask = Wmask = 28）。
        # 值在[0, 1]范围内。
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        # boxes (Boxes或Tensor): 长度为Bimg的Boxes对象或形状为(Bimg, 4)的Tensor。
        # boxes[i]和masks[i]对应同一个对象实例。
        image_shape (tuple): height, width
        # image_shape (tuple): 图像高度和宽度
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.
        # threshold (float): 将(软)掩码转换为二值掩码的阈值，取值范围[0, 1]。

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    # 返回：
    #   img_masks (Tensor): 形状为(Bimg, Himage, Wimage)的张量，其中Bimg是检测到的
    #   对象实例数，Himage和Wimage是图像的宽度和高度。img_masks[i]是对象实例i的二值掩码。
    """

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    # 断言掩码的宽高相等，只支持正方形掩码预测
    N = len(masks)  # 掩码数量
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)  # 如果没有掩码，返回空张量
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor  # 如果boxes不是张量，转换为张量
    device = boxes.device  # 获取设备类型
    assert len(boxes) == N, boxes.shape  # 确保boxes和masks数量一致

    img_h, img_w = image_shape  # 获取图像高度和宽度

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    # 实际实现将输入分成多个块，然后逐块粘贴。
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        # CPU在使用skip_empty=True逐个粘贴时效率最高，这样操作数量最少。
        num_chunks = N  # 在CPU上，每个掩码单独处理
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        # GPU对较大的块有并行优势，但可能有内存问题
        # 使用int(img_h)是因为在追踪时shape可能是张量
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        # 根据内存限制计算块数
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
        # 确保块数不超过掩码数量
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)  # 将掩码索引分块

    # 创建输出掩码张量
    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        # 逐块处理掩码
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            # 如果设置了阈值，将掩码二值化
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            # 用于可视化和调试
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
                                      # 脚本模式不使用优化的代码路径
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk  # 将处理后的掩码放入结果张量
    return img_masks  # 返回最终的掩码


# The below are the original paste function (from Detectron1) which has
# larger quantization error.
# It is faster on CPU, while the aligned one is faster on GPU thanks to grid_sample.
# 以下是原始的粘贴函数(来自Detectron1)，它有较大的量化误差。
# 在CPU上速度更快，而对齐版本在GPU上由于使用了grid_sample而更快。


def paste_mask_in_image_old(mask, box, img_h, img_w, threshold):
    """
    Paste a single mask in an image.
    This is a per-box implementation of :func:`paste_masks_in_image`.
    This function has larger quantization error due to incorrect pixel
    modeling and is not used any more.
    在图像中粘贴单个掩码。
    这是:func:`paste_masks_in_image`的逐框实现。
    由于像素建模不正确，此函数具有较大的量化误差，不再使用。

    Args:
        mask (Tensor): A tensor of shape (Hmask, Wmask) storing the mask of a single
            object instance. Values are in [0, 1].
        # mask (Tensor): 形状为(Hmask, Wmask)的张量，存储单个对象实例的掩码。
        # 值在[0, 1]范围内。
        box (Tensor): A tensor of shape (4, ) storing the x0, y0, x1, y1 box corners
            of the object instance.
        # box (Tensor): 形状为(4, )的张量，存储对象实例的x0, y0, x1, y1边界框坐标。
        img_h, img_w (int): Image height and width.
        # img_h, img_w (int): 图像高度和宽度。
        threshold (float): Mask binarization threshold in [0, 1].
        # threshold (float): 掩码二值化阈值，范围[0, 1]。

    Returns:
        im_mask (Tensor):
            The resized and binarized object mask pasted into the original
            image plane (a tensor of shape (img_h, img_w)).
    # 返回：
    #   im_mask (Tensor): 调整大小和二值化后的对象掩码，粘贴到原始图像平面上
    #   (形状为(img_h, img_w)的张量)。
    """
    # Conversion from continuous box coordinates to discrete pixel coordinates
    # via truncation (cast to int32). This determines which pixels to paste the
    # mask onto.
    # 通过截断(转换为int32)将连续框坐标转换为离散像素坐标。
    # 这决定了要将掩码粘贴到哪些像素上。
    box = box.to(dtype=torch.int32)  # Continuous to discrete coordinate conversion
                                     # 连续坐标到离散坐标的转换
    # An example (1D) box with continuous coordinates (x0=0.7, x1=4.3) will map to
    # a discrete coordinates (x0=0, x1=4). Note that box is mapped to 5 = x1 - x0 + 1
    # pixels (not x1 - x0 pixels).
    # 一个具有连续坐标的(1D)框(x0=0.7, x1=4.3)将映射到
    # 离散坐标(x0=0, x1=4)。注意，框被映射到5 = x1 - x0 + 1个像素
    # (而不是x1 - x0个像素)。
    samples_w = box[2] - box[0] + 1  # Number of pixel samples, *not* geometric width
                                     # 像素样本数量，*不是*几何宽度
    samples_h = box[3] - box[1] + 1  # Number of pixel samples, *not* geometric height
                                     # 像素样本数量，*不是*几何高度

    # Resample the mask from it's original grid to the new samples_w x samples_h grid
    # 将掩码从原始网格重新采样到新的samples_w x samples_h网格
    mask = Image.fromarray(mask.cpu().numpy())  # 转换为PIL图像
    mask = mask.resize((samples_w, samples_h), resample=Image.BILINEAR)  # 调整大小
    mask = np.array(mask, copy=False)  # 转换回numpy数组

    if threshold >= 0:
        # 如果设置了阈值，将掩码二值化
        mask = np.array(mask > threshold, dtype=np.uint8)
        mask = torch.from_numpy(mask)
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        # 为了可视化和调试，我们也允许返回未修改的掩码
        mask = torch.from_numpy(mask * 255).to(torch.uint8)

    # 创建空的目标掩码
    im_mask = torch.zeros((img_h, img_w), dtype=torch.uint8)
    # 计算有效区域的坐标
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, img_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, img_h)

    # 将掩码粘贴到目标区域
    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


# Our pixel modeling requires extrapolation for any continuous
# coordinate < 0.5 or > length - 0.5. When sampling pixels on the masks,
# we would like this extrapolation to be an interpolation between boundary values and zero,
# instead of using absolute zero or boundary values.
# Therefore `paste_mask_in_image_old` is often used with zero padding around the masks like this:
# masks, scale = pad_masks(masks[:, 0, :, :], 1)
# boxes = scale_boxes(boxes.tensor, scale)
# 我们的像素建模需要对任何连续坐标<0.5或>length-0.5进行外插。
# 当对掩码进行像素采样时，我们希望这种外插是边界值和零之间的插值，
# 而不是使用绝对零或边界值。
# 因此，`paste_mask_in_image_old`通常与掩码周围的零填充一起使用，如下所示：
# masks, scale = pad_masks(masks[:, 0, :, :], 1)
# boxes = scale_boxes(boxes.tensor, scale)


def pad_masks(masks, padding):
    """
    Args:
        masks (tensor): A tensor of shape (B, M, M) representing B masks.
        # masks (tensor): 形状为(B, M, M)的张量，表示B个掩码。
        padding (int): Number of cells to pad on all sides.
        # padding (int): 在所有边上填充的单元格数。

    Returns:
        The padded masks and the scale factor of the padding size / original size.
    # 返回：
    #   填充后的掩码和填充大小/原始大小的比例因子。
    """
    B = masks.shape[0]  # 掩码数量
    M = masks.shape[-1]  # 掩码大小
    pad2 = 2 * padding  # 两边填充的总大小
    scale = float(M + pad2) / M  # 计算缩放比例
    padded_masks = masks.new_zeros((B, M + pad2, M + pad2))  # 创建填充后的掩码张量
    padded_masks[:, padding:-padding, padding:-padding] = masks  # 将原始掩码复制到填充掩码的中心
    return padded_masks, scale  # 返回填充后的掩码和缩放比例


def scale_boxes(boxes, scale):
    """
    Args:
        boxes (tensor): A tensor of shape (B, 4) representing B boxes with 4
            coords representing the corners x0, y0, x1, y1,
        # boxes (tensor): 形状为(B, 4)的张量，表示B个框，每个框有4个坐标
        # 表示角点x0, y0, x1, y1，
        scale (float): The box scaling factor.
        # scale (float): 框缩放因子。

    Returns:
        Scaled boxes.
    # 返回：
    #   缩放后的框。
    """
    # 计算框的宽度和高度的一半
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    # 计算框的中心点
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    # 应用缩放
    w_half *= scale
    h_half *= scale

    # 创建缩放后的框
    scaled_boxes = torch.zeros_like(boxes)
    scaled_boxes[:, 0] = x_c - w_half  # 左上角x
    scaled_boxes[:, 2] = x_c + w_half  # 右下角x
    scaled_boxes[:, 1] = y_c - h_half  # 左上角y
    scaled_boxes[:, 3] = y_c + h_half  # 右下角y
    return scaled_boxes  # 返回缩放后的框
