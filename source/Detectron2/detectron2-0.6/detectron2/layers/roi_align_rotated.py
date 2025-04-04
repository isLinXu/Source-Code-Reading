# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from detectron2 import _C  # 导入C++/CUDA实现的操作


class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        # 保存反向传播所需的信息
        ctx.save_for_backward(roi)  # 保存ROI用于反向传播
        ctx.output_size = _pair(output_size)  # 将输出大小转换为(h, w)对
        ctx.spatial_scale = spatial_scale  # 保存空间尺度因子
        ctx.sampling_ratio = sampling_ratio  # 保存采样比率
        ctx.input_shape = input.size()  # 保存输入张量的形状
        # 调用C++/CUDA实现的前向传播函数
        output = _C.roi_align_rotated_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output  # 返回对齐后的特征

    @staticmethod
    @once_differentiable  # 标记为一次可微分，优化内存使用
    def backward(ctx, grad_output):
        # 从上下文中获取保存的信息
        (rois,) = ctx.saved_tensors  # 获取保存的ROI
        output_size = ctx.output_size  # 获取输出大小
        spatial_scale = ctx.spatial_scale  # 获取空间尺度因子
        sampling_ratio = ctx.sampling_ratio  # 获取采样比率
        bs, ch, h, w = ctx.input_shape  # 获取输入形状的各个维度
        # 调用C++/CUDA实现的反向传播函数计算梯度
        grad_input = _C.roi_align_rotated_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        # 返回各个输入参数的梯度，对于不需要梯度的参数返回None
        return grad_input, None, None, None, None, None


roi_align_rotated = _ROIAlignRotated.apply  # 创建函数接口


class ROIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        """
        Args:
            output_size (tuple): h, w
            # output_size (tuple): 输出特征图的高度和宽度
            spatial_scale (float): scale the input boxes by this number
            # spatial_scale (float): 用于缩放输入框的系数
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            # sampling_ratio (int): 每个输出样本需要采样的输入样本数。
            # 设为0表示密集采样。

        Note:
            ROIAlignRotated supports continuous coordinate by default:
            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5).
        # 注意：
        #     ROIAlignRotated默认支持连续坐标：
        #     给定一个连续坐标c，其两个相邻像素索引（在我们的像素模型中）
        #     通过floor(c - 0.5)和ceil(c - 0.5)计算。例如，
        #     c=1.3的像素邻居离散索引为[0]和[1]（它们是从
        #     连续坐标0.5和1.5的底层信号采样而来）。
        """
        super(ROIAlignRotated, self).__init__()
        self.output_size = output_size  # 存储输出大小
        self.spatial_scale = spatial_scale  # 存储空间尺度因子
        self.sampling_ratio = sampling_ratio  # 存储采样比率

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            # input: NCHW格式的图像特征
            rois: Bx6 boxes. First column is the index into N.
                The other 5 columns are (x_ctr, y_ctr, width, height, angle_degrees).
            # rois: Bx6的框。第一列是批次索引N。
            #     其他5列是(x中心, y中心, 宽度, 高度, 角度(度))。
        """
        assert rois.dim() == 2 and rois.size(1) == 6  # 确保ROI的形状正确
        orig_dtype = input.dtype  # 记录原始数据类型
        if orig_dtype == torch.float16:
            # 如果是半精度浮点数，转换为单精度以提高数值稳定性
            input = input.float()
            rois = rois.float()
        # 调用ROI对齐旋转函数并转换回原始数据类型
        return roi_align_rotated(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        ).to(dtype=orig_dtype)

    def __repr__(self):
        # 生成模块的字符串表示，用于打印模块信息
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
