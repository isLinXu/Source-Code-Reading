# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import lru_cache
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

from detectron2 import _C

from .wrappers import _NewEmptyTensorOp


class _DeformConv(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        weight,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=64,
    ):
        # 变形卷积的前向传播函数
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(input.dim())
            )
            # 输入必须是4D张量，否则抛出错误
        ctx.stride = _pair(stride)  # 将步长转换为(height_stride, width_stride)格式
        ctx.padding = _pair(padding)  # 将填充转换为(height_padding, width_padding)格式
        ctx.dilation = _pair(dilation)  # 将空洞率转换为(height_dilation, width_dilation)格式
        ctx.groups = groups  # 卷积分组数
        ctx.deformable_groups = deformable_groups  # 可变形分组数
        ctx.im2col_step = im2col_step  # 图像到列转换的步长

        ctx.save_for_backward(input, offset, weight)  # 保存用于反向传播的张量

        # 创建输出张量，大小由_output_size函数计算
        output = input.new_empty(
            _DeformConv._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride)
        )

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones
                                                              # 创建两个空缓冲区：columns和ones

        if not input.is_cuda:
            # 如果输入不在GPU上，则使用CPU实现
            if deformable_groups != 1:
                raise NotImplementedError(
                    "Deformable Conv with deformable_groups != 1 is not supported on CPUs!"
                )
                # CPU上不支持deformable_groups不为1的情况
            return deform_conv2d(
                input, offset, weight, stride=stride, padding=padding, dilation=dilation
            )
        else:
            # 如果输入在GPU上，则使用CUDA实现
            cur_im2col_step = _DeformConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            # 计算当前im2col步长
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"
            # 确保im2col步长能够整除批量大小

            # 调用CUDA核函数执行变形卷积的前向传播
            _C.deform_conv_forward(
                input,
                weight,
                offset,
                output,
                ctx.bufs_[0],
                ctx.bufs_[1],
                weight.size(3),  # 卷积核宽度
                weight.size(2),  # 卷积核高度
                ctx.stride[1],   # 宽度步长
                ctx.stride[0],   # 高度步长
                ctx.padding[1],  # 宽度填充
                ctx.padding[0],  # 高度填充
                ctx.dilation[1], # 宽度空洞率
                ctx.dilation[0], # 高度空洞率
                ctx.groups,      # 卷积分组
                ctx.deformable_groups,  # 可变形分组
                cur_im2col_step, # 当前im2col步长
            )
        return output  # 返回卷积结果

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # 变形卷积的反向传播函数，计算各个输入的梯度
        input, offset, weight = ctx.saved_tensors  # 获取保存的张量

        grad_input = grad_offset = grad_weight = None  # 初始化梯度为None

        if not grad_output.is_cuda:
            # CPU上不支持变形卷积的反向传播
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        else:
            # 在GPU上执行反向传播
            cur_im2col_step = _DeformConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            # 计算当前im2col步长
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"
            # 确保im2col步长能够整除批量大小

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                # 如果需要计算输入或偏移的梯度
                grad_input = torch.zeros_like(input)   # 创建与输入相同形状的梯度张量
                grad_offset = torch.zeros_like(offset) # 创建与偏移相同形状的梯度张量
                # 调用CUDA核函数计算输入和偏移的梯度
                _C.deform_conv_backward_input(
                    input,
                    offset,
                    grad_output,
                    grad_input,
                    grad_offset,
                    weight,
                    ctx.bufs_[0],
                    weight.size(3),  # 卷积核宽度
                    weight.size(2),  # 卷积核高度
                    ctx.stride[1],   # 宽度步长
                    ctx.stride[0],   # 高度步长
                    ctx.padding[1],  # 宽度填充
                    ctx.padding[0],  # 高度填充
                    ctx.dilation[1], # 宽度空洞率
                    ctx.dilation[0], # 高度空洞率
                    ctx.groups,      # 卷积分组
                    ctx.deformable_groups,  # 可变形分组
                    cur_im2col_step, # 当前im2col步长
                )

            if ctx.needs_input_grad[2]:
                # 如果需要计算权重的梯度
                grad_weight = torch.zeros_like(weight)  # 创建与权重相同形状的梯度张量
                # 调用CUDA核函数计算权重的梯度
                _C.deform_conv_backward_filter(
                    input,
                    offset,
                    grad_output,
                    grad_weight,
                    ctx.bufs_[0],
                    ctx.bufs_[1],
                    weight.size(3),  # 卷积核宽度
                    weight.size(2),  # 卷积核高度
                    ctx.stride[1],   # 宽度步长
                    ctx.stride[0],   # 高度步长
                    ctx.padding[1],  # 宽度填充
                    ctx.padding[0],  # 高度填充
                    ctx.dilation[1], # 宽度空洞率
                    ctx.dilation[0], # 高度空洞率
                    ctx.groups,      # 卷积分组
                    ctx.deformable_groups,  # 可变形分组
                    1,               # 权重的im2col步长
                    cur_im2col_step, # 当前im2col步长
                )

        # 返回各个输入的梯度，None表示不需要计算该参数的梯度
        return grad_input, grad_offset, grad_weight, None, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        # 计算卷积输出的大小
        channels = weight.size(0)  # 输出通道数
        output_size = (input.size(0), channels)  # 批大小和输出通道数
        for d in range(input.dim() - 2):
            # 对于高度和宽度维度
            in_size = input.size(d + 2)  # 输入大小
            pad = padding[d]  # 填充大小
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1  # 考虑空洞率的卷积核大小
            stride_ = stride[d]  # 步长
            # 计算输出大小
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            # 如果输出大小的任何维度小于等于0，则抛出错误
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    "x".join(map(str, output_size))
                )
            )
        return output_size  # 返回输出大小

    @staticmethod
    @lru_cache(maxsize=128)
    def _cal_im2col_step(input_size, default_size):
        """
        Calculate proper im2col step size, which should be divisible by input_size and not larger
        than prefer_size. Meanwhile the step size should be as large as possible to be more
        efficient. So we choose the largest one among all divisors of input_size which are smaller
        than prefer_size.
        计算适当的im2col步长大小，该步长应能被input_size整除且不大于prefer_size。同时，步长应尽可能大以提高效率。
        因此，我们选择input_size的所有小于prefer_size的除数中最大的一个。
        
        :param input_size: input batch size .
                          输入批量大小。
        :param default_size: default preferred im2col step size.
                            默认首选的im2col步长大小。
        :return: the largest proper step size.
                 最大的适当步长大小。
        """
        if input_size <= default_size:
            # 如果输入大小小于等于默认大小，直接返回输入大小
            return input_size
        best_step = 1  # 初始化最佳步长为1
        # 遍历可能的步长
        for step in range(2, min(int(math.sqrt(input_size)) + 1, default_size)):
            if input_size % step == 0:  # 如果步长能整除输入大小
                if input_size // step <= default_size:
                    # 如果输入大小除以步长小于等于默认大小，直接返回这个结果
                    return input_size // step
                best_step = step  # 否则更新最佳步长

        return best_step  # 返回找到的最佳步长


class _ModulatedDeformConv(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
    ):
        # 调制变形卷积的前向传播函数
        ctx.stride = stride  # 卷积步长
        ctx.padding = padding  # 填充大小
        ctx.dilation = dilation  # 空洞率
        ctx.groups = groups  # 卷积分组数
        ctx.deformable_groups = deformable_groups  # 可变形分组数
        ctx.with_bias = bias is not None  # 是否使用偏置
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
                                       # 如果不使用偏置，创建一个假的偏置张量
        if not input.is_cuda:
            # CPU上不支持变形卷积
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        if (
            weight.requires_grad
            or mask.requires_grad
            or offset.requires_grad
            or input.requires_grad
        ):
            # 如果任何输入需要梯度，则保存它们用于反向传播
            ctx.save_for_backward(input, offset, mask, weight, bias)
        # 创建输出张量
        output = input.new_empty(_ModulatedDeformConv._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]  # 创建两个空缓冲区
        # 调用CUDA核函数执行调制变形卷积的前向传播
        _C.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            weight.shape[2],  # 卷积核高度
            weight.shape[3],  # 卷积核宽度
            ctx.stride,       # 高度步长
            ctx.stride,       # 宽度步长
            ctx.padding,      # 高度填充
            ctx.padding,      # 宽度填充
            ctx.dilation,     # 高度空洞率
            ctx.dilation,     # 宽度空洞率
            ctx.groups,       # 卷积分组
            ctx.deformable_groups,  # 可变形分组
            ctx.with_bias,    # 是否使用偏置
        )
        return output  # 返回卷积结果

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # 调制变形卷积的反向传播函数，计算各个输入的梯度
        if not grad_output.is_cuda:
            # CPU上不支持变形卷积的反向传播
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        input, offset, mask, weight, bias = ctx.saved_tensors  # 获取保存的张量
        # 创建与各个输入相同形状的梯度张量
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        # 调用CUDA核函数计算所有梯度
        _C.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],  # 卷积核高度
            weight.shape[3],  # 卷积核宽度
            ctx.stride,       # 高度步长
            ctx.stride,       # 宽度步长
            ctx.padding,      # 高度填充
            ctx.padding,      # 宽度填充
            ctx.dilation,     # 高度空洞率
            ctx.dilation,     # 宽度空洞率
            ctx.groups,       # 卷积分组
            ctx.deformable_groups,  # 可变形分组
            ctx.with_bias,    # 是否使用偏置
        )
        if not ctx.with_bias:
            grad_bias = None  # 如果不使用偏置，则偏置的梯度为None

        # 返回各个输入的梯度，None表示不需要计算该参数的梯度
        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def _infer_shape(ctx, input, weight):
        # 推断调制变形卷积输出的形状
        n = input.size(0)  # 批大小
        channels_out = weight.size(0)  # 输出通道数
        height, width = input.shape[2:4]  # 输入高度和宽度
        kernel_h, kernel_w = weight.shape[2:4]  # 卷积核高度和宽度
        # 计算输出高度
        height_out = (
            height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)
        ) // ctx.stride + 1
        # 计算输出宽度
        width_out = (
            width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)
        ) // ctx.stride + 1
        return n, channels_out, height_out, width_out  # 返回输出形状


deform_conv = _DeformConv.apply  # 创建变形卷积函数
modulated_deform_conv = _ModulatedDeformConv.apply  # 创建调制变形卷积函数


class DeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        """
        Deformable convolution from :paper:`deformconv`.
        来自论文:paper:`deformconv`的可变形卷积。

        Arguments are similar to :class:`Conv2D`. Extra arguments:
        参数与:class:`Conv2D`类似。额外的参数：

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            # 可变形分组数：用于可变形卷积的分组数量。
            norm (nn.Module, optional): a normalization layer
            # 归一化层：可选的归一化层。
            activation (callable(Tensor) -> Tensor): a callable activation function
            # 激活函数：可调用的激活函数。
        """
        super(DeformConv, self).__init__()

        assert not bias  # 断言不使用偏置项
        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )  # 确保输入通道能被分组数整除
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)
        # 确保输出通道能被分组数整除

        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.kernel_size = _pair(kernel_size)  # 卷积核大小，转换为(height, width)格式
        self.stride = _pair(stride)  # 步长，转换为(height_stride, width_stride)格式
        self.padding = _pair(padding)  # 填充，转换为(height_padding, width_padding)格式
        self.dilation = _pair(dilation)  # 空洞率，转换为(height_dilation, width_dilation)格式
        self.groups = groups  # 卷积分组数
        self.deformable_groups = deformable_groups  # 可变形分组数
        self.norm = norm  # 归一化层
        self.activation = activation  # 激活函数

        # 创建权重参数
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
        )
        self.bias = None  # 不使用偏置

        # 使用kaiming初始化权重
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x, offset):
        """
        前向传播函数
        
        Args:
            x: 输入特征图
            offset: 偏移量场
        """
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            # 当输入为空时，我们希望返回具有"正确"形状的空张量，
            # 这样，后续操作在检查张量形状时不会出错。
            # 下面计算输出张量的高度和宽度
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]  # 计算输出特征图的空间尺寸
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape  # 组合批大小、通道数和空间尺寸
            return _NewEmptyTensorOp.apply(x, output_shape)  # 返回正确形状的空张量

        # 应用可变形卷积操作
        x = deform_conv(
            x,
            offset,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        if self.norm is not None:
            x = self.norm(x)  # 应用归一化
        if self.activation is not None:
            x = self.activation(x)  # 应用激活函数
        return x

    def extra_repr(self):
        """
        返回模块的额外表示信息，用于打印模块信息
        """
        tmpstr = "in_channels=" + str(self.in_channels)  # 输入通道数
        tmpstr += ", out_channels=" + str(self.out_channels)  # 输出通道数
        tmpstr += ", kernel_size=" + str(self.kernel_size)  # 卷积核大小
        tmpstr += ", stride=" + str(self.stride)  # 步长
        tmpstr += ", padding=" + str(self.padding)  # 填充
        tmpstr += ", dilation=" + str(self.dilation)  # 空洞率
        tmpstr += ", groups=" + str(self.groups)  # 卷积分组数
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)  # 可变形分组数
        tmpstr += ", bias=False"  # 不使用偏置
        return tmpstr


class ModulatedDeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        """
        Modulated deformable convolution from :paper:`deformconv2`.
        来自论文:paper:`deformconv2`的调制可变形卷积。

        Arguments are similar to :class:`Conv2D`. Extra arguments:
        参数与:class:`Conv2D`类似。额外的参数：

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            # 可变形分组数：用于可变形卷积的分组数量。
            norm (nn.Module, optional): a normalization layer
            # 归一化层：可选的归一化层。
            activation (callable(Tensor) -> Tensor): a callable activation function
            # 激活函数：可调用的激活函数。
        """
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.kernel_size = _pair(kernel_size)  # 卷积核大小，转换为(height, width)格式
        self.stride = stride  # 步长
        self.padding = padding  # 填充
        self.dilation = dilation  # 空洞率
        self.groups = groups  # 卷积分组数
        self.deformable_groups = deformable_groups  # 可变形分组数
        self.with_bias = bias  # 是否使用偏置
        self.norm = norm  # 归一化层
        self.activation = activation  # 激活函数

        # 创建权重参数
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))  # 创建偏置参数
        else:
            self.bias = None  # 不使用偏置

        # 初始化权重和偏置
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")  # 使用kaiming初始化权重
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)  # 将偏置初始化为0

    def forward(self, x, offset, mask):
        """
        前向传播函数
        
        Args:
            x: 输入特征图
            offset: 偏移量场
            mask: 调制掩码
        """
        if x.numel() == 0:
            # 当输入为空时，返回正确形状的空张量
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]  # 计算输出特征图的空间尺寸
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape  # 组合批大小、通道数和空间尺寸
            return _NewEmptyTensorOp.apply(x, output_shape)  # 返回正确形状的空张量

        # 应用调制可变形卷积操作
        x = modulated_deform_conv(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        if self.norm is not None:
            x = self.norm(x)  # 应用归一化
        if self.activation is not None:
            x = self.activation(x)  # 应用激活函数
        return x

    def extra_repr(self):
        """
        返回模块的额外表示信息，用于打印模块信息
        """
        tmpstr = "in_channels=" + str(self.in_channels)  # 输入通道数
        tmpstr += ", out_channels=" + str(self.out_channels)  # 输出通道数
        tmpstr += ", kernel_size=" + str(self.kernel_size)  # 卷积核大小
        tmpstr += ", stride=" + str(self.stride)  # 步长
        tmpstr += ", padding=" + str(self.padding)  # 填充
        tmpstr += ", dilation=" + str(self.dilation)  # 空洞率
        tmpstr += ", groups=" + str(self.groups)  # 卷积分组数
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)  # 可变形分组数
        tmpstr += ", bias=" + str(self.with_bias)  # 是否使用偏置
        return tmpstr
