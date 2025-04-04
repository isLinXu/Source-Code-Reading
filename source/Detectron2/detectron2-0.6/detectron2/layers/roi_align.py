# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn
from torchvision.ops import roi_align  # 从torchvision导入roi_align操作


# NOTE: torchvision's RoIAlign has a different default aligned=False
# 注意：torchvision的RoIAlign默认值aligned=False，与本实现不同
class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            # output_size (tuple): 输出特征的高度和宽度
            spatial_scale (float): scale the input boxes by this number
            # spatial_scale (float): 用此数值缩放输入框坐标
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            # sampling_ratio (int): 每个输出样本采集的输入样本数量。
            # 0表示密集采样。
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.
            # aligned (bool): 如果为False，使用Detectron中的传统实现。
            # 如果为True，则更完美地对齐结果。

        Note:
            The meaning of aligned=True:
            # aligned=True的含义：

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.
            # 给定连续坐标c，其两个相邻像素索引（在我们的像素模型中）
            # 通过floor(c - 0.5)和ceil(c - 0.5)计算。例如，
            # c=1.3的相邻像素离散索引为[0]和[1]（它们是从连续坐标0.5和1.5处的
            # 底层信号采样而来）。但原始的roi_align(aligned=False)在计算相邻像素索引时
            # 不减去0.5，因此在执行双线性插值时使用了稍微不正确对齐的像素
            # （相对于我们的像素模型）。

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.
            # 使用`aligned=True`时，
            # 我们首先适当地缩放ROI，然后在调用roi_align之前将其移动-0.5。
            # 这产生了正确的邻居像素；请参见detectron2/tests/test_roi_align.py进行验证。

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
            # 如果ROIAlign与卷积层一起使用，这种差异不会影响模型的性能。
        """
        super().__init__()
        self.output_size = output_size  # 保存输出大小
        self.spatial_scale = spatial_scale  # 保存空间缩放比例
        self.sampling_ratio = sampling_ratio  # 保存采样比率
        self.aligned = aligned  # 保存对齐标志

        from torchvision import __version__  # 导入torchvision版本

        version = tuple(int(x) for x in __version__.split(".")[:2])  # 解析torchvision版本号
        # https://github.com/pytorch/vision/pull/2438
        assert version >= (0, 7), "Require torchvision >= 0.7"  # 确保torchvision版本至少为0.7

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            # input: NCHW格式的图像特征
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
            # rois: Bx5格式的框。第一列是批次索引N，其他4列是xyxy坐标。
        """
        assert rois.dim() == 2 and rois.size(1) == 5  # 确保ROI的维度和大小正确
        if input.is_quantized:
            input = input.dequantize()  # 如果输入是量化的，则反量化
        return roi_align(
            input,
            rois.to(dtype=input.dtype),  # 将ROI转换为与输入相同的数据类型
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )  # 调用torchvision的roi_align函数

    def __repr__(self):
        # 返回模块的字符串表示，用于打印模块信息
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
