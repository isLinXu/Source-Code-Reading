# Copyright (c) Facebook, Inc. and its affiliates.
# 版权所有 (c) Facebook, Inc. 及其附属公司。
import torch
import torch.distributed as dist
from fvcore.nn.distributed import differentiable_all_reduce
from torch import nn
from torch.nn import functional as F

from detectron2.utils import comm, env

from .wrappers import BatchNorm2d


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    BatchNorm2d，其中批统计量和仿射参数是固定的。

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    它包含名为"weight"、"bias"、"running_mean"、"running_var"的不可训练缓冲区，
    初始化为执行恒等变换。

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    从Caffe2加载的预训练骨干网络模型只包含"weight"和"bias"，
    这些是从BN的原始四个参数计算得出的。
    仿射变换`x * weight + bias`将执行等效计算：
    `(x - running_mean) / sqrt(running_var) * weight + bias`。
    当从Caffe2加载骨干网络模型时，"running_mean"和"running_var"
    将保持不变作为恒等变换。

    Other pre-trained backbone models may contain all 4 parameters.
    其他预训练的骨干网络模型可能包含所有4个参数。

    The forward is implemented by `F.batch_norm(..., training=False)`.
    前向传播通过`F.batch_norm(..., training=False)`实现。
    """

    _version = 3  # 版本号，用于兼容性检查

    def __init__(self, num_features, eps=1e-5):
        """
        初始化FrozenBatchNorm2d
        
        Args:
            num_features: 特征数量（通道数）
            eps: 用于数值稳定性的小值
        """
        super().__init__()
        self.num_features = num_features  # 特征数量
        self.eps = eps  # 用于数值稳定性的小值
        # 注册不可训练的缓冲区
        self.register_buffer("weight", torch.ones(num_features))  # 权重初始化为1
        self.register_buffer("bias", torch.zeros(num_features))   # 偏置初始化为0
        self.register_buffer("running_mean", torch.zeros(num_features))  # 均值初始化为0
        self.register_buffer("running_var", torch.ones(num_features) - eps)  # 方差初始化为1-eps

    def forward(self, x):
        """
        前向传播函数
        """
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            # 当需要梯度时，F.batch_norm会使用额外的内存，
            # 因为其反向操作也计算weight/bias的梯度。
            scale = self.weight * (self.running_var + self.eps).rsqrt()  # 计算缩放因子
            bias = self.bias - self.running_mean * scale  # 计算偏置
            scale = scale.reshape(1, -1, 1, 1)  # 重塑尺寸以便广播
            bias = bias.reshape(1, -1, 1, 1)    # 重塑尺寸以便广播
            out_dtype = x.dtype  # may be half
                                 # 可能是半精度
            return x * scale.to(out_dtype) + bias.to(out_dtype)  # 应用缩放和偏置，转换为输入的数据类型
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            # 当不需要梯度时，F.batch_norm是一个单一的融合操作，
            # 提供更多的优化机会。
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,  # 始终使用评估模式
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        从状态字典加载参数，处理版本兼容性问题
        """
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            # 早期版本中没有running_mean/var
            # 这将消除警告
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        """
        返回模块的字符串表示
        """
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        将模块中的所有BatchNorm/SyncBatchNorm转换为FrozenBatchNorm。

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
            如果模块是BatchNorm/SyncBatchNorm，则返回一个新模块。
            否则，原地转换模块并返回它。

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        类似于https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        中的convert_sync_batchnorm
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)  # 定义要转换的BN类型
        res = module  # 结果模块，默认为输入模块
        if isinstance(module, bn_module):
            # 如果当前模块是BN模块，则转换为FrozenBatchNorm2d
            res = cls(module.num_features)  # 创建新的FrozenBatchNorm2d实例
            if module.affine:
                # 如果原模块有可学习参数，则复制参数
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            # 复制运行统计量
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            # 如果当前模块不是BN模块，则递归处理其子模块
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)  # 替换转换后的子模块
        return res


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
            norm (字符串或可调用对象)：可以是BN、SyncBN、FrozenBN、GN之一；
            或者是一个接受通道数并返回归一化层作为nn.Module的可调用对象。

    Returns:
        nn.Module or None: the normalization layer
        nn.Module或None：归一化层
    """
    if norm is None:
        return None  # 如果norm为None，则不使用归一化
    if isinstance(norm, str):
        if len(norm) == 0:
            return None  # 如果norm是空字符串，则不使用归一化
        # 根据字符串选择对应的归一化类型
        norm = {
            "BN": BatchNorm2d,  # 批归一化
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            # 在https://github.com/pytorch/pytorch/pull/36382中修复
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,  # 根据PyTorch版本选择同步批归一化实现
            "FrozenBN": FrozenBatchNorm2d,  # 冻结的批归一化
            "GN": lambda channels: nn.GroupNorm(32, channels),  # 组归一化，固定32组
            # for debugging:
            # 用于调试：
            "nnSyncBN": nn.SyncBatchNorm,  # PyTorch原生同步批归一化
            "naiveSyncBN": NaiveSyncBatchNorm,  # 简单实现的同步批归一化
            # expose stats_mode N as an option to caller, required for zero-len inputs
            # 将stats_mode N作为选项暴露给调用者，零长度输入时需要
            "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),  # 使用N模式的简单同步批归一化
        }[norm]
    return norm(out_channels)  # 返回指定通道数的归一化层


class NaiveSyncBatchNorm(BatchNorm2d):
    """
    In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
    when the batch size on each worker is different.
    在PyTorch<=1.5中，当每个工作进程的批大小不同时，
    ``nn.SyncBatchNorm``的梯度计算不正确。
    (e.g., when scale augmentation is used, or when it is applied to mask head).
    （例如，当使用尺度增强时，或当应用于掩码头时）。

    This is a slower but correct alternative to `nn.SyncBatchNorm`.
    这是`nn.SyncBatchNorm`的一个较慢但正确的替代方案。

    Note:
        There isn't a single definition of Sync BatchNorm.
        同步批归一化没有单一的定义。

        When ``stats_mode==""``, this module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.
        当``stats_mode=="``时，此模块通过使用每个工作进程的统计数据（具有相同权重）来计算总体统计数据。
        只有当所有工作进程具有相同的(N, H, W)时，结果才是所有样本的真实统计数据（就像它们都在一个工作进程上一样）。
        此模式不支持批大小为零的输入。

        When ``stats_mode=="N"``, this module computes overall statistics by weighting
        the statistics of each worker by their ``N``. The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (H, W). It is slower than ``stats_mode==""``.
        当``stats_mode=="N"``时，此模块通过根据每个工作进程的``N``加权其统计数据来计算总体统计数据。
        只有当所有工作进程具有相同的(H, W)时，结果才是所有样本的真实统计数据（就像它们都在一个工作进程上一样）。
        它比``stats_mode==""``慢。

        Even though the result of this module may not be the true statistics of all samples,
        it may still be reasonable because it might be preferrable to assign equal weights
        to all workers, regardless of their (H, W) dimension, instead of putting larger weight
        on larger images. From preliminary experiments, little difference is found between such
        a simplified implementation and an accurate computation of overall mean & variance.
        尽管此模块的结果可能不是所有样本的真实统计数据，但它可能仍然是合理的，
        因为无论工作进程的(H, W)维度如何，可能更倾向于为所有工作进程分配相等的权重，
        而不是对更大的图像赋予更大的权重。从初步实验来看，这种简化实现与总体均值和方差的精确计算之间几乎没有差异。
    """

    def __init__(self, *args, stats_mode="", **kwargs):
        """
        初始化NaiveSyncBatchNorm
        
        Args:
            stats_mode: 统计模式，可以是""或"N"
            以及BatchNorm2d的其他参数
        """
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]  # 确保stats_mode是有效值
        self._stats_mode = stats_mode  # 设置统计模式

    def forward(self, input):
        """
        前向传播函数
        """
        if comm.get_world_size() == 1 or not self.training:
            # 如果是单进程或者不在训练模式，则使用普通BN
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]  # 获取批大小和通道数

        half_input = input.dtype == torch.float16  # 检查是否为半精度浮点数
        if half_input:
            # fp16 does not have good enough numerics for the reduction here
            # fp16的数值精度不足以进行这里的归约操作
            input = input.float()  # 转换为单精度浮点数
            
        # 计算每个通道的均值和均方值
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        if self._stats_mode == "":
            # 在""模式下，每个工作进程的权重相等
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = torch.cat([mean, meansqr], dim=0)  # 连接均值和均方值
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())  # 全局归约并平均
            mean, meansqr = torch.split(vec, C)  # 分离均值和均方值
            momentum = self.momentum  # 使用原始动量
        else:
            # 在"N"模式下，根据批大小加权
            if B == 0:
                # 处理批大小为0的情况
                vec = torch.zeros([2 * C + 1], device=mean.device, dtype=mean.dtype)
                vec = vec + input.sum()  # make sure there is gradient w.r.t input
                                        # 确保有关于输入的梯度
            else:
                # 连接均值、均方值和批大小计数器（为1）
                vec = torch.cat(
                    [mean, meansqr, torch.ones([1], device=mean.device, dtype=mean.dtype)], dim=0
                )
            vec = differentiable_all_reduce(vec * B)  # 全局归约，考虑批大小

            total_batch = vec[-1].detach()  # 获取总批大小
            momentum = total_batch.clamp(max=1) * self.momentum  # no update if total_batch is 0
                                                                # 如果总批大小为0，则不更新
            mean, meansqr, _ = torch.split(vec / total_batch.clamp(min=1), C)  # avoid div-by-zero
                                                                              # 避免除以零

        # 计算方差、标准差倒数、缩放和偏置
        var = meansqr - mean * mean  # 计算方差
        invstd = torch.rsqrt(var + self.eps)  # 计算标准差的倒数
        scale = self.weight * invstd  # 计算缩放因子
        bias = self.bias - mean * scale  # 计算偏置
        scale = scale.reshape(1, -1, 1, 1)  # 重塑尺寸以便广播
        bias = bias.reshape(1, -1, 1, 1)    # 重塑尺寸以便广播

        # 更新运行统计量
        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)
        
        # 应用缩放和偏置
        ret = input * scale + bias
        if half_input:
            ret = ret.half()  # 如果输入是半精度，则将结果转回半精度
        return ret


class CycleBatchNormList(nn.ModuleList):
    """
    Implement domain-specific BatchNorm by cycling.
    通过循环实现特定领域的BatchNorm。

    When a BatchNorm layer is used for multiple input domains or input
    features, it might need to maintain a separate test-time statistics
    for each domain. See Sec 5.2 in :paper:`rethinking-batchnorm`.
    当一个BatchNorm层用于多个输入域或输入特征时，它可能需要为每个域维护单独的测试时统计数据。
    参见:paper:`rethinking-batchnorm`中的第5.2节。

    This module implements it by using N separate BN layers
    and it cycles through them every time a forward() is called.
    该模块通过使用N个单独的BN层实现，并且在每次调用forward()时都会循环遍历它们。

    NOTE: The caller of this module MUST guarantee to always call
    this module by multiple of N times. Otherwise its test-time statistics
    will be incorrect.
    注意：此模块的调用者必须保证始终按N的倍数调用此模块。
    否则，其测试时统计数据将不正确。
    """

    def __init__(self, length: int, bn_class=nn.BatchNorm2d, **kwargs):
        """
        Args:
            length: number of BatchNorm layers to cycle.
                   要循环的BatchNorm层的数量。
            bn_class: the BatchNorm class to use
                     要使用的BatchNorm类
            kwargs: arguments of the BatchNorm class, such as num_features.
                   BatchNorm类的参数，如num_features。
        """
        self._affine = kwargs.pop("affine", True)  # 提取并保存affine参数，默认为True
        super().__init__([bn_class(**kwargs, affine=False) for k in range(length)])  # 创建不带仿射变换的BN层列表
        if self._affine:
            # shared affine, domain-specific BN
            # 共享的仿射变换，特定领域的BN
            channels = self[0].num_features  # 获取特征通道数
            self.weight = nn.Parameter(torch.ones(channels))  # 创建共享权重参数
            self.bias = nn.Parameter(torch.zeros(channels))   # 创建共享偏置参数
        self._pos = 0  # 当前使用的BN层索引

    def forward(self, x):
        """
        前向传播函数
        """
        ret = self[self._pos](x)  # 使用当前BN层处理输入
        self._pos = (self._pos + 1) % len(self)  # 更新索引，循环使用

        if self._affine:
            # 如果启用了仿射变换，应用共享的权重和偏置
            w = self.weight.reshape(1, -1, 1, 1)  # 重塑权重以便广播
            b = self.bias.reshape(1, -1, 1, 1)    # 重塑偏置以便广播
            return ret * w + b  # 应用仿射变换
        else:
            return ret  # 直接返回结果

    def extra_repr(self):
        """
        返回模块的额外表示信息
        """
        return f"affine={self._affine}"
