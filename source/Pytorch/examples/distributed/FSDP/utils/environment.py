# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.  # 版权声明，Meta Platforms及其附属公司所有
# All rights reserved.  # 保留所有权利
#
# This source code is licensed under the Apache-style license found in the  # 本源代码根据Apache风格许可证进行授权
# LICENSE file in the root directory of this source tree.  # 许可证文件位于本源代码树的根目录中

# This is a simple check to confirm that your current server has full bfloat support -  # 这是一个简单的检查，确认当前服务器是否具有完整的bfloat支持
#  both GPU native support, and Network communication support.  # 包括GPU本地支持和网络通信支持

# Be warned that if you run on V100 without a check like this, you will be running without native Bfloat16  # 请注意，如果在没有此检查的情况下在V100上运行，将无法使用本地Bfloat16
# support and will find significant performance degradation (but it will not complain via an error).  # 这将导致显著的性能下降（但不会通过错误进行抱怨）
# Hence the reason for a checker!  # 这就是需要检查器的原因！

from pkg_resources import packaging  # 从pkg_resources模块导入packaging，用于版本管理
import torch  # 导入PyTorch库
import torch.cuda.nccl as nccl  # 导入NCCL库，用于GPU间的通信
import torch.distributed as dist  # 导入PyTorch的分布式模块

# global flag that confirms ampere architecture, cuda version and  # 确认安培架构、CUDA版本和
# nccl version to verify bfloat16 native support is ready  # NCCL版本以验证bfloat16本地支持是否准备好

def bfloat_support():  # 定义检查bfloat支持的函数
    return (  # 返回以下条件的布尔值
        torch.version.cuda  # 检查CUDA版本是否存在
        and torch.cuda.is_bf16_supported()  # 检查当前CUDA是否支持bfloat16
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)  # 检查CUDA版本是否大于等于11.0
        and dist.is_nccl_available()  # 检查NCCL是否可用
        and nccl.version() >= (2, 10)  # 检查NCCL版本是否大于等于2.10
    )  # 如果所有条件都满足，则返回True，否则返回False