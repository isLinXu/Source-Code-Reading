# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType  # 从 types 导入 MethodType
from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型

from transformers import Trainer  # 从 transformers 导入 Trainer 类
from typing_extensions import override  # 从 typing_extensions 导入 override 装饰器

from ...extras.packages import is_transformers_version_equal_to_4_46  # 从 extras.packages 导入检查 transformers 版本的函数
from ..callbacks import PissaConvertCallback, SaveProcessorCallback  # 从 callbacks 导入 PissaConvertCallback 和 SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler  # 从 trainer_utils 导入创建自定义优化器和调度器的函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    import torch  # 导入 torch 库
    from transformers import ProcessorMixin  # 从 transformers 导入 ProcessorMixin

    from ...hparams import FinetuningArguments  # 从 hparams 导入 FinetuningArguments


class CustomTrainer(Trainer):  # 定义自定义训练器类，继承自 Trainer
    r"""
    Inherits Trainer for custom optimizer.  # 继承 Trainer 以实现自定义优化器。
    """

    def __init__(  # 初始化方法
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.finetuning_args = finetuning_args  # 保存微调参数

        if processor is not None:  # 如果处理器不为 None
            self.add_callback(SaveProcessorCallback(processor))  # 添加保存处理器的回调

        if finetuning_args.pissa_convert:  # 如果启用 Pissa 转换
            self.add_callback(PissaConvertCallback)  # 添加 Pissa 转换的回调

        if finetuning_args.use_badam:  # 如果使用 BAdam 优化器
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore  # 导入 BAdamCallback 和旧版本的梯度裁剪函数

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)  # 将旧版本的梯度裁剪方法绑定到加速器
            self.add_callback(BAdamCallback)  # 添加 BAdam 回调

    @override  # 重写父类方法
    def create_optimizer(self) -> "torch.optim.Optimizer":  # 创建优化器的方法
        if self.optimizer is None:  # 如果优化器尚未创建
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)  # 创建自定义优化器
        return super().create_optimizer()  # 调用父类的方法创建优化器

    @override  # 重写父类方法
    def create_scheduler(  # 创建学习率调度器的方法
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)  # 创建自定义调度器
        return super().create_scheduler(num_training_steps, optimizer)  # 调用父类的方法创建调度器

    @override  # 重写父类方法
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # 计算损失的方法
        r"""
        Fixes the loss value for transformers 4.46.0.  # 修复 transformers 4.46.0 的损失值。
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs, **kwargs)  # 调用父类的方法计算损失
        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):  # 如果是 transformers 4.46.0 版本且模型不接受损失参数
            # other model should not scale the loss  # 其他模型不应该缩放损失
            if return_outputs:  # 如果需要返回输出
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])  # 返回缩放后的损失
            else:
                return loss / self.args.gradient_accumulation_steps  # 返回缩放后的损失

        return loss  # 返回原始损失