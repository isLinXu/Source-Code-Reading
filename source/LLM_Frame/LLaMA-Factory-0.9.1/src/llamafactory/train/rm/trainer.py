# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
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

import json  # 导入 JSON 库
import os  # 导入操作系统库
from types import MethodType  # 从 types 导入 MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union  # 导入类型检查、字典、列表、可选类型、元组和联合类型

import torch  # 导入 PyTorch 库
from transformers import Trainer  # 从 transformers 导入 Trainer 类
from typing_extensions import override  # 从 typing_extensions 导入 override 装饰器

from ...extras import logging  # 从 extras 导入日志模块
from ...extras.packages import is_transformers_version_equal_to_4_46  # 从 extras.packages 导入检查 transformers 版本的函数
from ..callbacks import FixValueHeadModelCallback, PissaConvertCallback, SaveProcessorCallback  # 从 callbacks 导入回调类
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler  # 从 trainer_utils 导入创建自定义优化器和调度器的函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import PreTrainedModel, ProcessorMixin  # 导入预训练模型和处理器混合类型
    from transformers.trainer import PredictionOutput  # 从 transformers.trainer 导入预测输出类型

    from ...hparams import FinetuningArguments  # 从 hparams 导入 FinetuningArguments


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class PairwiseTrainer(Trainer):  # 定义 PairwiseTrainer 类，继承自 Trainer
    r"""
    Inherits Trainer to compute pairwise loss.  # 继承 Trainer 以计算成对损失。
    """

    def __init__(  # 初始化方法
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.finetuning_args = finetuning_args  # 保存微调参数
        self.can_return_loss = True  # override property to return eval_loss  # 重写属性以返回 eval_loss
        self.add_callback(FixValueHeadModelCallback)  # 添加 FixValueHeadModelCallback 回调

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
    def compute_loss(  # 计算损失的方法
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.  # 计算成对损失。选择前 n 个示例，拒绝最后 n 个示例。

        Subclass and override to inject custom behavior.  # 子类并重写以注入自定义行为。

        Note that the first element will be removed from the output tuple.  # 注意，输出元组的第一个元素将被移除。
        See: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py#L3842
        """
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)  # 获取模型输出和隐藏状态
        batch_size = inputs["input_ids"].size(0) // 2  # 计算批次大小
        chosen_masks, rejected_masks = torch.split(inputs["attention_mask"], batch_size, dim=0)  # 分割注意力掩码
        chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)  # 分割奖励值
        chosen_scores = chosen_rewards.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1))  # 获取选择的分数
        rejected_scores = rejected_rewards.gather(dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1))  # 获取拒绝的分数
        chosen_scores, rejected_scores = chosen_scores.squeeze(), rejected_scores.squeeze()  # 压缩分数张量

        loss = -torch.nn.functional.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()  # 计算成对损失

        if is_transformers_version_equal_to_4_46() and kwargs.pop("num_items_in_batch", False):  # 如果是 transformers 4.46.0 版本且指定了批次中的项目数量
            loss /= self.args.gradient_accumulation_steps  # fixes the loss value for transformers 4.46.0  # 修复 transformers 4.46.0 的损失值

        if return_outputs:  # 如果需要返回输出
            return loss, (loss, chosen_scores, rejected_scores)  # 返回损失和分数
        else:
            return loss  # 返回损失

    def save_predictions(self, predict_results: "PredictionOutput") -> None:  # 保存预测结果的方法
        r"""
        Saves model predictions to `output_dir`.  # 将模型预测结果保存到 `output_dir`。

        A custom behavior that not contained in Seq2SeqTrainer.  # 自定义行为，不包含在 Seq2SeqTrainer 中。
        """
        if not self.is_world_process_zero():  # 如果不是主进程
            return  # 直接返回

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")  # 设置输出预测文件路径
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")  # 记录保存预测结果的信息
        chosen_scores, rejected_scores = predict_results.predictions  # 获取选择和拒绝的分数

        with open(output_prediction_file, "w", encoding="utf-8") as writer:  # 打开文件以写入预测结果
            res: List[str] = []  # 初始化结果列表
            for c_score, r_score in zip(chosen_scores, rejected_scores):  # 遍历选择和拒绝的分数
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))  # 将分数转换为 JSON 格式并添加到结果列表

            writer.write("\n".join(res))  # 将结果写入文件