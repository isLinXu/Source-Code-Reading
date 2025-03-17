# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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

# Import base libraries | 导入基础库
import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer  # TRL's DPO trainer base class | TRL的DPO训练器基类
from trl.trainer import disable_dropout_in_model  # Disable dropout | 禁用dropout
from typing_extensions import override  # Override decorator | 重写装饰器

# Project custom modules | 项目自定义模块
from ...extras.constants import IGNORE_INDEX  # Special padding index | 特殊填充索引
from ...extras.packages import is_transformers_version_equal_to_4_46  # Version check | Transformers版本检查
from ..callbacks import PissaConvertCallback, SaveProcessorCallback  # Callbacks | 回调函数
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps  # Custom utilities | 自定义工具函数


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin  # Type hints | 类型提示
    from ...hparams import FinetuningArguments  # Finetuning parameters | 微调参数配置


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],  # Model to train | 待训练模型
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],  # Reference model | 参考模型
        finetuning_args: "FinetuningArguments",  # Finetuning parameters | 微调参数
        processor: Optional["ProcessorMixin"],  # Processor for multimodal data | 多模态数据处理器
        disable_dropout: bool = True,  # Whether to disable dropout | 是否禁用dropout
        **kwargs,
    ):
        # Disable dropout in models during initialization | 初始化时禁用模型中的dropout层
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        # Configure training parameters | 配置训练参数
        self.finetuning_args = finetuning_args  # 微调参数实例
        self.f_divergence_type = "reverse_kl"  # Divergence type for DPO | DPO散度类型（反向KL散度）
        self.reference_free = False  # Whether to use reference-free mode | 是否无参考模型模式
        self.use_dpo_data_collator = True  # Hack to avoid warnings | 避免数据整理器警告的临时方案
        self.generate_during_eval = False  # Disable generation during evaluation | 评估时禁用生成
        self.label_pad_token_id = IGNORE_INDEX  # Padding token ID for labels | 标签填充索引（忽略损失计算）
        self.padding_value = 0  # Padding value | 输入填充值
        self.is_encoder_decoder = model.config.is_encoder_decoder  # Encoder-decoder flag | 编码器-解码器结构标记
        self.precompute_ref_log_probs = False  # Precompute reference log probs | 是否预计算参考模型log概率
        self._peft_has_been_casted_to_bf16 = False  # PEFT model casting flag | PEFT模型bfloat16转换标记

        self.ref_model = ref_model  # Reference model | 参考模型实例
        self._stored_metrics = defaultdict(lambda: defaultdict(list))  # Metric storage | 指标存储字典

        # DPO hyperparameters | DPO超参数
        self.beta = finetuning_args.pref_beta  # Temperature parameter β | 温度参数β
        self.loss_type = finetuning_args.pref_loss  # Loss type (dpo/ipo/orpo/simpo) | 损失类型
        self.ftx_gamma = finetuning_args.pref_ftx  # Supervised fine-tuning coefficient γ | 监督微调混合系数
        self.label_smoothing = finetuning_args.dpo_label_smoothing  # Label smoothing factor | 标签平滑系数
        self.simpo_gamma = finetuning_args.simpo_gamma  # Gamma for SimPO | SimPO的gamma参数

        Trainer.__init__(self, model=model, **kwargs)  # Initialize parent Trainer | 初始化父类Trainer
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # Suppress reference model GC warnings | 忽略参考模型的垃圾回收警告

        # Prepare reference model for distributed training | 分布式训练准备参考模型
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)):
                    # Quantized models are already on correct device | 量化模型已在正确设备上
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        # Add processor saving callback | 添加处理器保存回调
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # Add PiSSA matrix conversion callback | 添加PiSSA矩阵转换回调
        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        # Enable BAdam optimization | 启用BAdam优化器
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            # Monkey-patch gradient clipping | 猴子补丁方式修改梯度裁剪
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        """Create custom optimizer | 创建自定义优化器"""
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        """Create custom scheduler | 创建自定义学习率调度器"""
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        """Compute odds ratio loss for ORPO | 计算ORPO的胜率损失"""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        return -F.logsigmoid(self.beta * log_odds)

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        """Compute SimPO loss | 计算SimPO损失"""
        return -F.logsigmoid(self.beta * (chosen_logps - rejected_logps - self.simpo_gamma))

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Combined forward pass for chosen and rejected samples | 合并处理正负样本的前向传播"""
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # Avoid gradient flow | 阻断梯度流向参考模型

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        
        # Length normalization for specific loss types | 特定损失类型的长度归一化
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        # Split results for chosen and rejected samples | 分割正负样本结果
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Compute reference model's log probabilities | 计算参考模型的log概率"""
        if not self.finetuning_args.use_ref_model:
            return None, None

        ref_model = self.ref_model if self.ref_model else model
        ref_context = self.accelerator.unwrap_model(model).disable_adapter() if self.ref_model is None else nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """Compute DPO loss and metrics | 计算DPO损失和相关指标"""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss  # Add supervised loss | 添加监督损失项

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()  # 正样本平均奖励
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()  # 负样本平均奖励
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()  # 奖励准确率
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()  # 奖励边际值
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()  # 正样本策略log概率
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()  # 负样本策略log概率
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()  # 正样本策略logits
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()  # 负样本策略logits
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()  # 监督微调损失
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()  # 胜率比损失

        return losses.mean(), metrics

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fix loss calculation for transformers 4.46.0 | 修复transformers 4.46.0的损失计算问题"""
        loss = super().compute_loss(model, inputs, return_outputs)
        if is_transformers_version_equal_to_4_46() and kwargs.pop("num_items_in_batch", False):
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps
        return loss

    @override
    def log(self, logs: Dict[str, float]) -> None:
        """Custom logging logic | 自定义日志记录逻辑"""
        # logs either has "loss" or "eval_loss" | 日志包含训练或评估损失
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs | 将平均后的指标加入日志
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce | 填充指标列表以便分布式归约
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()  # 分布式归约指标
        for key, metric in zip(key_list, metric_list):  # add remaining items | 添加剩余指标
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs)
