# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json  # 导入 JSON 模块
import os  # 导入操作系统模块
from types import MethodType  # 从 types 导入 MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union  # 导入类型检查、任意类型、字典、列表、可选类型、元组和联合类型

import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 库
from transformers import Seq2SeqTrainer  # 从 transformers 导入 Seq2SeqTrainer
from typing_extensions import override  # 从 typing_extensions 导入 override

from ...extras import logging  # 从 extras 导入日志模块
from ...extras.constants import IGNORE_INDEX  # 从 extras.constants 导入 IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46  # 从 extras.packages 导入版本检查函数
from ..callbacks import PissaConvertCallback, SaveProcessorCallback  # 导入回调函数
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler  # 导入自定义优化器和调度器创建函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from torch.utils.data import Dataset  # 导入 Dataset 类型
    from transformers import ProcessorMixin  # 导入 ProcessorMixin 类型
    from transformers.trainer import PredictionOutput  # 导入 PredictionOutput 类型

    from ...hparams import FinetuningArguments  # 导入微调参数类型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class CustomSeq2SeqTrainer(Seq2SeqTrainer):  # 定义自定义 Seq2SeqTrainer 类
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.  # 继承 Seq2SeqTrainer 以计算生成指标，如 BLEU 和 ROUGE。
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

        if finetuning_args.use_badam:  # 如果使用 BAdam
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore  # 导入 BAdam 回调和旧版本的梯度裁剪函数

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)  # 设置梯度裁剪方法
            self.add_callback(BAdamCallback)  # 添加 BAdam 的回调

    @override  # 重写父类方法
    def create_optimizer(self) -> "torch.optim.Optimizer":  # 创建优化器
        if self.optimizer is None:  # 如果优化器为 None
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)  # 创建自定义优化器
        return super().create_optimizer()  # 调用父类的创建优化器方法

    @override  # 重写父类方法
    def create_scheduler(  # 创建学习率调度器
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)  # 创建自定义调度器
        return super().create_scheduler(num_training_steps, optimizer)  # 调用父类的创建调度器方法

    @override  # 重写父类方法
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # 计算损失
        r"""
        Fixes the loss value for transformers 4.46.0.  # 修复 transformers 4.46.0 的损失值。
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs, **kwargs)  # 调用父类计算损失的方法
        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):  # 如果是 4.46 版本且模型不接受损失参数
            # other model should not scale the loss
            if return_outputs:  # 如果返回输出
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])  # 返回缩放后的损失
            else:
                return loss / self.args.gradient_accumulation_steps  # 返回缩放后的损失

        return loss  # 返回损失

    @override  # 重写父类方法
    def prediction_step(  # 预测步骤
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.  # 移除生成令牌中的提示部分。

        Subclass and override to inject custom behavior.  # 子类并重写以注入自定义行为。
        """
        labels = inputs["labels"] if "labels" in inputs else None  # 获取标签
        if self.args.predict_with_generate:  # 如果使用生成预测
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."  # 确保张量是左填充的
            labels = labels.detach().clone() if labels is not None else None  # 备份标签
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)  # 获取提示和标签的长度
            if prompt_len > label_len:  # 如果提示长度大于标签长度
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])  # 将标签填充到目标长度
            if label_len > prompt_len:  # 如果标签长度大于提示长度
                inputs["labels"] = inputs["labels"][:, :prompt_len]  # 截断标签而不是填充输入

        loss, generated_tokens, _ = super().prediction_step(  # 调用父类的预测步骤方法
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:  # 如果生成的令牌不为 None 且使用生成预测
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id  # 将生成的令牌的前部分设置为填充标记
            generated_tokens = generated_tokens.contiguous()  # 确保张量在连续内存中

        return loss, generated_tokens, labels  # 返回损失、生成的令牌和标签

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":  # 填充张量到目标长度
        r"""
        Pads the tensor to the same length as the target tensor.  # 将张量填充到与目标张量相同的长度。
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."  # 确保填充标记存在
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)  # 创建填充张量
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding  # 采用左填充
        return padded_tensor.contiguous()  # 返回连续内存中的张量

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:  # 保存预测结果
        r"""
        Saves model predictions to `output_dir`.  # 将模型预测结果保存到输出目录。

        A custom behavior that not contained in Seq2SeqTrainer.  # 这是 Seq2SeqTrainer 中未包含的自定义行为。
        """
        if not self.is_world_process_zero():  # 如果不是主进程
            return  # 直接返回

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")  # 设置输出文件路径
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")  # 记录保存预测结果的信息

        labels = np.where(  # 获取标签
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(  # 获取预测结果
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):  # 遍历每个预测结果
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]  # 获取非填充标记的位置
            if len(pad_len):  # 如果存在非填充标记
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)  # 将填充标记移动到最后

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)  # 解码输入
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)  # 解码标签
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)  # 解码预测结果

        with open(output_prediction_file, "w", encoding="utf-8") as writer:  # 打开输出文件进行写入
            res: List[str] = []  # 初始化结果列表
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):  # 遍历解码后的输入、标签和预测结果
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))  # 将结果添加到列表中

            writer.write("\n".join(res))  # 将结果写入文件