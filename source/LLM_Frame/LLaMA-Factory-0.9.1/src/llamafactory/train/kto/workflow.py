# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/kto.py
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

from typing import TYPE_CHECKING, List, Optional

# 导入数据处理模块
# Import data processing modules
from ...data import KTODataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX  # 特殊填充索引
from ...extras.ploting import plot_loss  # 损失曲线可视化
from ...hparams import ModelArguments  # 模型超参数
from ...model import load_model, load_tokenizer  # 模型加载工具
from ..trainer_utils import create_modelcard_and_push, create_ref_model  # 模型工具
from .trainer import CustomKTOTrainer  # 自定义KTO训练器


if TYPE_CHECKING:
    # 类型提示仅在类型检查时导入
    # Type hints for type checking only
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments


def run_kto(
    model_args: "ModelArguments",          # 模型配置（路径、名称等）
    data_args: "DataArguments",            # 数据配置（路径、预处理）
    training_args: "Seq2SeqTrainingArguments",  # 训练参数（学习率、批次大小）
    finetuning_args: "FinetuningArguments",# 微调参数（KTO超参数）
    callbacks: Optional[List["TrainerCallback"]] = None,  # 训练回调
):
    # 加载分词器及相关模块
    # Load tokenizer and related modules
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # 获取对话模板并修复分词器
    # Get conversation template and fix tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # 加载KTO专用数据集
    # Load KTO-specific dataset
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="kto", **tokenizer_module)
    # 加载基础模型
    # Load base model
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # 初始化KTO数据整理器
    # Initialize KTO data collator
    data_collator = KTODataCollatorWithPadding(
        template=template,
        pad_to_multiple_of=8,  # 填充到8的倍数（优化硬件性能）
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,  # 标签填充处理
        **tokenizer_module,
    )

    # 创建参考模型
    # Create reference model
    if finetuning_args.ref_model is None and (not training_args.do_train):  # 评估时使用自身作为参考模型
        ref_model = model
    else:
        ref_model = create_ref_model(model_args, finetuning_args)  # 显式创建参考模型

    # 更新训练参数
    # Update training arguments
    training_args.remove_unused_columns = False  # 保留所有列（多模态数据需要）

    # 初始化KTO训练器
    # Initialize KTO trainer
    trainer = CustomKTOTrainer(
        model=model,            # 待训练模型
        ref_model=ref_model,    # 参考模型
        args=training_args,     # 训练参数
        finetuning_args=finetuning_args,  # 微调参数
        data_collator=data_collator,  # 数据整理器
        callbacks=callbacks,    # 回调函数
        **dataset_module,       # 注入数据集
        **tokenizer_module,     # 注入分词器
    )

    # 训练流程
    # Training process
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)  # 开始/恢复训练
        trainer.save_model()  # 保存最终模型
        trainer.log_metrics("train", train_result.metrics)  # 记录训练指标
        trainer.save_metrics("train", train_result.metrics)  # 保存训练指标
        trainer.save_state()  # 保存训练状态
        # 主进程绘制损失曲线
        # Plot loss curves in main process
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/chosen"])

    # 评估流程
    # Evaluation process
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")  # 执行评估
        # 当参考模型是自身时无法计算奖励指标
        # Cannot compute rewards when reference model is self
        if id(model) == id(ref_model):
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)  # 记录评估指标
        trainer.save_metrics("eval", metrics)  # 保存评估指标

    # 创建并推送模型卡片
    # Create and push model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
