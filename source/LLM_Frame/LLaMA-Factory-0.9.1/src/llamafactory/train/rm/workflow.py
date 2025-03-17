# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

# Import data processing modules
# 导入数据处理相关模块
from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss  # Visualization of loss curves
from ...model import load_model, load_tokenizer  # Model loading utilities
from ..callbacks import fix_valuehead_checkpoint  # Fix value head checkpoint
from ..trainer_utils import create_modelcard_and_push  # Model card creation
from .metric import ComputeAccuracy  # Accuracy metric
from .trainer import PairwiseTrainer  # Pairwise comparison trainer


if TYPE_CHECKING:
    # Type hints only for type checkers
    # 类型提示仅在类型检查时导入
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_rm(
    model_args: "ModelArguments",          # Model config (path, name, etc)
    data_args: "DataArguments",            # Dataset config (path, preprocessing)
    training_args: "Seq2SeqTrainingArguments",  # Training settings (lr, batch size)
    finetuning_args: "FinetuningArguments",# Finetuning config (LoRA, QLoRA)
    callbacks: Optional[List["TrainerCallback"]] = None,  # Training callbacks
):
    # Load tokenizer and related modules
    # 加载分词器及相关模块
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # Get template and fix tokenizer (add special tokens)
    # 获取模板并修复分词器（添加特殊token）
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Load dataset module (train/valid/test)
    # 加载数据集模块（包含训练/验证/测试集）
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    
    # Load model with value head for reward modeling
    # 加载模型（添加价值头用于奖励模型训练）
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    
    # Initialize pairwise data collator with padding
    # 初始化成对数据填充整理器（处理不同长度样本）
    data_collator = PairwiseDataCollatorWithPadding(template=template, pad_to_multiple_of=8, **tokenizer_module)

    # Update arguments
    training_args.remove_unused_columns = False  # Important for multimodal and pairwise dataset // 重要：保留所有列

    # Initialize pairwise trainer
    # 初始化成对训练器
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,  # Batch processing
        callbacks=callbacks,  # Training callbacks
        compute_metrics=ComputeAccuracy(),  # Accuracy metric
        **dataset_module,  # Inject datasets
        **tokenizer_module,  # Inject tokenizer
    )

    # Training
    if training_args.do_train:
        # Execute training process
        # 执行训练流程
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Save final model
        
        # Fix value head checkpoint format
        # 修复价值头检查点格式（适配HuggingFace格式）
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

        # Log and save metrics
        # 记录和保存训练指标
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()  # Save training state
        
        # Plot loss curves in main process
        # 主进程绘制损失曲线
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Evaluation
    if training_args.do_eval:
        # Run evaluation on validation set
        # 执行验证集评估
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        # Run prediction on test set
        # 执行测试集预测
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)  # Save predictions

    # Create model card and push to Hub
    # 创建模型卡片并推送到HuggingFace Hub
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
