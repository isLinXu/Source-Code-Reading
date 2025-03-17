# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

# Type hints imports | 类型提示相关导入
from typing import TYPE_CHECKING, List, Optional

# Project modules | 项目自定义模块
from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer  # Data processing tools | 数据处理工具
from ...extras.constants import IGNORE_INDEX  # Special padding index | 特殊填充索引
from ...extras.misc import cal_effective_tokens  # Calculate effective tokens | 计算有效token数
from ...extras.ploting import plot_loss  # Loss visualization | 损失曲线可视化
from ...hparams import ModelArguments  # Model hyperparameters | 模型超参数
from ...model import load_model, load_tokenizer  # Model loading utils | 模型加载工具
from ..trainer_utils import create_modelcard_and_push, create_ref_model  # Training utilities | 训练工具函数
from .trainer import CustomDPOTrainer  # Custom DPO trainer | 自定义DPO训练器


if TYPE_CHECKING:
    # Type hints for type checking | 类型检查用类型提示
    from transformers import Seq2SeqTrainingArguments, TrainerCallback  # Training arguments types | 训练参数类型
    from ...hparams import DataArguments, FinetuningArguments  # Data and finetuning args | 数据参数和微调参数


def run_dpo(
    model_args: "ModelArguments",  # Model configuration | 模型配置参数
    data_args: "DataArguments",  # Data configuration | 数据配置参数
    training_args: "Seq2SeqTrainingArguments",  # Training configuration | 训练参数
    finetuning_args: "FinetuningArguments",  # Finetuning configuration | 微调参数
    callbacks: Optional[List["TrainerCallback"]] = None,  # Training callbacks | 训练回调函数
):
    """Main DPO training workflow | DPO训练主流程"""
    # Load and prepare tokenizer | 加载并准备tokenizer
    tokenizer_module = load_tokenizer(model_args)  # Load tokenizer | 加载tokenizer
    tokenizer = tokenizer_module["tokenizer"]  # Get tokenizer instance | 获取分词器实例
    template = get_template_and_fix_tokenizer(tokenizer, data_args)  # Get conversation template | 获取对话模板并修复分词器

    # Prepare dataset | 准备数据集
    dataset_module = get_dataset(  # Get training dataset | 获取训练数据集
        template, 
        model_args,
        data_args,
        training_args,
        stage="rm",  # Specify reward modeling stage | 指定为奖励建模阶段
        **tokenizer_module
    )

    # Load main model | 加载主模型
    model = load_model(  # Load pretrained model | 加载预训练模型
        tokenizer,
        model_args,
        finetuning_args,
        training_args.do_train  # Whether in training mode | 是否训练模式
    )

    # Initialize data collator | 初始化数据整理器
    data_collator = PairwiseDataCollatorWithPadding(  # Pairwise data padding collator | 成对数据填充整理器
        template=template,  # Conversation template | 对话模板
        pad_to_multiple_of=8,  # Pad to multiple of 8 (optimize GPU) | 按8的倍数填充（优化GPU性能）
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,  # Label padding ID | 标签填充ID
        **tokenizer_module,
    )

    # Create reference model | 创建参考模型
    if finetuning_args.use_ref_model:  # Whether to use reference model | 是否使用参考模型
        if finetuning_args.ref_model is None and (not training_args.do_train):  # Evaluation without specified ref model | 评估阶段且未指定参考模型
            ref_model = model  # Use current model as reference | 使用当前模型作为参考模型
        else:
            ref_model = create_ref_model(model_args, finetuning_args)  # Create new reference model | 创建新的参考模型
    else:
        ref_model = None  # Disable reference model | 不使用参考模型

    # Update training arguments | 更新训练参数
    training_args.remove_unused_columns = False  # Keep all columns (important for multimodal) | 保留所有列（重要对于多模态数据）

    # Calculate effective tokens (for throughput) | 计算有效token数量（用于吞吐量统计）
    effective_token_num = 0.0
    if finetuning_args.include_effective_tokens_per_second:  # If need to calculate tokens/sec | 需要计算有效token/秒
        for data in dataset_module["train_dataset"]:  # Iterate training data | 遍历训练数据集
            effective_token_num += len(data["chosen_input_ids"])  # Accumulate chosen tokens | 累计正样本token数
            effective_token_num += len(data["rejected_input_ids"])  # Accumulate rejected tokens | 累计负样本token数

    # Initialize DPO trainer | 初始化DPO训练器
    trainer = CustomDPOTrainer(
        model=model,  # Main model | 主模型
        ref_model=ref_model,  # Reference model | 参考模型
        args=training_args,  # Training arguments | 训练参数
        finetuning_args=finetuning_args,  # Finetuning arguments | 微调参数
        data_collator=data_collator,  # Data collator | 数据整理器
        callbacks=callbacks,  # Training callbacks | 训练回调函数
        **dataset_module,  # Dataset | 数据集
        **tokenizer_module,  # Tokenizer | 分词器
    )

    # Training phase | 训练阶段
    if training_args.do_train:  # If training enabled | 是否进行训练
        train_result = trainer.train(  # Start training | 开始训练
            resume_from_checkpoint=training_args.resume_from_checkpoint  # Resume from checkpoint | 从检查点恢复
        )

        # Calculate effective tokens per second | 计算有效token吞吐量
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = cal_effective_tokens(
                effective_token_num,  # Total effective tokens | 总有效token数
                train_result.metrics["epoch"],  # Training epochs | 训练轮数
                train_result.metrics["train_runtime"]  # Training duration | 训练时间
            )

        # Save training results | 保存训练结果
        trainer.save_model()  # Save model | 保存模型
        trainer.log_metrics("train", train_result.metrics)  # Log metrics | 记录训练指标
        trainer.save_metrics("train", train_result.metrics)  # Save metrics | 保存训练指标
        trainer.save_state()  # Save training state | 保存训练状态
        
        # Plot loss curves | 绘制损失曲线
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:  # Main process and need plot | 主进程且需要绘图
            plot_loss(  # Plot training curves | 绘制训练曲线
                training_args.output_dir, 
                keys=["loss", "eval_loss", "rewards/accuracies"]  # Metrics to plot | 需要绘制的指标
            )

    # Evaluation phase | 评估阶段
    if training_args.do_eval:  # If evaluation enabled | 是否进行评估
        metrics = trainer.evaluate(metric_key_prefix="eval")  # Run evaluation | 执行评估
        if id(model) == id(ref_model):  # When reference model is itself | 当参考模型是自身时
            remove_keys = [key for key in metrics.keys() if "rewards" in key]  # Remove reward metrics | 移除奖励相关指标
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)  # Log evaluation metrics | 记录评估指标
        trainer.save_metrics("eval", metrics)  # Save evaluation metrics | 保存评估指标

    # Create and push model card | 创建并推送模型卡片
    create_modelcard_and_push(  # Generate model card | 生成模型卡片
        trainer,  # Trainer instance | 训练器实例
        model_args,  # Model arguments | 模型参数
        data_args,  # Data arguments | 数据参数
        training_args,  # Training arguments | 训练参数
        finetuning_args  # Finetuning arguments | 微调参数
    )
