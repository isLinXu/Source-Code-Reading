# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/ppo.py
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
from ...data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss  # Visualization of loss curves
from ...model import load_model, load_tokenizer  # Model loading utilities
from ..callbacks import fix_valuehead_checkpoint  # Fix value head checkpoint
from ..trainer_utils import create_ref_model, create_reward_model  # Model creation helpers
from .trainer import CustomPPOTrainer  # Custom PPO trainer


if TYPE_CHECKING:
    # Type hints only for type checkers
    # 类型提示仅在类型检查时导入
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_ppo(
    model_args: "ModelArguments",          # Model config (path, name, etc)
    data_args: "DataArguments",            # Dataset config (path, preprocessing)
    training_args: "Seq2SeqTrainingArguments",  # Training settings (lr, batch size)
    finetuning_args: "FinetuningArguments",# Finetuning config (LoRA, PPO params)
    generating_args: "GeneratingArguments",# Generation settings (temp, top_p)
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
    # 加载数据集模块（PPO阶段专用）
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    
    # Load model with value head for PPO
    # 加载带价值头的模型（PPO训练必需）
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    # Configure padding side
    # 设置填充方向：生成时左填充，训练时右填充
    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = MultiModalDataCollatorForSeq2Seq(template=template, **tokenizer_module)

    # Create reference model and reward model
    # 创建参考模型和奖励模型
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)  # 参考模型（初始策略）
    reward_model = create_reward_model(model, model_args, finetuning_args)  # 奖励模型（用于计算回报）

    # Initialize PPO trainer
    # 初始化自定义PPO训练器
    ppo_trainer: "CustomPPOTrainer" = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        model=model,            # 待训练的策略模型
        reward_model=reward_model,  # 奖励模型
        ref_model=ref_model,    # 参考模型（防止策略漂移）
        data_collator=data_collator,  # 多模态数据整理器
        **dataset_module,       # 注入数据集
        **tokenizer_module,     # 注入分词器
    )

    # Training process
    # 训练流程
    if training_args.do_train:
        # Execute PPO training
        # 执行PPO训练
        ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ppo_trainer.save_model()  # 保存最终模型
        
        # Fix value head checkpoint format
        # 修复价值头检查点格式（适配HuggingFace格式）
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

        ppo_trainer.save_state()  # must be called after save_model to have a folder // 保存训练状态（需在save_model之后调用）
        
        # Plot loss curves in main process
        # 主进程绘制损失曲线
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
