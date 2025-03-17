# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import math  # 导入数学库
from typing import TYPE_CHECKING, List, Optional  # 导入类型检查、列表和可选类型

from transformers import DataCollatorForLanguageModeling  # 从 transformers 导入语言模型的数据收集器

from ...data import get_dataset, get_template_and_fix_tokenizer  # 从数据模块导入获取数据集和获取模板及修复分词器的函数
from ...extras.ploting import plot_loss  # 从 extras.ploting 导入绘制损失的函数
from ...model import load_model, load_tokenizer  # 从模型模块导入加载模型和加载分词器的函数
from ..trainer_utils import create_modelcard_and_push  # 从 trainer_utils 导入创建模型卡和推送的函数
from .trainer import CustomTrainer  # 从 trainer 导入自定义训练器类


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import Seq2SeqTrainingArguments, TrainerCallback  # 导入 Seq2Seq 训练参数和训练回调类型

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments  # 导入数据、微调和模型参数类型


def run_pt(  # 定义运行 PT 的函数
    model_args: "ModelArguments",  # 模型参数
    data_args: "DataArguments",  # 数据参数
    training_args: "Seq2SeqTrainingArguments",  # 训练参数
    finetuning_args: "FinetuningArguments",  # 微调参数
    callbacks: Optional[List["TrainerCallback"]] = None,  # 可选的训练回调
):
    tokenizer_module = load_tokenizer(model_args)  # 加载分词器模块
    tokenizer = tokenizer_module["tokenizer"]  # 获取分词器
    template = get_template_and_fix_tokenizer(tokenizer, data_args)  # 获取模板并修复分词器
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)  # 获取数据集模块
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)  # 加载模型
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # 创建语言模型的数据收集器

    # Initialize our Trainer
    trainer = CustomTrainer(  # 初始化自定义训练器
        model=model,  # 模型
        args=training_args,  # 训练参数
        finetuning_args=finetuning_args,  # 微调参数
        data_collator=data_collator,  # 数据收集器
        callbacks=callbacks,  # 训练回调
        **dataset_module,  # 数据集模块参数
        **tokenizer_module,  # 分词器模块参数
    )

    # Training
    if training_args.do_train:  # 如果进行训练
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)  # 开始训练
        trainer.save_model()  # 保存模型
        trainer.log_metrics("train", train_result.metrics)  # 记录训练指标
        trainer.save_metrics("train", train_result.metrics)  # 保存训练指标
        trainer.save_state()  # 保存训练状态
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:  # 如果是主进程并且需要绘制损失
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])  # 绘制损失图

    # Evaluation
    if training_args.do_eval:  # 如果进行评估
        metrics = trainer.evaluate(metric_key_prefix="eval")  # 评估模型
        try:
            perplexity = math.exp(metrics["eval_loss"])  # 计算困惑度
        except OverflowError:  # 如果发生溢出错误
            perplexity = float("inf")  # 设置困惑度为无穷大

        metrics["perplexity"] = perplexity  # 将困惑度添加到指标中
        trainer.log_metrics("eval", metrics)  # 记录评估指标
        trainer.save_metrics("eval", metrics)  # 保存评估指标

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)  # 创建模型卡并推送