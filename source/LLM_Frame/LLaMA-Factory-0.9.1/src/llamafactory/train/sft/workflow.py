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

from typing import TYPE_CHECKING, List, Optional  # 导入类型检查、列表和可选类型

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer  # 导入数据处理相关函数和类
from ...extras.constants import IGNORE_INDEX  # 从 extras.constants 导入 IGNORE_INDEX
from ...extras.misc import cal_effective_tokens, get_logits_processor  # 从 extras.misc 导入计算有效令牌和获取 logits 处理器的函数
from ...extras.ploting import plot_loss  # 从 extras.ploting 导入绘制损失的函数
from ...model import load_model, load_tokenizer  # 从 model 导入加载模型和加载分词器的函数
from ..trainer_utils import create_modelcard_and_push  # 从 trainer_utils 导入创建模型卡和推送的函数
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor  # 导入计算准确率、相似度和评估 logit 处理器的类和函数
from .trainer import CustomSeq2SeqTrainer  # 导入自定义 Seq2SeqTrainer 类


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import Seq2SeqTrainingArguments, TrainerCallback  # 导入 Seq2Seq 训练参数和训练回调类型

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments  # 导入数据、微调、生成和模型参数类型


def run_sft(  # 定义运行 SFT 的函数
    model_args: "ModelArguments",  # 模型参数
    data_args: "DataArguments",  # 数据参数
    training_args: "Seq2SeqTrainingArguments",  # 训练参数
    finetuning_args: "FinetuningArguments",  # 微调参数
    generating_args: "GeneratingArguments",  # 生成参数
    callbacks: Optional[List["TrainerCallback"]] = None,  # 可选的训练回调
):
    tokenizer_module = load_tokenizer(model_args)  # 加载分词器模块
    tokenizer = tokenizer_module["tokenizer"]  # 获取分词器
    template = get_template_and_fix_tokenizer(tokenizer, data_args)  # 获取模板并修复分词器
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)  # 获取数据集模块
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)  # 加载模型

    if getattr(model, "is_quantized", False) and not training_args.do_train:  # 如果模型被量化且不进行训练
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction  # 黑科技：使模型兼容预测

    data_collator = SFTDataCollatorWith4DAttentionMask(  # 创建数据收集器
        template=template,  # 模板
        pad_to_multiple_of=8 if training_args.do_train else None,  # 为短注意力填充到 8 的倍数
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,  # 设置标签填充标记
        block_diag_attn=model_args.block_diag_attn,  # 块对角注意力
        attn_implementation=getattr(model.config, "_attn_implementation", None),  # 获取注意力实现
        compute_dtype=model_args.compute_dtype,  # 计算数据类型
        **tokenizer_module,  # 传递分词器模块参数
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len  # 设置生成最大长度
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams  # 设置生成束数
    training_args.remove_unused_columns = False  # important for multimodal dataset  # 对于多模态数据集很重要

    effective_token_num = 0.0  # 初始化有效令牌数量
    if finetuning_args.include_effective_tokens_per_second:  # 如果包含每秒有效令牌
        for data in dataset_module["train_dataset"]:  # 遍历训练数据集
            effective_token_num += len(data["input_ids"])  # 累加有效令牌数量

    # Metric utils
    metric_module = {}  # 初始化指标模块
    if training_args.predict_with_generate:  # 如果使用生成预测
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)  # 使用相似度计算
    elif finetuning_args.compute_accuracy:  # 如果计算准确率
        metric_module["compute_metrics"] = ComputeAccuracy()  # 使用准确率计算
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor  # 设置 logits 处理器

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(  # 初始化自定义 Seq2SeqTrainer
        model=model,  # 模型
        args=training_args,  # 训练参数
        finetuning_args=finetuning_args,  # 微调参数
        data_collator=data_collator,  # 数据收集器
        callbacks=callbacks,  # 训练回调
        **dataset_module,  # 数据集模块参数
        **tokenizer_module,  # 分词器模块参数
        **metric_module,  # 指标模块参数
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()  # 将生成参数转换为字典
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids  # 设置结束标记 ID
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id  # 设置填充标记 ID
    gen_kwargs["logits_processor"] = get_logits_processor()  # 获取 logits 处理器

    # Training
    if training_args.do_train:  # 如果进行训练
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)  # 开始训练
        if finetuning_args.include_effective_tokens_per_second:  # 如果包含每秒有效令牌
            train_result.metrics["effective_tokens_per_sec"] = cal_effective_tokens(  # 计算每秒有效令牌
                effective_token_num, train_result.metrics["epoch"], train_result.metrics["train_runtime"]
            )

        trainer.save_model()  # 保存模型
        trainer.log_metrics("train", train_result.metrics)  # 记录训练指标
        trainer.save_metrics("train", train_result.metrics)  # 保存训练指标
        trainer.save_state()  # 保存训练状态
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:  # 如果是主进程并且需要绘制损失
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])  # 绘制损失图

    if training_args.predict_with_generate:  # 如果使用生成预测
        tokenizer.padding_side = "left"  # 使用左填充进行生成

    # Evaluation
    if training_args.do_eval:  # 如果进行评估
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)  # 评估模型
        if training_args.predict_with_generate:  # 如果启用生成预测
            metrics.pop("eval_loss", None)  # 删除评估损失
        trainer.log_metrics("eval", metrics)  # 记录评估指标
        trainer.save_metrics("eval", metrics)  # 保存评估指标

    # Predict
    if training_args.do_predict:  # 如果进行预测
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)  # 进行预测
        if training_args.predict_with_generate:  # 如果启用生成预测
            predict_results.metrics.pop("predict_loss", None)  # 删除预测损失
        trainer.log_metrics("predict", predict_results.metrics)  # 记录预测指标
        trainer.save_metrics("predict", predict_results.metrics)  # 保存预测指标
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)  # 保存预测结果

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)  # 创建模型卡并推送