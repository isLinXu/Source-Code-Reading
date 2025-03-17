# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
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

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from accelerate.utils import DistributedDataParallelKwargs  # 分布式训练参数
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint  # 移除虚拟检查点
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer  # TRL库的PPO组件
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation  # 模型解封装
from typing_extensions import override  # 重写装饰器

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback  # 自定义回调
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler  # 自定义优化器/调度器
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead  # 带价值头的模型

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments  # 超参数配置


logger = logging.get_logger(__name__)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""
    Inherits PPOTrainer.
    自定义PPO训练器，继承自TRL的PPOTrainer和HuggingFace Trainer
    """
    def __init__(
        self,
        model_args: "ModelArguments",          # 模型路径/名称等参数
        training_args: "Seq2SeqTrainingArguments",  # 训练参数（批次大小、学习率等）
        finetuning_args: "FinetuningArguments",# 微调参数（LoRA、QLoRA等）
        generating_args: "GeneratingArguments",# 生成参数（温度、top_p等）
        callbacks: Optional[List["TrainerCallback"]],  # 回调函数列表
        model: "AutoModelForCausalLMWithValueHead",  # 带价值头的语言模型
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],  # 奖励模型
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],  # 参考模型
        tokenizer: "PreTrainedTokenizer",  # 分词器
        processor: Optional["ProcessorMixin"],  # 处理器（多模态用）
        data_collator: "DataCollatorWithPadding",  # 数据整理器
        train_dataset: Optional["Dataset"] = None,  # 训练数据集
        eval_dataset: Optional["Dataset"] = None,  # 验证数据集（PPO暂不支持）
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        # 计算有效批次大小
        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        # 配置PPO参数
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # 处理DeepSpeed配置
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # 计算总训练步数
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        # 创建优化器和调度器
        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        # 初始化PPOTrainer
        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        # 初始化训练相关参数
        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # 当前设备（适配DeepSpeed）

        # 配置生成参数
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        # 初始化训练状态和控制
        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        # 混合精度上下文和警告过滤
        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # 过滤参考模型的垃圾回收警告

        # 准备完整奖励模型（非LoRA情况）
        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # 量化模型已位于正确设备
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        # 添加自定义回调
        self.add_callback(FixValueHeadModelCallback)  # 修复价值头模型
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))  # 保存处理器

        # 添加BAdam优化器支持
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        实现PPO阶段的训练循环，类似HuggingFace Trainer的_inner_training_loop
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        # 计算训练相关参数
        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        # 初始化训练状态
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # 打印训练信息
        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {:,}".format(
                total_train_batch_size
            )
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        # 初始化训练指标
        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()  # 损失平均值计算器
        reward_meter = AverageMeter()  # 奖励平均值计算器
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # 训练主循环
        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # 模型生成阶段
            self.model.eval()
            self.tokenizer.padding_side = "right"  # 设置右侧填充
            queries, responses, rewards = [], [], []
            # 分小批次处理
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses = self.get_inputs(
                    batch[idx : idx + self.config.mini_batch_size]
                )
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # PPO优化阶段
            self.model.train()
            stats = self.step(queries, responses, rewards)  # 执行PPO步骤
            self.tokenizer.padding_side = "left"  # 恢复左侧填充
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            # 日志记录
            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)  # 记录统计信息
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            # 更新训练状态
            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            # 定期日志输出
            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            # 定期保存模型
            if (step + 1) % self.args.save_steps == 0:
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            # 检查是否提前终止
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        # 创建自定义优化器
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            # 分离衰减参数和非衰减参数
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            # 创建默认优化器
            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        # 创建自定义学习率调度器
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, "torch.Tensor"]) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
        r"""
        Generates model's responses given queries.
        根据输入生成模型响应
        """
        # 处理填充问题（针对llama2）
        if batch["input_ids"].size(0) == 1:  # 处理梯度累积>1的情况
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        # 模型生成阶段
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)  # 保存LayerNorm参数

            # 执行生成
            generate_output: "torch.Tensor" = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)  # 恢复LayerNorm参数

        # 处理生成结果
        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            # 去除填充
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            # 处理空响应和EOS token
            if len(response_indexes) == 0:  # 允许空响应
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # 包含EOS token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # 去除左侧填充
            responses.append(response[i, :response_length])  # 去除右侧填充

        return queries, responses

    @torch.no_grad()
    def get_rewards(
        self,
        queries: List["torch.Tensor"],
        responses: List["torch.Tensor"],
    ) -> List["torch.Tensor"]:
        r"""
        Computes scores using given reward model.
        使用奖励模型计算得分
        """
        # API类型奖励模型
        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        # 准备模型输入
        batch: Dict[str, "torch.Tensor"] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)

        # 切换奖励模型类型
        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")  # 切换到奖励头
            reward_model = self.model
        else:
            reward_model = self.reward_model

        # 计算奖励值
        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:
            values: "torch.Tensor" = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")  # 恢复默认头

        # 提取最终奖励值
        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # 使用fp32类型

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: Dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""
        Calculates model outputs in multiple batches.
        分批次计算模型输出（处理大batch内存问题）
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size  # 小批次大小
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        # 分小批次处理
        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            # 前向计算
            with self.amp_context:  # 支持bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            # 计算对数概率和掩码
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            # 处理每个样本的掩码
            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # 处理左侧填充
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            # 收集结果
            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.
        保存模型检查点（处理分布式训练场景）
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # 处理不同分布式训练框架的保存逻辑
        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # 获取分布式状态字典
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # 移除虚拟检查点
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
