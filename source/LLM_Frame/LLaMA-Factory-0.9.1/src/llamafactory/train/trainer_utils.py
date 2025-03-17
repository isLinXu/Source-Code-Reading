# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the original GaLore's implementation: https://github.com/jiaweizzhao/GaLore
# and the original LoRA+'s implementation: https://github.com/nikhil-ghosh-berkeley/loraplus
# and the original BAdam's implementation: https://github.com/Ledzy/BAdam
# and the HuggingFace's TRL library: https://github.com/huggingface/trl
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

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from typing_extensions import override

from ..extras import logging
from ..extras.constants import IGNORE_INDEX
from ..extras.packages import is_galore_available
from ..hparams import FinetuningArguments, ModelArguments
from ..model import find_all_linear_modules, load_model, load_tokenizer, load_valuehead_params


if is_galore_available():
    from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit


if TYPE_CHECKING:
    from transformers import PreTrainedModel, Seq2SeqTrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


# 虚拟优化器（用于GaLore分层优化）
class DummyOptimizer(torch.optim.Optimizer):
    r"""
    A dummy optimizer used for the GaLore algorithm.
    GaLore算法使用的虚拟优化器，实际优化由各层独立优化器完成
    """
    def __init__(
        self, lr: float = 1e-3, optimizer_dict: Optional[Dict["torch.nn.Parameter", "torch.optim.Optimizer"]] = None
    ) -> None:
        dummy_tensor = torch.randn(1, 1)  # 创建虚拟张量以满足父类初始化
        self.optimizer_dict = optimizer_dict  # 存储各参数的独立优化器
        super().__init__([dummy_tensor], {"lr": lr})  # 调用父类初始化

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        pass  # 空实现，梯度清零由各层优化器自行处理

    @override
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass  # 空实现，参数更新由各层优化器自行处理


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    """创建模型卡并推送至Hugging Face Hub"""
    kwargs = {
        "tasks": "text-generation",  # 任务类型
        "finetuned_from": model_args.model_name_or_path,  # 基础模型路径
        "tags": ["llama-factory", finetuning_args.finetuning_type],  # 模型标签
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = data_args.dataset  # 添加数据集信息

    if model_args.use_unsloth:  # 检查是否使用Unsloth优化
        kwargs["tags"] = kwargs["tags"] + ["unsloth"]  # 添加优化标签

    if not training_args.do_train:  # 检查是否训练模式
        pass
    elif training_args.push_to_hub:  # 检查是否推送到Hub
        trainer.push_to_hub(**kwargs)  # 推送模型
    else:
        trainer.create_model_card(license="other", **kwargs)  # 本地创建模型卡


def create_ref_model(
    model_args: "ModelArguments", finetuning_args: "FinetuningArguments", add_valuehead: bool = False
) -> Optional[Union["PreTrainedModel", "AutoModelForCausalLMWithValueHead"]]:
    r"""
    Creates reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    创建参考模型用于PPO/DPO训练，不支持评估模式
    """
    if finetuning_args.ref_model is not None:  # 检查是否指定参考模型
        ref_model_args = ModelArguments.copyfrom(  # 复制模型参数
            model_args,
            model_name_or_path=finetuning_args.ref_model,  # 覆盖模型路径
            adapter_name_or_path=finetuning_args.ref_model_adapters,  # 适配器路径
            quantization_bit=finetuning_args.ref_model_quantization_bit,  # 量化配置
        )
        ref_finetuning_args = FinetuningArguments()  # 创建默认微调参数
        tokenizer = load_tokenizer(ref_model_args)["tokenizer"]  # 加载分词器
        ref_model = load_model(  # 加载参考模型
            tokenizer, 
            ref_model_args, 
            ref_finetuning_args, 
            is_trainable=False,  # 不可训练
            add_valuehead=add_valuehead  # 是否添加value head
        )
        logger.info_rank0(f"Created reference model from {finetuning_args.ref_model}")
    else:  # 未指定参考模型的情况
        if finetuning_args.finetuning_type == "lora":  # LoRA微调时不需要参考模型
            ref_model = None
        else:  # 全参数微调时使用原模型作为参考
            ref_model_args = ModelArguments.copyfrom(model_args)
            ref_finetuning_args = FinetuningArguments()
            tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
            ref_model = load_model(
                tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            logger.info_rank0("Created reference model from the model itself.")

    return ref_model


def create_reward_model(
    model: "AutoModelForCausalLMWithValueHead", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["AutoModelForCausalLMWithValueHead"]:
    r"""
    Creates reward model for PPO training.
    创建奖励模型用于PPO训练
    """
    if finetuning_args.reward_model_type == "api":  # API类型奖励模型
        assert finetuning_args.reward_model.startswith("http"), "Please provide full url."
        logger.info_rank0(f"Use reward server {finetuning_args.reward_model}")
        return finetuning_args.reward_model  # 直接返回API地址
    elif finetuning_args.reward_model_type == "lora":  # LoRA适配器奖励模型
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")  # 加载奖励适配器
        # 确保可训练参数为float32
        for name, param in model.named_parameters():  # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32)  # 可训练参数转为float32
        # 加载value head参数
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        # 注册buffer参数
        model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
        model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
        model.register_buffer(
            "default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False
        )
        model.register_buffer(
            "default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False
        )
        logger.info_rank0(f"Loaded adapter weights of reward model from {finetuning_args.reward_model}")
        return None
    else:  # 完整奖励模型
        reward_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.reward_model,  # 覆盖模型路径
            adapter_name_or_path=finetuning_args.reward_model_adapters,  # 适配器路径
            quantization_bit=finetuning_args.reward_model_quantization_bit,  # 量化配置
        )
        reward_finetuning_args = FinetuningArguments()
        tokenizer = load_tokenizer(reward_model_args)["tokenizer"]
        reward_model = load_model(  # 加载完整奖励模型
            tokenizer, reward_model_args, reward_finetuning_args, is_trainable=False, add_valuehead=True
        )
        logger.info_rank0(f"Loaded full weights of reward model from {finetuning_args.reward_model}")
        logger.warning_rank0("Please ensure the ppo model and reward model share SAME tokenizer and vocabulary.")
        return reward_model


def _get_decay_parameter_names(model: "PreTrainedModel") -> List[str]:
    r"""
    Returns a list of names of parameters with weight decay. (weights in non-layernorm layers)
    获取需要权重衰减的参数名称列表（非LayerNorm层的权重参数）
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)  # 排除LayerNorm参数
    decay_parameters = [name for name in decay_parameters if "bias" not in name]  # 排除偏置项
    return decay_parameters


def _create_galore_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    """创建GaLore优化器"""
    # 确定GaLore目标层
    if len(finetuning_args.galore_target) == 1 and finetuning_args.galore_target[0] == "all":
        galore_targets = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)  # 查找所有线性层
    else:
        galore_targets = finetuning_args.galore_target  # 使用用户指定目标层

    # 收集GaLore参数
    galore_params: List["torch.nn.Parameter"] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(target in name for target in galore_targets):
            for param in module.parameters():
                if param.requires_grad and len(param.shape) > 1:  # 仅处理二维参数（权重矩阵）
                    galore_params.append(param)

    # 配置GaLore参数
    galore_kwargs = {
        "rank": finetuning_args.galore_rank,  # 低秩维度
        "update_proj_gap": finetuning_args.galore_update_interval,  # 投影更新间隔
        "scale": finetuning_args.galore_scale,  # 缩放因子
        "proj_type": finetuning_args.galore_proj_type,  # 投影类型
    }

    # 参数分类
    id_galore_params = {id(param) for param in galore_params}
    decay_params, nodecay_params = [], []  # 非GaLore参数
    trainable_params: List["torch.nn.Parameter"] = []  # 所有可训练参数
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            if id(param) not in id_galore_params:  # 非GaLore参数
                if name in decay_param_names:
                    decay_params.append(param)  # 需要衰减的参数
                else:
                    nodecay_params.append(param)  # 不需要衰减的参数

    # 获取优化器基类
    _, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

    # 选择GaLore优化器类型
    if training_args.optim == "adamw_torch":
        optim_class = GaLoreAdamW
    elif training_args.optim in ["adamw_bnb_8bit", "adamw_8bit", "paged_adamw_8bit"]:
        optim_class = GaLoreAdamW8bit
    elif training_args.optim == "adafactor":
        optim_class = GaLoreAdafactor
    else:
        raise NotImplementedError(f"Unknow optim: {training_args.optim}")

    # 分层优化模式
    if finetuning_args.galore_layerwise:
        if training_args.gradient_accumulation_steps != 1:
            raise ValueError("Per-layer GaLore does not support gradient accumulation.")

        # 为每个参数创建独立优化器
        optimizer_dict: Dict["torch.Tensor", "torch.optim.Optimizer"] = {}
        # 处理不需要衰减的参数
        for param in nodecay_params:
            param_groups = [dict(params=[param], weight_decay=0.0)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        # 处理需要衰减的参数
        for param in decay_params:
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        # 处理GaLore参数
        for param in galore_params:
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay, **galore_kwargs)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)

        # 注册梯度后处理钩子
        def optimizer_hook(param: "torch.nn.Parameter"):
            if param.grad is not None:
                optimizer_dict[param].step()  # 执行参数更新
                optimizer_dict[param].zero_grad()  # 清空梯度

        for param in trainable_params:
            param.register_post_accumulate_grad_hook(optimizer_hook)  # 注册钩子

        optimizer = DummyOptimizer(lr=training_args.learning_rate, optimizer_dict=optimizer_dict)
    else:  # 普通GaLore模式
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0),
            dict(params=decay_params, weight_decay=training_args.weight_decay),
            dict(params=galore_params, weight_decay=training_args.weight_decay, **galore_kwargs),
        ]
        optimizer = optim_class(param_groups, **optim_kwargs)

    logger.info_rank0("Using GaLore optimizer, may cause hanging at the start of training, wait patiently.")
    return optimizer


def _create_loraplus_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    """创建LoRA+优化器"""
    # 设置不同参数组的学习率
    default_lr = training_args.learning_rate  # 默认学习率
    loraplus_lr = default_lr * finetuning_args.loraplus_lr_ratio  # LoRA B层学习率
    embedding_lr = finetuning_args.loraplus_lr_embedding  # 嵌入层学习率

    # 参数分类
    decay_param_names = _get_decay_parameter_names(model)
    param_dict: Dict[str, List["torch.nn.Parameter"]] = {
        "lora_a": [],  # LoRA A层参数
        "lora_b": [],  # LoRA B层（需要衰减）
        "lora_b_nodecay": [],  # LoRA B层（不需要衰减）
        "embedding": [],  # 嵌入层参数
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_embedding_B" in name:  # 嵌入层适配器参数
                param_dict["embedding"].append(param)
            elif "lora_B" in name or param.ndim == 1:  # B层或一维参数（偏置）
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            else:  # A层参数
                param_dict["lora_a"].append(param)

    # 创建优化器
    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=param_dict["lora_a"], lr=default_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b"], lr=loraplus_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
        dict(params=param_dict["embedding"], lr=embedding_lr, weight_decay=training_args.weight_decay),
    ]
    optimizer = optim_class(param_groups, **optim_kwargs)
    logger.info_rank0(f"Using LoRA+ optimizer with loraplus lr ratio {finetuning_args.loraplus_lr_ratio:.2f}.")
    return optimizer


def _create_badam_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    """创建BAdam优化器（块自适应优化算法）"""
    decay_params, nodecay_params = [], []  # 初始化参数列表
    decay_param_names = _get_decay_parameter_names(model)  # 获取需要权重衰减的参数名
    
    # 参数分类逻辑
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in decay_param_names:
                decay_params.append(param)  # 需要权重衰减的参数
            else:
                nodecay_params.append(param)  # 不需要权重衰减的参数

    # 获取基础优化器配置
    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=nodecay_params, weight_decay=0.0),
        dict(params=decay_params, weight_decay=training_args.weight_decay),
    ]

    # 分层模式BAdam
    if finetuning_args.badam_mode == "layer":
        from badam import BlockOptimizer

        base_optimizer = optim_class(param_groups, **optim_kwargs)  # 创建基础优化器
        optimizer = BlockOptimizer(
            base_optimizer=base_optimizer,
            named_parameters_list=list(model.named_parameters()),  # 模型参数列表
            block_prefix_list=None,  # 自动检测层前缀
            switch_block_every=finetuning_args.badam_switch_interval,  # 层切换间隔
            start_block=finetuning_args.badam_start_block,  # 起始层索引
            switch_mode=finetuning_args.badam_switch_mode,  # 切换模式（顺序/随机）
            verbose=finetuning_args.badam_verbose,  # 是否输出调试信息
            ds_zero3_enabled=is_deepspeed_zero3_enabled(),  # 检查DeepSpeed Zero3状态
        )
        logger.info_rank0(
            f"Using BAdam optimizer with layer-wise update, switch mode is {finetuning_args.badam_switch_mode}, "
            f"switch block every {finetuning_args.badam_switch_interval} steps, "
            f"default start block is {finetuning_args.badam_start_block}"
        )

    # 比例模式BAdam
    elif finetuning_args.badam_mode == "ratio":
        from badam import BlockOptimizerRatio

        assert finetuning_args.badam_update_ratio > 1e-6  # 确保更新比例有效
        optimizer = BlockOptimizerRatio(
            param_groups=param_groups,
            named_parameters_list=list(model.named_parameters()),
            update_ratio=finetuning_args.badam_update_ratio,  # 参数更新比例
            mask_mode=finetuning_args.badam_mask_mode,  # 掩码模式（topk/random）
            verbose=finetuning_args.badam_verbose,  # 是否输出调试信息
            include_embedding=False,  # 是否包含嵌入层
            **optim_kwargs,
        )
        logger.info_rank0(
            f"Using BAdam optimizer with ratio-based update, update ratio is {finetuning_args.badam_update_ratio}, "
            f"mask mode is {finetuning_args.badam_mask_mode}"
        )

    return optimizer


def _create_adam_mini_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
) -> "torch.optim.Optimizer":
    """创建Adam-mini优化器（针对LLM优化的轻量级Adam）"""
    from adam_mini import Adam_mini

    # 获取模型结构参数
    hidden_size = getattr(model.config, "hidden_size", None)  # 隐藏层维度
    num_q_head = getattr(model.config, "num_attention_heads", None)  # 注意力头数
    num_kv_head = getattr(model.config, "num_key_value_heads", None)  # KV头数

    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),  # 模型参数
        lr=training_args.learning_rate,  # 学习率
        betas=(training_args.adam_beta1, training_args.adam_beta2),  # 动量参数
        eps=training_args.adam_epsilon,  # 数值稳定项
        weight_decay=training_args.weight_decay,  # 权重衰减
        model_sharding=is_fsdp_enabled() or is_deepspeed_zero3_enabled(),  # 检查分布式训练状态
        dim=hidden_size,  # 模型维度
        n_heads=num_q_head,  # 注意力头数
        n_kv_heads=num_kv_head,  # KV头数
    )
    logger.info_rank0("Using Adam-mini optimizer.")
    return optimizer


def create_custom_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Optional["torch.optim.Optimizer"]:
    """创建自定义优化器的工厂方法"""
    if finetuning_args.use_galore:  # GaLore优化器
        return _create_galore_optimizer(model, training_args, finetuning_args)

    if finetuning_args.loraplus_lr_ratio is not None:  # LoRA+优化器
        return _create_loraplus_optimizer(model, training_args, finetuning_args)

    if finetuning_args.use_badam:  # BAdam优化器
        return _create_badam_optimizer(model, training_args, finetuning_args)

    if finetuning_args.use_adam_mini:  # Adam-mini优化器
        return _create_adam_mini_optimizer(model, training_args)


def create_custom_scheduler(
    training_args: "Seq2SeqTrainingArguments",
    num_training_steps: int,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> None:
    """为GaLore创建自定义学习率调度器"""
    if optimizer is not None and isinstance(optimizer, DummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict  # 获取分层优化器字典
        scheduler_dict: Dict["torch.nn.Parameter", "torch.optim.lr_scheduler.LRScheduler"] = {}

        # 为每个参数创建独立调度器
        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                training_args.lr_scheduler_type,  # 调度器类型
                optimizer=optimizer_dict[param],  # 对应优化器
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),  # warmup步数
                num_training_steps=num_training_steps,  # 总训练步数
                scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,  # 调度器参数
            )

        # 定义调度器更新钩子
        def scheduler_hook(param: "torch.nn.Parameter"):
            scheduler_dict[param].step()  # 更新学习率

        # 注册梯度累积后钩子
        for param in optimizer_dict.keys():
            param.register_post_accumulate_grad_hook(scheduler_hook)


def get_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    计算给定标签在模型输出下的对数概率
    """
    # 维度校验
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    # 对齐logits和labels
    labels = labels[:, 1:].clone()  # 移除第一个token的标签
    logits = logits[:, :-1, :]  # 移除最后一个logit
    
    # 创建损失掩码
    loss_mask = labels != label_pad_token_id  # 标识有效token位置
    labels[labels == label_pad_token_id] = 0  # 用虚拟token替换填充符
    
    # 计算每个token的对数概率
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    # 返回批次总对数概率和有效长度
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)
