# Copyright 2024 the LlamaFactory team.
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

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    与冻结（部分参数）训练相关的参数。
    """

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
                "（冻结（部分参数）微调的可训练层数。正数表示最后n层设为可训练，负数表示前n层设为可训练。）"
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
                "（冻结（部分参数）微调的可训练模块名称。使用逗号分隔多个模块。使用`all`指定所有可用模块。）"
            )
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
                "（除隐藏层外要设置为可训练的模块名称，用于冻结（部分参数）微调。使用逗号分隔多个模块。）"
            )
        },
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    与LoRA训练相关的参数。
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
                "（除LoRA层外要设置为可训练并保存在最终检查点中的模块名称。使用逗号分隔多个模块。）"
            )
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2). (LoRA微调的缩放因子，默认为lora_rank * 2)"},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning. (LoRA微调的dropout率)"},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning. (LoRA微调的内在维度)"},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
                "（应用LoRA的目标模块名称。使用逗号分隔多个模块。使用`all`指定所有线性模块。）"
            )
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A). (LoRA plus学习率比例 (lr_B / lr_A))"},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers. (LoRA嵌入层的LoRA plus学习率)"},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer. (是否对LoRA层使用秩稳定缩放因子)"},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA). (是否使用权重分解的LoRA方法(DoRA))"},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter. (是否初始化PiSSA适配器)"},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it. (PiSSA中FSVD执行的迭代步数。使用-1禁用)"},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter. (是否将PiSSA适配器转换为普通LoRA适配器)"},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight. (是否创建随机初始化权重的新适配器)"},
    )


@dataclass
class RLHFArguments:
    r"""
    Arguments pertaining to the PPO, DPO and KTO training.
    与PPO、DPO和KTO训练相关的参数。
    """

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss. (偏好损失中的beta参数)"},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training. (DPO训练中监督微调损失系数)"},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use. (使用的DPO损失类型)"},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5. (cDPO中的稳健DPO标签平滑参数，应在0和0.5之间)"},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training. (KTO训练中期望损失的权重因子)"},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training. (KTO训练中非期望损失的权重因子)"},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss. (SimPO损失中的目标奖励边际项)"},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step. (PPO优化步骤中创建经验缓冲区的小批量数量)"},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step. (PPO优化步骤中执行的轮次数)"},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training. (在PPO训练中使用分数归一化)"},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training. (PPO训练中自适应KL控制的目标KL值)"},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training. (在PPO训练中计算优势前对奖励进行白化)"},
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training. (PPO或DPO训练中使用的参考模型路径)"},
    )
    ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model. (参考模型适配器的路径)"},
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model. (量化参考模型的位数)"},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training. (PPO训练中使用的奖励模型路径)"},
    )
    reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model. (奖励模型适配器的路径)"},
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model. (量化奖励模型的位数)"},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training. (PPO训练中奖励模型的类型。Lora模型仅支持lora训练)"},
    )


@dataclass
class GaloreArguments:
    r"""
    Arguments pertaining to the GaLore algorithm.
    与GaLore算法相关的参数。
    """

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore). (是否使用梯度低秩投影(GaLore))"},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
                "（应用GaLore的模块名称。使用逗号分隔多个模块。使用`all`指定所有线性模块。）"
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients. (GaLore梯度的秩)"},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection. (更新GaLore投影的步数)"},
    )
    galore_scale: float = field(
        default=0.25,
        metadata={"help": "GaLore scaling coefficient. (GaLore缩放系数)"},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection. (GaLore投影类型)"},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory. (是否启用分层更新以进一步节省内存)"},
    )


@dataclass
class BAdamArgument:
    r"""
    Arguments pertaining to the BAdam optimizer.
    与BAdam优化器相关的参数。
    """

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer. (是否使用BAdam优化器)"},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer. (是否使用分层或比例BAdam优化器)"},
    )
    badam_start_block: Optional[int] = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam. (分层BAdam的起始块索引)"},
    )
    badam_switch_mode: Optional[Literal["ascending", "descending", "random", "fixed"]] = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam. (分层BAdam选择更新块的策略)"},
    )
    badam_switch_interval: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update. (分层BAdam更新块的步数。使用-1禁用块更新)"
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam. (比例BAdam的更新比例)"},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
                "（BAdam优化器的掩码模式。`adjacent`表示可训练参数相邻，`scatter`表示可训练参数从权重中随机选择。）"
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
                "（BAdam优化器的详细程度。0表示不打印，1表示打印块前缀，2表示打印可训练参数。）"
            )
        },
    )


@dataclass
class FinetuningArguments(FreezeArguments, LoraArguments, RLHFArguments, GaloreArguments, BAdamArgument):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    与我们将要使用的微调技术相关的参数。
    """

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP). (是否在纯bf16精度下训练模型(不使用AMP))"},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training. (训练中将执行哪个阶段)"},
    )
    finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use. (使用哪种微调方法)"},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable. (是否仅使扩展块中的参数可训练)"},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer. (是否使用Adam-mini优化器)"},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze vision tower in MLLM training. (在MLLM训练中是否冻结视觉塔)"},
    )
    train_mm_proj_only: bool = field(
        default=False,
        metadata={"help": "Whether or not to train the multimodal projector for MLLM only. (是否仅训练MLLM的多模态投影器)"},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation. (是否在评估时计算token级别的准确率)"},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves. (是否保存训练损失曲线)"},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second. (是否计算每秒有效token数)"},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: List[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: Optional[List[str]] = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: List[str] = split_arg(self.lora_target)
        self.additional_target: Optional[List[str]] = split_arg(self.additional_target)
        self.galore_target: List[str] = split_arg(self.galore_target)
        self.freeze_vision_tower = self.freeze_vision_tower or self.train_mm_proj_only
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method. (无效的微调方法)"
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization. (我们只接受4位或8位量化)"
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization. (我们只接受4位或8位量化)"

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training. (PPO训练必须设置`reward_model`)")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training. (Freeze/Full PPO训练中`reward_model_type`不能为lora)")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function. (`dpo_label_smoothing`仅对sigmoid损失函数有效)")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training. (`use_llama_pro`仅对Freeze或LoRA训练有效)")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore or BAdam together. (LoRA不能与GaLore或BAdam一起使用)")

        if self.use_galore and self.use_badam:
            raise ValueError("Cannot use GaLore with BAdam together. (不能同时使用GaLore和BAdam)")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage. (当前训练阶段不能使用PiSSA)")

        if self.train_mm_proj_only and self.finetuning_type != "full":
            raise ValueError("`train_mm_proj_only` is only valid for full training. (`train_mm_proj_only`仅对full训练有效)")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training. (`loraplus_lr_ratio`仅对LoRA训练有效)")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training. (`use_rslora`仅对LoRA训练有效)")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training. (`use_dora`仅对LoRA训练有效)")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training. (`pissa_init`仅对LoRA训练有效)")
