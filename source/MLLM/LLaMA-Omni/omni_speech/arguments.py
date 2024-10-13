import transformers

from dataclasses import dataclass, field
from typing import Optional

# 定义ModelArguments数据类，用于配置模型参数
@dataclass
class ModelArguments:
    # model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    # version: Optional[str] = field(default="v0")
    # freeze_backbone: bool = field(default=False)
    # tune_speech_projector: bool = field(default=False)
    # tune_speech_encoder: bool = field(default=False)
    # tune_speech_generator_only: bool = field(default=False)
    # speech_encoder_type: Optional[str] = field(default=None)
    # speech_encoder: Optional[str] = field(default=None)
    # pretrain_speech_projector: Optional[str] = field(default=None)
    # speech_projector_type: Optional[str] = field(default='linear')
    # speech_generator_type: Optional[str] = field(default='ctc')
    # ctc_decoder_config: str = "(2,4096,32,11008)"
    # ctc_upsample_factor: int = 1
    # ctc_loss_weight: float = 1.0
    # unit_vocab_size: int = 1000
    # speech_encoder_ds_rate: int = 5
    # speech_encoder_hidden_size: int = 1280
    # 模型名称或路径，默认使用Facebook的OPT-125m模型
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    # 模型版本，默认为v0
    version: Optional[str] = field(default="v0")
    # 是否冻结模型主干，默认不冻结
    freeze_backbone: bool = field(default=False)
    # 是否调整语音投影器，默认不调整
    tune_speech_projector: bool = field(default=False)
    # 是否调整语音编码器，默认不调整
    tune_speech_encoder: bool = field(default=False)
    # 是否仅调整语音生成器，默认不调整
    tune_speech_generator_only: bool = field(default=False)
    # 语音编码器类型，默认为None
    speech_encoder_type: Optional[str] = field(default=None)
    # 语音编码器模型路径，默认为None
    speech_encoder: Optional[str] = field(default=None)
    # 预训练语音投影器路径，默认为None
    pretrain_speech_projector: Optional[str] = field(default=None)
    # 语音投影器类型，默认为'linear'
    speech_projector_type: Optional[str] = field(default='linear')
    # 语音生成器类型，默认为'ctc'
    speech_generator_type: Optional[str] = field(default='ctc')
    # CTC解码器配置，默认为"(2,4096,32,11008)"
    ctc_decoder_config: str = "(2,4096,32,11008)"
    # CTC上采样因子，默认为1
    ctc_upsample_factor: int = 1
    # CTC损失权重，默认为1.0
    ctc_loss_weight: float = 1.0
    # 单元词汇表大小，默认为1000
    unit_vocab_size: int = 1000
    # 语音编码器下采样率，默认为5
    speech_encoder_ds_rate: int = 5
    # 语音编码器隐藏层大小，默认为1280
    speech_encoder_hidden_size: int = 1280

# 定义DataArguments数据类，用于配置数据参数
@dataclass
class DataArguments:
    # 数据路径，默认为None
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # 是否多模态，默认为False
    is_multimodal: bool = False
    # 输入类型，默认为"mel"
    input_type: str = field(default="mel")
    # 是否对语音进行标准化，默认为False
    speech_normalize: bool = False
    # Mel频谱大小，默认为128
    mel_size: int = 128
    # 是否有目标单元，默认为False
    has_tgt_units: bool = False


# 定义TrainingArguments数据类，扩展自transformers.TrainingArguments，用于配置训练参数
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # 缓存目录，默认为None
    cache_dir: Optional[str] = field(default=None)
    # 优化器，默认使用"adamw_torch"
    optim: str = field(default="adamw_torch")
    # 是否冻结语音投影器，默认为False
    freeze_speech_projector: bool = field(default=False)
    # 模型最大序列长度，默认为512
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # 是否使用双量化，默认为True
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    # 量化类型，默认为"nf4"
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    # 使用的位数，默认为16
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    # 是否启用LoRA，默认为False
    lora_enable: bool = False
    # LoRA的r参数，默认为64
    lora_r: int = 64
    # LoRA的alpha参数，默认为16
    lora_alpha: int = 16
    # LoRA的dropout，默认为0.05
    lora_dropout: float = 0.05
    # LoRA的权重路径，默认为空
    lora_weight_path: str = ""
    # LoRA的偏置，默认为"none"
    lora_bias: str = "none"
    # 语音投影器的学习率，默认为None
    speech_projector_lr: Optional[float] = None
    # 是否按模态长度分组，默认为False
    group_by_modality_length: bool = field(default=False)