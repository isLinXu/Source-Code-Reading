from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import transformers

# 如果是类型检查，导入transformers模块
if TYPE_CHECKING:
    import transformers

# 定义ModelArguments类，用于存储模型参数
@dataclass
class ModelArguments:
    # 缓存目录，默认为None
    cache_dir: Optional[str] = field(default=None)
    # 模型名称或路径，默认为"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name_or_path: Optional[str] = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # 分词器名称或路径，默认为None
    tokenizer_name_or_path: Optional[str] = field(default=None)
    # 注意力实现方式，默认为None
    attn_implementation: Optional[str] = field(default=None)
    # 视觉塔1，默认为空字符串
    vision_tower: Optional[str] = field(default='')
    # 视觉塔2，默认为空字符串
    vision_tower2: Optional[str] = field(default='')
    # 连接器类型，默认为'linear'
    connector_type: str = field(default='linear')

    # 混合视觉选择层，默认为-1，表示最后一层
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    # 混合补丁合并类型，默认为'flat'
    mm_patch_merge_type: Optional[str] = field(default='flat')
    # 混合视觉选择特征，默认为"patch"
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # 重采样器隐藏大小，默认为768
    resampler_hidden_size: Optional[int] = field(default=768)
    # 查询数量，默认为128
    num_queries: Optional[int] = field(default=128)
    # 重采样器层数，默认为3
    num_resampler_layers: Optional[int] = field(default=3)

    # 模型最大长度，默认为512，序列将被右填充（可能被截断）
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # 分词器是否使用快速模式，默认为False
    tokenizer_use_fast: bool = field(default=False)
    # 分词器填充侧，默认为'right'
    tokenizer_padding_side: str = field(default='right')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    conv_version: str = 'pretrain'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_recipe: str = field(default='common')                                                              # 训练配方，默认为'common'
    tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8     # LLM调谐类型，默认为'frozen'，支持：frozen, full, lora, qlora_int4, qlora_int8
    tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune          # 视觉塔调谐类型，默认为'frozen'，支持：frozen, full, partially-tune
    tune_vision_tower_from_layer: Optional[int] = field(default=10)                                             # 从哪一层开始调谐视觉塔，默认为第10层
    tune_type_connector: str = field(default="full") # support only: frozen, full                               # 连接器调谐类型，默认为'full'，支持：frozen, full
    tune_embed_tokens: Optional[int] = field(default=False)                                                     # 是否调谐嵌入标记，默认不调谐
    
    optim: str = field(default="adamw_torch")                                                                   # 优化器类型，默认为'adamw_torch'
    remove_unused_columns: bool = field(default=False)                                                          # 是否移除未使用的列，默认不移除
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}                  # 通过双量化压缩量化统计数据
    )                                                                                                           # 是否进行双量化，默认为True
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )                                                                                                           # 量化数据类型，默认为'nf4'
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )                                                                                                           # 使用的位数，默认为16位
    lora_r: int = 64                                                                                            # LORA的r参数，默认为64
    lora_alpha: int = 16                                                                                        # LORA的alpha参数，默认为16
    lora_dropout: float = 0.05                                                                                  # LORA的dropout参数，默认为0.05
    lora_weight_path: str = ""                                                                                  # LORA权重路径，默认为空字符串
    lora_bias: str = "none"                                                                                     # LORA偏置设置，默认为'none'
    mm_projector_lr: Optional[float] = None                                                                     # 模态投影器学习率，默认为None
    group_by_modality_length: bool = field(default=False)                                                       # 是否按模态长度分组，默认不分组
    vision_tower_lr: Optional[float] = None                                                                     # 视觉塔学习率，默认为None
    pretrained_model_path: Optional[str] = field(default=None)                                                  # 预训练模型路径，默认为None


    
    
   
