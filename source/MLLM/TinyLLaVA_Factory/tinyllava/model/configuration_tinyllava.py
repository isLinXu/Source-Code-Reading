from transformers import PretrainedConfig, LlavaConfig
from transformers import CONFIG_MAPPING
from transformers import AutoConfig
from tinyllava.utils.constants import *

class TinyLlavaConfig(PretrainedConfig):
    """
    TinyLlavaConfig 类继承自 PretrainedConfig，用于配置 TinyLlava 模型的参数。
    """
    model_type = "tinyllava"
    def __init__(
        self,
        llm_model_name_or_path = '',
        tokenizer_name_or_path = None,
        vision_model_name_or_path = '',
        vision_model_name_or_path2 = '',
        connector_type = None,
        text_config=None,
        hidden_size=2048,
        vocab_size=32000,
        ignore_index=-100,
        image_token_index=32000,
        pad_token = None,
        pad_token_id = None,
        tokenizer_padding_side = 'right',
        tokenizer_model_max_length = 2048,
        vision_config = None,
        vision_hidden_size = None,
        vision_feature_layer = -2,
        vision_feature_select_strategy = 'patch',
        image_aspect_ratio = 'square',
        resampler_hidden_size = None,
        num_queries = None,
        num_resampler_layers = None,
        use_cache = False,
        cache_dir = None,
        tokenizer_use_fast = False,
        tune_type_llm = 'frozen',
        tune_type_connector = 'frozen',
        tune_type_vision_tower = 'frozen',
        tune_vision_tower_from_layer = -1,
        # speech
        audio_model_name_or_path='',
        audio_encoder_type='whisper',
        audio_feature_layer=-1,
        **kwargs

    ):
        """
        初始化 TinyLlavaConfig 类的实例。

        参数:
        - llm_model_name_or_path: 预训练的语言模型名称或路径。
        - tokenizer_name_or_path: 分词器的名称或路径，默认与语言模型相同。
        - vision_model_name_or_path: 预训练的视觉模型名称或路径。
        - vision_model_name_or_path2: 第二个预训练的视觉模型名称或路径。
        - ... (其他参数)
        """
        # 初始化实例变量
        self.llm_model_name_or_path = llm_model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or self.llm_model_name_or_path
        self.vision_model_name_or_path = vision_model_name_or_path
        self.vision_model_name_or_path2 = vision_model_name_or_path2
        self.connector_type = connector_type
        self.tune_type_llm = tune_type_llm
        self.tune_type_connector = tune_type_connector
        self.tune_type_vision_tower = tune_type_vision_tower
        self.tune_vision_tower_from_layer = tune_vision_tower_from_layer
        
        self.ignore_index = IGNORE_INDEX
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.tokenizer_padding_side = tokenizer_padding_side
        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_aspect_ratio = image_aspect_ratio
        self.resampler_hidden_size = resampler_hidden_size
        self.num_queries = num_queries
        self.num_resampler_layers = num_resampler_layers
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.tokenizer_use_fast = tokenizer_use_fast
        # 音频初始化
        self.audio_model_name_or_path = audio_model_name_or_path
        self.audio_encoder_type = audio_encoder_type
        self.audio_feature_layer = audio_feature_layer

        # 加载文本和视觉配置
        self._load_text_config(text_config)
        self._load_vision_config(vision_config)

        # 调用父类的初始化方法
        super().__init__(**kwargs)
    
    def load_from_config(self, config):
        # 从配置对象中加载参数
        self.llm_model_name_or_path = getattr(config, 'model_name_or_path',  '')
        self.tokenizer_name_or_path = getattr(config, 'tokenizer_name_or_path', None) or self.llm_model_name_or_path
        self.vision_model_name_or_path = getattr(config, 'vision_tower',  '')
        self.vision_model_name_or_path2 = getattr(config, 'vision_tower2',  '')
        self.connector_type = getattr(config, 'connector_type',  None)
        self.vision_feature_layer = getattr(config, 'mm_vision_select_layer',  -2)
        self.vision_feature_select_strategy = getattr(config, 'mm_vision_select_feature',  "patch")
        self.image_aspect_ratio = getattr(config, 'image_aspect_ratio',  "pad")
        self.resampler_hidden_size = getattr(config, 'resampler_hidden_size',  None)
        self.num_queries = getattr(config, 'num_queries',  None)
        self.num_resampler_layers = getattr(config, 'num_resampler_layers',  None)
        
        self.cache_dir = getattr(config, 'cache_dir', None)
        self.tokenizer_use_fast = getattr(config, 'tokenizer_use_fast', False)
        self.tokenizer_model_max_length = getattr(config, 'model_max_length', 2048)
        self.tokenizer_padding_side = getattr(config, 'tokenizer_padding_side', 'right')
        # 加载文本和视觉配置
        self._load_text_config()
        self._load_vision_config()
      
    
    def _load_text_config(self, text_config=None):
        """
        加载视觉配置。

        参数:
        - vision_config: 可选的视觉配置字典。
        """
        if self.llm_model_name_or_path is None or self.llm_model_name_or_path == '':
            self.text_config = CONFIG_MAPPING['llama']()
           
        else:
            self.text_config = AutoConfig.from_pretrained(self.llm_model_name_or_path, trust_remote_code=True)
            if text_config is not None:
                self.text_config = self.text_config.from_dict(text_config)
        # 设置隐藏层大小和词汇表大小
        self.hidden_size = getattr(self.text_config, 'hidden_size',  getattr(self.text_config, 'model_dim', None))
        self.vocab_size = getattr(self.text_config, 'vocab_size',  None)
    
    
    
    def _load_vision_config(self, vision_config=None):
        if self.vision_model_name_or_path is None or self.vision_model_name_or_path == '':
            self.vision_config = CONFIG_MAPPING['clip_vision_model'](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )
            
        else:
            self.vision_config = AutoConfig.from_pretrained(self.vision_model_name_or_path.split(':')[-1])
            self.vision_config = getattr(self.vision_config, 'vision_config', self.vision_config)
            if vision_config is not None:
                self.vision_config = self.vision_config.from_dict(vision_config)
        # 设置视觉模型的名称或路径和其他相关参数
        self.vision_config.model_name_or_path = self.vision_model_name_or_path.split(':')[-1]
        self.vision_config.model_name_or_path2 = self.vision_model_name_or_path2.split(':')[-1]
        self.vision_hidden_size = getattr(self.vision_config, 'hidden_size',  None)  
        

