from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from . import LLMFactory, ConnectorFactory, VisionTowerFactory, AudioTowerFactory
from .configuration_tinyllava import TinyLlavaConfig
from ..utils.constants import *
# from tinyllava.utils.data_utils import get_value_from_kwargs


def get_value_from_kwargs(kwargs, name):
    """
    从关键字参数中获取指定名称的值，并将其从关键字参数中移除。

    Args:
        kwargs (dict): 关键字参数，需要从中获取值的字典。
        name (str): 关键字参数的名称，需要获取其对应的值。

    Returns:
        Union[Any, None]: 返回获取到的值，若未找到则返回None。

    """
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None
    


class TinyLlavaPreTrainedModel(PreTrainedModel):
    """
    TinyLlava Pre-trained Model class.

    Args:
        PreTrainedModel: Base class for all pre-trained models.

    Attributes:
        config_class (Type[TinyLlavaConfig]): The configuration class to use for initialization.
        base_model_prefix (str): Prefix for the base model in the state dictionary.
        supports_gradient_checkpointing (bool): Whether the model supports gradient checkpointing.
        _no_split_modules (List[str]): List of module names that should not be split across GPUs.
        _skip_keys_device_placement (str): Keys that should not be moved to the device.
        _supports_flash_attn_2 (bool): Whether the model supports FlashAttention 2.

    Returns:
        None
    """
    # 定义配置类为TinyLlavaConfig，这个类包含了模型的配置信息
    config_class = TinyLlavaConfig
    # 定义基础模型的前缀为"model"，这可能是模型权重文件名的一部分
    base_model_prefix = "model"
    # 支持梯度检查点技术，这是一种减少内存使用的技术，可以在训练大型模型时使用
    supports_gradient_checkpointing = True
    # 定义不需要分割的模块列表，这些模块在模型并行处理时不会被分割
    _no_split_modules = ["LlavaVisionAttention"]
    # 定义跳过设备放置的键，这些键在模型加载时不会考虑设备放置
    _skip_keys_device_placement = "past_key_values"
    # 支持Flash Attention 2技术，这是一种用于加速注意力计算的技术
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """
        初始化模型权重。

        Args:
            module (nn.Module): 需要初始化权重的模型模块。

        Returns:
            None

        """
        # 设置权重初始化的标准差
        # 如果模型配置中有initializer_range属性，则使用它，否则使用text_config中的initializer_range
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        # 如果模块中有class_embedding属性，则将其权重初始化为正态分布，均值为0，标准差为std
        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)
        # 如果模块是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 将权重初始化为正态分布，均值为0，标准差为std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，则将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 将权重初始化为正态分布，均值为0，标准差为std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在padding_idx，则将对应位置的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        判断当前语言模型是否支持SDPA（Sparse Dense Product Attention）

        Args:
            无

        Returns:
            bool: 如果当前语言模型支持SDPA，则返回True；否则返回False。

        """
        return self.language_model._supports_sdpa


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel):
    """
    TinyLlavaForConditionalGeneration 类是 TinyLlavaPreTrainedModel 的子类，用于条件生成任务。

    Args:
        config (TinyLlavaConfig): TinyLlava 的配置类实例。

    Attributes:
        language_model (nn.Module): 语言模型模块，由 LLMFactory 创建。
        vision_tower (nn.Module): 视觉塔模块，由 VisionTowerFactory 创建。
        connector (nn.Module): 连接器模块，由 ConnectorFactory 创建。
        tokenizer (PreTrainedTokenizer): 分词器，用于文本预处理。

    Returns:
        None

    """
    def __init__(self, config: TinyLlavaConfig):
        
        super().__init__(config)  # 调用父类的初始化方法
        # 创建语言模型模块
        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config)
        # 创建视觉塔模块
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)
        # 创建音频塔模块
        self.audio_tower = AudioTowerFactory(config.audio_model_name_or_path)(config.audio_config)
        # 创建音频投影器模块
        self.audio_projector = build_audio_projector(config)

        # 创建连接器模块
        self.connector = ConnectorFactory(config.connector_type)(config)
        # 获取分词器类和加载后处理函数
        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        # 初始化分词器
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,   # 分词器缓存目录
            model_max_length = config.tokenizer_model_max_length, # 模型最大长度
            padding_side = config.tokenizer_padding_side, # 填充方向
            use_fast = config.tokenizer_use_fast, # 是否使用快速分词器
        ))
        self.post_init() # 执行初始化后的操作

    
    def encode_audio(self, audio, audio_lengths):
        """
        对输入的音频数据进行编码，并返回编码后的特征向量。

        参数:
            audio (torch.Tensor): 输入的音频数据张量。
            audio_lengths (torch.Tensor): 每个音频样本的有效长度。

        返回:
            List[torch.Tensor]: 编码后的音频特征向量列表。
        """
        audio_features = self.audio_tower(audio)
        audio_features = self.audio_projector(audio_features)
        return [audio_features[i, :audio_lengths[i]] for i in range(len(audio_features))]


    def get_input_embeddings(self):
        """
        获取语言模型的输入嵌入层。

        Args:
            无参数。

        Returns:
            nn.Module: 语言模型的输入嵌入层。

        """
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        设置语言模型的输入嵌入层。

        Args:
            value (nn.Module): 要设置的输入嵌入层。

        Returns:
            None
        """
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """
        获取语言模型的输出嵌入层。

        Args:
            无参数。

        Returns:
            nn.Module: 语言模型的输出嵌入层。

        """
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """
        设置语言模型的输出嵌入层。

        Args:
            new_embeddings (nn.Module): 新的输出嵌入层。

        Returns:
            None
        """
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        """
        设置语言模型的解码器。

        Args:
            decoder (nn.Module): 新的解码器模块。

        Returns:
            None
        """
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        """
        获取语言模型的解码器。

        Args:
            无参数。

        Returns:
            nn.Module: 语言模型的解码器模块。
        """
        return self.language_model.get_decoder()

    def tie_weights(self):
        """
        将语言模型的输入嵌入和输出嵌入权重进行绑定。

        Args:
            无参数。

        Returns:
            None: 该函数无返回值，但会修改语言模型的嵌入权重，实现权重绑定。

        """
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        """
        调整语言模型的嵌入层大小以匹配新的词汇表大小。

        Args:
            new_num_tokens (Optional[int], optional): 新的词汇表大小。如果未提供，则使用原始词汇表大小。默认为None。
            pad_to_multiple_of (Optional[int], optional): 调整嵌入层大小以匹配该数字的倍数。如果未提供，则使用原始嵌入层大小。默认为None。

        Returns:
            nn.Embedding: 调整大小后的嵌入层。

        """
        # 调整语言模型的token嵌入以适应新的token数量，同时确保嵌入矩阵的大小是pad_to_multiple_of的倍数
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        # 更新配置中的词汇表大小
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        # 返回调整后的token嵌入
        return model_embeds

    def process_audio(self, audio):
        if isinstance(audio, str):  # 如果是音频文件路径
            audio_features = self.audio_preprocess(audio)
        else:  # 如果已经是预处理后的特征
            audio_features = audio
        return self.audio_tower(audio_features)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        实现模型的forward方法。

        Args:
            input_ids (torch.LongTensor, optional): 输入的文本id。默认为None。
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码。默认为None。
            position_ids (Optional[torch.LongTensor], optional): 位置id。默认为None。
            past_key_values (Optional[List[torch.FloatTensor]], optional): 上一层的key和value缓存。默认为None。
            inputs_embeds (Optional[torch.FloatTensor], optional): 输入的嵌入张量。默认为None。
            labels (Optional[torch.LongTensor], optional): 标签。默认为None。
            use_cache (Optional[bool], optional): 是否使用缓存。默认为None。
            output_attentions (Optional[bool], optional): 是否输出注意力权重。默认为None。
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态。默认为None。
            images (Optional[torch.FloatTensor], optional): 图像张量。默认为None。
            image_sizes (Optional[List[List[int]]], optional): 图像尺寸列表。默认为None。
            return_dict (Optional[bool], optional): 是否返回字典。默认为None。

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: 模型输出。如果return_dict为True，则返回一个包含多个键值对的字典，否则返回一个包含多个元素的元组。

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        生成文本或图像描述文本。

        Args:
            inputs (Optional[torch.Tensor], optional): 输入文本或None。默认为None。
            images (Optional[torch.Tensor], optional): 输入图像或None。默认为None。
            image_sizes (Optional[torch.Tensor], optional): 输入图像的大小或None。默认为None。
            **kwargs: 其他可选参数。

        Returns:
            Union[GenerateOutput, torch.LongTensor]: 生成的文本或图像描述文本，具体返回类型根据输入参数和模型配置而定。

        Raises:
            NotImplementedError: 如果kwargs中包含"inputs_embeds"参数，则抛出此异常。
        """
        # 从文件中提取位置ID和注意力掩码，如果未提供则默认为None
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        # 如果kwargs中包含inputs_embeds，则抛出未实现错误，因为当前不支持此参数
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # 如果提供了图像数据，则调用prepare_inputs_labels_for_multimodal方法准备多模态输入
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            # 如果没有图像数据，则使用语言模型的输入嵌入方法获取输入嵌入
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)
        # 调用语言模型的generate方法生成输出，传入必要的参数
        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def encode_images(self, images):
        """
        对输入的图像进行编码，并返回编码后的特征向量。

        Args:
            images (torch.Tensor): 输入的图像张量，形状为 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 编码后的特征向量张量，形状为 (batch_size, feature_dim)。

        """
        # 初始化一个空字典用于存储关键字参数
        kwargs = {}
        # 将配置中的视觉特征层添加到kwargs字典中
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        # 将配置中的视觉特征选择策略添加到kwargs字典中
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        # 将图像数据移动到指定的设备，并转换成指定的数据类型
        images = images.to(device=self.device, dtype=self.dtype)
        # 使用视觉塔（vision tower）处理图像数据，并传入kwargs中的参数
        image_features = self.vision_tower(images, **kwargs)
        # 使用连接器（connector）处理图像特征
        image_features = self.connector(image_features)
        # 返回处理后的图像特征
        return image_features
    
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        为生成文本准备输入数据。

        Args:
            input_ids (torch.LongTensor): 输入的文本id。
            past_key_values (Optional[List[torch.FloatTensor]]): 上一步的key和value缓存。
            inputs_embeds (Optional[torch.FloatTensor]): 输入的嵌入张量。
            **kwargs: 其他可选参数。

        Returns:
            Dict[str, Any]: 包含输入数据的字典，包含input_ids, attention_mask, past_key_values,
                              inputs_embeds, position_ids等字段，如果传入了images和image_sizes，则还包含这两个字段。

        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, position_ids, attention_mask, past_key_values, labels,
    #     images, image_lengths, audio, audio_lengths
    # ):
    #     """
    #     对输入数据进行处理，为跨模态任务准备输入和标签。
    #     工作原理：
    #     该函数的主要目的是为了准备多模态输入数据和对应的标签，以便于模型能够处理这些数据并进行训练或推理。多模态输入通常指的是来自不同感官模态的数据，例如文本、图像、音频等。函数会根据配置和输入数据的不同情况，对输入数据进行相应的处理，包括截断、填充、生成注意力掩码和位置ID等。
    #     实现细节
    #     1.获取配置参数：
    #         函数首先尝试从配置中获取 tokenizer_model_max_length 参数，这个参数用于限制输入序列的最大长度。如果配置中没有设置该参数，则默认为 None。
    #     2.截断输入数据：
    #         如果 tokenizer_model_max_length 不为 None，函数会对输入的嵌入序列 new_input_embeds 和标签序列 new_labels 进行截断，只保留前 tokenizer_model_max_length 个元素。
    #     3.计算最大长度：
    #         函数计算所有输入嵌入序列的最大长度 max_len，并获取批量大小 batch_size。
    #     4.初始化填充后的数据结构：
    #         创建用于存储填充后输入嵌入序列的列表 new_input_embeds_padded。
    #         创建一个与最大长度相同的全零标签张量 new_labels_padded，用于存储填充后的标签。
    #         创建一个全零的注意力掩码张量 attention_mask，用于指示哪些位置是实际的输入数据。
    #         创建一个全零的位置ID张量 position_ids，用于表示每个位置在序列中的位置。
    #     5.填充输入数据和标签：
    #         函数遍历每个输入嵌入序列和对应的标签序列。
    #         根据配置中的 tokenizer_padding_side 参数（默认为 'right'），决定是在左侧还是右侧进行填充。
    #         对于每个序列，创建一个全零张量用于填充，并将其与原始序列拼接。
    #         如果序列长度大于0，则将标签填充到 new_labels_padded 的相应位置，并更新 attention_mask 和 position_ids。
    #     6.堆叠填充后的数据结构：
    #         使用 torch.stack 函数将填充后的输入嵌入序列列表 new_input_embeds_padded 堆叠成一个张量 new_input_embeds。
    #     7.处理可选的输出：
    #         如果 _labels 为 None，则 new_labels 也设置为 None；否则，将填充后的标签张量 new_labels_padded 赋值给 new_labels。
    #         如果 _attention_mask 为 None，则 attention_mask 也设置为 None；否则，将 attention_mask 转换为 _attention_mask 的数据类型。
    #         如果 _position_ids 为 None，则 position_ids 也设置为 None。
    #         返回结果：
    #         函数返回 None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels，这些数据结构将被用于模型的训练或推理过程。
    #
    #     Args:
    #         input_ids (torch.LongTensor): 输入的文本id张量。
    #         position_ids (Optional[torch.LongTensor]): 输入的位置id张量。
    #         attention_mask (Optional[torch.Tensor]): 注意力掩码张量。
    #         past_key_values (Optional[List[torch.FloatTensor]]): 上一步的key和value缓存。
    #         labels (Optional[torch.LongTensor]): 文本标签张量。
    #         images (torch.FloatTensor): 图像张量。
    #         image_sizes (Optional[List[List[int]]]): 图像尺寸列表。
    #
    #     Returns:
    #         Tuple[None, Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[torch.FloatTensor]],
    #                Optional[torch.Tensor], Optional[torch.Tensor]]: 返回处理后的输入数据元组，包括：
    #             - None: 当前未使用，占位符。
    #             - position_ids (Optional[torch.Tensor]): 处理后的位置id张量。
    #             - attention_mask (Optional[torch.Tensor]): 处理后的注意力掩码张量。
    #             - past_key_values (Optional[List[torch.FloatTensor]]): 处理后的上一步的key和value缓存。
    #             - new_input_embeds (Optional[torch.Tensor]): 处理后的输入嵌入张量。
    #             - new_labels (Optional[torch.Tensor]): 处理后的标签张量。
    #
    #     Raises:
    #         NotImplementedError: 如果config的tune_mm_mlp_adapter属性为True，则抛出此异常，表示该特性尚未实现。
    #
    #     """
    #     vision_tower = self.vision_tower  # 获取视觉塔对象
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:  # 检查视觉塔对象、图像数据或输入ID的形状是否满足条件
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels  # 如果不满足条件，返回相应的变量
    #
    #     image_features = self.encode_images(images)  # 对图像进行编码，提取特征
    #
    #     # TODO: 图像开始/结束功能在此处未实现以支持预训练。
    #     if getattr(self.config, 'tune_mm_mlp_adapter', False):  # 检查配置中是否启用了调整MM MLP适配器的选项
    #         raise NotImplementedError  # 如果启用，抛出未实现错误
    #
    #     # Let's just add dummy tensors if they do not exist,
    #     # it is a headache to deal with None all the time.
    #     # But it is not ideal, and if you have a better idea,
    #     # please open an issue / submit a PR, thanks.
    #     # 定义_labels变量，用于存储标签信息
    #     _labels = labels
    #     # 定义_position_ids变量，用于存储位置ID信息
    #     _position_ids = position_ids
    #     # 定义_attention_mask变量，用于存储注意力掩码信息
    #     _attention_mask = attention_mask
    #
    #     # 如果attention_mask为None，则创建一个与input_ids形状相同的全1张量作为默认值，并将其数据类型设置为torch.bool
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         # 如果attention_mask不为None，则将其转换为布尔类型
    #         attention_mask = attention_mask.bool()
    #
    #     # 如果position_ids为None，则创建一个从0开始，长度与input_ids第二维相同的张量作为默认值，并将其数据类型设置为torch.long
    #     if position_ids is None:
    #         position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #
    #     # 如果labels为None，则创建一个与input_ids形状相同的全IGNORE_INDEX张量作为默认值
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)
    #
    #     # remove the padding using attention_mask -- FIXME
    #     # 将输入的input_ids赋值给_input_ids变量，用于后续可能的引用或备份
    #     # 使用列表推导式，根据attention_mask过滤input_ids列表中的元素
    #     # 对于每一组cur_input_ids和cur_attention_mask，只保留cur_attention_mask为True的元素
    #     _input_ids = input_ids
    #     input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     # 使用列表推导式，根据attention_mask过滤labels列表中的元素
    #     # 对于每一组cur_labels和cur_attention_mask，只保留cur_attention_mask为True的元素
    #     labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
    #
    #     # 初始化新的输入嵌入和标签列表
    #     new_input_embeds = []
    #     new_labels = []
    #     # 当前图像索引初始化为0
    #     cur_image_idx = 0
    #     # 遍历输入ID的批次索引和当前输入ID
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         # 计算当前批次中图像标记的数量
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         # 如果当前批次中没有图像标记
    #         if num_images == 0:
    #             # 获取当前图像特征
    #             cur_image_features = image_features[cur_image_idx]
    #             # 获取当前输入ID的嵌入
    #             cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
    #             # 将当前输入嵌入和图像特征拼接起来
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
    #             # 将拼接后的输入嵌入添加到新输入嵌入列表
    #             new_input_embeds.append(cur_input_embeds)
    #             # 将当前批次的标签添加到新标签列表
    #             new_labels.append(labels[batch_idx])
    #             # 当前图像索引加1
    #             cur_image_idx += 1
    #             # 继续下一个批次
    #             continue
    #
    #         # image_token_indices 用于存储图像标记的索引，初始化为-1（表示序列开始前的位置）
    #         # 然后通过torch.where查找cur_input_ids中等于IMAGE_TOKEN_INDEX的所有索引，并转换为列表
    #         # 最后添加cur_input_ids的长度（表示序列结束后的位置）
    #         image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #         # cur_input_ids_noim 用于存储去除图像标记后的输入ID序列
    #         cur_input_ids_noim = []
    #         # cur_labels 用于存储当前批次的标签
    #         cur_labels = labels[batch_idx]
    #         # cur_labels_noim 用于存储去除图像标记后的标签序列
    #         cur_labels_noim = []
    #         # 遍历image_token_indices列表，获取每个图像标记之间的输入ID和标签
    #         for i in range(len(image_token_indices) - 1):
    #             # 将当前图像标记之间的输入ID添加到cur_input_ids_noim列表中
    #             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #             # 将当前图像标记之间的标签添加到cur_labels_noim列表中
    #             cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    #         # 计算cur_labels_noim中每个元素的形状大小，存储在split_sizes列表中
    #         split_sizes = [x.shape[0] for x in cur_labels_noim]
    #         # 获取输入嵌入，使用torch.cat将cur_input_ids_noim连接起来
    #         cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
    #         # 根据split_sizes将cur_input_embeds分割成多个部分，存储在cur_input_embeds_no_im列表中
    #         cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #         # 初始化新的输入嵌入列表和新的标签列表
    #         cur_new_input_embeds = []
    #         cur_new_labels = []
    #
    #         # 遍历所有图片的数量加一（可能是为了处理边界情况）
    #         for i in range(num_images + 1):
    #             # 将当前输入嵌入（不包括图片）添加到新的输入嵌入列表中
    #             cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #             # 将当前标签（不包括图片）添加到新的标签列表中
    #             cur_new_labels.append(cur_labels_noim[i])
    #             # 如果当前索引小于图片数量
    #             if i < num_images:
    #                 # 获取当前图片特征
    #                 cur_image_features = image_features[cur_image_idx]
    #                 # 图片索引加一，以便下次循环获取下一张图片特征
    #                 cur_image_idx += 1
    #                 # 将当前图片特征添加到新的输入嵌入列表中
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 # 创建一个与当前图片特征形状相同的张量，填充忽略索引值，并添加到新的标签列表中
    #                 # 这可能是为了标记图片部分在后续处理中不被考虑为有效标签
    #                 cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
    #
    #         # 将cur_new_input_embeds中的每个元素移动到self.device指定的设备上（如GPU）
    #         cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
    #         # 将cur_new_input_embeds列表中的所有张量沿第0维进行拼接
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #         # 将cur_new_labels列表中的所有张量沿第0维进行拼接
    #         cur_new_labels = torch.cat(cur_new_labels)
    #         # 将拼接后的cur_new_input_embeds添加到new_input_embeds列表中
    #         new_input_embeds.append(cur_new_input_embeds)
    #         # 将拼接后的cur_new_labels添加到new_labels列表中
    #         new_labels.append(cur_new_labels)
    #
    #     # Truncate sequences to max length as image embeddings can make the sequence longer
    #     # 获取配置中的tokenizer_model_max_length参数，如果没有设置则默认为None
    #     tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     # 如果tokenizer_model_max_length不为None，则对new_input_embeds和new_labels进行截断
    #     if tokenizer_model_max_length is not None:
    #         # 对new_input_embeds中的每个元素进行截断，只保留前tokenizer_model_max_length个元素
    #         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #         # 对new_labels中的每个元素进行截断，只保留前tokenizer_model_max_length个元素
    #         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
    #
    #     # Combine them
    #     # 计算输入嵌入序列的最大长度
    #     max_len = max(x.shape[0] for x in new_input_embeds)
    #     # 获取批量大小
    #     batch_size = len(new_input_embeds)
    #     # 初始化填充后的输入嵌入序列列表
    #     new_input_embeds_padded = []
    #     # 创建一个与最大长度相同，且所有元素都为IGNORE_INDEX的标签张量
    #     new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    #     # 创建一个全零的注意力掩码张量
    #     attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    #     # 创建一个全零的位置ID张量
    #     position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    #
    #     # 遍历新的输入嵌入和新的标签
    #     for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    #         # 获取当前嵌入的长度
    #         cur_len = cur_new_embed.shape[0]
    #         # 如果配置中的tokenizer_padding_side为'left'，则进行左填充
    #         if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #             # 创建一个全零张量，其形状为(max_len - cur_len, cur_new_embed.shape[1])，数据类型和设备与cur_new_embed相同
    #             new_input_embeds_padded.append(torch.cat((
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
    #                 cur_new_embed # 将当前嵌入追加到全零张量的右侧
    #             ), dim=0))
    #
    #             # 如果当前嵌入长度大于0，则进行以下操作
    #             if cur_len > 0:
    #                 # 将新的标签填充到new_labels_padded的相应位置
    #                 new_labels_padded[i, -cur_len:] = cur_new_labels
    #                 # 更新attention_mask，将当前嵌入对应的部分设置为True
    #                 attention_mask[i, -cur_len:] = True
    #                 # 更新position_ids，生成一个从0到cur_len-1的张量，并将其填充到position_ids的相应位置
    #                 position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #         else:
    #             # 如果配置中的tokenizer_padding_side不为'left'，则进行右填充
    #             new_input_embeds_padded.append(torch.cat((
    #                 cur_new_embed, # # 将当前嵌入追加到全零张量的左侧
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
    #             ), dim=0))
    #             # 如果当前嵌入长度大于0，则进行以下操作
    #             if cur_len > 0:
    #                 # 将新的标签填充到new_labels_padded的相应位置，从左侧开始填充
    #                 new_labels_padded[i, :cur_len] = cur_new_labels
    #                 # 更新attention_mask，将当前嵌入对应的部分设置为True，从左侧开始
    #                 attention_mask[i, :cur_len] = True
    #                 # 更新position_ids，生成一个从0到cur_len-1的张量，并将其填充到position_ids的相应位置，从左侧开始
    #                 position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #
    #     # 使用torch.stack函数将new_input_embeds_padded列表中的张量沿第0维堆叠起来，生成新的张量new_input_embeds
    #     new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    #
    #     # 如果_labels为None，则new_labels也设置为None；否则，将new_labels_padded赋值给new_labels
    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded
    #
    #     # 如果_attention_mask为None，则attention_mask也设置为None；否则，将attention_mask转换为_attention_mask的数据类型
    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
    #     # 如果_position_ids为None，则position_ids也设置为None
    #     if _position_ids is None:
    #         position_ids = None
    #     # 返回None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    #

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_lengths, audio, audio_lengths
    ):
        # 获取设备信息
        device = input_ids.device if input_ids is not None else self.device
        
        # 如果没有多模态输入,直接返回原始输入
        if images is None and audio is None:
            return input_ids, position_ids, attention_mask, past_key_values, labels
        
        # 初始化新的输入嵌入和标签列表
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_audio_idx = 0
        
        # 处理每个批次的输入
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 计算图像和音频标记的数量
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            num_audio = (cur_input_ids == AUDIO_TOKEN_INDEX).sum().item()
            
            # 如果没有图像或音频标记,直接添加当前输入
            if num_images == 0 and num_audio == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
            
            # 分割输入,处理图像和音频标记
            token_indices = ([-1] + 
                            torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() +
                            torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0].tolist() +
                            [cur_input_ids.shape[0]])
            token_indices.sort()
            
            cur_input_ids_list = []
            cur_labels_list = []
            for i in range(len(token_indices) - 1):
                cur_input_ids_list.append(cur_input_ids[token_indices[i] + 1:token_indices[i + 1]])
                cur_labels_list.append(labels[batch_idx][token_indices[i] + 1:token_indices[i + 1]])
            
            # 计算每个文本部分的长度
            split_sizes = [x.shape[0] for x in cur_labels_list]
            
            # 嵌入文本部分
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_list))
            cur_input_embeds_no_mmtoken = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # 处理文本、图像和音频
            cur_new_input_embeds = []
            cur_new_labels = []
            
            for i, (cur_input_ids_chunk, cur_labels_chunk) in enumerate(zip(cur_input_ids_list, cur_labels_list)):
                cur_new_input_embeds.append(cur_input_embeds_no_mmtoken[i])
                cur_new_labels.append(cur_labels_chunk)
                
                # 处理图像标记
                if i < num_images + num_audio and cur_input_ids_chunk[-1] == IMAGE_TOKEN_INDEX:
                # if i < num_images and cur_input_ids_chunk[-1] == IMAGE_TOKEN_INDEX:
                    cur_image_features = self.encode_images(images[cur_image_idx].unsqueeze(0))[0]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, 
                                                    device=cur_labels_chunk.device, dtype=cur_labels_chunk.dtype))
                
                # 处理音频标记
                elif i < num_images + num_audio and cur_input_ids_chunk[-1] == AUDIO_TOKEN_INDEX:
                    cur_audio_features = self.encode_audio(audio[cur_audio_idx].unsqueeze(0), 
                                                        audio_lengths[cur_audio_idx].unsqueeze(0))[0]
                    cur_audio_idx += 1
                    cur_new_input_embeds.append(cur_audio_features)
                    cur_new_labels.append(torch.full((cur_audio_features.shape[0],), IGNORE_INDEX, 
                                                    device=cur_labels_chunk.device, dtype=cur_labels_chunk.dtype))
            
            # 将处理后的嵌入和标签移至正确的设备
            cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
            
            # 合并新的输入嵌入和标签
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # 截断序列到最大长度
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        
        # 填充处理
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=device)
        
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
        
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        # 返回处理后的数据
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels_padded




    def load_llm(self, **kwargs):
        """
        加载预训练语言模型。

        Args:
            **kwargs: 关键字参数，包括：
                - model_name_or_path (str): 预训练语言模型的名称或路径。
                - pretrained_llm_path (str): 预训练语言模型的路径，优先级高于`model_name_or_path`。
                - torch_dtype (torch.dtype): 语言模型的torch数据类型。

        Returns:
            None

        """
        # 从kwargs中获取模型名称或路径
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        # 从kwargs中获取预训练的语言模型路径
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        # 如果提供了预训练的语言模型路径，则使用该路径作为模型名称
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        # 如果模型名称不为空，则从预训练的模型加载语言模型
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        # 打印加载的语言模型名称
        print('loading language model from ', language_model_name)
        # 设置语言模型的requires_grad属性为False，表示在训练过程中不更新其参数
        self.language_model.requires_grad_(False)
        # 设置配置中的文本配置的torch_dtype，如果没有提供则默认为None
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        # 获取tokenizer的pad_token属性，如果没有则默认为None
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        # 获取tokenizer的pad_token_id属性，如果没有则默认为None
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
        
    def load_vision_tower(self, **kwargs):
        """
        加载视觉塔模型。

        Args:
            **kwargs: 可选参数，用于指定模型名称或路径以及其他加载模型所需的参数。

        Returns:
            None

        """
        # 从kwargs中获取模型名称或路径
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        # 加载指定名称或路径的模型
        self.vision_tower.load_model(vision_tower_name, **kwargs)

        
    def load_connector(self, **kwargs):
        """
        加载连接器的模型。

        Args:
            **kwargs: 加载模型时需要的参数，具体参数取决于连接器的类型。

        Returns:
            None

        """
        # 调用连接器对象的load_model方法，加载模型。
        # **kwargs 允许传递任意数量的关键字参数给load_model方法。
        # 这些参数可能包括模型的路径、配置选项等。
        self.connector.load_model(**kwargs)

            

        
        
