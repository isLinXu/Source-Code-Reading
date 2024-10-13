#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# 定义LlavaConfig类，继承自LlamaConfig
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"  # 指定模型类型为"llava_llama"

# 定义LlavaLlamaModel类，继承自LlavaMetaModel和LlamaModel
class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig  # 指定配置类为LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config) # 调用父类的构造函数


# 定义LlavaLlamaForCausalLM类，继承自LlamaForCausalLM和LlavaMetaForCausalLM
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig  # 指定配置类为LlavaConfig

    def __init__(self, config):
        """
        初始化LlavaLlamaForCausalLM类的实例。

        Args:
            config (LlamaConfig): 模型的配置对象。
        """
        super(LlamaForCausalLM, self).__init__(config)  # 调用父类的构造函数
        self.model = LlavaLlamaModel(config)            # 初始化LlavaLlamaModel实例
        self.pretraining_tp = config.pretraining_tp     # 获取预训练参数
        self.vocab_size = config.vocab_size             # 获取词汇表大小
        # 初始化线性层，用于语言模型的头部，输入为隐藏层大小，输出为词汇表大小，不使用权重偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        """
        返回LlavaLlamaModel实例。

        Returns:
            LlavaLlamaModel: 模型的实例。
        """
        return self.model

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
        前向传播函数，定义了模型如何处理输入数据并产生输出。

        Args:
            input_ids (torch.LongTensor): 输入的token id序列。
            attention_mask (Optional[torch.Tensor]): 注意力掩码，用于指示哪些位置需要被注意力机制考虑。
            position_ids (Optional[torch.LongTensor]): 位置编码的id序列。
            past_key_values (Optional[List[torch.FloatTensor]]): 过去的key和value值，用于缓存。
            inputs_embeds (Optional[torch.FloatTensor]): 输入的嵌入表示。
            labels (Optional[torch.LongTensor]): 标签，用于监督学习。
            use_cache (Optional[bool]): 是否使用缓存。
            output_attentions (Optional[bool]): 是否输出注意力权重。
            output_hidden_states (Optional[bool]): 是否输出隐藏状态。
            images (Optional[torch.FloatTensor]): 图像数据。
            image_sizes (Optional[List[List[int]]]): 图像尺寸。
            return_dict (Optional[bool]): 是否返回字典格式的结果。

        Returns:
            Any: 根据return_dict参数决定返回值格式。
        """
        # 如果输入嵌入（inputs_embeds）为None，则调用prepare_inputs_labels_for_multimodal方法准备多模态输入和标签
        if inputs_embeds is None:
            # 准备多模态输入和标签，包括输入ID、位置ID、注意力掩码、过去的键值对、输入嵌入和标签
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

        # 调用父类的forward方法，传递准备好的输入参数
        return super().forward(
            input_ids=input_ids,  # 输入ID
            attention_mask=attention_mask,  # 注意力掩码
            position_ids=position_ids,  # 位置ID
            past_key_values=past_key_values,  # 过去的键值对
            inputs_embeds=inputs_embeds,  # 输入嵌入
            labels=labels,  # 标签
            use_cache=use_cache,  # 是否使用缓存
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict  # 是否返回字典
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 从kwargs中移除position_ids和attention_mask
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        # 如果kwargs中包含inputs_embeds，则抛出NotImplementedError异常
        if "inputs_embeds" in kwargs:
            # 不支持inputs_embeds参数
            raise NotImplementedError("`inputs_embeds` is not supported")
        # 如果images不为空
        if images is not None:
            # 准备多模态输入、位置ID、注意力掩码等
            # 注意：这里忽略了prepare_inputs_labels_for_multimodal的返回值中的部分元素，用_代替
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
            # 如果images为空，则通过模型的embed_tokens方法获取inputs_embeds
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 调用父类的generate方法，传入位置ID、注意力掩码和inputs_embeds等参数
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        为生成任务准备输入数据

        此函数扩展了基类的prepare_inputs_for_generation函数，用于处理与图像相关的输入数据
        它接收输入的token id序列、过去的key-value对、输入的嵌入向量，以及包含图像数据的额外关键字参数

        参数:
            input_ids (torch.Tensor): 输入的token id序列
            past_key_values (tuple, optional): 过去的key-value对，用于生成过程中缓存计算结果，以提高效率，默认为None
            inputs_embeds (torch.Tensor, optional): 输入的嵌入向量，如果提供，则不使用input_ids，默认为None
            **kwargs: 额外的关键字参数，可能包含images（图像数据）和image_sizes（图像尺寸）等信息

        返回:
            dict: 包含所有必要输入数据的字典，可能包括token ids、过去的key-value对、输入的嵌入向量、图像数据等
        """
        # 获取图像数据和图像尺寸，如果存在的话
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)

        # 调用基类的prepare_inputs_for_generation函数，准备其他基本输入数据
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        # 如果图像数据存在，则添加到输入数据中
        if images is not None:
            inputs['images'] = images

        # 如果图像尺寸数据存在，则添加到输入数据中
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        # 返回包含所有必要输入数据的字典
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
