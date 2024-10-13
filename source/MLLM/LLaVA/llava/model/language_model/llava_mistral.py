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
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# 定义一个继承自MistralConfig的类LlavaMistralConfig，用于配置LlavaMistral模型
class LlavaMistralConfig(MistralConfig):
    # 指定模型类型为llava_mistral
    model_type = "llava_mistral"

# 继承LlavaMetaModel和MistralModel，定义LlavaMistralModel类，该类整合了Llava和Mistral的模型结构
class LlavaMistralModel(LlavaMetaModel, MistralModel):
    # 指定配置类为LlavaMistralConfig
    config_class = LlavaMistralConfig

    # 初始化函数，用于创建LlavaMistralModel类的实例
    def __init__(self, config: MistralConfig):
        # 调用父类的初始化函数，用给定的配置初始化模型
        super(LlavaMistralModel, self).__init__(config)


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    """
    LlavaMistralForCausalLM 类继承自 MistralForCausalLM 和 LlavaMetaForCausalLM，用于创建一个适用于因果语言模型的 LlavaMistral 模型。
    """
    config_class = LlavaMistralConfig

    def __init__(self, config):
        """
        初始化函数，用于创建模型的实例。

        参数:
        - config: 模型的配置对象，用于设置模型的各类参数。
        """
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        """
        获取模型实例的方法。

        返回:
        - self.model: 当前的 LlavaMistral 模型实例。
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
        定义模型的前向传播方法。

        参数:
        - input_ids: 输入的令牌ID序列。
        - attention_mask: 注意力掩码，用于指示每个位置是否应被忽略。
        - position_ids: 位置ID序列。
        - past_key_values: 之前的键值对，用于加速解码。
        - inputs_embeds: 输入的嵌入向量。
        - labels: 目标标签，用于计算损失。
        - use_cache: 是否使用缓存的键值对。
        - output_attentions: 是否输出注意力权重。
        - output_hidden_states: 是否输出隐藏状态。
        - images: 输入的图像数据。
        - image_sizes: 输入图像的尺寸信息。
        - return_dict: 是否以字典形式返回输出。

        返回:
        - Union[Tuple, CausalLMOutputWithPast]: 返回一个元组或 CausalLMOutputWithPast 对象，包含模型的输出结果。
        """
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

        return super().forward(
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
        生成方法，用于根据输入生成文本。

        参数:
        - inputs: 输入的文本数据。
        - images: 输入的图像数据。
        - image_sizes: 输入图像的尺寸信息。
        - **kwargs: 其他关键词参数，如生成长度、温度等。

        返回:
        - Union[GenerateOutput, torch.LongTensor]: 返回生成的结果，可以是 GenerateOutput 对象或长整型张量。
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

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
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        准备生成的输入数据。

        参数:
        - input_ids: 输入的令牌ID序列。
        - past_key_values: 之前的键值对，用于加速解码。
        - inputs_embeds: 输入的嵌入向量。
        - **kwargs: 其他关键词参数。

        返回:
        - 修改后的输入数据字典。
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
# 注册配置类和模型类
AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
