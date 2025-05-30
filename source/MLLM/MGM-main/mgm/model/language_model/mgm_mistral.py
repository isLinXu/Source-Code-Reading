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
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# ------------------------------------------------------------------------
# 导入必要的库和模块
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.generation.utils import logging

from ..mgm_arch import MGMMetaModel, MGMMetaForCausalLM

logger = logging.get_logger(__name__)

# 定义MGM的配置类，继承自MistralConfig
class MGMConfig(MistralConfig):
    model_type = "mgm_mistral"  # 指定模型类型标识符


# 定义MGM的Mistral模型实现，继承自MGMMetaModel和MistralModel
class MGMMistralModel(MGMMetaModel, MistralModel):
    config_class = MGMConfig  # 指定使用的配置类
    
    def __init__(self, config: MistralConfig):
        super(MGMMistralModel, self).__init__(config)
        # self.max_pos_idx = 0

# 定义用于因果语言建模的MGM模型，继承自MistralForCausalLM和MGMMetaForCausalLM
class MGMMistralForCausalLM(MistralForCausalLM, MGMMetaForCausalLM):
    config_class = MGMConfig

    def __init__(self, config):
        # 初始化父类（MistralForCausalLM）
        super(MistralForCausalLM, self).__init__(config)
        self.model = MGMMistralModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理（
        self.post_init()

    def get_model(self):
        return self.model

    # 前向传播函数，处理多模态输入
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
        images: Optional[torch.FloatTensor] = None,     # 主图像输入
        images_aux: Optional[torch.FloatTensor] = None, # 辅助图像输入
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # 如果没有输入嵌入，准备多模态输入
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(  # 准备多模态输入
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_aux
            )

        # 调用父类的前向传播
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

    # 生成方法（禁用梯度计算）
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported") # 不支持直接输入嵌入

        # 处理图像输入
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal( # 准备多模态输入
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                images_aux
            )
        else:
            # 调用父类的生成方法
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 调用父类的生成方法
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    # 准备生成所需的输入
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)              # 提取图像输入
        images_aux = kwargs.pop("images_aux", None)      # 提取辅助图像输入
        _inputs = super().prepare_inputs_for_generation( # 调用父类方法
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # 添加图像参数到输入字典
        if images is not None:
            _inputs['images'] = images
        if images_aux is not None:
            _inputs['images_aux'] = images_aux
        return _inputs

# 注册自定义配置和模型到Auto类
AutoConfig.register("mgm_mistral", MGMConfig)
AutoModelForCausalLM.register(MGMConfig, MGMMistralForCausalLM)