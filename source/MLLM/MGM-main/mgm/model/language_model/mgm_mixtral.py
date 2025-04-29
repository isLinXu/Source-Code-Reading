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

# 导入类型提示模块
from typing import List, Optional, Tuple, Union
# 导入PyTorch核心库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入Hugging Face Transformers相关组件
from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM
# 导入模型输出类型
from transformers.modeling_outputs import CausalLMOutputWithPast
# 导入生成相关工具
from transformers.generation.utils import GenerateOutput
from transformers.generation.utils import logging
# 导入自定义的多模态模型架构
from ..mgm_arch import MGMMetaModel, MGMMetaForCausalLM

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 自定义配置类（继承自MistralConfig）
class MGMConfig(MistralConfig):
    model_type = "mgm_mistral"  # 模型类型标识符，用于AutoConfig自动识别

# 多模态模型实现（继承元模型和原始Mistral模型）
class MGMMistralModel(MGMMetaModel, MistralModel):
    config_class = MGMConfig  # 指定关联的配置类
    
    def __init__(self, config: MistralConfig):
        # 调用父类初始化方法（MistralModel）
        super(MGMMistralModel, self).__init__(config)
        # 可扩展的位置索引参数（当前被注释）
        # self.max_pos_idx = 0

# 因果语言模型类（集成多模态能力）
class MGMMistralForCausalLM(MistralForCausalLM, MGMMetaForCausalLM):
    config_class = MGMConfig  # 绑定自定义配置类

    def __init__(self, config):
        # 初始化父类（MistralForCausalLM）
        super(MistralForCausalLM, self).__init__(config)
        # 实例化核心模型组件
        self.model = MGMMistralModel(config)
        # 预训练并行参数（当前被注释）
        # self.pretraining_tp = config.pretraining_tp
        # 设置词汇表大小
        self.vocab_size = config.vocab_size
        # 初始化语言模型头（hidden_size -> vocab_size）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重初始化及后处理（保留原始注释）
        self.post_init()

    def get_model(self):
        return self.model  # 获取底层模型实例

    # 前向传播方法（处理多模态输入）
    def forward(
        self,
        input_ids: torch.LongTensor = None,          # 文本token ID序列
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None, # 位置ID
        past_key_values: Optional[List[torch.FloatTensor]] = None, # 缓存键值
        inputs_embeds: Optional[torch.FloatTensor] = None, # 预计算嵌入
        labels: Optional[torch.LongTensor] = None,   # 训练标签
        use_cache: Optional[bool] = None,            # 是否使用缓存
        output_attentions: Optional[bool] = None,    # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        images: Optional[torch.FloatTensor] = None,  # 主视觉输入（Bx3xHxW）
        images_aux: Optional[torch.FloatTensor] = None, # 辅助视觉输入
        return_dict: Optional[bool] = None,          # 是否返回字典格式
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # 多模态输入预处理（当未提供预计算嵌入时）
        if inputs_embeds is None:
            # 解包预处理结果
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(  # 关键多模态处理方法
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,        # 主图像数据
                images_aux     # 辅助图像数据
            )

        # 调用父类前向传播（执行标准语言模型计算）
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

    # 文本生成方法（禁用梯度计算）
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,     # 输入token
        images: Optional[torch.Tensor] = None,     # 生成用图像输入
        images_aux: Optional[torch.FloatTensor] = None, # 辅助图像
        **kwargs,                                  # 其他生成参数
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 提取位置ID和注意力掩码
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        # 检查不支持的输入模式
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")  # 强制使用多模态预处理

        # 图像输入处理分支
        if images is not None:
            # 执行多模态输入预处理
            (
                inputs,
                position_ids,
                attention_mask,
                _,  # 忽略past_key_values
                inputs_embeds,  # 获取融合嵌入
                _   # 忽略labels
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,  # past_key_values
                None,  # labels
                images,
                images_aux
            )
        else:
            # 纯文本模式：直接获取token嵌入
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 调用父类生成方法（执行实际生成逻辑）
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,  # 传入处理后的嵌入
            **kwargs
        )

    # 生成准备方法（扩展父类功能）
    def prepare_inputs_for_generation(
        self, 
        input_ids,                # 当前生成的token序列
        past_key_values=None,     # 缓存的注意力键值
        inputs_embeds=None,       # 预计算嵌入
        **kwargs
    ):
        # 提取图像输入参数
        images = kwargs.pop("images", None)       # 主图像
        images_aux = kwargs.pop("images_aux", None) # 辅助图像
        # 调用父类准备方法
        _inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            **kwargs
        )
        # 注入多模态参数
        if images is not None:
            _inputs['images'] = images          # 添加主图像到输入字典
        if images_aux is not None:
            _inputs['images_aux'] = images_aux  # 添加辅助图像到输入字典
        return _inputs

# 将自定义配置注册到AutoConfig系统
AutoConfig.register("mgm_mistral", MGMConfig)
# 将模型类关联到配置类
AutoModelForCausalLM.register(MGMConfig, MGMMistralForCausalLM)