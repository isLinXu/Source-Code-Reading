#    Copyright 2023 Haotian Liu
#    版权所有 2023 刘昊天
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    根据 Apache 许可证 2.0 版本授权
#    you may not use this file except in compliance with the License.
#    除非符合许可证要求，否则不得使用此文件
#    You may obtain a copy of the License at
#    您可以通过以下网址获取许可证副本：
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    除非适用法律要求或书面同意，否则软件
#    distributed under the License is distributed on an "AS IS" BASIS,
#    按"原样"分发，没有任何形式的明示或暗示保证
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    请参阅许可证中的特定语言条款和权限说明
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# 基于LLaVA项目修改 (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# 版权所有 2024 李彦伟
# ------------------------------------------------------------------------

# 导入类型提示模块
from typing import List, Optional, Tuple, Union

# 导入PyTorch核心库
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn

try:
    # 尝试导入Hugging Face Transformers库组件
    # AutoConfig: 自动配置加载器
    # AutoModelForCausalLM: 自动因果语言模型加载器
    # Gemma系列模型相关类
    from transformers import AutoConfig, AutoModelForCausalLM, \
                            GemmaConfig, GemmaModel, GemmaForCausalLM
except:
    # 导入失败时提示升级Transformers版本
    print("New model not imported. Try to update Transformers to 4.38.0 or later.")
    print("新模型未导入，请尝试升级Transformers到4.38.0或更高版本")

# 导入Transformers模型输出类型
from transformers.modeling_outputs import CausalLMOutputWithPast  # 带历史记录的因果语言模型输出
from transformers.generation.utils import GenerateOutput  # 生成输出容器
from transformers.generation.utils import logging  # 日志工具

# 从上级目录导入MGM基础架构
from ..mgm_arch import MGMMetaModel, MGMMetaForCausalLM  # MGM元模型和因果语言模型基类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

class MGMConfig(GemmaConfig):
    """自定义MGM模型配置类，继承自Gemma配置"""
    model_type = "mgm_gemma"  # 注册新的模型类型标识


class MGMGemmaModel(MGMMetaModel, GemmaModel):
    """MGM模型实现，组合元模型和原始Gemma模型"""
    config_class = MGMConfig  # 指定关联的配置类
    
    def __init__(self, config: GemmaConfig):
        """初始化方法，继承父类构造器"""
        super(MGMGemmaModel, self).__init__(config)  # 调用GemmaModel的初始化


class MGMGemmaForCausalLM(GemmaForCausalLM, MGMMetaForCausalLM):
    """用于因果语言建模的完整MGM模型"""
    config_class = MGMConfig  # 指定配置类

    def __init__(self, config):
        """初始化语言模型头部"""
        # 显式调用GemmaForCausalLM的父类初始化（避免多继承问题）
        super(GemmaForCausalLM, self).__init__(config)
        # 初始化多模态模型主体
        self.model = MGMGemmaModel(config)
        # 设置词汇表大小（从配置中获取）
        self.vocab_size = config.vocab_size
        # 创建语言模型头部（隐藏层到词汇表的线性投影）
        self.lm_head = nn.Linear(
            config.hidden_size,  # 输入维度：隐藏层大小
            config.vocab_size,   # 输出维度：词汇表大小
            bias=False           # 禁用偏置项
        )

        # 权重初始化后处理（来自Transformers的特定初始化逻辑）
        self.post_init()

    def get_model(self):
        """获取模型主体（用于嵌入层访问）"""
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,          # 输入的token ID序列
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码（指定有效token）
        position_ids: Optional[torch.LongTensor] = None, # 位置编码ID
        past_key_values: Optional[List[torch.FloatTensor]] = None, # 缓存的键值对（加速生成）
        inputs_embeds: Optional[torch.FloatTensor] = None, # 直接输入的嵌入表示
        labels: Optional[torch.LongTensor] = None,   # 训练标签
        use_cache: Optional[bool] = None,            # 是否使用键值缓存
        cache_position: Optional[torch.LongTensor] = None, # 缓存位置索引
        output_attentions: Optional[bool] = None,    # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None, # 是否返回所有隐藏状态
        images: Optional[torch.FloatTensor] = None,  # 主图像输入（形状：[batch, channels, H, W]）
        images_aux: Optional[torch.FloatTensor] = None, # 辅助图像输入
        return_dict: Optional[bool] = None,          # 是否以字典形式返回结果
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """前向传播流程，处理多模态输入"""

        # 当没有直接提供输入嵌入时，准备多模态输入
        if inputs_embeds is None:
            # 解包多模态预处理结果
            (
                input_ids,          # 处理后的token ID
                position_ids,       # 调整后的位置编码
                attention_mask,     # 更新后的注意力掩码
                past_key_values,    # 更新后的键值缓存
                inputs_embeds,      # 融合多模态的输入嵌入
                labels,             # 调整后的标签
            ) = self.prepare_inputs_labels_for_multimodal(  # 来自MGMMetaForCausalLM的方法
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,        # 主视觉输入
                images_aux     # 辅助视觉输入
            )

        # 调用父类（GemmaForCausalLM）的前向计算
        return super().forward(
            input_ids=input_ids,          # 传递处理后的输入
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # 使用融合后的嵌入
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()  # 禁用梯度计算（节省内存）
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,   # 文本输入张量
        images: Optional[torch.Tensor] = None,   # 图像输入张量
        images_aux: Optional[torch.FloatTensor] = None, # 辅助图像输入
        **kwargs,  # 其他生成参数（如max_length, temperature等）
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """生成文本的入口方法"""
        
        # 从kwargs中提取并移除位置ID（避免重复传递）
        position_ids = kwargs.pop("position_ids", None)
        # 从kwargs中提取并移除注意力掩码
        attention_mask = kwargs.pop("attention_mask", None)
        
        # 检查是否直接传入inputs_embeds（当前不支持）
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
            raise NotImplementedError("暂不支持直接传入inputs_embeds参数")

        # 处理多模态输入
        if images is not None:
            # 解包多模态预处理结果（忽略不需要的返回值）
            (
                inputs,          # 更新后的输入ID（可能为None）
                position_ids,    # 调整后的位置编码
                attention_mask,  # 更新后的注意力掩码
                _,               # 占位符（past_key_values）
                inputs_embeds,   # 融合多模态的嵌入表示
                _                # 占位符（labels）
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,          # 原始输入
                position_ids,
                attention_mask,
                None,            # past_key_values（生成时初始为None）
                None,            # labels（生成时不需要）
                images,          # 主图像
                images_aux       # 辅助图像
            )
        else:
            # 纯文本模式：直接获取token嵌入
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 调用父类（GemmaForCausalLM）的生成方法
        return super().generate(
            position_ids=position_ids,    # 传递调整后的位置编码
            attention_mask=attention_mask, # 更新后的注意力掩码
            inputs_embeds=inputs_embeds,  # 输入嵌入（文本或多模态）
            **kwargs                      # 其他生成参数
        )

    def prepare_inputs_for_generation(
        self, 
        input_ids,             # 当前生成的token ID序列
        past_key_values=None,  # 缓存的键值对
        inputs_embeds=None,    # 已有的输入嵌入
        **kwargs
    ):
        """准备生成所需的输入数据"""
        # 提取并移除图像相关参数
        images = kwargs.pop("images", None)      # 主图像
        images_aux = kwargs.pop("images_aux", None) # 辅助图像
        
        # 调用父类方法准备基础输入
        _inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            **kwargs
        )
        
        # 添加多模态参数到输入字典
        if images is not None:
            _inputs['images'] = images       # 注入主图像数据
        if images_aux is not None:
            _inputs['images_aux'] = images_aux # 注入辅助图像数据
            
        return _inputs

# 将自定义配置注册到AutoConfig系统
AutoConfig.register("mgm_gemma", MGMConfig)
# 将模型类关联到配置类，实现自动模型加载
AutoModelForCausalLM.register(MGMConfig, MGMGemmaForCausalLM)