# 版权声明和法律条款（中英双语）
#    Copyright 2023 Haotian Liu
#    版权所有 2023 刘昊天（保留所有权利）
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    根据 Apache 许可证 2.0 版本授权（法律约束声明）
#    you may not use this file except in compliance with the License.
#    除非符合许可证要求，否则不得使用此文件（使用限制说明）
#    You may obtain a copy of the License at
#    您可以通过以下网址获取许可证副本：（法律条款指引）
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    除非适用法律要求或书面同意，否则软件（免责声明开始）
#    distributed under the License is distributed on an "AS IS" BASIS,
#    按"原样"分发，没有任何形式的明示或暗示保证（质量担保说明）
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    请参阅许可证中的特定语言条款和权限说明（详细条款指引）
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# 基于LLaVA项目修改 (https://github.com/haotian-liu/LLaVA)（项目来源声明）
# Copyright 2024 Yanwei Li
# 版权所有 2024 李彦伟（当前项目版权声明）
# ------------------------------------------------------------------------

# 类型提示导入（增强代码可读性和静态检查）
from typing import List, Optional, Tuple, Union  # 列表/可选/元组/联合类型

# PyTorch核心库导入
import torch  # 基础张量操作
import torch.nn as nn  # 神经网络模块

# Transformers组件导入（带异常处理）
try:
    from transformers import (
        AutoConfig,           # 自动配置加载器（根据模型名称加载配置）
        AutoModelForCausalLM, # 自动因果语言模型加载器
        GemmaConfig,          # Gemma模型配置类
        GemmaModel,           # Gemma基础模型实现
        GemmaForCausalLM      # Gemma因果语言模型
    )
except ImportError:
    # 兼容性处理：当Transformers版本过低时提示升级
    print("New model not imported. Try to update Transformers to 4.38.0 or later.")
    print("新模型未导入，请尝试升级Transformers到4.38.0或更高版本")

# 模型输出类型导入
from transformers.modeling_outputs import CausalLMOutputWithPast  # 带历史键值的输出结构
from transformers.generation.utils import GenerateOutput  # 生成结果容器
from transformers.generation.utils import logging  # 日志工具

# 多模态基础架构导入（相对路径）
from ..mgm_arch import (
    MGMMetaModel,       # 多模态元模型（视觉编码器集成）
    MGMMetaForCausalLM  # 多模态因果语言模型基类
)

# 日志记录器初始化
logger = logging.get_logger(__name__)  # 获取与当前模块同名的日志记录器

class MGMConfig(GemmaConfig):
    """MGM自定义配置类（继承Gemma配置）
    
    扩展功能：
    - 添加多模态处理相关配置参数
    - 注册新的模型类型标识符
    """
    model_type = "mgm_gemma"  # 注册到AutoConfig系统的新模型类型标识

class MGMGemmaModel(MGMMetaModel, GemmaModel):
    """多模态Gemma模型实现（继承自Meta模型和原始Gemma模型）
    
    功能：
    - 整合视觉编码器和语言模型
    - 处理多模态输入的特征融合
    """
    config_class = MGMConfig  # 指定关联的配置类
    
    def __init__(self, config: GemmaConfig):
        """模型初始化构造器
        
        参数：
        config : GemmaConfig
            模型配置对象，包含隐藏层维度等参数
        """
        super(MGMGemmaModel, self).__init__(config)  # 调用GemmaModel的初始化

class MGMGemmaForCausalLM(GemmaForCausalLM, MGMMetaForCausalLM):
    """多模态因果语言模型（支持图像条件生成）
    
    功能扩展：
    - 处理图像和文本的联合输入
    - 生成时支持多模态条件
    """
    config_class = MGMConfig  # 绑定配置类

    def __init__(self, config):
        """语言模型初始化
        
        参数：
        config : MGMConfig
            包含模型所有配置参数的对象
        """
        # 显式调用GemmaForCausalLM的父类初始化（解决多继承问题）
        super(GemmaForCausalLM, self).__init__(config)
        
        # 初始化多模态主干网络
        self.model = MGMGemmaModel(config)  # 创建模型实例
        
        # 词汇表参数设置
        self.vocab_size = config.vocab_size  # 从配置获取词汇表大小
        
        # 语言模型头部（隐藏层到词汇表的投影）
        self.lm_head = nn.Linear(
            config.hidden_size,  # 输入维度：Transformer隐藏层维度（如768）
            config.vocab_size,   # 输出维度：词汇表大小（如32000）
            bias=False           # 禁用偏置项（与原始模型保持一致）
        )

        # 权重后初始化（应用特殊初始化策略）
        self.post_init()  # 来自Transformers的初始化方法

    def get_model(self):
        """获取主干模型（用于访问嵌入层）"""
        return self.model  # 返回模型实例

    def forward(
        self,
        input_ids: torch.LongTensor = None,          # 输入token IDs [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码 [batch_size, seq_len]
        position_ids: Optional[torch.LongTensor] = None, # 位置编码 [batch_size, seq_len]
        past_key_values: Optional[List[torch.FloatTensor]] = None, # 历史键值缓存（加速生成）
        inputs_embeds: Optional[torch.FloatTensor] = None, # 预计算嵌入 [batch_size, seq_len, hidden_size]
        labels: Optional[torch.LongTensor] = None,   # 训练标签 [batch_size, seq_len]
        use_cache: Optional[bool] = None,            # 是否使用键值缓存（推理时建议True）
        cache_position: Optional[torch.LongTensor] = None, # 缓存位置索引（处理长序列）
        output_attentions: Optional[bool] = None,    # 是否返回注意力权重（用于可视化）
        output_hidden_states: Optional[bool] = None, # 是否返回所有隐藏状态（用于特征提取）
        images: Optional[torch.FloatTensor] = None,  # 主图像输入 [batch_size, channels, height, width]
        images_aux: Optional[torch.FloatTensor] = None, # 辅助图像输入（多图像场景）
        return_dict: Optional[bool] = None,          # 是否返回字典格式（兼容新旧版本）
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """前向传播（处理多模态输入）
        
        返回：
        Union[Tuple, CausalLMOutputWithPast]
            包含损失、logits、隐藏状态等的输出对象
        """

        # 多模态输入预处理（当未提供预计算嵌入时）
        if inputs_embeds is None:
            # 解包多模态预处理结果
            (
                input_ids,        # 处理后的token IDs（可能包含图像特殊标记）
                position_ids,     # 调整后的位置编码（考虑图像token位置）
                attention_mask,   # 扩展的注意力掩码（包含图像区域）
                past_key_values,  # 更新的键值缓存（包含视觉特征）
                inputs_embeds,    # 融合后的嵌入表示（文本+视觉）
                labels            # 对齐后的训练标签
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,     # 主图像特征（经过视觉编码器处理）
                images_aux  # 辅助图像特征（可选）
            )

        # 调用父类前向传播（执行Transformer计算）
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()  # 禁用梯度计算（推理优化）
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,   # 输入token IDs [batch_size, seq_len]
        images: Optional[torch.Tensor] = None,   # 图像输入 [batch_size, channels, H, W]
        images_aux: Optional[torch.FloatTensor] = None, # 辅助图像输入
        **kwargs  # 生成参数（max_length, temperature, top_p等）
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """多模态条件文本生成入口
        
        返回：
        Union[GenerateOutput, torch.LongTensor]
            生成结果（包含序列和分数）或纯token ID张量
        """
        
        # 参数预处理
        position_ids = kwargs.pop("position_ids", None)  # 提取并移除位置参数
        attention_mask = kwargs.pop("attention_mask", None)  # 提取并移除注意力掩码
        
        # 检查不支持的参数
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds`参数暂不支持")

        # 多模态输入处理
        if images is not None:
            # 解包多模态预处理结果
            (
                inputs,         # 更新后的token IDs（可能为None）
                position_ids,   # 调整后的位置编码
                attention_mask, # 扩展的注意力掩码
                _,              # 占位符（past_key_values）
                inputs_embeds,  # 融合后的嵌入表示
                _               # 占位符（labels）
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,  # 初始生成无历史缓存
                None,  # 生成任务无需标签
                images,
                images_aux
            )
        else:
            # 纯文本模式：直接获取token嵌入
            inputs_embeds = self.model.embed_tokens(inputs)

        # 执行生成（继承Gemma的生成策略）
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.LongTensor,       # 当前生成的token序列 [batch, seq]
        past_key_values: Optional[List] = None, # 历史键值缓存（加速生成）
        inputs_embeds: Optional[torch.Tensor] = None, # 预计算嵌入
        **kwargs
    ):
        """生成输入预处理（注入多模态参数）
        
        返回：
        dict
            包含所有生成所需参数的字典
        """
        # 提取多模态参数
        images = kwargs.pop("images", None)      # 主图像
        images_aux = kwargs.pop("images_aux", None) # 辅助图像
        
        # 调用父类方法准备基础输入
        _inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
        # 注入多模态参数
        if images is not None:
            _inputs['images'] = images       # 添加主图像到输入字典
        if images_aux is not None:
            _inputs['images_aux'] = images_aux # 添加辅助图像
        
        return _inputs  # 返回完整输入字典

# 模型注册到Transformers系统
AutoConfig.register("mgm_gemma", MGMConfig)  # 注册配置类
AutoModelForCausalLM.register(MGMConfig, MGMGemmaForCausalLM)  # 注册模型类