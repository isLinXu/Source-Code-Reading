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


from typing import Optional, Tuple

import torch

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MptConfig, MptForCausalLM, MptModel
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMptConfig(MptConfig):
    """
    LlavaMpt模型的配置类，继承自MptConfig。
    该类指定了模型的类型为"llava_mpt"。
    """
    model_type = "llava_mpt"


class LlavaMptModel(LlavaMetaModel, MptModel):
    """
    LlavaMpt模型类，继承自LlavaMetaModel和MptModel。
    该类结合了Llava和MPT模型的特点，用于处理特定的模型架构。
    """
    config_class = LlavaMptConfig

    def __init__(self, config: MptConfig):
        """
        初始化模型配置，并对配置进行必要的调整。

        参数:
            config (MptConfig): 模型的配置对象，用于初始化模型参数。
        """
        # 调整配置中的hidden_size为d_model，以确保配置的一致性
        config.hidden_size = config.d_model
        # 调用父类的初始化方法，传入调整后的配置
        super(LlavaMptModel, self).__init__(config)

    def embed_tokens(self, x):
        """
        对输入的token进行嵌入操作。

        参数:
            x: 输入的token，可以是文本或其他类型的输入。

        返回:
            输入token的嵌入表示。
        """
        # 使用word embedding方法对输入token进行嵌入
        return self.wte(x)


class LlavaMptForCausalLM(MptForCausalLM, LlavaMetaForCausalLM):
    """
    LlavaMptForCausalLM 类结合了 MptForCausalLM 和 LlavaMetaForCausalLM 的特性，
    用于创建一个适用于因果语言模型的变压器模型。该模型支持梯度检查点。
    """
    config_class = LlavaMptConfig               # 定义该类使用的配置类
    supports_gradient_checkpointing = True      # 表明该类支持梯度检查点


    def __init__(self, config):
        """
        初始化函数，用于创建模型的参数和结构。

        参数:
        - config: 配置对象，包含模型的各种超参数和配置设置。
        """
        super(MptForCausalLM, self).__init__(config)    # 调用基类初始化

        self.transformer = LlavaMptModel(config)        # 创建变压器模型
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    def get_model(self):
        """
        获取变压器模型。

        返回:
        - self.transformer: 返回变压器模型实例。
        """
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        """
        设置模块的梯度检查点。

        参数:
        - module: 要设置的模块。
        - value: 是否启用梯度检查点，默认为 False。
        """
        if isinstance(module, LlavaMptModel):        # 检查模块是否为 LlavaMptModel 类型
            module.gradient_checkpointing = value    # 设置梯度检查点

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images=None):
        """
        定义模型的前向传播过程。

        参数:
        - input_ids: 输入的令牌ID序列。
        - past_key_values: 用于解码的过去键值对。
        - attention_mask: 注意力掩码，用于指示哪些位置应该被忽略。
        - inputs_embeds: 输入的嵌入向量。
        - labels: 语言模型的标签。
        - use_cache: 是否使用缓存来加速解码过程。
        - output_attentions: 是否输出注意力权重。
        - output_hidden_states: 是否输出隐藏状态。
        - return_dict: 是否以字典形式返回输出。
        - images: 输入的图像数据。

        返回:
        - 通过模型前向传播后的输出结果。
        """
        # 准备多模态输入和标签
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # 调用基类的前向传播函数
        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        """
        为生成任务准备输入数据。这个方法是为了在生成过程中，除了处理输入的文本ID序列，还能处理过去的关键值和输入的嵌入，
        并且支持通过关键字参数的形式接收额外的参数。

        :param input_ids: 输入的文本ID序列，是生成过程中的主要输入。
        :param past_key_values: 以往生成步中计算的关键值，用于加速生成过程和提高效率。
        :param inputs_embeds: 输入的嵌入表示，有时候用于替代输入的文本ID序列。
        :param kwargs: 关键字参数，用于接收其他可能对生成过程有影响的参数。
        :return: 准备好的输入数据，包括文本ID序列、过去的关键值、输入的嵌入表示以及其他额外的参数。
        """
        # 从关键字参数中提取图片数据，如果没有提供，则默认为None
        images = kwargs.pop("images", None)
        # 调用基类的prepare_inputs_for_generation方法，处理输入的文本ID序列、过去的关键值和输入的嵌入表示
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # 将图片数据添加到处理好的输入数据中
        _inputs['images'] = images
        # 返回准备好的输入数据
        return _inputs

    # 注册LLAVA-MPT配置类到AutoConfig，使得可以通过AutoConfig直接获取LLAVA-MPT的配置类
    AutoConfig.register("llava_mpt", LlavaMptConfig)
    # 注册LLAVA-MPT模型类到AutoModelForCausalLM，使得可以通过AutoModelForCausalLM直接获取LLAVA-MPT的模型类
    AutoModelForCausalLM.register(LlavaMptConfig, LlavaMptForCausalLM)
