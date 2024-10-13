from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from omni_speech.model.language_model.omni_speech_llama import OmniSpeechLlamaForCausalLM
from omni_speech.model.speech_generator.builder import build_speech_generator
from omni_speech.model.speech_generator.generation import GenerationWithCTC


class OmniSpeech2SConfig(LlamaConfig):
    # 定义模型类型为 "omni_speech2s_llama"
    model_type = "omni_speech2s_llama"


class OmniSpeech2SLlamaForCausalLM(OmniSpeechLlamaForCausalLM, GenerationWithCTC):
    # 指定配置类为 OmniSpeech2SConfig
    config_class = OmniSpeech2SConfig

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化权重并应用最终处理
        # Initialize weights and apply final processing
        self.post_init()
        # 如果配置中包含 speech_generator_type，则初始化语音生成器
        if hasattr(config, "speech_generator_type"):
            self.speech_generator = build_speech_generator(config)
    
    def initialize_speech_generator(self, model_args):
        """
        初始化语音生成器的相关配置
        :param model_args: 模型参数
        """
        # 从模型参数中获取或设置默认的语音生成器类型
        self.config.speech_generator_type = getattr(model_args, 'speech_generator_type', 'ctc')
        # 获取或设置默认的CTC解码器配置
        self.config.ctc_decoder_config = getattr(model_args, 'ctc_decoder_config', '(4,4096,32,11008)')
        # 获取或设置默认的CTC上采样因子
        self.config.ctc_upsample_factor = getattr(model_args, 'ctc_upsample_factor', 1)
        # 获取或设置默认的CTC损失权重
        self.config.ctc_loss_weight = getattr(model_args, 'ctc_loss_weight', 1.0)
        # 获取或设置默认的单位词汇表大小
        self.config.unit_vocab_size = getattr(model_args, 'unit_vocab_size', 1000)
        # 获取或设置是否仅调整语音生成器
        self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', False)
        # 如果语音生成器尚未初始化，则根据配置构建语音生成器
        if getattr(self, "speech_generator", None) is None:
            self.speech_generator = build_speech_generator(self.config)

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
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播函数，用于计算模型的输出和损失。

        参数:
            input_ids: 输入的token id张量。
            attention_mask: 注意力掩码张量。
            position_ids: 位置编码张量。
            past_key_values: 过去的key和value张量列表。
            inputs_embeds: 输入的嵌入张量。
            labels: 标签张量。
            use_cache: 是否使用缓存。
            output_attentions: 是否输出注意力。
            output_hidden_states: 是否输出隐藏状态。
            speech: 语音输入张量。
            speech_lengths: 语音长度张量。
            tgt_units: 目标单位张量。
            return_dict: 是否返回字典格式的结果。
            cache_position: 缓存位置张量。

        返回:
            CausalLMOutputWithPast对象，包含损失、logits、过去的key和value、隐藏状态和注意力。
        """
        # 如果输入嵌入为None，则准备输入和标签
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )
        # 如果处于训练模式，
        if self.training:
            # 如果只调整语音生成器
            if self.tune_speech_generator_only:
                # 使用torch.no_grad()上下文管理器来禁用梯度计算，以节省内存和提高速度
                with torch.no_grad():
                    # 调用父类的forward方法计算llama模型的输出
                    llama_output = super(OmniSpeechLlamaForCausalLM, self).forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=True,
                        return_dict=return_dict
                    )
                # 计算语音生成器的损失
                loss = self.speech_generator(llama_output['hidden_states'][-1], labels, tgt_units)
            else:
                # 调用父类的forward方法计算llama模型的输出
                llama_output = super(OmniSpeechLlamaForCausalLM, self).forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict
                )
                # 获取语言模型的损失
                lm_loss = llama_output.loss
                # 计算语音生成器的损失
                ctc_loss = self.speech_generator(llama_output['hidden_states'][-1], labels, tgt_units)
                # 计算总损失，结合语言模型损失和语音生成器损失
                loss = lm_loss + ctc_loss * self.config.ctc_loss_weight
        else:
            # 如果模型不处于训练模式，调用父类的forward方法计算llama模型的输出
            llama_output = super(OmniSpeechLlamaForCausalLM, self).forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict
            )
            # 获取语言模型的损失
            loss = llama_output.loss

        # 返回CausalLMOutputWithPast对象，包含损失、logits、过去的key和value、隐藏状态和注意力
        return CausalLMOutputWithPast(
            loss=loss,
            logits=llama_output.logits,
            past_key_values=llama_output.past_key_values,
            hidden_states=llama_output.hidden_states,
            attentions=llama_output.attentions
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        streaming_unit_gen=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        生成函数，用于生成模型的输出序列。

        参数:
            inputs: 输入张量。
            speech: 语音输入张量。
            speech_lengths: 语音长度张量。
            streaming_unit_gen: 是否流式生成。
            **kwargs: 其他关键字参数。

        返回:
            GenerateOutput对象或torch.LongTensor类型的生成序列。
        """
        # 从kwargs中提取position_ids和attention_mask，如果不存在则默认为None
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        # 如果存在inputs_embeds参数，则抛出不支持的异常
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # 如果speech不为None，则调用prepare_inputs_labels_for_speech_and_text方法处理输入
        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            # 如果speech为None，则使用embed_tokens方法获取输入的嵌入表示
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # 调用GenerationWithCTC的generate方法生成序列
        outputs = GenerationWithCTC.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            streaming_unit_gen=streaming_unit_gen,
            **kwargs
        )
        # 处理隐藏状态并预测CTC
        hidden_states = outputs['hidden_states']
        hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)
        ctc_pred = self.speech_generator.predict(hidden_states.squeeze(0))
        # 返回生成的序列和CTC预测结果
        return outputs.sequences, ctc_pred

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        准备用于生成模型的输入数据。

        参数:
            input_ids (List[int]): 输入的token ID列表。
            past_key_values (Optional[Tuple[torch.Tensor]]): 上一个时间步的key和value张量，用于Transformer的自注意力机制。
            inputs_embeds (Optional[torch.Tensor]): 输入的嵌入张量。
            **kwargs: 其他可能的关键字参数。

        返回:
            Dict[str, Any]: 包含所有必要输入数据的字典。
        """
        # 从kwargs中弹出speech和speech_lengths参数，如果没有提供则默认为None
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)

        # 调用父类的方法来准备基本的输入数据
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        # 如果提供了speech数据，则将其添加到输入字典中
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs

# 注册配置和模型类
AutoConfig.register("omni_speech2s_llama", OmniSpeech2SConfig)
AutoModelForCausalLM.register(OmniSpeech2SConfig, OmniSpeech2SLlamaForCausalLM)
