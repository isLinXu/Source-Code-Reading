# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
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

from abc import ABC, abstractmethod

import torch

from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector
from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.utils import lengths_to_padding_mask

class OmniSpeechMetaModel:
    """
    OmniSpeechMetaModel类用于构建和管理语音编码器和投影器模型组件。
    它的主要职责是根据配置初始化和获取这些组件，并支持从预训练模型加载权重。
    """

    def __init__(self, config):
        """
        初始化方法，用于创建模型的实例。

        参数:
        - config: 配置对象，包含模型的各种配置参数。
        """
        super(OmniSpeechMetaModel, self).__init__(config)

        # 检查配置中是否包含语音编码器的设置，如果有，则构建语音编码器和投影器
        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)

    def get_speech_encoder(self):
        """
        获取语音编码器实例。

        返回:
        - speech_encoder: 语音编码器实例，如果没有设置，则返回None。
        """
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder

    def initialize_speech_modules(self, model_args, fsdp=None):
        """
        根据传入的模型参数初始化或重新配置语音编码器和投影器。

        参数:
        - model_args: 包含模型参数的对象，用于获取语音编码器的配置。
        - fsdp: 可选参数，用于处理模型的并行训练相关设置。
        """
        # 从模型参数中获取语音编码器及其类型、投影器类型等配置
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        # 如果语音编码器未设置，则根据配置构建新的编码器
        if self.get_speech_encoder() is None:
            speech_encoder = build_speech_encoder(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder

        # 如果语音投影器未设置，则根据配置构建新的投影器
        if getattr(self, 'speech_projector', None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # 如果投影器已被冻结（例如，通过LoRA），则解冻以进行训练
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        # 如果指定了预训练的投影器权重文件，则加载这些权重
        if model_args.pretrain_speech_projector is not None:
            pretrain_speech_projector_weights = torch.load(model_args.pretrain_speech_projector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.speech_projector.load_state_dict(get_w(pretrain_speech_projector_weights, 'speech_projector'))

# 定义一个抽象基类 OmniSpeechMetaForCausalLM
class OmniSpeechMetaForCausalLM(ABC):
    # 定义一个抽象方法 get_model，子类必须实现
    @abstractmethod
    def get_model(self):
        """
        抽象方法，返回模型实例
        """
        pass

    # 获取语音编码器
    def get_speech_encoder(self):
        """
        返回语音编码器实例
        """
        return self.get_model().get_speech_encoder()

    # 获取语音投影器
    def get_speech_projector(self):
        """
        返回语音投影器实例
        """
        return self.get_model().speech_projector

    def encode_speech(self, speech, speech_lengths):
        """
        对语音进行编码
        :param speech: 语音数据
        :param speech_lengths: 语音长度
        :return: 编码后的语音特征
        """
        # 获取语音编码器类型和实例
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()

        # 如果是whisper类型的编码器
        if "whisper" in speech_encoder_type.lower():
            # 对语音数据进行编码
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            # 更新语音长度
            speech_lengths = (speech_lengths + 1) // 2
        else:
            # 如果是未知的编码器类型，抛出异常
            raise ValueError(f'Unknown speech encoder: {speech_encoder}')

        # 获取语音投影器类型和实例
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()

        # 如果是线性类型的投影器
        if speech_projector_type == "linear":
            # 对编码结果进行投影
            encoder_outs = speech_projector(encoder_outs)
            # 根据投影器的配置更新语音长度
            speech_lengths = speech_lengths // speech_projector.k
        else:
            # 如果是未知的投影器类型，抛出异常
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')

        # 根据语音长度提取特征序列
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]

        # 返回编码后的语音特征
        return speech_features

    # 准备输入和标签
    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        """
        准备输入和标签，用于语音和文本的联合训练
        :param input_ids: 输入ID
        :param position_ids: 位置ID
        :param attention_mask: 注意力掩码
        :param past_key_values: 过去的键值
        :param labels: 标签
        :param speech: 语音数据
        :param speech_lengths: 语音长度
        :return: 处理后的输入和标签
        """
        # 检查语音编码器是否可用以及是否有语音数据，如果不是，则直接返回原始输入和标签
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        # 对语音数据进行编码，转换为特征表示
        speech_features = self.encode_speech(speech, speech_lengths)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        # 初始化标签、位置ID和注意力掩码，如果它们不存在
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # 使用注意力掩码去除padding部分的输入和标签
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # 初始化新的输入嵌入和标签列表
        new_input_embeds = []
        new_labels = []
        # 用于追踪语音数据的索引
        cur_speech_idx = 0

        # 对输入和标签进行处理
        # 遍历输入ID列表，处理每个批次
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 计算当前输入中演讲令牌的数量
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()

            # 如果没有speech令牌，则直接处理并继续下一个批次
            if num_speech == 0:
                cur_speech_features = speech_features[cur_speech_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_speech_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                continue

            # 获取演讲令牌的索引位置，以便分割和处理
            speech_token_indices = [-1] + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []

            # 移除speech令牌，只保留文本部分的输入ID和标签
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i] + 1:speech_token_indices[i + 1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i] + 1:speech_token_indices[i + 1]])

            # 计算每个文本部分的长度，用于后续的分割
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # 处理每个文本部分和对应的speech特征，生成新的输入嵌入和标签
            for i in range(num_speech + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])
                if i < num_speech:
                    cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(
                        torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            # 将新的输入嵌入和标签移至正确的设备
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # 合并新的输入嵌入和标签
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # 将处理后的输入嵌入和标签添加到输出列表中
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 对序列进行截断
        # Truncate sequences to max length as speech features can make the sequence longer
        # 获取配置中的tokenizer最大长度设置
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            # 如果设置了最大长度，对输入和标签进行截断
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # 对输入和标签进行填充，以便于批量处理
        # 对它们进行组合
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # 根据配置决定填充的方式（左侧或右侧）
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                # 如果配置为左侧填充，将当前嵌入、标签、注意力掩码和位置ID填充到右侧
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                # 如果配置为右侧填充，将当前嵌入、标签、注意力掩码和位置ID填充到左侧
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        # 将填充后的输入嵌入堆叠成批次
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # 根据传入的参数决定是否返回标签、注意力掩码和位置ID
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # 返回处理后的数据
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels