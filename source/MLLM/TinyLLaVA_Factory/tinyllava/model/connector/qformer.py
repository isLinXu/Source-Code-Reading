
import torch
import torch.nn as nn

from transformers.models.bert.configuration_bert import BertConfig

from . import register_connector
from .base import Connector



class QFormer(nn.Module):
    """
    QFormer类是一个基于BERT模型的神经网络模块，用于处理视觉查询任务。
    """
    def __init__(self, config):
        """
        初始化QFormer模块。

        :param config: 包含模型配置参数的对象。
        """
        super().__init__()
        # 加载预训练的BERT配置，并根据config参数进行调整
        bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
        bert_config.encoder_width = config.vision_hidden_size
        # insert cross-attention layer every other block
        bert_config.add_cross_attention = True
        bert_config.cross_attention_freq = 2
        bert_config.query_length = config.num_queries

        # 创建BERT模型实例，并移除BERT的嵌入层
        self.bert = BertModel(config=bert_config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None
        self.bert.embeddings.position_embeddings = None
        self.bert.embeddings.LayerNorm.weight = None # # 移除LayerNorm的权重
        self.bert.embeddings.LayerNorm.bias = None # # 移除LayerNorm的偏置

        # 移除BERT编码器中每一层的输出和中间状态
        for layer in self.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # 初始化查询令牌参数
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_queries, bert_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=bert_config.initializer_range)

        # 初始化投影层
        self.projector = nn.Linear(bert_config.hidden_size, config.hidden_size)

     

    def forward(self, x):
        """
        前向传播函数。

        :param x: 输入张量。
        :return: 处理后的图像嵌入张量。
        """
        device = x.device
        image_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(device)     # 创建图像注意力掩码
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1).to(device)  # 扩展查询令牌到输入批次大小

        # 使用BERT模型处理查询令牌和输入张量
        query_output = self.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state       # 获取最后的隐藏状态作为图像嵌入
        image_embeds = self.projector(image_embeds)         # 通过投影层处理图像嵌入
        return image_embeds

 

@register_connector('qformer')
class QFormerConnector(Connector):
    def __init__(self, config):
        """
        初始化QFormerConnector类。

        :param config: 配置参数
        """
        super().__init__()
        self._connector = QFormer(config)   # 初始化QFormer模型


    def load_model(self, **kwargs):
        """
        加载预训练模型的权重。

        :param kwargs: 可能包含预训练模型路径的参数
        """
        pretrained_connector_path = kwargs.get('pretrained_connector_path', None)                       # 获取预训练模型路径
        if pretrained_connector_path is not None:
            pretrained_connector_path = os.path.join(pretrained_connector_path, 'pytorch_model.bin')    # 构建完整的权重文件路径
            connector_weights = torch.load(pretrained_connector_path, map_location='cpu')               # 加载权重文件

            # 定义一个函数，用于从加载的权重中提取与_connector相关的部分
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self._connector.load_state_dict(get_w(connector_weights, '_connector'), strict=False)       # 加载权重到模型
            print(f'Loading connector from {pretrained_connector_path}...')                             # 打印加载信息

        for p in self._connector.parameters():
            p.requires_grad = False                                                                     # 将模型参数设置为不需要梯度计算，即不参与训练
 
# =================================qformer bert related =================================
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


from torch import Tensor, device, dtype, nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""
    """
    构建从单词和位置嵌入中得到的嵌入表示。

    Attributes:
        config (BertConfig): Bert配置对象。
    """
    def __init__(self, config):
        super().__init__()
        # 初始化单词嵌入层, 词嵌入维度为config.hidden_size，词典大小为config.vocab_size，padding_idx为config.pad_token_id
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        # 初始化位置嵌入层, 位置嵌入维度为config.hidden_size，位置嵌入长度为config.max_position_embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # 初始化LayerNorm层，用于归一化嵌入表示，epsilon为config.layer_norm_eps
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册position_ids作为buffer，它是一个连续的内存块，在序列化时会被导出
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        # 获取位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        # 保存配置对象
        self.config = config

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        """
        前向传播函数，计算嵌入表示。

        Args:
            input_ids (torch.Tensor): 输入的token id序列。
            position_ids (torch.Tensor): 输入的位置id序列。
            query_embeds (torch.Tensor): 查询嵌入表示。
            past_key_values_length (int): 过去的key-value长度。

        Returns:
            torch.Tensor: 计算得到的嵌入表示。
        """
        # 如果input_ids不为空，则获取序列长度
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0
        # 如果position_ids为空，则根据当前序列长度生成新的position_ids
        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ].clone()
        # 如果input_ids不为空，则计算单词嵌入
        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            # 如果位置嵌入类型为"absolute"，则计算位置嵌入并与单词嵌入相加
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings
            # 如果query_embeds不为空，则将其与嵌入表示拼接起来
            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            # 如果input_ids为空，则直接使用query_embeds作为嵌入表示
            embeddings = query_embeds
        # 对嵌入表示进行LayerNorm归一化
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入表示进行Dropout操作
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        """
        初始化BertSelfAttention模块。

        :param config: 配置对象，包含模型参数。
        :param is_cross_attention: 布尔值，指示是否使用交叉注意力。
        """
        super().__init__()
        self.config = config
        # 检查隐藏层大小是否能被子注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads                               # 注意力头数
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)     # 每个头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size            # 所有头的总大小

        # 定义线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            # 如果使用交叉注意力，则key和value的输入维度为encoder_width
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            # 否则，key和value的输入维度为hidden_size
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)                      # 定义dropout层
        # 定义位置嵌入类型
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为相对key或相对key_query，则定义距离嵌入
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.save_attention = False                                                         # 是否保存注意力权重

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        """
        将输入张量x转置为适合注意力分数计算的形状。

        Args:
            x (torch.Tensor): 输入张量，其形状应为(batch_size, seq_length, hidden_size)。

        Returns:
            torch.Tensor: 转置后的张量，形状为(batch_size, num_attention_heads, seq_length, attention_head_size)。
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        # 如果是跨注意力模块，键和值来自编码器；注意力掩码需要确保不关注编码器的填充标记。
        is_cross_attention = encoder_hidden_states is not None
        # 根据是否是跨注意力或是否有过去的键值对来处理键层和值层
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # 如果需要保存注意力图并注册梯度钩子，以便于后续计算梯度
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # 应用dropout，实际上是在丢弃整个token以进行注意力分配
        attention_probs_dropped = self.dropout(attention_probs)

        # 如果存在头部掩码，则应用
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        # 计算上下文层
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        # 调整上下文层的形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 根据是否输出注意力分数来构建输出
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        # 将过去的键值对添加到输出中
        outputs = outputs + (past_key_value,)
        # 返回输出
        return outputs


class BertSelfOutput(nn.Module):
    """
    BertSelfOutput类是BERT模型中的一个自输出模块，用于处理隐藏状态并返回处理后的结果。

    Attributes:
        dense (nn.Linear): 一个线性层，用于将输入的隐藏状态进行线性变换。
        LayerNorm (nn.LayerNorm): 层归一化层，用于对线性变换后的隐藏状态进行归一化处理。
        dropout (nn.Dropout): 一个dropout层，用于防止过拟合。
    """
    def __init__(self, config):
        """
        BertSelfOutput类的构造函数。

        Args:
            config (object): BERT模型的配置对象，包含隐藏层大小、层归一化epsilon值和dropout概率等参数。
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        前向传播函数，用于处理隐藏状态并返回结果。

        Args:
            hidden_states (tensor): 输入的隐藏状态张量。
            input_tensor (tensor): 输入张量，将与处理后的隐藏状态相加。

        Returns:
            tensor: 处理后的隐藏状态张量。
        """
        hidden_states = self.dense(hidden_states)                       # 线性变换
        hidden_states = self.dropout(hidden_states)                     # dropout处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)    # 归一化处理并加上输入张量
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        """
        初始化BertAttention模块。

        :param config: Bert配置对象，包含模型的所有超参数。
        :param is_cross_attention: 布尔值，指示是否使用交叉注意力。
        """
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        剪枝指定的注意力头。

        :param heads: 需要剪枝的注意力头的索引列表。
        """
        if len(heads) == 0:
            return
        # 找到可剪枝的头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )
        # 剪枝线性层
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        前向传播函数。

        :param hidden_states: 输入的隐藏状态。
        :param attention_mask: 注意力掩码，用于屏蔽某些位置。
        :param head_mask: 头掩码，用于屏蔽某些头。
        :param encoder_hidden_states: 编码器的隐藏状态（仅在交叉注意力中使用）。
        :param encoder_attention_mask: 编码器的注意力掩码。
        :param past_key_value: 过去的键值对（用于缓存）。
        :param output_attentions: 是否输出注意力权重。
        :return: 包含输出隐藏状态和其他可能输出的元组。
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        # 如果需要，添加注意力权重
        return outputs


class BertIntermediate(nn.Module):
    """
    BertIntermediate类是Bert模型中的一个中间层，用于处理隐藏状态。

    Args:
        config (BertConfig): Bert模型的配置对象。
    """
    def __init__(self, config):
        super().__init__()                                                          # 调用父类的构造函数
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)        # 定义一个线性层
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """
        前向传播函数，处理输入的隐藏状态。

        Args:
            hidden_states (Tensor): 输入的隐藏状态张量。

        Returns:
            Tensor: 处理后的隐藏状态张量。
        """
        hidden_states = self.dense(hidden_states)                   # 通过线性层
        hidden_states = self.intermediate_act_fn(hidden_states)     # 应用激活函数
        return hidden_states                                        # 返回处理后的隐藏状态


class BertOutput(nn.Module):
    """
    BertOutput类是BERT模型中的一个输出层，它负责处理隐藏状态并返回最终的输出。

    Args:
        config (BertConfig): BERT模型的配置对象。
    """
    def __init__(self, config):
        super().__init__()                                                              # 调用父类的构造函数
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)            # 全连接层，用于改变隐藏状态的维度
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)    # 层归一化，用于稳定训练过程
        self.dropout = nn.Dropout(config.hidden_dropout_prob)                           # 随机丢弃一部分神经元，用于防止过拟合

    def forward(self, hidden_states, input_tensor):
        """
        前向传播函数，处理输入的隐藏状态并返回最终的输出。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态张量。
            input_tensor (torch.Tensor): 输入的张量。

        Returns:
            torch.Tensor: 处理后的隐藏状态张量。
        """
        hidden_states = self.dense(hidden_states)                                       # 通过全连接层改变维度
        hidden_states = self.dropout(hidden_states)                                     # 应用dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)                    # 层归一化并加上输入张量
        return hidden_states                                                            # 返回处理后的隐藏状态


class BertLayer(nn.Module):
    """
    BertLayer类是BERT模型的一个层，包含了自注意力机制、交叉注意力机制（可选）、前馈神经网络等组件。
    """
    def __init__(self, config, layer_num):
        """
        BertLayer类的构造函数。

        :param config: BERT模型的配置对象。
        :param layer_num: 当前层的编号。
        """
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        # 如果配置中启用了交叉注意力，并且当前层号是交叉注意力频率的倍数，则添加交叉注意力机制
        if (
            self.config.add_cross_attention
            and layer_num % self.config.cross_attention_freq == 0
        ):
            self.crossattention = BertAttention(
                config, is_cross_attention=self.config.add_cross_attention
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        """
        BertLayer类的前向传播函数。

        :param hidden_states: 输入的隐藏状态。
        :param attention_mask: 注意力掩码。
        :param head_mask: 头部掩码。
        :param encoder_hidden_states: 编码器的隐藏状态（用于交叉注意力）。
        :param encoder_attention_mask: 编码器的注意力掩码。
        :param past_key_value: 过去的键值对（用于缓存）。
        :param output_attentions: 是否输出注意力权重。
        :param query_length: 查询序列的长度。
        :return: 前向传播的输出结果。
        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # 计算自注意力机制的输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            # 如果查询序列长度大于0，则处理查询序列的自注意力输出
            query_attention_output = attention_output[:, :query_length, :]
            # 如果存在交叉注意力机制，则计算交叉注意力的输出
            if self.has_cross_attention:
                assert (
                    encoder_hidden_states is not None
                ), "encoder_hidden_states must be given for cross-attention layers"
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                outputs = (
                    outputs + cross_attention_outputs[1:-1]
                )  # add cross attentions if we output attention weights

            # 对查询序列的自注意力输出应用前馈神经网络
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            # 如果原始注意力输出的序列长度大于查询序列长度，则处理剩余部分的前馈神经网络输出
            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            # 如果查询序列长度为0，则直接对自注意力输出应用前馈神经网络
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        对注意力输出应用前馈神经网络，并返回结果。

        :param attention_output: 注意力输出。
        :return: 前馈神经网络的输出结果。
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        """
        对查询序列的注意力输出应用前馈神经网络，并返回结果。

        :param attention_output: 查询序列的注意力输出。
        :return: 前馈神经网络的输出结果。
        """
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """
    BertEncoder类是BERT模型的核心编码器部分，它由多个BertLayer层组成。
    """
    def __init__(self, config):
        """
        BertEncoder的构造函数，初始化编码器的配置和层列表。

        :param config: BertConfig对象，包含BERT模型的所有配置参数。
        """
        super().__init__()
        self.config = config
        # 创建一个包含多个BertLayer层的ModuleList
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        """
        BertEncoder的前向传播函数，处理输入并产生输出。

        :param hidden_states: 输入的隐藏状态张量。
        :param attention_mask: 注意力掩码张量，用于屏蔽某些位置。
        :param head_mask: 头部掩码张量，用于屏蔽某些头部的注意力。
        :param encoder_hidden_states: 编码器的隐藏状态张量。
        :param encoder_attention_mask: 编码器的注意力掩码张量。
        :param past_key_values: 过去的键值对，用于缓存。
        :param use_cache: 是否使用缓存。
        :param output_attentions: 是否输出注意力权重。
        :param output_hidden_states: 是否输出隐藏状态。
        :param return_dict: 是否返回字典格式的输出。
        :param query_length: 查询长度。
        :return: 包含输出隐藏状态、键值对、隐藏状态列表、自注意力权重和交叉注意力权重的对象。
        """
        # 初始化用于收集输出的变量
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        # 遍历每一层BertLayer
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            # 如果需要输出隐藏状态，则收集当前层的隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码和过去的键值对
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并且处于训练模式
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                # 如果使用缓存，则发出警告并禁用缓存
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                # 定义一个自定义的前向传播函数
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, past_key_value, output_attentions, query_length
                        )

                    return custom_forward

                # 使用torch.utils.checkpoint.checkpoint进行梯度检查点
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # 正常前向传播
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            # 更新隐藏状态
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则收集当前层的键值对
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力权重，则收集自注意力和交叉注意力权重
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据return_dict参数决定返回格式
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    """
    BertPooler类用于实现BERT模型中的池化层。

    Args:
        config (obj): BERT模型的配置对象，包含隐藏层大小等信息。
    """
    def __init__(self, config):
        super().__init__()
        # 定义一个线性层，输入和输出的维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个激活函数，使用双曲正切函数tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """
        forward方法实现了前向传播过程。

        Args:
            hidden_states (tensor): 来自BERT模型的隐藏状态张量。

        Returns:
            pooled_output (tensor): 经过池化和激活函数处理后的输出张量。
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # 池化策略是取第一个token对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个token的隐藏状态通过线性层
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数tanh
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """
    Bert模型的预测头变换层，用于处理隐藏状态并应用激活函数和层归一化。

    Args:
        config (obj): Bert配置对象，包含模型参数。
    """
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度均为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据config.hidden_act的类型，确定激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 定义层归一化，eps参数用于防止除0错误
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """
        前向传播函数，处理输入的隐藏状态。

        Args:
            hidden_states (tensor): 输入的隐藏状态张量。

        Returns:
            tensor: 处理后的隐藏状态张量。
        """
        # 通过全连接层
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用层归一化
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """
    BertLMPredictionHead类是用于BERT模型的预测头，它负责将隐藏状态转换为词汇表上的概率分布。
    """
    def __init__(self, config):
        """
        初始化BertLMPredictionHead类的实例。

        :param config: BERT模型的配置对象，包含模型参数等信息。
        """
        super().__init__()
        # 初始化一个BertPredictionHeadTransform对象，用于对隐藏状态进行变换。
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # 输出权重与输入嵌入相同，但每个token都有一个仅输出的偏置。
        # decoder是一个线性层，将隐藏状态映射到词汇表大小的空间，不包含偏置项。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化一个偏置参数，大小为词汇表大小。
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # 需要将decoder的偏置链接到bias变量，以便在使用resize_token_embeddings时正确调整偏置大小。
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        """
        前向传播函数，将输入的隐藏状态转换为词汇表上的概率分布。

        :param hidden_states: 输入的隐藏状态张量。
        :return: 词汇表上的概率分布张量。
        """
        # 对隐藏状态进行变换。
        hidden_states = self.transform(hidden_states)
        # 通过decoder线性层将变换后的隐藏状态映射到词汇表大小的空间。
        hidden_states = self.decoder(hidden_states)
        # 返回最终的输出。
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    """
    该类继承自nn.Module，用于实现BERT模型的MLM（Masked Language Model）头部分。
    """
    def __init__(self, config):
        """
        BertOnlyMLMHead类的构造函数。

        :param config: BERT模型的配置参数。
        """
        super().__init__()                                  # 调用父类的构造函数
        self.predictions = BertLMPredictionHead(config)     # 初始化预测头

    def forward(self, sequence_output):
        """
        前向传播函数。

        :param sequence_output: BERT模型的序列输出。
        :return: 预测得分。
        """
        prediction_scores = self.predictions(sequence_output)   # 获取预测得分
        return prediction_scores                                # 返回预测得分


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    # 在加载模型时忽略的键列表，例如position_ids
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        """
        初始化权重
        :param module: 需要初始化权重的模块
        """
        # 如果模块是线性层或嵌入层，则使用正态分布初始化权重
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 与TensorFlow版本略有不同，TensorFlow使用截断正态分布进行初始化
            # 参见 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        # 如果模块是层归一化层，则将偏置置零，权重填充为1.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果模块是线性层且存在偏置，则将偏置置零
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    BertModel 类实现了 BERT 模型，可以作为编码器（仅使用自注意力）或解码器（在自注意力层之间添加一层交叉注意力）。
    参考文献：`Attention is all you need <https://arxiv.org/abs/1706.03762>`__。
    """
    def __init__(self, config, add_pooling_layer=False):
        """
        初始化 BertModel 实例。

        :param config: BertConfig 实例，包含模型的配置信息。
        :param add_pooling_layer: 布尔值，是否添加池化层。
        """
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)                            # 初始化嵌入层
        self.encoder = BertEncoder(config)                                  # 初始化编码器
        self.pooler = BertPooler(config) if add_pooling_layer else None     # 初始化池化层（如果需要）
        self.init_weights()                                                 # 初始化权重

    def get_input_embeddings(self):
        """
        获取输入嵌入。

        :return: 词嵌入矩阵。
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        设置输入嵌入。

        :param value: 新的词嵌入矩阵。
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        剪枝模型的注意力头。

        :param heads_to_prune: 需要剪枝的头的字典，格式为 {层号: [需要剪枝的头列表]}。
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: Tuple[int],
        device: device,
        is_decoder: bool,
        has_query: bool = False,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.

                创建可广播的注意力和因果掩码，以便忽略未来的和被遮盖的标记。

        :param attention_mask: 注意力掩码，1 表示要关注的标记，0 表示要忽略的标记。
        :param input_shape: 输入模型的形状。
        :param device: 输入模型的设备。
        :param is_decoder: 布尔值，模型是否作为解码器。
        :param has_query: 布尔值，是否具有查询。
        :return: 扩展的注意力掩码。
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 如果attention_mask的维度是3，我们只需将其扩展到所有头（heads）上可广播
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        # 如果attention_mask的维度是2
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # 如果提供了维度为[batch_size, seq_length]的填充掩码
            # - 如果模型是解码器，在填充掩码的基础上应用因果掩码
            # - 如果模型是编码器，将掩码扩展到[batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                # 创建一个从0到seq_length-1的序列，并将其移动到正确的设备上
                seq_ids = torch.arange(seq_length, device=device)
                # 创建因果掩码，确保每个位置只能看到它之前的位置
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )

                # add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                # 确保因果掩码和attention_mask的数据类型一致
                causal_mask = causal_mask.to(attention_mask.dtype)

                # 如果因果掩码的长度小于attention_mask的长度，需要添加前缀掩码
                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    if has_query:  # UniLM style attention mask # UniLM风格的attention_mask
                        # 在因果掩码前添加全0的前缀掩码
                        causal_mask = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prefix_seq_len, seq_length),
                                    device=device,
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask,
                            ],
                            axis=1,
                        )
                    # 在因果掩码后添加全1的前缀掩码
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, causal_mask.shape[1], prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                # 将因果掩码和attention_mask相乘，得到最终的extended_attention_mask
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                # 对于编码器，直接将attention_mask扩展到所需的维度
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            # 如果输入的attention_mask维度不对，抛出错误
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # 将extended_attention_mask转换为模型的数据类型，以兼容fp16
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        # 将掩码中需要关注的位置设为0，需要屏蔽的位置设为一个很大的负数（-10000.0）
        # 这样在softmax之前加上这个掩码，就相当于完全屏蔽了这些位置
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        前向传播函数。

        :param input_ids: 输入的标记 ID。
        :param attention_mask: 注意力掩码。
        :param position_ids: 标记的位置 ID。
        :param head_mask: 头部掩码。
        :param query_embeds: 查询嵌入。
        :param encoder_hidden_states: 编码器的隐藏状态。
        :param encoder_attention_mask: 编码器的注意力掩码。
        :param past_key_values: 过去的键值对。
        :param use_cache: 是否使用缓存。
        :param output_attentions: 是否输出注意力。
        :param output_hidden_states: 是否输出隐藏状态。
        :param return_dict: 是否返回字典。
        :param is_decoder: 模型是否作为解码器。
        :return: 前向传播的结果。
        """
        # 设置输出注意力矩阵和隐藏状态矩阵的默认值
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果input_ids为空，则必须提供query_embeds
        if input_ids is None:
            assert (
                query_embeds is not None
            ), "You have to specify query_embeds when input_ids is None"

        # past_key_values_length
        # 计算past_key_values的长度
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length
            if past_key_values is not None
            else 0
        )
        # 获取query_embeds的长度
        query_length = query_embeds.shape[1] if query_embeds is not None else 0
        # 获取嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeds=query_embeds,
            past_key_values_length=past_key_values_length,
        )
        # 获取输入形状和设备信息
        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device
        # 如果attention_mask为空，则创建一个全1的attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 根据是否为解码器，获取扩展的attention_mask
        if is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                input_ids.shape,
                device,
                is_decoder,
                has_query=(query_embeds is not None),
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, device, is_decoder
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # 如果提供了用于交叉注意力的2D或3D attention_mask，需要将其广播到[batch_size, num_heads, seq_length, seq_length]
        # 如果encoder_hidden_states不为None，则需要计算encoder_extended_attention_mask
        if encoder_hidden_states is not None:
            # 如果encoder_hidden_states是列表类型
            if type(encoder_hidden_states) == list:
                # 获取encoder_hidden_states中第一个元素的形状信息
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0
                ].size()
            else:
                # 获取encoder_hidden_states的形状信息
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            # 定义encoder_hidden_shape为encoder的批次大小和序列长度
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            # 如果encoder_attention_mask是列表类型
            if type(encoder_attention_mask) == list:
                # 对列表中的每个mask调用invert_attention_mask函数，并返回新的列表
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            # 如果encoder_attention_mask为None
            elif encoder_attention_mask is None:
                # 创建一个全1的张量，形状与encoder_hidden_shape相同，设备与当前设备相同
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                # 调用invert_attention_mask函数，并将结果赋值给encoder_extended_attention_mask
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
            else:
                # 调用invert_attention_mask函数，并将结果赋值给encoder_extended_attention_mask
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            # 如果encoder_hidden_states为None，则encoder_extended_attention_mask也为None
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # 准备head_mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 通过编码器处理嵌入输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        # 获取序列输出和池化输出
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )
        # 根据return_dict的值返回相应的结果
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertLMHeadModel(BertPreTrainedModel):
    # 定义在加载模型时忽略的键
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        # 初始化BERT模型，不包含池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 初始化MLM头，用于预测
        self.cls = BertOnlyMLMHead(config)
        # 初始化权重
        self.init_weights()

    def get_output_embeddings(self):
        """
        获取输出嵌入层
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        设置新的输出嵌入层
        :param new_embeddings: 新的嵌入层
        """
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=True,
        reduction="mean",
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None
        # 通过BERT模型获取输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 如果query_embeds不为None，则对sequence_output进行切片操作
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]

        # 计算预测分数
        prediction_scores = self.cls(sequence_output)

        # 如果return_logits为True，则返回预测分数的特定切片
        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        # 初始化lm_loss为None
        lm_loss = None
        # 如果labels不为None，则进行下一词预测
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            # 将预测分数和输入id进行移位操作
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            # 初始化损失函数
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            # 计算语言模型损失
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            # 如果reduction为"none"，则对lm_loss进行视图变换，并对其求和
            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        # 返回结果
        if not return_dict:
            # 如果return_dict为False，则返回元组形式的结果
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        # 如果return_dict为True，则返回CausalLMOutputWithCrossAttentions对象
        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, query_embeds, past=None, attention_mask=None, **model_kwargs
    ):
        """
        准备用于生成模型的输入数据。

        参数:
            input_ids (torch.Tensor): 输入的token ID序列。
            query_embeds (torch.Tensor): 查询嵌入向量。
            past (torch.Tensor, 可选): 过去的key-value对，用于解码器。
            attention_mask (torch.Tensor, 可选): 注意力掩码，用于指示模型哪些位置需要关注。
            **model_kwargs: 其他传递给模型的关键字参数。

        返回:
            dict: 包含准备好的输入数据的字典。
        """
        # 如果模型作为编码器-解码器模型中的解码器使用，则动态创建解码器注意力掩码
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        # 创建查询掩码，形状与query_embeds相同，除了最后一个维度
        query_mask = input_ids.new_ones(query_embeds.shape[:-1])
        # 将查询掩码和注意力掩码在最后一个维度上拼接
        attention_mask = torch.cat([query_mask, attention_mask], dim=-1)

        # 如果使用了past，则裁剪decoder_input_ids
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        # 返回包含所有输入数据的字典
        return {
            "input_ids": input_ids,
            "query_embeds": query_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        """
        重新排序缓存
        :param past: 缓存，一个包含多个层缓存的元组，每个层缓存是状态张量的元组
        :param beam_idx: 束索引，一个张量，指示每个序列在当前束中的位置
        :return: 重新排序后的缓存，结构与输入的past相同，但其中的张量已根据beam_idx重新排序
        """
        reordered_past = ()                                                             # 初始化重新排序后的缓存为空元组
        for layer_past in past:                                                         # 遍历每一层的缓存
            # 对当前层的每个状态张量进行索引选择操作，根据beam_idx重新排序
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past   # 将重新排序后的层缓存添加到总缓存中
                ),
            )
        return reordered_past                                                           # 返回重新排序后的缓存


class BertForMaskedLM(BertPreTrainedModel):
    # 定义在加载模型时忽略的键
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        # 初始化BertModel，不包含池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 初始化MLM头，用于预测被掩盖的词
        self.cls = BertOnlyMLMHead(config)
        # 初始化权重
        self.init_weights()

    def get_output_embeddings(self):
        """
        获取输出嵌入层
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        设置新的输出嵌入层
        """
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=False,
    ):
        r"""
        前向传播函数
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        # 确定是否返回字典格式的结果
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # 通过BertModel获取输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )
        # 如果提供了query_embeds，则使用它们来获取序列输出
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]
        # 通过MLM头获取预测分数
        prediction_scores = self.cls(sequence_output)
        # 如果只需要返回logits，则直接返回
        if return_logits:
            return prediction_scores
        # 计算掩码语言模型的损失
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token # -100索引表示填充token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        # 根据是否返回字典格式来组织输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

