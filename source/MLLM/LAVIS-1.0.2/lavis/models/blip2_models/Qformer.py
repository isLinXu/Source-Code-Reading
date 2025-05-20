"""
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on huggingface code base
 * https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert
"""

import math # 导入 math 模块
import os # 导入 os 模块
import warnings # 导入 warnings 模块
from dataclasses import dataclass # 从 dataclasses 模块导入 dataclass
from typing import Optional, Tuple, Dict, Any # 从 typing 模块导入 Optional, Tuple, Dict, Any 类型提示

import torch # 导入 PyTorch 库
from torch import Tensor, device, dtype, nn # 从 torch 导入 Tensor, device, dtype, nn
import torch.utils.checkpoint # 导入 PyTorch 的 checkpoint 工具
from torch import nn # 再次导入 nn (冗余)
from torch.nn import CrossEntropyLoss # 从 torch.nn 导入 CrossEntropyLoss 损失函数
import torch.nn.functional as F # 导入 torch.nn.functional 并命名为 F

from transformers.activations import ACT2FN # 从 transformers.activations 导入 ACT2FN 激活函数映射
from transformers.file_utils import ( # 从 transformers.file_utils 导入文件工具
    ModelOutput, # 模型输出基类
)
from transformers.modeling_outputs import ( # 从 transformers.modeling_outputs 导入模型输出类
    BaseModelOutputWithPastAndCrossAttentions, # 包含 past key values 和 cross attentions 的基础模型输出
    BaseModelOutputWithPoolingAndCrossAttentions, # 包含 pooling 和 cross attentions 的基础模型输出
    CausalLMOutputWithCrossAttentions, # 包含 cross attentions 的因果语言模型输出
    MaskedLMOutput, # Masked Language Model 输出
    MultipleChoiceModelOutput, # 多项选择模型输出
    NextSentencePredictorOutput, # 下一句预测模型输出
    QuestionAnsweringModelOutput, # 问答模型输出
    SequenceClassifierOutput, # 序列分类模型输出
    TokenClassifierOutput, # Token 分类模型输出
)
from transformers.modeling_utils import ( # 从 transformers.modeling_utils 导入模型工具函数
    PreTrainedModel, # 预训练模型基类
    apply_chunking_to_forward, # 应用 chunking 到前向传播的函数
    find_pruneable_heads_and_indices, # 查找可剪枝的 attention heads 和索引的函数
    prune_linear_layer, # 剪枝线性层的函数
)
from transformers.utils import logging # 从 transformers.utils 导入 logging 工具
from transformers.models.bert.configuration_bert import BertConfig # 从 transformers.models.bert.configuration_bert 导入 BertConfig 配置类

logger = logging.get_logger(__name__) # 获取当前模块的 logger


class BertEmbeddings(nn.Module): # 定义 BertEmbeddings 类，继承自 nn.Module
    """Construct the embeddings from word and position embeddings.""" # 从词嵌入和位置嵌入构建嵌入表示。

    def __init__(self, config): # 类的初始化方法，接收一个 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.word_embeddings = nn.Embedding( # 定义词嵌入层
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id # 嵌入矩阵大小为 vocab_size x hidden_size，指定 padding 索引
        )
        self.position_embeddings = nn.Embedding( # 定义位置嵌入层
            config.max_position_embeddings, config.hidden_size # 嵌入矩阵大小为 max_position_embeddings x hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # self.LayerNorm 没有使用 snake_case 命名，以与 TensorFlow 模型变量名保持一致，并能够加载任何 TensorFlow checkpoint 文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 定义 Layer Normalization 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 定义 Dropout 层

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # position_ids (1, 位置嵌入长度) 在内存中是连续的，并在序列化时导出
        self.register_buffer( # 注册一个 buffer，不会被视为模型参数，但会随模型保存和加载
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) # 创建一个从 0 到 max_position_embeddings-1 的序列，并扩展为形状 (1, max_position_embeddings)
        )
        self.position_embedding_type = getattr( # 获取位置嵌入类型，默认为 "absolute"
            config, "position_embedding_type", "absolute"
        )

        self.config = config # 保存配置对象

    def forward( # 定义前向传播方法
        self,
        input_ids=None, # 输入 token ID
        position_ids=None, # 位置 ID
        query_embeds=None, # Query 嵌入 (Qformer 特有)
        past_key_values_length=0, # 过去的 key/value 长度 (用于增量解码)
    ):
        if input_ids is not None: # 如果提供了 input_ids
            seq_length = input_ids.size()[1] # 获取序列长度
        else: # 如果没有提供 input_ids
            seq_length = 0 # 序列长度为 0

        if position_ids is None: # 如果没有提供 position_ids
            position_ids = self.position_ids[ # 根据序列长度和 past_key_values_length 生成 position_ids
                :, past_key_values_length : seq_length + past_key_values_length
            ].clone() # 获取对应范围的 position_ids 并克隆

        if input_ids is not None: # 如果提供了 input_ids
            embeddings = self.word_embeddings(input_ids) # 通过 word_embeddings 层获取嵌入表示
            if self.position_embedding_type == "absolute": # 如果位置嵌入类型是绝对位置嵌入
                position_embeddings = self.position_embeddings(position_ids) # 通过 position_embeddings 层获取位置嵌入
                embeddings = embeddings + position_embeddings # 将位置嵌入加到词嵌入上

            if query_embeds is not None: # 如果提供了 query_embeds
                embeddings = torch.cat((query_embeds, embeddings), dim=1) # 将 query_embeds 和 embeddings 在序列长度维度上拼接
        else: # 如果没有提供 input_ids
            embeddings = query_embeds # 直接使用 query_embeds 作为 embeddings

        embeddings = self.LayerNorm(embeddings) # 对 embeddings 进行 Layer Normalization
        embeddings = self.dropout(embeddings) # 对 embeddings 应用 Dropout
        return embeddings # 返回最终的嵌入表示


class BertSelfAttention(nn.Module): # 定义 BertSelfAttention 类，继承自 nn.Module
    def __init__(self, config, is_cross_attention): # 类的初始化方法，接收 config 对象和 is_cross_attention 标志
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.config = config # 保存配置对象
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr( # 检查隐藏层大小是否能被 attention heads 数量整除
            config, "embedding_size" # 并且检查 config 是否没有 embedding_size 属性
        ):
            raise ValueError( # 如果不满足条件，抛出 ValueError 异常
                "The hidden size (%d) is not a multiple of the number of attention " # 错误信息：隐藏层大小不是 attention heads 数量的倍数
                "heads (%d)" % (config.hidden_size, config.num_attention_heads) # 格式化错误信息，包含实际的 hidden_size 和 num_attention_heads
            )

        self.num_attention_heads = config.num_attention_heads # 保存 attention heads 的数量
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 计算每个 attention head 的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 计算所有 attention heads 的总大小

        self.query = nn.Linear(config.hidden_size, self.all_head_size) # 定义 Query 的线性变换层
        if is_cross_attention: # 如果是交叉注意力
            self.key = nn.Linear(config.encoder_width, self.all_head_size) # 定义 Key 的线性变换层，输入维度为 encoder_width
            self.value = nn.Linear(config.encoder_width, self.all_head_size) # 定义 Value 的线性变换层，输入维度为 encoder_width
        else: # 如果是自注意力
            self.key = nn.Linear(config.hidden_size, self.all_head_size) # 定义 Key 的线性变换层，输入维度为 hidden_size
            self.value = nn.Linear(config.hidden_size, self.all_head_size) # 定义 Value 的线性变换层，输入维度为 hidden_size

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob) # 定义 Dropout 层，用于 attention 概率
        self.position_embedding_type = getattr( # 获取位置嵌入类型，默认为 "absolute"
            config, "position_embedding_type", "absolute"
        )
        if ( # 如果位置嵌入类型是 "relative_key" 或 "relative_key_query"
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings # 保存最大位置嵌入数
            self.distance_embedding = nn.Embedding( # 定义距离嵌入层
                2 * config.max_position_embeddings - 1, self.attention_head_size # 嵌入矩阵大小为 (2*max_pos-1) x head_size
            )
        self.save_attention = False # 初始化 save_attention 标志为 False

    def save_attn_gradients(self, attn_gradients): # 定义保存 attention 梯度的函数
        self.attn_gradients = attn_gradients # 将传入的梯度保存到实例变量

    def get_attn_gradients(self): # 定义获取 attention 梯度的函数
        return self.attn_gradients # 返回保存的 attention 梯度

    def save_attention_map(self, attention_map): # 定义保存 attention map 的函数
        self.attention_map = attention_map # 将传入的 attention map 保存到实例变量

    def get_attention_map(self): # 定义获取 attention map 的函数
        return self.attention_map # 返回保存的 attention map

    def transpose_for_scores(self, x): # 定义用于计算 attention scores 的转置函数
        new_x_shape = x.size()[:-1] + ( # 计算新的形状，将最后一个维度拆分为 num_attention_heads 和 attention_head_size
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape) # 重塑张量形状
        return x.permute(0, 2, 1, 3) # 将维度顺序调整为 (batch_size, num_heads, sequence_length, head_size)

    def forward( # 定义前向传播方法
        self,
        hidden_states, # 输入的隐藏状态
        attention_mask=None, # attention mask
        head_mask=None, # head mask
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的 attention mask (用于交叉注意力)
        past_key_value=None, # 过去的 key 和 value (用于增量解码)
        output_attentions=False, # 是否输出 attention map
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        # 如果这是一个交叉注意力模块的实例，key 和 value 来自编码器；
        # attention mask 需要确保编码器的 padding token 不被关注。
        is_cross_attention = encoder_hidden_states is not None # 判断是否是交叉注意力

        if is_cross_attention: # 如果是交叉注意力
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states)) # 计算 Key，来自编码器隐藏状态
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states)) # 计算 Value，来自编码器隐藏状态
            attention_mask = encoder_attention_mask # 使用编码器的 attention mask
        elif past_key_value is not None: # 如果不是交叉注意力且提供了 past_key_value (用于增量解码)
            key_layer = self.transpose_for_scores(self.key(hidden_states)) # 计算 Key，来自当前隐藏状态
            value_layer = self.transpose_for_scores(self.value(hidden_states)) # 计算 Value，来自当前隐藏状态
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2) # 将过去的 Key 和当前的 Key 拼接
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2) # 将过去的 Value 和当前的 Value 拼接
        else: # 如果是自注意力且没有 past_key_value
            key_layer = self.transpose_for_scores(self.key(hidden_states)) # 计算 Key，来自当前隐藏状态
            value_layer = self.transpose_for_scores(self.value(hidden_states)) # 计算 Value，来自当前隐藏状态

        mixed_query_layer = self.query(hidden_states) # 计算 Query 的线性变换结果

        query_layer = self.transpose_for_scores(mixed_query_layer) # 对 Query 进行转置和重塑

        past_key_value = (key_layer, value_layer) # 保存当前的 Key 和 Value 作为 past_key_value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算 "query" 和 "key" 的点积，得到原始 attention scores。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # 计算 Query 和 Key 的转置的点积

        if ( # 如果位置嵌入类型是 "relative_key" 或 "relative_key_query"
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1] # 获取序列长度
            position_ids_l = torch.arange( # 创建左侧位置 ID 序列
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange( # 创建右侧位置 ID 序列
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r # 计算位置之间的距离
            positional_embedding = self.distance_embedding( # 通过距离嵌入层获取位置嵌入
                distance + self.max_position_embeddings - 1 # 调整距离范围以匹配嵌入矩阵索引
            )
            positional_embedding = positional_embedding.to( # 转换为与 query_layer 相同的数据类型
                dtype=query_layer.dtype
            )  # fp16 compatibility # fp16 兼容性

            if self.position_embedding_type == "relative_key": # 如果是 "relative_key" 位置嵌入
                relative_position_scores = torch.einsum( # 计算相对位置分数
                    "bhld,lrd->bhlr", query_layer, positional_embedding # 使用 einsum 计算
                )
                attention_scores = attention_scores + relative_position_scores # 将相对位置分数加到 attention scores 上
            elif self.position_embedding_type == "relative_key_query": # 如果是 "relative_key_query" 位置嵌入
                relative_position_scores_query = torch.einsum( # 计算 Query 相关的相对位置分数
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum( # 计算 Key 相关的相对位置分数
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = ( # 将两种相对位置分数都加到 attention scores 上
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # 对 attention scores 进行缩放
        if attention_mask is not None: # 如果提供了 attention mask
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # 应用 attention mask (在 BertModel 的 forward 函数中为所有层预先计算)
            attention_scores = attention_scores + attention_mask # 将 attention mask 加到 attention scores 上

        # Normalize the attention scores to probabilities.
        # 将 attention scores 归一化为概率。
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # 对 attention scores 应用 Softmax 得到 attention 概率

        if is_cross_attention and self.save_attention: # 如果是交叉注意力且 save_attention 为 True
            self.save_attention_map(attention_probs) # 保存 attention map
            attention_probs.register_hook(self.save_attn_gradients) # 注册 hook，用于保存 attention 梯度

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # 这实际上是 dropout 掉整个 token 的注意力，这可能看起来有点不寻常，但它来自原始的 Transformer 论文。
        attention_probs_dropped = self.dropout(attention_probs) # 对 attention 概率应用 Dropout

        # Mask heads if we want to
        # 如果需要，对 heads 进行 mask
        if head_mask is not None: # 如果提供了 head mask
            attention_probs_dropped = attention_probs_dropped * head_mask # 将 head mask 应用到 attention 概率上

        context_layer = torch.matmul(attention_probs_dropped, value_layer) # 计算 attention 概率和 Value 的矩阵乘法，得到 context layer

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # 将维度顺序调整回 (batch_size, sequence_length, num_heads, head_size) 并使其连续
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # 计算新的形状，将最后两个维度合并为 all_head_size
        context_layer = context_layer.view(*new_context_layer_shape) # 重塑张量形状

        outputs = ( # 构建输出元组
            (context_layer, attention_probs) if output_attentions else (context_layer,) # 如果 output_attentions 为 True，则包含 context_layer 和 attention_probs，否则只包含 context_layer
        )

        outputs = outputs + (past_key_value,) # 将 past_key_value 添加到输出元组中
        return outputs # 返回输出元组


class BertSelfOutput(nn.Module): # 定义 BertSelfOutput 类，继承自 nn.Module
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 定义一个线性层，输入和输出维度都是 hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 定义 Layer Normalization 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 定义 Dropout 层

    def forward(self, hidden_states, input_tensor): # 定义前向传播方法，接收 hidden_states 和 input_tensor
        hidden_states = self.dense(hidden_states) # 对 hidden_states 应用线性变换
        hidden_states = self.dropout(hidden_states) # 对结果应用 Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # 将处理后的 hidden_states 与 input_tensor (残差连接) 相加，然后应用 Layer Normalization
        return hidden_states # 返回最终的 hidden_states


class BertAttention(nn.Module): # 定义 BertAttention 类，继承自 nn.Module
    def __init__(self, config, is_cross_attention=False): # 类的初始化方法，接收 config 对象和 is_cross_attention 标志
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.self = BertSelfAttention(config, is_cross_attention) # 定义 BertSelfAttention 模块
        self.output = BertSelfOutput(config) # 定义 BertSelfOutput 模块
        self.pruned_heads = set() # 初始化一个集合，用于存储被剪枝的 attention heads

    def prune_heads(self, heads): # 定义剪枝 attention heads 的方法
        if len(heads) == 0: # 如果 heads 列表为空，则直接返回
            return
        heads, index = find_pruneable_heads_and_indices( # 查找可剪枝的 heads 和对应的索引
            heads, # 要剪枝的 heads 列表
            self.self.num_attention_heads, # attention heads 的总数量
            self.self.attention_head_size, # 每个 attention head 的大小
            self.pruned_heads, # 已经剪枝的 heads 集合
        )

        # Prune linear layers
        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index) # 剪枝 Query 线性层
        self.self.key = prune_linear_layer(self.self.key, index) # 剪枝 Key 线性层
        self.self.value = prune_linear_layer(self.self.value, index) # 剪枝 Value 线性层
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1) # 剪枝输出线性层

        # Update hyper params and store pruned heads
        # 更新超参数并存储被剪枝的 heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads) # 更新 attention heads 的数量
        self.self.all_head_size = ( # 更新所有 attention heads 的总大小
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads) # 将新剪枝的 heads 添加到集合中

    def forward( # 定义前向传播方法
        self,
        hidden_states, # 输入的隐藏状态
        attention_mask=None, # attention mask
        head_mask=None, # head mask
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的 attention mask (用于交叉注意力)
        past_key_value=None, # 过去的 key 和 value (用于增量解码)
        output_attentions=False, # 是否输出 attention map
    ):
        self_outputs = self.self( # 通过 BertSelfAttention 模块进行前向传播
            hidden_states, # 输入隐藏状态
            attention_mask, # attention mask
            head_mask, # head mask
            encoder_hidden_states, # 编码器隐藏状态
            encoder_attention_mask, # 编码器 attention mask
            past_key_value, # past_key_value
            output_attentions, # 是否输出 attention
        )
        attention_output = self.output(self_outputs[0], hidden_states) # 通过 BertSelfOutput 模块处理 attention 输出，传入 attention 模块的输出和原始 hidden_states

        outputs = (attention_output,) + self_outputs[ # 构建输出元组，包含 attention_output 和 attention 模块的其他输出 (如 attention map)
            1:
        ]  # add attentions if we output them # 如果输出 attention，则添加到输出中
        return outputs # 返回输出元组


class BertIntermediate(nn.Module): # 定义 BertIntermediate 类，继承自 nn.Module (前馈网络的第一部分)
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # 定义一个线性层，将 hidden_size 映射到 intermediate_size
        if isinstance(config.hidden_act, str): # 如果激活函数是字符串类型
            self.intermediate_act_fn = ACT2FN[config.hidden_act] # 从 ACT2FN 字典中获取对应的激活函数
        else: # 如果激活函数不是字符串类型
            self.intermediate_act_fn = config.hidden_act # 直接使用 config 中指定的激活函数

    def forward(self, hidden_states): # 定义前向传播方法，接收 hidden_states
        hidden_states = self.dense(hidden_states) # 对 hidden_states 应用线性变换
        hidden_states = self.intermediate_act_fn(hidden_states) # 对结果应用激活函数
        return hidden_states # 返回处理后的 hidden_states


class BertOutput(nn.Module): # 定义 BertOutput 类，继承自 nn.Module (前馈网络的第二部分)
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size) # 定义一个线性层，将 intermediate_size 映射回 hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 定义 Layer Normalization 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 定义 Dropout 层

    def forward(self, hidden_states, input_tensor): # 定义前向传播方法，接收 hidden_states (来自 intermediate 层) 和 input_tensor (来自 attention 层)
        hidden_states = self.dense(hidden_states) # 对 hidden_states 应用线性变换
        hidden_states = self.dropout(hidden_states) # 对结果应用 Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # 将处理后的 hidden_states 与 input_tensor (残差连接) 相加，然后应用 Layer Normalization
        return hidden_states # 返回最终的 hidden_states


class BertLayer(nn.Module): # 定义 BertLayer 类，继承自 nn.Module (单个编码器层)
    def __init__(self, config, layer_num): # 类的初始化方法，接收 config 对象和层编号 layer_num
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.config = config # 保存配置对象
        self.chunk_size_feed_forward = config.chunk_size_feed_forward # 保存前馈网络的 chunk 大小
        self.seq_len_dim = 1 # 序列长度所在的维度索引
        self.attention = BertAttention(config) # 定义自注意力模块
        self.layer_num = layer_num # 保存当前层编号
        if ( # 如果配置中指定添加交叉注意力且当前层编号满足交叉注意力频率
            self.config.add_cross_attention
            and layer_num % self.config.cross_attention_freq == 0
        ):
            self.crossattention = BertAttention( # 定义交叉注意力模块
                config, is_cross_attention=self.config.add_cross_attention # 设置为交叉注意力
            )
            self.has_cross_attention = True # 设置标志表示有交叉注意力
        else: # 如果不满足条件
            self.has_cross_attention = False # 设置标志表示没有交叉注意力
        self.intermediate = BertIntermediate(config) # 定义中间层 (前馈网络的第一部分)
        self.output = BertOutput(config) # 定义输出层 (前馈网络的第二部分)

        self.intermediate_query = BertIntermediate(config) # 定义用于 Query 的中间层 (Qformer 特有)
        self.output_query = BertOutput(config) # 定义用于 Query 的输出层 (Qformer 特有)

    def forward( # 定义前向传播方法
        self,
        hidden_states, # 输入的隐藏状态
        attention_mask=None, # attention mask
        head_mask=None, # head mask
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的 attention mask (用于交叉注意力)
        past_key_value=None, # 过去的 key 和 value (用于增量解码)
        output_attentions=False, # 是否输出 attention map
        query_length=0, # Query 的序列长度 (Qformer 特有)
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 解码器单向自注意力的缓存 key/values 元组位于位置 1,2
        self_attn_past_key_value = ( # 获取自注意力的 past_key_value
            past_key_value[:2] if past_key_value is not None else None # 如果 past_key_value 存在，取前两个元素
        )
        self_attention_outputs = self.attention( # 通过自注意力模块进行前向传播
            hidden_states, # 输入隐藏状态
            attention_mask, # attention mask
            head_mask, # head mask
            output_attentions=output_attentions, # 是否输出 attention
            past_key_value=self_attn_past_key_value, # 自注意力的 past_key_value
        )
        attention_output = self_attention_outputs[0] # 获取自注意力的输出
        outputs = self_attention_outputs[1:-1] # 获取自注意力的其他输出 (如 attention map)

        present_key_value = self_attention_outputs[-1] # 获取当前的 present_key_value

        if query_length > 0: # 如果 Query 序列长度大于 0 (处理 Query 部分)
            query_attention_output = attention_output[:, :query_length, :] # 提取 Query 部分的 attention 输出

            if self.has_cross_attention: # 如果当前层有交叉注意力
                assert ( # 断言编码器隐藏状态必须提供
                    encoder_hidden_states is not None
                ), "encoder_hidden_states must be given for cross-attention layers" # 错误信息
                cross_attention_outputs = self.crossattention( # 通过交叉注意力模块进行前向传播
                    query_attention_output, # 输入 Query 部分的 attention 输出
                    attention_mask, # attention mask
                    head_mask, # head mask
                    encoder_hidden_states, # 编码器隐藏状态
                    encoder_attention_mask, # 编码器 attention mask
                    output_attentions=output_attentions, # 是否输出 attention
                )
                query_attention_output = cross_attention_outputs[0] # 获取交叉注意力的输出
                outputs = ( # 将交叉注意力的其他输出添加到 outputs 中
                    outputs + cross_attention_outputs[1:-1]
                )  # add cross attentions if we output attention weights # 如果输出 attention 权重，则添加交叉注意力 map

            layer_output = apply_chunking_to_forward( # 对 Query 部分应用前馈网络，使用 chunking
                self.feed_forward_chunk_query, # 用于 Query 的前馈网络函数
                self.chunk_size_feed_forward, # chunk 大小
                self.seq_len_dim, # 序列长度维度
                query_attention_output, # 输入 Query 部分的 attention 输出
            )
            if attention_output.shape[1] > query_length: # 如果 attention 输出包含文本部分
                layer_output_text = apply_chunking_to_forward( # 对文本部分应用前馈网络，使用 chunking
                    self.feed_forward_chunk, # 用于文本的前馈网络函数
                    self.chunk_size_feed_forward, # chunk 大小
                    self.seq_len_dim, # 序列长度维度
                    attention_output[:, query_length:, :], # 输入文本部分的 attention 输出
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1) # 将 Query 部分和文本部分的输出拼接
        else: # 如果 Query 序列长度为 0 (只处理文本部分)
            layer_output = apply_chunking_to_forward( # 对整个 attention 输出应用前馈网络，使用 chunking
                self.feed_forward_chunk, # 前馈网络函数
                self.chunk_size_feed_forward, # chunk 大小
                self.seq_len_dim, # 序列长度维度
                attention_output, # 输入 attention 输出
            )
        outputs = (layer_output,) + outputs # 将最终的层输出添加到 outputs 的开头

        outputs = outputs + (present_key_value,) # 将 present_key_value 添加到 outputs 的末尾

        return outputs # 返回输出元组

    def feed_forward_chunk(self, attention_output): # 定义用于文本部分的前馈网络 chunk 函数
        intermediate_output = self.intermediate(attention_output) # 通过 intermediate 层处理 attention 输出
        layer_output = self.output(intermediate_output, attention_output) # 通过 output 层处理 intermediate 输出和 attention 输出 (残差连接)
        return layer_output # 返回层输出

    def feed_forward_chunk_query(self, attention_output): # 定义用于 Query 部分的前馈网络 chunk 函数
        intermediate_output = self.intermediate_query(attention_output) # 通过 intermediate_query 层处理 attention 输出
        layer_output = self.output_query(intermediate_output, attention_output) # 通过 output_query 层处理 intermediate_query 输出和 attention 输出 (残差连接)
        return layer_output # 返回层输出


class BertEncoder(nn.Module): # 定义 BertEncoder 类，继承自 nn.Module
    def __init__(self, config): # 构造函数，接收一个配置对象
        super().__init__() # 调用父类构造函数
        self.config = config # 保存配置对象
        self.layer = nn.ModuleList( # 创建一个 ModuleList，包含多个 BertLayer
            [BertLayer(config, i) for i in range(config.num_hidden_layers)] # 根据配置中的层数创建 BertLayer 实例
        )

    def forward( # 定义前向传播方法
        self,
        hidden_states, # 输入的隐藏状态
        attention_mask=None, # 注意力掩码
        head_mask=None, # 头部掩码
        encoder_hidden_states=None, # 编码器隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器注意力掩码 (用于交叉注意力)
        past_key_values=None, # 过去的键值对 (用于解码器或生成)
        use_cache=None, # 是否使用缓存
        output_attentions=False, # 是否输出注意力权重
        output_hidden_states=False, # 是否输出所有层的隐藏状态
        return_dict=True, # 是否以字典形式返回输出
        query_length=0, # 查询序列的长度
    ):
        all_hidden_states = () if output_hidden_states else None # 初始化存储所有隐藏状态的元组
        all_self_attentions = () if output_attentions else None # 初始化存储所有自注意力权重的元组
        all_cross_attentions = ( # 初始化存储所有交叉注意力权重的元组
            () if output_attentions and self.config.add_cross_attention else None # 如果配置中添加了交叉注意力且需要输出注意力，则初始化
        )

        next_decoder_cache = () if use_cache else None # 初始化存储下一时刻解码器缓存的元组

        for i in range(self.config.num_hidden_layers): # 遍历每一层
            layer_module = self.layer[i] # 获取当前层的模块
            if output_hidden_states: # 如果需要输出所有隐藏状态
                all_hidden_states = all_hidden_states + (hidden_states,) # 将当前层的输入隐藏状态添加到元组中

            layer_head_mask = head_mask[i] if head_mask is not None else None # 获取当前层的头部掩码
            past_key_value = past_key_values[i] if past_key_values is not None else None # 获取当前层的过去键值对

            if getattr(self.config, "gradient_checkpointing", False) and self.training: # 如果配置中启用了梯度检查点并且模型处于训练模式

                if use_cache: # 如果同时启用了缓存
                    logger.warn( # 记录警告
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..." # 警告信息：use_cache=True 与梯度检查点不兼容，将 use_cache 设置为 False
                    )
                    use_cache = False # 将 use_cache 设置为 False

                def create_custom_forward(module): # 定义一个创建自定义前向传播函数的内部函数
                    def custom_forward(*inputs): # 定义自定义前向传播函数
                        return module( # 调用当前层的模块
                            *inputs, past_key_value, output_attentions, query_length # 传递输入参数和额外的参数
                        )

                    return custom_forward # 返回自定义前向传播函数

                layer_outputs = torch.utils.checkpoint.checkpoint( # 使用梯度检查点执行前向传播
                    create_custom_forward(layer_module), # 传递自定义前向传播函数
                    hidden_states, # 传递隐藏状态
                    attention_mask, # 传递注意力掩码
                    layer_head_mask, # 传递头部掩码
                    encoder_hidden_states, # 传递编码器隐藏状态
                    encoder_attention_mask, # 传递编码器注意力掩码
                )
            else: # 如果没有启用梯度检查点
                layer_outputs = layer_module( # 直接调用当前层模块进行前向传播
                    hidden_states, # 传递隐藏状态
                    attention_mask, # 传递注意力掩码
                    layer_head_mask, # 传递头部掩码
                    encoder_hidden_states, # 传递编码器隐藏状态
                    encoder_attention_mask, # 传递编码器注意力掩码
                    past_key_value, # 传递过去键值对
                    output_attentions, # 传递是否输出注意力权重
                    query_length, # 传递查询序列长度
                )

            hidden_states = layer_outputs[0] # 获取当前层的输出隐藏状态
            if use_cache: # 如果使用缓存
                next_decoder_cache += (layer_outputs[-1],) # 将当前层的输出缓存添加到下一时刻的缓存元组中
            if output_attentions: # 如果需要输出注意力权重
                all_self_attentions = all_self_attentions + (layer_outputs[1],) # 将当前层的自注意力权重添加到元组中
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],) # 将当前层的交叉注意力权重添加到元组中

        if output_hidden_states: # 在循环结束后，如果需要输出所有隐藏状态
            all_hidden_states = all_hidden_states + (hidden_states,) # 将最后一层的输出隐藏状态添加到元组中

        if not return_dict: # 如果不需要以字典形式返回
            return tuple( # 返回一个元组
                v # 包含所有非 None 的输出结果
                for v in [
                    hidden_states, # 最后一层的隐藏状态
                    next_decoder_cache, # 下一时刻的缓存
                    all_hidden_states, # 所有层的隐藏状态
                    all_self_attentions, # 所有层的自注意力权重
                    all_cross_attentions, # 所有层的交叉注意力权重
                ]
                if v is not None # 过滤掉 None 值
            )
        return BaseModelOutputWithPastAndCrossAttentions( # 如果需要以字典形式返回，则返回 BaseModelOutputWithPastAndCrossAttentions 对象
            last_hidden_state=hidden_states, # 最后一层的隐藏状态
            past_key_values=next_decoder_cache, # 下一时刻的缓存
            hidden_states=all_hidden_states, # 所有层的隐藏状态
            attentions=all_self_attentions, # 所有层的自注意力权重
            cross_attentions=all_cross_attentions, # 所有层的交叉注意力权重
        )


class BertPooler(nn.Module): # 定义 BertPooler 类，继承自 nn.Module
    def __init__(self, config): # 类的初始化方法，接收一个 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 定义一个线性层，输入和输出维度都是 hidden_size
        self.activation = nn.Tanh() # 定义一个 Tanh 激活函数

    def forward(self, hidden_states): # 定义前向传播方法，接收 hidden_states
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # 我们通过简单地获取对应于第一个 token 的隐藏状态来对模型进行“池化”。
        first_token_tensor = hidden_states[:, 0] # 获取序列中第一个 token 的隐藏状态
        pooled_output = self.dense(first_token_tensor) # 对第一个 token 的隐藏状态应用线性变换
        pooled_output = self.activation(pooled_output) # 对结果应用 Tanh 激活函数
        return pooled_output # 返回池化后的输出


class BertPredictionHeadTransform(nn.Module): # 定义 BertPredictionHeadTransform 类，继承自 nn.Module (用于预测头的变换层)
    def __init__(self, config): # 类的初始化方法，接收一个 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 定义一个线性层，输入和输出维度都是 hidden_size
        if isinstance(config.hidden_act, str): # 如果激活函数是字符串类型
            self.transform_act_fn = ACT2FN[config.hidden_act] # 从 ACT2FN 字典中获取对应的激活函数
        else: # 如果激活函数不是字符串类型
            self.transform_act_fn = config.hidden_act # 直接使用 config 中指定的激活函数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 定义 Layer Normalization 层

    def forward(self, hidden_states): # 定义前向传播方法，接收 hidden_states
        hidden_states = self.dense(hidden_states) # 对 hidden_states 应用线性变换
        hidden_states = self.transform_act_fn(hidden_states) # 对结果应用激活函数
        hidden_states = self.LayerNorm(hidden_states) # 对结果应用 Layer Normalization
        return hidden_states # 返回处理后的 hidden_states


class BertLMPredictionHead(nn.Module): # 定义 BertLMPredictionHead 类，继承自 nn.Module (用于语言模型预测的头部)
    def __init__(self, config): # 类的初始化方法，接收一个 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.transform = BertPredictionHeadTransform(config) # 定义一个 BertPredictionHeadTransform 模块

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # 输出权重与输入嵌入相同，但每个 token 有一个仅用于输出的偏置。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 定义一个线性层，将 hidden_size 映射到 vocab_size，不使用偏置

        self.bias = nn.Parameter(torch.zeros(config.vocab_size)) # 定义一个可学习的偏置参数，形状为 vocab_size

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # 需要在这两个变量之间建立链接，以便在使用 `resize_token_embeddings` 时正确调整偏置的大小
        self.decoder.bias = self.bias # 将定义的偏置参数赋值给 decoder 线性层的偏置

    def forward(self, hidden_states): # 定义前向传播方法，接收 hidden_states
        hidden_states = self.transform(hidden_states) # 通过 transform 模块处理 hidden_states
        hidden_states = self.decoder(hidden_states) # 通过 decoder 线性层进行预测
        return hidden_states # 返回预测结果


class BertOnlyMLMHead(nn.Module): # 定义 BertOnlyMLMHead 类，继承自 nn.Module (仅用于 Masked Language Model 的头部)
    def __init__(self, config): # 类的初始化方法，接收一个 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.predictions = BertLMPredictionHead(config) # 定义一个 BertLMPredictionHead 模块

    def forward(self, sequence_output): # 定义前向传播方法，接收 sequence_output
        prediction_scores = self.predictions(sequence_output) # 通过 predictions 模块计算预测分数
        return prediction_scores # 返回预测分数


class BertPreTrainedModel(PreTrainedModel): # 定义 BertPreTrainedModel 抽象类，继承自 PreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    config_class = BertConfig # 指定模型的配置类为 BertConfig
    base_model_prefix = "bert" # 指定模型的基模型前缀为 "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"] # 指定加载模型时忽略的缺失键列表，这里是 position_ids

    def _init_weights(self, module): # 定义权重初始化方法
        """Initialize the weights""" # 初始化权重
        if isinstance(module, (nn.Linear, nn.Embedding)): # 如果模块是线性层或嵌入层
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 与使用 truncated_normal 进行初始化的 TensorFlow 版本略有不同
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) # 使用正态分布初始化权重
        elif isinstance(module, nn.LayerNorm): # 如果模块是 Layer Normalization 层
            module.bias.data.zero_() # 将偏置初始化为零
            module.weight.data.fill_(1.0) # 将权重初始化为一
        if isinstance(module, nn.Linear) and module.bias is not None: # 如果模块是线性层且有偏置
            module.bias.data.zero_() # 将偏置初始化为零


class BertModel(BertPreTrainedModel): # 定义 BertModel 类，继承自 BertPreTrainedModel
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    该模型可以作为编码器（仅包含自注意力）或解码器。作为解码器时，在自注意力层之间会添加一层交叉注意力，遵循 Ashish Vaswani 等人在 `Attention is all you need <https://arxiv.org/abs/1706.03762>`__ 中描述的架构。
    参数和 :obj:`add_cross_attention` 设置为 :obj:`True`；此时在前向传播中需要输入 :obj:`encoder_hidden_states`。
    """

    def __init__(self, config, add_pooling_layer=False): # 类的初始化方法，接收 config 对象和是否添加池化层的标志
        super().__init__(config) # 调用父类 BertPreTrainedModel 的初始化方法
        self.config = config # 保存 config 对象

        self.embeddings = BertEmbeddings(config) # 定义 BertEmbeddings 模块，用于处理输入嵌入

        self.encoder = BertEncoder(config) # 定义 BertEncoder 模块，包含多个 Transformer 层

        self.pooler = BertPooler(config) if add_pooling_layer else None # 根据 add_pooling_layer 标志决定是否定义 BertPooler 模块

        self.init_weights() # 调用父类的权重初始化方法

    def get_input_embeddings(self): # 获取输入嵌入层
        return self.embeddings.word_embeddings # 返回词嵌入层

    def set_input_embeddings(self, value): # 设置输入嵌入层
        self.embeddings.word_embeddings = value # 将词嵌入层设置为指定的值

    def _prune_heads(self, heads_to_prune): # 剪枝注意力头的方法
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        剪枝模型的注意力头。heads_to_prune: 字典，格式为 {层编号: [该层要剪枝的注意力头列表]}。详见基类 PreTrainedModel。
        """
        for layer, heads in heads_to_prune.items(): # 遍历要剪枝的层和对应的注意力头列表
            self.encoder.layer[layer].attention.prune_heads(heads) # 调用对应层的注意力模块的剪枝方法

    def get_extended_attention_mask( # 获取扩展的注意力掩码
        self,
        attention_mask: Tensor, # 输入的注意力掩码
        input_shape: Tuple[int], # 输入的形状
        device: device, # 设备
        is_decoder: bool, # 是否是解码器
        has_query: bool = False, # 是否包含 query (用于 UniLM 风格的注意力)
    ) -> Tensor: # 返回扩展的注意力掩码
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
        使注意力掩码和因果掩码可广播，以便忽略未来和被掩码的 token。

        参数:
            attention_mask (:obj:`torch.Tensor`):
                掩码，用 1 表示要关注的 token，用 0 表示要忽略的 token。
            input_shape (:obj:`Tuple[int]`):
                模型输入的形状。
            device: (:obj:`torch.device`):
                模型输入的设备。

        返回:
            :obj:`torch.Tensor` 扩展的注意力掩码，其数据类型与 :obj:`attention_mask.dtype` 相同。
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 我们可以自己提供一个维度为 [batch_size, from_seq_length, to_seq_length] 的自注意力掩码，
        # 此时我们只需要使其可广播到所有注意力头。
        if attention_mask.dim() == 3: # 如果注意力掩码是 3 维的
            extended_attention_mask = attention_mask[:, None, :, :] # 扩展维度使其可广播到注意力头维度
        elif attention_mask.dim() == 2: # 如果注意力掩码是 2 维的 (通常是 padding mask)
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # 提供了维度为 [batch_size, seq_length] 的 padding mask
            # - 如果模型是解码器，除了 padding mask 外还应用因果掩码
            # - 如果模型是编码器，使掩码可广播到 [batch_size, num_heads, seq_length, seq_length]
            if is_decoder: # 如果是解码器
                batch_size, seq_length = input_shape # 获取 batch_size 和序列长度

                seq_ids = torch.arange(seq_length, device=device) # 创建序列 ID 张量
                causal_mask = ( # 创建因果掩码
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1) # 扩展并重复序列 ID
                    <= seq_ids[None, :, None] # 与扩展的序列 ID 进行比较，生成下三角掩码
                )

                # add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                # 在因果掩码前添加一个全 1 的前缀掩码
                # 在 pytorch < 1.3 版本中，因果掩码和注意力掩码必须具有相同类型
                causal_mask = causal_mask.to(attention_mask.dtype) # 将因果掩码转换为与注意力掩码相同的数据类型

                if causal_mask.shape[1] < attention_mask.shape[1]: # 如果因果掩码的序列长度小于注意力掩码的序列长度 (通常是因为有 query token)
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1] # 计算前缀序列长度
                    if has_query:  # UniLM style attention mask # 如果包含 query (UniLM 风格的注意力掩码)
                        causal_mask = torch.cat( # 拼接掩码
                            [
                                torch.zeros( # 添加一个全零的掩码用于 query 对 input 的注意力
                                    (batch_size, prefix_seq_len, seq_length),
                                    device=device,
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask, # 原始的因果掩码
                            ],
                            axis=1, # 沿序列维度拼接
                        )
                    causal_mask = torch.cat( # 拼接掩码
                        [
                            torch.ones( # 添加一个全 1 的掩码用于 input 对 query 的注意力
                                (batch_size, causal_mask.shape[1], prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask, # 之前拼接或原始的因果掩码
                        ],
                        axis=-1, # 沿序列维度拼接
                    )
                extended_attention_mask = ( # 结合因果掩码和 padding mask
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :] # 扩展维度并相乘
                )
            else: # 如果是编码器
                extended_attention_mask = attention_mask[:, None, None, :] # 扩展维度使其可广播到注意力头和序列维度
        else: # 如果注意力掩码维度不是 2 或 3
            raise ValueError( # 抛出值错误
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format( # 错误信息：input_ids 或 attention_mask 的形状错误
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # 由于 attention_mask 对于我们想要关注的位置是 1.0，对于被掩码的位置是 0.0，
        # 这个操作将创建一个张量，对于我们想要关注的位置是 0.0，对于被掩码的位置是 -10000.0。
        # 由于我们在 softmax 之前的原始分数上加上它，这实际上等同于完全移除这些位置。
        extended_attention_mask = extended_attention_mask.to( # 转换数据类型
            dtype=self.dtype # 转换为模型的数据类型 (例如 fp16 兼容性)
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # 将 1/0 掩码转换为 0/-10000 掩码
        return extended_attention_mask # 返回扩展的注意力掩码

    def forward( # 定义前向传播方法
        self,
        input_ids=None, # 输入 token ID
        attention_mask=None, # 注意力掩码
        position_ids=None, # 位置 ID
        head_mask=None, # 注意力头掩码
        query_embeds=None, # Query 嵌入 (用于 Q-Former)
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的注意力掩码 (用于交叉注意力)
        past_key_values=None, # 过去的 key 和 value (用于解码加速)
        use_cache=None, # 是否使用缓存
        output_attentions=None, # 是否输出注意力权重
        output_hidden_states=None, # 是否输出所有层的隐藏状态
        return_dict=None, # 是否以字典形式返回输出
        is_decoder=False, # 是否作为解码器运行
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
        encoder_hidden_states (:obj:`torch.FloatTensor` 形状为 :obj:`(batch_size, sequence_length, hidden_size)`, `可选`):
            编码器最后一层输出的隐藏状态序列。如果模型配置为解码器，则在交叉注意力中使用。
        encoder_attention_mask (:obj:`torch.FloatTensor` 形状为 :obj:`(batch_size, sequence_length)`, `可选`):
            用于避免对编码器输入的 padding token 索引执行注意力的掩码。如果模型配置为解码器，则在交叉注意力中使用此掩码。掩码值选择在 ``[0, 1]`` 之间:
            - 1 表示 **未被掩码** 的 token，
            - 0 表示 **被掩码** 的 token。
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` 长度为 :obj:`config.n_layers`，每个 tuple 包含 4 个形状为 :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)` 的张量):
            包含预先计算好的注意力块的 key 和 value 隐藏状态。可用于加速解码。
            如果使用 :obj:`past_key_values`，用户可以选择只输入形状为 :obj:`(batch_size, 1)` 的最后一个 :obj:`decoder_input_ids`
            （那些没有将其过去的 key value 状态提供给此模型的 token），而不是形状为 :obj:`(batch_size, sequence_length)` 的所有 :obj:`decoder_input_ids`。
        use_cache (:obj:`bool`, `可选`):
            如果设置为 :obj:`True`，则返回 :obj:`past_key_values` key value 状态，可用于加速解码（参见 :obj:`past_key_values`）。
        """
        output_attentions = ( # 确定是否输出注意力权重
            output_attentions # 如果已指定，则使用指定值
            if output_attentions is not None
            else self.config.output_attentions # 否则使用 config 中的默认值
        )
        output_hidden_states = ( # 确定是否输出所有层的隐藏状态
            output_hidden_states # 如果已指定，则使用指定值
            if output_hidden_states is not None
            else self.config.output_hidden_states # 否则使用 config 中的默认值
        )
        return_dict = ( # 确定是否以字典形式返回输出
            return_dict # 如果已指定，则使用指定值
            if return_dict is not None
            else self.config.use_return_dict # 否则使用 config 中的默认值
        )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache # 确定是否使用缓存 (此行被注释掉)

        if input_ids is None: # 如果 input_ids 为 None
            assert ( # 断言 query_embeds 不为 None
                query_embeds is not None
            ), "You have to specify query_embeds when input_ids is None" # 错误信息：当 input_ids 为 None 时，必须指定 query_embeds

        # past_key_values_length # 过去的 key value 长度
        past_key_values_length = ( # 计算过去的 key value 长度
            past_key_values[0][0].shape[2] - self.config.query_length # 如果 past_key_values 不为 None，则计算其序列长度减去 query 长度
            if past_key_values is not None
            else 0 # 否则为 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0 # 计算 query 的序列长度，如果 query_embeds 为 None 则为 0

        embedding_output = self.embeddings( # 获取嵌入输出
            input_ids=input_ids, # 输入 token ID
            position_ids=position_ids, # 位置 ID
            query_embeds=query_embeds, # Query 嵌入
            past_key_values_length=past_key_values_length, # 过去的 key value 长度
        )

        input_shape = embedding_output.size()[:-1] # 获取输入形状 (不包含 hidden_size)
        batch_size, seq_length = input_shape # 获取 batch_size 和序列长度
        device = embedding_output.device # 获取设备

        if attention_mask is None: # 如果注意力掩码为 None
            attention_mask = torch.ones( # 创建一个全 1 的注意力掩码
                ((batch_size, seq_length + past_key_values_length)), device=device # 形状为 (batch_size, 序列长度 + 过去的 key value 长度)
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 我们可以自己提供一个维度为 [batch_size, from_seq_length, to_seq_length] 的自注意力掩码，
        # 此时我们只需要使其可广播到所有注意力头。
        if is_decoder: # 如果是解码器
            extended_attention_mask = self.get_extended_attention_mask( # 获取扩展的注意力掩码
                attention_mask, # 输入的注意力掩码
                input_ids.shape, # 输入 ID 的形状
                device, # 设备
                is_decoder, # 是否是解码器
                has_query=(query_embeds is not None), # 是否包含 query
            )
        else: # 如果是编码器
            extended_attention_mask = self.get_extended_attention_mask( # 获取扩展的注意力掩码
                attention_mask, input_shape, device, is_decoder # 输入的注意力掩码、输入形状、设备、是否是解码器
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # 如果为交叉注意力提供了 2D 或 3D 注意力掩码
        # 我们需要使其可广播到 [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None: # 如果提供了编码器的隐藏状态 (进行交叉注意力)
            if type(encoder_hidden_states) == list: # 如果编码器隐藏状态是列表 (可能来自多个编码器)
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[ # 获取第一个编码器隐藏状态的形状
                    0
                ].size()
            else: # 如果编码器隐藏状态不是列表
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size() # 获取编码器隐藏状态的形状
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length) # 构造编码器隐藏状态的形状元组

            if type(encoder_attention_mask) == list: # 如果编码器注意力掩码是列表
                encoder_extended_attention_mask = [ # 遍历列表并反转掩码
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            elif encoder_attention_mask is None: # 如果编码器注意力掩码为 None
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device) # 创建一个全 1 的编码器注意力掩码
                encoder_extended_attention_mask = self.invert_attention_mask( # 反转掩码
                    encoder_attention_mask
                )
            else: # 如果编码器注意力掩码不是列表也不是 None
                encoder_extended_attention_mask = self.invert_attention_mask( # 反转掩码
                    encoder_attention_mask
                )
        else: # 如果没有提供编码器的隐藏状态
            encoder_extended_attention_mask = None # 编码器的扩展注意力掩码为 None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # 如果需要，准备注意力头掩码
        # head_mask 中的 1.0 表示保留该注意力头
        # attention_probs 的形状是 bsz x n_heads x N x N
        # 输入的 head_mask 形状是 [num_heads] 或 [num_hidden_layers x num_heads]
        # head_mask 会被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers) # 获取处理后的注意力头掩码

        encoder_outputs = self.encoder( # 通过编码器进行前向传播
            embedding_output, # 嵌入输出
            attention_mask=extended_attention_mask, # 扩展的自注意力掩码
            head_mask=head_mask, # 注意力头掩码
            encoder_hidden_states=encoder_hidden_states, # 编码器的隐藏状态 (用于交叉注意力)
            encoder_attention_mask=encoder_extended_attention_mask, # 编码器的扩展注意力掩码 (用于交叉注意力)
            past_key_values=past_key_values, # 过去的 key 和 value
            use_cache=use_cache, # 是否使用缓存
            output_attentions=output_attentions, # 是否输出注意力权重
            output_hidden_states=output_hidden_states, # 是否输出所有层的隐藏状态
            return_dict=return_dict, # 是否以字典形式返回输出
            query_length=query_length, # Query 的序列长度
        )
        sequence_output = encoder_outputs[0] # 获取编码器的序列输出 (通常是最后一层的隐藏状态)
        pooled_output = ( # 获取池化输出
            self.pooler(sequence_output) if self.pooler is not None else None # 如果存在 pooler，则对其应用，否则为 None
        )

        if not return_dict: # 如果不以字典形式返回
            return (sequence_output, pooled_output) + encoder_outputs[1:] # 返回序列输出、池化输出以及编码器的其他输出

        return BaseModelOutputWithPoolingAndCrossAttentions( # 以字典形式返回输出 (使用 BaseModelOutputWithPoolingAndCrossAttentions 数据类)
            last_hidden_state=sequence_output, # 最后一层的隐藏状态
            pooler_output=pooled_output, # 池化输出
            past_key_values=encoder_outputs.past_key_values, # 过去的 key 和 value
            hidden_states=encoder_outputs.hidden_states, # 所有层的隐藏状态
            attentions=encoder_outputs.attentions, # 注意力权重
            cross_attentions=encoder_outputs.cross_attentions, # 交叉注意力权重
        )


class BertLMHeadModel(BertPreTrainedModel): # 定义 BertLMHeadModel 类，继承自 BertPreTrainedModel (用于语言模型任务)

    _keys_to_ignore_on_load_unexpected = [r"pooler"] # 加载预训练模型时，忽略意外出现的键列表，这里是 pooler
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"] # 加载预训练模型时，忽略缺失的键列表，这里是 position_ids 和 predictions.decoder.bias

    def __init__(self, config): # 类的初始化方法，接收一个 config 对象
        super().__init__(config) # 调用父类 BertPreTrainedModel 的初始化方法

        self.bert = BertModel(config, add_pooling_layer=False) # 定义 BertModel 模块作为基础模型，不添加池化层
        self.cls = BertOnlyMLMHead(config) # 定义 BertOnlyMLMHead 模块作为语言模型预测头部

        self.init_weights() # 调用父类的权重初始化方法

    def get_output_embeddings(self): # 获取输出嵌入层 (即预测层的权重)
        return self.cls.predictions.decoder # 返回预测头部中的 decoder 线性层

    def set_output_embeddings(self, new_embeddings): # 设置输出嵌入层
        self.cls.predictions.decoder = new_embeddings # 将预测头部中的 decoder 线性层设置为新的嵌入

    def forward( # 定义前向传播方法
        self,
        input_ids=None, # 输入 token ID
        attention_mask=None, # 注意力掩码
        position_ids=None, # 位置 ID
        head_mask=None, # 注意力头掩码
        query_embeds=None, # Query 嵌入 (用于 Q-Former)
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的注意力掩码 (用于交叉注意力)
        labels=None, # 标签 (用于计算损失)
        past_key_values=None, # 过去的 key 和 value (用于解码加速)
        use_cache=True, # 是否使用缓存 (默认为 True)
        output_attentions=None, # 是否输出注意力权重
        output_hidden_states=None, # 是否输出所有层的隐藏状态
        return_dict=None, # 是否以字典形式返回输出
        return_logits=False, # 是否只返回 logits
        is_decoder=True, # 是否作为解码器运行 (默认为 True)
        reduction="mean", # 损失函数的 reduction 方式 (默认为 "mean")
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
        encoder_hidden_states (:obj:`torch.FloatTensor` 形状为 :obj:`(batch_size, sequence_length, hidden_size)`, `可选`):
            编码器最后一层输出的隐藏状态序列。如果模型配置为解码器，则在交叉注意力中使用。
        encoder_attention_mask (:obj:`torch.FloatTensor` 形状为 :obj:`(batch_size, sequence_length)`, `可选`):
            用于避免对编码器输入的 padding token 索引执行注意力的掩码。如果模型配置为解码器，则在交叉注意力中使用此掩码。掩码值选择在 ``[0, 1]`` 之间:
            - 1 表示 **未被掩码** 的 token，
            - 0 表示 **被掩码** 的 token。
        labels (:obj:`torch.LongTensor` 形状为 :obj:`(batch_size, sequence_length)`, `可选`):
            用于计算从左到右语言模型损失（下一个词预测）的标签。索引应在 ``[-100, 0, ..., config.vocab_size]`` 之间（参见 ``input_ids`` 文档字符串）。索引设置为 ``-100`` 的 token 将被忽略（被掩码），损失仅针对标签在 ``[0, ..., config.vocab_size]`` 中的 token 计算。
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` 长度为 :obj:`config.n_layers`，每个 tuple 包含 4 个形状为 :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)` 的张量):
            包含预先计算好的注意力块的 key 和 value 隐藏状态。可用于加速解码。
            如果使用 :obj:`past_key_values`，用户可以选择只输入形状为 :obj:`(batch_size, 1)` 的最后一个 :obj:`decoder_input_ids`
            （那些没有将其过去的 key value 状态提供给此模型的 token），而不是形状为 :obj:`(batch_size, sequence_length)` 的所有 :obj:`decoder_input_ids`。
        use_cache (:obj:`bool`, `可选`):
            如果设置为 :obj:`True`，则返回 :obj:`past_key_values` key value 状态，可用于加速解码（参见 :obj:`past_key_values`）。
        返回:
        示例::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = ( # 确定是否以字典形式返回输出
            return_dict # 如果已指定，则使用指定值
            if return_dict is not None
            else self.config.use_return_dict # 否则使用 config 中的默认值
        )
        if labels is not None: # 如果提供了标签
            use_cache = False # 则不使用缓存 (因为通常是训练模式)
        if past_key_values is not None: # 如果提供了过去的 key value
            query_embeds = None # 则忽略 query_embeds (因为通常是生成模式，query 已经融入 past_key_values)

        outputs = self.bert( # 通过 BertModel 进行前向传播
            input_ids, # 输入 token ID
            attention_mask=attention_mask, # 注意力掩码
            position_ids=position_ids, # 位置 ID
            head_mask=head_mask, # 注意力头掩码
            query_embeds=query_embeds, # Query 嵌入
            encoder_hidden_states=encoder_hidden_states, # 编码器的隐藏状态
            encoder_attention_mask=encoder_attention_mask, # 编码器的注意力掩码
            past_key_values=past_key_values, # 过去的 key value
            use_cache=use_cache, # 是否使用缓存
            output_attentions=output_attentions, # 是否输出注意力权重
            output_hidden_states=output_hidden_states, # 是否输出所有层的隐藏状态
            return_dict=return_dict, # 是否以字典形式返回
            is_decoder=is_decoder, # 是否是解码器
        )

        sequence_output = outputs[0] # 获取 BertModel 的序列输出 (通常是最后一层的隐藏状态)
        if query_embeds is not None: # 如果使用了 query 嵌入
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :] # 移除 query 部分的输出，只保留输入序列的输出

        prediction_scores = self.cls(sequence_output) # 通过语言模型预测头部计算预测分数

        if return_logits: # 如果只返回 logits
            return prediction_scores[:, :-1, :].contiguous() # 返回预测分数，移除最后一个 token 的预测 (因为是下一个词预测)，并确保内存连续

        lm_loss = None # 初始化语言模型损失为 None
        if labels is not None: # 如果提供了标签
            # we are doing next-token prediction; shift prediction scores and input ids by one
            # 我们正在进行下一个 token 预测；将预测分数和输入 ID 向前移动一个位置
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous() # 移除最后一个 token 的预测分数，并确保内存连续
            labels = labels[:, 1:].contiguous() # 移除第一个 token 的标签，并确保内存连续 (标签对应于下一个 token)
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1) # 定义交叉熵损失函数，指定 reduction 方式和标签平滑
            lm_loss = loss_fct( # 计算语言模型损失
                shifted_prediction_scores.view(-1, self.config.vocab_size), # 将预测分数展平为 (batch_size * sequence_length, vocab_size)
                labels.view(-1), # 将标签展平为 (batch_size * sequence_length)
            )
            if reduction == "none": # 如果 reduction 方式是 "none"
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1) # 将损失重新 reshape 并按 batch 维度求和

        if not return_dict: # 如果不以字典形式返回
            output = (prediction_scores,) + outputs[2:] # 构建输出 tuple，包含预测分数和 BertModel 的其他输出 (除了第一个元素 sequence_output)
            return ((lm_loss,) + output) if lm_loss is not None else output # 如果计算了损失，则将损失添加到输出 tuple 的开头，否则直接返回输出 tuple

        return CausalLMOutputWithCrossAttentions( # 以字典形式返回输出 (使用 CausalLMOutputWithCrossAttentions 数据类)
            loss=lm_loss, # 语言模型损失
            logits=prediction_scores, # 预测分数
            past_key_values=outputs.past_key_values, # 过去的 key value
            hidden_states=outputs.hidden_states, # 所有层的隐藏状态
            attentions=outputs.attentions, # 注意力权重
            cross_attentions=outputs.cross_attentions, # 交叉注意力权重
        )

    def prepare_inputs_for_generation( # 为生成任务准备输入
        self, input_ids, query_embeds, past=None, attention_mask=None, **model_kwargs # 输入 token ID, query 嵌入, 过去的 key value, 注意力掩码, 其他模型参数
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        # 如果模型在编码器-解码器模型中用作解码器，则解码器注意力掩码是动态创建的
        if attention_mask is None: # 如果注意力掩码为 None
            attention_mask = input_ids.new_ones(input_ids.shape) # 创建一个与 input_ids 形状相同的全 1 掩码
        query_mask = input_ids.new_ones(query_embeds.shape[:-1]) # 创建一个与 query_embeds 形状（不含 hidden_size）相同的全 1 掩码
        attention_mask = torch.cat([query_mask, attention_mask], dim=-1) # 将 query 掩码和输入掩码拼接起来

        # cut decoder_input_ids if past is used
        # 如果使用了 past，则截断 decoder_input_ids
        if past is not None: # 如果提供了过去的 key value
            input_ids = input_ids[:, -1:] # 只保留 input_ids 的最后一个 token (用于自回归生成)

        return { # 返回一个字典，包含为生成任务准备好的输入
            "input_ids": input_ids, # 输入 token ID
            "query_embeds": query_embeds, # Query 嵌入
            "attention_mask": attention_mask, # 注意力掩码
            "past_key_values": past, # 过去的 key value
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None), # 编码器的隐藏状态 (从 model_kwargs 中获取)
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None), # 编码器的注意力掩码 (从 model_kwargs 中获取)
            "is_decoder": True, # 标记为解码器模式
        }

    def _reorder_cache(self, past, beam_idx): # 根据 beam search 的索引重新排序缓存
        reordered_past = () # 初始化重新排序后的缓存 tuple
        for layer_past in past: # 遍历每一层的缓存
            reordered_past += ( # 将重新排序后的当前层缓存添加到结果 tuple 中
                tuple( # 创建当前层重新排序后的缓存 tuple
                    past_state.index_select(0, beam_idx) for past_state in layer_past # 对当前层缓存中的每个张量根据 beam_idx 重新排序 (沿 batch 维度)
                ),
            )
        return reordered_past # 返回重新排序后的缓存


class BertForMaskedLM(BertPreTrainedModel): # 定义 BertForMaskedLM 类，继承自 BertPreTrainedModel (用于 Masked Language Model 任务)

    _keys_to_ignore_on_load_unexpected = [r"pooler"] # 加载预训练模型时，忽略意外出现的键列表，这里是 pooler
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"] # 加载预训练模型时，忽略缺失的键列表，这里是 position_ids 和 predictions.decoder.bias

    def __init__(self, config): # 类的初始化方法，接收一个 config 对象
        super().__init__(config) # 调用父类 BertPreTrainedModel 的初始化方法

        self.bert = BertModel(config, add_pooling_layer=False) # 定义 BertModel 模块作为基础模型，不添加池化层
        self.cls = BertOnlyMLMHead(config) # 定义 BertOnlyMLMHead 模块作为 Masked Language Model 预测头部

        self.init_weights() # 调用父类的权重初始化方法

    def get_output_embeddings(self): # 获取输出嵌入层 (即预测层的权重)
        return self.cls.predictions.decoder # 返回预测头部中的 decoder 线性层

    def set_output_embeddings(self, new_embeddings): # 设置输出嵌入层
        self.cls.predictions.decoder = new_embeddings # 将预测头部中的 decoder 线性层设置为新的嵌入

    def forward( # 定义前向传播方法
        self,
        input_ids=None, # 输入 token ID
        attention_mask=None, # 注意力掩码
        position_ids=None, # 位置 ID
        head_mask=None, # 注意力头掩码
        query_embeds=None, # Query 嵌入 (用于 Q-Former)
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的注意力掩码 (用于交叉注意力)
        labels=None, # 标签 (用于计算损失)
        output_attentions=None, # 是否输出注意力权重
        output_hidden_states=None, # 是否输出所有层的隐藏状态
        return_dict=None, # 是否以字典形式返回输出
        return_logits=False, # 是否只返回 logits
        is_decoder=False, # 是否作为解码器运行 (默认为 False，因为是 MLM 任务)
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        labels (:obj:`torch.LongTensor` 形状为 :obj:`(batch_size, sequence_length)`, `可选`):
            用于计算 Masked Language Model 损失的标签。索引应在 ``[-100, 0, ..., config.vocab_size]`` 之间（参见 ``input_ids`` 文档字符串）。索引设置为 ``-100`` 的 token 将被忽略（被掩码），损失仅针对标签在 ``[0, ..., config.vocab_size]`` 中的 token 计算。
        """

        return_dict = ( # 确定是否以字典形式返回输出
            return_dict # 如果已指定，则使用指定值
            if return_dict is not None
            else self.config.use_return_dict # 否则使用 config 中的默认值
        )

        outputs = self.bert( # 通过 BertModel 进行前向传播
            input_ids, # 输入 token ID
            attention_mask=attention_mask, # 注意力掩码
            position_ids=position_ids, # 位置 ID
            head_mask=head_mask, # 注意力头掩码
            query_embeds=query_embeds, # Query 嵌入
            encoder_hidden_states=encoder_hidden_states, # 编码器的隐藏状态
            encoder_attention_mask=encoder_attention_mask, # 编码器的注意力掩码
            output_attentions=output_attentions, # 是否输出注意力权重
            output_hidden_states=output_hidden_states, # 是否输出所有层的隐藏状态
            return_dict=return_dict, # 是否以字典形式返回
            is_decoder=is_decoder, # 是否是解码器
        )

        if query_embeds is not None: # 如果使用了 query 嵌入
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :] # 移除 query 部分的输出，只保留输入序列的输出
        prediction_scores = self.cls(sequence_output) # 通过 Masked Language Model 预测头部计算预测分数

        if return_logits: # 如果只返回 logits
            return prediction_scores # 返回预测分数

        masked_lm_loss = None # 初始化 Masked Language Model 损失为 None
        if labels is not None: # 如果提供了标签
            loss_fct = CrossEntropyLoss()  # -100 index = padding token # 定义交叉熵损失函数，-100 索引表示 padding token，将被忽略
            masked_lm_loss = loss_fct( # 计算 Masked Language Model 损失
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1) # 将预测分数和标签展平后计算损失
            )

        if not return_dict: # 如果不以字典形式返回
            output = (prediction_scores,) + outputs[2:] # 构建输出 tuple，包含预测分数和 BertModel 的其他输出 (除了第一个元素 sequence_output)
            return ( # 返回结果
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output # 如果计算了损失，则将损失添加到输出 tuple 的开头，否则直接返回输出 tuple
            )

        return MaskedLMOutput( # 以字典形式返回输出 (使用 MaskedLMOutput 数据类)
            loss=masked_lm_loss, # Masked Language Model 损失
            logits=prediction_scores, # 预测分数
            hidden_states=outputs.hidden_states, # 所有层的隐藏状态
            attentions=outputs.attentions, # 注意力权重
        )
