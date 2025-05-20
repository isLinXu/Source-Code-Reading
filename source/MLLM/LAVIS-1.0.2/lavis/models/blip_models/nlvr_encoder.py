"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math # 导入 math 模块
from typing import Tuple # 从 typing 模块导入 Tuple 类型提示

import torch # 导入 PyTorch 库
import torch.utils.checkpoint # 导入 PyTorch 的 checkpoint 工具
from torch import Tensor, device, nn # 从 torch 导入 Tensor, device, nn
from transformers.activations import ACT2FN # 从 transformers.activations 导入 ACT2FN 激活函数映射
from transformers.modeling_outputs import ( # 从 transformers.modeling_outputs 导入模型输出类
    BaseModelOutputWithPastAndCrossAttentions, # 包含 past key values 和 cross attentions 的基础模型输出
    BaseModelOutputWithPoolingAndCrossAttentions, # 包含 pooling 和 cross attentions 的基础模型输出
)
from transformers.modeling_utils import ( # 从 transformers.modeling_utils 导入模型工具函数
    PreTrainedModel, # 预训练模型基类
    apply_chunking_to_forward, # 应用 chunking 到前向传播的函数
    find_pruneable_heads_and_indices, # 查找可剪枝的 attention heads 和索引的函数
    prune_linear_layer, # 剪枝线性层的函数
)
from transformers.models.bert.configuration_bert import BertConfig # 从 transformers.models.bert.configuration_bert 导入 BertConfig 配置类
from transformers.utils import logging # 从 transformers.utils 导入 logging 工具

logger = logging.get_logger(__name__) # 获取当前模块的 logger

class BertEmbeddings(nn.Module): # 定义 BertEmbeddings 类，继承自 nn.Module
    """Construct the embeddings from word and position embeddings.""" # 从词嵌入和位置嵌入构建嵌入表示。

    def __init__(self, config): # 类的初始化方法，接收一个 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.word_embeddings = nn.Embedding( # 定义词嵌入层
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id # 词汇表大小，隐藏层大小，padding 索引
        )
        self.position_embeddings = nn.Embedding( # 定义位置嵌入层
            config.max_position_embeddings, config.hidden_size # 最大位置数，隐藏层大小
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # self.LayerNorm 没有使用 snake_case 命名，是为了与 TensorFlow 模型变量名保持一致，以便加载任何 TensorFlow checkpoint 文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 定义 Layer Normalization 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 定义 Dropout 层

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # position_ids (1, 位置嵌入长度) 在内存中是连续的，并在序列化时导出
        self.register_buffer( # 注册一个 buffer，它不是模型参数，但会随模型状态一起保存和加载
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) # 创建一个从 0 到 max_position_embeddings-1 的张量，并扩展形状为 (1, -1)
        )
        self.position_embedding_type = getattr( # 获取位置嵌入类型，默认为 "absolute"
            config, "position_embedding_type", "absolute"
        )

        self.config = config # 保存配置对象

    def forward( # 定义前向传播方法
        self,
        input_ids=None, # 输入的 token ID
        position_ids=None, # 输入的位置 ID
        inputs_embeds=None, # 输入的嵌入表示
        past_key_values_length=0, # 过去 key values 的长度，用于增量解码
    ):
        if input_ids is not None: # 如果提供了 input_ids
            input_shape = input_ids.size() # 获取输入形状
        else: # 如果没有提供 input_ids，则使用 inputs_embeds
            input_shape = inputs_embeds.size()[:-1] # 获取输入形状（去除最后一个维度）

        seq_length = input_shape[1] # 获取序列长度

        if position_ids is None: # 如果没有提供 position_ids
            position_ids = self.position_ids[ # 根据序列长度和 past_key_values_length 生成 position_ids
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if inputs_embeds is None: # 如果没有提供 inputs_embeds
            inputs_embeds = self.word_embeddings(input_ids) # 通过 word_embeddings 层获取嵌入表示

        embeddings = inputs_embeds # 初始化 embeddings 为 inputs_embeds

        if self.position_embedding_type == "absolute": # 如果位置嵌入类型是绝对位置嵌入
            position_embeddings = self.position_embeddings(position_ids) # 通过 position_embeddings 层获取位置嵌入
            embeddings += position_embeddings # 将位置嵌入加到 inputs_embeds 上
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
        mixed_query_layer = self.query(hidden_states) # 对 hidden_states 进行 Query 线性变换

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        # 如果这是一个交叉注意力模块的实例，Key 和 Value 来自编码器；attention mask 需要确保编码器的 padding token 不被关注。
        is_cross_attention = encoder_hidden_states is not None # 判断是否是交叉注意力

        if is_cross_attention: # 如果是交叉注意力
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states)) # 对编码器隐藏状态进行 Key 线性变换并转置
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states)) # 对编码器隐藏状态进行 Value 线性变换并转置
            attention_mask = encoder_attention_mask # 使用编码器的 attention mask
        elif past_key_value is not None: # 如果不是交叉注意力且提供了 past_key_value (用于增量解码)
            key_layer = self.transpose_for_scores(self.key(hidden_states)) # 对当前 hidden_states 进行 Key 线性变换并转置
            value_layer = self.transpose_for_scores(self.value(hidden_states)) # 对当前 hidden_states 进行 Value 线性变换并转置
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2) # 将过去的 Key 和当前的 Key 拼接
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2) # 将过去的 Value 和当前的 Value 拼接
        else: # 如果是自注意力且没有提供 past_key_value
            key_layer = self.transpose_for_scores(self.key(hidden_states)) # 对 hidden_states 进行 Key 线性变换并转置
            value_layer = self.transpose_for_scores(self.value(hidden_states)) # 对 hidden_states 进行 Value 线性变换并转置

        query_layer = self.transpose_for_scores(mixed_query_layer) # 对 Query 线性变换结果进行转置

        past_key_value = (key_layer, value_layer) # 更新 past_key_value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算 Query 和 Key 的点积，得到原始的 attention scores。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # 计算 Query 和 Key 的矩阵乘法

        if ( # 如果位置嵌入类型是 "relative_key" 或 "relative_key_query"
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1] # 获取序列长度
            position_ids_l = torch.arange( # 创建左侧位置 ID 张量
                seq_length, dtype=torch.long, device=hidden_states.device # 形状为 (seq_length, 1)
            ).view(-1, 1)
            position_ids_r = torch.arange( # 创建右侧位置 ID 张量
                seq_length, dtype=torch.long, device=hidden_states.device # 形状为 (1, seq_length)
            ).view(1, -1)
            distance = position_ids_l - position_ids_r # 计算位置之间的距离矩阵
            positional_embedding = self.distance_embedding( # 通过距离嵌入层获取位置嵌入
                distance + self.max_position_embeddings - 1 # 调整距离，使其索引从 0 开始
            )
            positional_embedding = positional_embedding.to( # 转换为与 query_layer 相同的 dtype
                dtype=query_layer.dtype
            )  # fp16 compatibility # 兼容 fp16

            if self.position_embedding_type == "relative_key": # 如果是 "relative_key" 位置嵌入
                relative_position_scores = torch.einsum( # 使用 einsum 计算相对位置分数
                    "bhld,lrd->bhlr", query_layer, positional_embedding # 形状为 (batch, heads, seq_len, seq_len)
                )
                attention_scores = attention_scores + relative_position_scores # 将相对位置分数加到 attention scores 上
            elif self.position_embedding_type == "relative_key_query": # 如果是 "relative_key_query" 位置嵌入
                relative_position_scores_query = torch.einsum( # 计算 Query 与位置嵌入的相对位置分数
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum( # 计算 Key 与位置嵌入的相对位置分数
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
    def __init__(self, config, twin=False, merge=False): # 类的初始化方法，接收 config 对象，twin 标志和 merge 标志
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 定义 Layer Normalization 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 定义 Dropout 层
        if twin: # 如果 twin 为 True (通常用于交叉注意力)
            self.dense0 = nn.Linear(config.hidden_size, config.hidden_size) # 定义第一个线性层
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size) # 定义第二个线性层
        else: # 如果 twin 为 False (通常用于自注意力)
            self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 定义一个线性层
        if merge: # 如果 merge 为 True (通常用于交叉注意力且层数 >= 6)
            self.act = ACT2FN[config.hidden_act] # 获取激活函数
            self.merge_layer = nn.Linear(config.hidden_size * 2, config.hidden_size) # 定义合并线性层，输入维度是 hidden_size 的两倍
            self.merge = True # 设置 merge 标志为 True
        else: # 如果 merge 为 False
            self.merge = False # 设置 merge 标志为 False

    def forward(self, hidden_states, input_tensor): # 定义前向传播方法，接收 hidden_states 和 input_tensor
        if type(hidden_states) == list: # 如果 hidden_states 是一个列表 (通常来自 twin attention heads)
            hidden_states0 = self.dense0(hidden_states[0]) # 对列表的第一个元素应用 dense0 线性层
            hidden_states1 = self.dense1(hidden_states[1]) # 对列表的第二个元素应用 dense1 线性层
            if self.merge: # 如果 merge 为 True
                # hidden_states = self.merge_layer(self.act(torch.cat([hidden_states0,hidden_states1],dim=-1))) # 原始代码，注释掉了激活函数
                hidden_states = self.merge_layer( # 应用 merge_layer 线性层
                    torch.cat([hidden_states0, hidden_states1], dim=-1) # 将两个 hidden_states 在最后一个维度上拼接
                )
            else: # 如果 merge 为 False
                hidden_states = (hidden_states0 + hidden_states1) / 2 # 将两个 hidden_states 相加并取平均
        else: # 如果 hidden_states 不是列表 (通常来自单个 attention head)
            hidden_states = self.dense(hidden_states) # 对 hidden_states 应用 dense 线性层
        hidden_states = self.dropout(hidden_states) # 对 hidden_states 应用 Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # 将处理后的 hidden_states 与 input_tensor 相加，然后应用 Layer Normalization
        return hidden_states # 返回最终的 hidden_states


class BertAttention(nn.Module): # 定义 BertAttention 类，继承自 nn.Module
    def __init__(self, config, is_cross_attention=False, layer_num=-1): # 类的初始化方法，接收 config 对象，is_cross_attention 标志和 layer_num
        super().__init__() # 调用父类 nn.Module 的初始化方法
        if is_cross_attention: # 如果是交叉注意力
            self.self0 = BertSelfAttention(config, is_cross_attention) # 定义第一个 BertSelfAttention 模块
            self.self1 = BertSelfAttention(config, is_cross_attention) # 定义第二个 BertSelfAttention 模块
        else: # 如果是自注意力
            self.self = BertSelfAttention(config, is_cross_attention) # 定义一个 BertSelfAttention 模块
        self.output = BertSelfOutput( # 定义 BertSelfOutput 模块
            config, # 传入 config
            twin=is_cross_attention, # twin 参数取决于是否是交叉注意力
            merge=(is_cross_attention and layer_num >= 6), # merge 参数取决于是否是交叉注意力且层数 >= 6
        )
        self.pruned_heads = set() # 初始化一个集合用于存储被剪枝的 heads

    def prune_heads(self, heads): # 定义剪枝 attention heads 的函数
        if len(heads) == 0: # 如果要剪枝的 heads 列表为空，则直接返回
            return
        heads, index = find_pruneable_heads_and_indices( # 找到可剪枝的 heads 和对应的索引
            heads, # 要剪枝的 heads
            self.self.num_attention_heads, # attention heads 的总数
            self.self.attention_head_size, # 每个 attention head 的大小
            self.pruned_heads, # 已经剪枝的 heads 集合
        )

        # Prune linear layers
        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index) # 剪枝 Query 线性层
        self.self.key = prune_linear_layer(self.self.key, index) # 剪枝 Key 线性层
        self.self.value = prune_linear_layer(self.self.value, index) # 剪枝 Value 线性层
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1) # 剪枝 output 线性层 (如果存在)

        # Update hyper params and store pruned heads
        # 更新超参数并存储被剪枝的 heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads) # 更新 attention heads 的数量
        self.self.all_head_size = ( # 更新所有 attention heads 的总大小
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads) # 将新剪枝的 heads 添加到 pruned_heads 集合中

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
        if type(encoder_hidden_states) == list: # 如果编码器隐藏状态是列表 (通常用于双流交叉注意力)
            self_outputs0 = self.self0( # 通过第一个 BertSelfAttention 模块进行前向传播
                hidden_states, # 输入隐藏状态
                attention_mask, # attention mask
                head_mask, # head mask
                encoder_hidden_states[0], # 第一个编码器隐藏状态
                encoder_attention_mask[0], # 第一个编码器 attention mask
                past_key_value, # past_key_value
                output_attentions, # 是否输出 attention
            )
            self_outputs1 = self.self1( # 通过第二个 BertSelfAttention 模块进行前向传播
                hidden_states, # 输入隐藏状态
                attention_mask, # attention mask
                head_mask, # head mask
                encoder_hidden_states[1], # 第二个编码器隐藏状态
                encoder_attention_mask[1], # 第二个编码器 attention mask
                past_key_value, # past_key_value
                output_attentions, # 是否输出 attention
            )
            attention_output = self.output( # 通过 BertSelfOutput 模块处理 attention 输出
                [self_outputs0[0], self_outputs1[0]], hidden_states # 传入两个 attention 模块的输出和原始 hidden_states
            )

            outputs = (attention_output,) + self_outputs0[ # 构建输出元组，包含 attention_output 和第一个 attention 模块的其他输出 (如 attention map)
                1:
            ]  # add attentions if we output them # 如果输出 attention，则添加到输出中
        else: # 如果编码器隐藏状态不是列表 (通常用于单流自注意力或交叉注意力)
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


class BertIntermediate(nn.Module): # 定义 BertIntermediate 类，继承自 nn.Module
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # 定义一个线性层，将 hidden_size 映射到 intermediate_size
        if isinstance(config.hidden_act, str): # 如果激活函数是字符串类型
            self.intermediate_act_fn = ACT2FN[config.hidden_act] # 从 ACT2FN 字典中获取对应的激活函数
        else: # 如果激活函数不是字符串类型 (可能是函数对象)
            self.intermediate_act_fn = config.hidden_act # 直接使用 config 中指定的激活函数

    def forward(self, hidden_states): # 定义前向传播方法，接收 hidden_states
        hidden_states = self.dense(hidden_states) # 对 hidden_states 应用线性变换
        hidden_states = self.intermediate_act_fn(hidden_states) # 对结果应用激活函数
        return hidden_states # 返回处理后的 hidden_states


class BertOutput(nn.Module): # 定义 BertOutput 类，继承自 nn.Module
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


class BertLayer(nn.Module): # 定义 BertLayer 类，继承自 nn.Module
    def __init__(self, config, layer_num): # 类的初始化方法，接收 config 对象和层编号 layer_num
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.config = config # 保存配置对象
        self.chunk_size_feed_forward = config.chunk_size_feed_forward # 保存前馈网络的 chunk 大小
        self.seq_len_dim = 1 # 序列长度所在的维度索引
        self.attention = BertAttention(config) # 定义自注意力模块
        self.layer_num = layer_num # 保存当前层编号
        if self.config.add_cross_attention: # 如果配置中指定添加交叉注意力
            self.crossattention = BertAttention( # 定义交叉注意力模块
                config, # 传入 config
                is_cross_attention=self.config.add_cross_attention, # 设置为交叉注意力
                layer_num=layer_num, # 传入层编号
            )
        self.intermediate = BertIntermediate(config) # 定义中间层 (前馈网络的第一部分)
        self.output = BertOutput(config) # 定义输出层 (前馈网络的第二部分)

    def forward( # 定义前向传播方法
        self,
        hidden_states, # 输入的隐藏状态
        attention_mask=None, # attention mask
        head_mask=None, # head mask
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的 attention mask (用于交叉注意力)
        past_key_value=None, # 过去的 key 和 value (用于增量解码)
        output_attentions=False, # 是否输出 attention map
        mode=None, # 模式 (例如 "multimodal")
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
            past_key_value=self_attn_past_key_value, # 传入自注意力的 past_key_value
        )
        attention_output = self_attention_outputs[0] # 获取自注意力的输出 (context layer)

        outputs = self_attention_outputs[1:-1] # 获取自注意力的其他输出 (如 attention map)
        present_key_value = self_attention_outputs[-1] # 获取当前层的 present_key_value

        if mode == "multimodal": # 如果模式是 "multimodal" (通常表示进行交叉注意力)
            assert ( # 断言编码器隐藏状态必须提供
                encoder_hidden_states is not None
            ), "encoder_hidden_states must be given for cross-attention layers" # 错误信息：交叉注意力层必须提供 encoder_hidden_states
            cross_attention_outputs = self.crossattention( # 通过交叉注意力模块进行前向传播
                attention_output, # 输入是自注意力的输出
                attention_mask, # attention mask
                head_mask, # head mask
                encoder_hidden_states, # 编码器隐藏状态
                encoder_attention_mask, # 编码器 attention mask
                output_attentions=output_attentions, # 是否输出 attention
            )
            attention_output = cross_attention_outputs[0] # 获取交叉注意力的输出 (context layer)
            outputs = ( # 更新 outputs，添加交叉注意力的其他输出
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights # 如果输出 attention 权重，则添加交叉注意力权重

        layer_output = apply_chunking_to_forward( # 应用 chunking 到前馈网络的前向传播
            self.feed_forward_chunk, # 要应用 chunking 的函数
            self.chunk_size_feed_forward, # chunk 大小
            self.seq_len_dim, # 序列长度所在的维度
            attention_output, # 输入到前馈网络的张量
        )
        outputs = (layer_output,) + outputs # 将前馈网络的输出添加到 outputs 的开头

        outputs = outputs + (present_key_value,) # 将 present_key_value 添加到 outputs 的末尾

        return outputs # 返回最终的输出元组

    def feed_forward_chunk(self, attention_output): # 定义前馈网络的 chunk 前向传播函数
        intermediate_output = self.intermediate(attention_output) # 通过中间层
        layer_output = self.output(intermediate_output, attention_output) # 通过输出层，并添加残差连接
        return layer_output # 返回前馈网络的输出


class BertEncoder(nn.Module): # 定义 BertEncoder 类，继承自 nn.Module
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.config = config # 保存配置对象
        self.layer = nn.ModuleList( # 定义一个 ModuleList 包含多个 BertLayer
            [BertLayer(config, i) for i in range(config.num_hidden_layers)] # 根据 num_hidden_layers 创建 BertLayer 实例
        )
        self.gradient_checkpointing = False # 初始化 gradient_checkpointing 标志为 False

    def forward( # 定义前向传播方法
        self,
        hidden_states, # 输入的隐藏状态
        attention_mask=None, # attention mask
        head_mask=None, # head mask
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的 attention mask (用于交叉注意力)
        past_key_values=None, # 过去的 key 和 value (用于增量解码)
        use_cache=None, # 是否使用缓存 (用于增量解码)
        output_attentions=False, # 是否输出 attention map
        output_hidden_states=False, # 是否输出所有层的隐藏状态
        return_dict=True, # 是否以字典形式返回输出
        mode="multimodal", # 模式 (例如 "multimodal")
    ):
        all_hidden_states = () if output_hidden_states else None # 初始化存储所有隐藏状态的元组
        all_self_attentions = () if output_attentions else None # 初始化存储所有自注意力 map 的元组
        all_cross_attentions = ( # 初始化存储所有交叉注意力 map 的元组
            () if output_attentions and self.config.add_cross_attention else None # 如果输出 attention 且添加了交叉注意力，则初始化
        )

        next_decoder_cache = () if use_cache else None # 初始化存储下一层的 key/value 缓存的元组

        for i in range(self.config.num_hidden_layers): # 遍历每一层
            layer_module = self.layer[i] # 获取当前层的模块
            if output_hidden_states: # 如果需要输出所有隐藏状态
                all_hidden_states = all_hidden_states + (hidden_states,) # 将当前层的输入 hidden_states 添加到元组中

            layer_head_mask = head_mask[i] if head_mask is not None else None # 获取当前层的 head mask
            past_key_value = past_key_values[i] if past_key_values is not None else None # 获取当前层的 past_key_value

            if self.gradient_checkpointing and self.training: # 如果启用梯度检查点且处于训练模式

                if use_cache: # 如果同时启用了 use_cache
                    logger.warn( # 发出警告
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..." # use_cache=True 与梯度检查点不兼容，将 use_cache 设置为 False
                    )
                    use_cache = False # 将 use_cache 设置为 False

                def create_custom_forward(module): # 定义一个创建自定义前向传播函数的函数
                    def custom_forward(*inputs): # 定义自定义前向传播函数
                        return module(*inputs, past_key_value, output_attentions) # 调用模块的前向传播，并传入 past_key_value 和 output_attentions

                    return custom_forward # 返回自定义前向传播函数

                layer_outputs = torch.utils.checkpoint.checkpoint( # 使用梯度检查点进行前向传播
                    create_custom_forward(layer_module), # 传入自定义前向传播函数
                    hidden_states, # 输入 hidden_states
                    attention_mask, # attention mask
                    layer_head_mask, # layer head mask
                    encoder_hidden_states, # 编码器隐藏状态
                    encoder_attention_mask, # 编码器 attention mask
                    mode=mode, # 模式
                )
            else: # 如果没有启用梯度检查点
                layer_outputs = layer_module( # 直接调用当前层模块的前向传播
                    hidden_states, # 输入 hidden_states
                    attention_mask, # attention mask
                    layer_head_mask, # layer head mask
                    encoder_hidden_states, # 编码器隐藏状态
                    encoder_attention_mask, # 编码器 attention mask
                    past_key_value, # past_key_value
                    output_attentions, # 是否输出 attention
                    mode=mode, # 模式
                )

            hidden_states = layer_outputs[0] # 获取当前层的输出 hidden_states
            if use_cache: # 如果使用缓存
                next_decoder_cache += (layer_outputs[-1],) # 将当前层的 present_key_value 添加到 next_decoder_cache 中
            if output_attentions: # 如果输出 attention
                all_self_attentions = all_self_attentions + (layer_outputs[1],) # 将当前层的自注意力 map 添加到元组中
                # Note: Cross-attention maps are added within BertLayer if mode is "multimodal"

        if output_hidden_states: # 如果需要输出所有隐藏状态
            all_hidden_states = all_hidden_states + (hidden_states,) # 将最后一层的输出 hidden_states 添加到元组中

        if not return_dict: # 如果不需要以字典形式返回
            return tuple( # 返回一个元组
                v # 包含所有非 None 的输出
                for v in [
                    hidden_states, # 最后一层的 hidden_states
                    next_decoder_cache, # 下一层的 key/value 缓存
                    all_hidden_states, # 所有层的 hidden_states
                    all_self_attentions, # 所有自注意力 map
                    all_cross_attentions, # 所有交叉注意力 map
                ]
                if v is not None # 过滤掉 None 值
            )
        return BaseModelOutputWithPastAndCrossAttentions( # 以字典形式返回 BaseModelOutputWithPastAndCrossAttentions 对象
            last_hidden_state=hidden_states, # 最后一层的 hidden_states
            past_key_values=next_decoder_cache, # 下一层的 key/value 缓存
            hidden_states=all_hidden_states, # 所有层的 hidden_states
            attentions=all_self_attentions, # 所有自注意力 map
            cross_attentions=all_cross_attentions, # 所有交叉注意力 map
        )


class BertPooler(nn.Module): # 定义 BertPooler 类，继承自 nn.Module
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 定义一个线性层，输入和输出维度都是 hidden_size
        self.activation = nn.Tanh() # 定义 Tanh 激活函数

    def forward(self, hidden_states): # 定义前向传播方法，接收 hidden_states
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # 我们通过简单地获取对应于第一个 token 的隐藏状态来对模型进行“池化”。
        first_token_tensor = hidden_states[:, 0] # 获取序列中第一个 token (通常是 [CLS] token) 的隐藏状态
        pooled_output = self.dense(first_token_tensor) # 对第一个 token 的隐藏状态应用线性变换
        pooled_output = self.activation(pooled_output) # 对结果应用 Tanh 激活函数
        return pooled_output # 返回池化后的输出


class BertPredictionHeadTransform(nn.Module): # 定义 BertPredictionHeadTransform 类，继承自 nn.Module
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 定义一个线性层
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


class BertLMPredictionHead(nn.Module): # 定义 BertLMPredictionHead 类，继承自 nn.Module (用于语言模型预测头)
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.transform = BertPredictionHeadTransform(config) # 定义一个 BertPredictionHeadTransform 模块

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # 输出权重与输入嵌入相同，但每个 token 有一个仅用于输出的偏置。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 定义一个线性层，将 hidden_size 映射到 vocab_size，不使用偏置

        self.bias = nn.Parameter(torch.zeros(config.vocab_size)) # 定义一个可学习的偏置参数，形状为 (vocab_size,)

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # 需要在这两个变量之间建立链接，以便在使用 `resize_token_embeddings` 时正确调整偏置的大小
        self.decoder.bias = self.bias # 将 decoder 的偏置设置为定义的 bias 参数

    def forward(self, hidden_states): # 定义前向传播方法，接收 hidden_states
        hidden_states = self.transform(hidden_states) # 通过 transform 模块处理 hidden_states
        hidden_states = self.decoder(hidden_states) # 通过 decoder 线性层进行预测
        return hidden_states # 返回预测分数


class BertOnlyMLMHead(nn.Module): # 定义 BertOnlyMLMHead 类，继承自 nn.Module (用于 Masked Language Model 预测头)
    def __init__(self, config): # 类的初始化方法，接收 config 对象
        super().__init__() # 调用父类 nn.Module 的初始化方法
        self.predictions = BertLMPredictionHead(config) # 定义一个 BertLMPredictionHead 模块

    def forward(self, sequence_output): # 定义前向传播方法，接收 sequence_output
        prediction_scores = self.predictions(sequence_output) # 通过 predictions 模块计算预测分数
        return prediction_scores # 返回预测分数


class BertPreTrainedModel(PreTrainedModel): # 定义 BertPreTrainedModel 抽象基类，继承自 PreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    config_class = BertConfig # 指定模型的配置类
    base_model_prefix = "bert" # 指定模型的基础前缀
    _keys_to_ignore_on_load_missing = [r"position_ids"] # 指定加载时忽略的缺失键的正则表达式列表

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
    该模型可以作为编码器（仅包含自注意力）或解码器，作为解码器时，会在自注意力层之间添加一层交叉注意力，遵循 Ashish Vaswani 等人在 `Attention is all you need <https://arxiv.org/abs/1706.03762>`__ 中描述的架构。
    参数和 :obj:`add_cross_attention` 设置为 :obj:`True`；然后期望 :obj:`encoder_hidden_states` 作为前向传播的输入。
    """

    def __init__(self, config, add_pooling_layer=True): # 类的初始化方法，接收 config 对象和 add_pooling_layer 标志
        super().__init__(config) # 调用父类 BertPreTrainedModel 的初始化方法
        self.config = config # 保存配置对象

        self.embeddings = BertEmbeddings(config) # 定义 BertEmbeddings 模块

        self.encoder = BertEncoder(config) # 定义 BertEncoder 模块

        self.pooler = BertPooler(config) if add_pooling_layer else None # 定义 BertPooler 模块 (如果 add_pooling_layer 为 True)

        self.init_weights() # 调用父类的权重初始化方法

    def get_input_embeddings(self): # 定义获取输入嵌入层的方法
        return self.embeddings.word_embeddings # 返回词嵌入层

    def set_input_embeddings(self, value): # 定义设置输入嵌入层的方法
        self.embeddings.word_embeddings = value # 设置词嵌入层

    def _prune_heads(self, heads_to_prune): # 定义剪枝 attention heads 的方法
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        剪枝模型的 heads。heads_to_prune: 字典，格式为 {层编号: 该层要剪枝的 heads 列表}。参见基类 PreTrainedModel。
        """
        for layer, heads in heads_to_prune.items(): # 遍历要剪枝的层和对应的 heads
            self.encoder.layer[layer].attention.prune_heads(heads) # 调用对应层的 attention 模块的 prune_heads 方法

    def get_extended_attention_mask( # 定义获取扩展 attention mask 的方法
        self,
        attention_mask: Tensor, # attention mask
        input_shape: Tuple[int], # 输入形状
        device: device, # 设备
        is_decoder: bool, # 是否是解码器
    ) -> Tensor: # 返回扩展 attention mask
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
                一个张量，其中 1 表示要关注的 token，0 表示要忽略的 token。
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
                模型输入的形状。
            device: (:obj:`torch.device`):
                The device of the input to the model.
                模型输入的设备。

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
            扩展的 attention mask，其 dtype 与 :obj:`attention_mask.dtype` 相同。
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 我们可以自己提供一个维度为 [batch_size, from_seq_length, to_seq_length] 的自注意力 mask，在这种情况下，我们只需要使其可以广播到所有 heads。
        if attention_mask.dim() == 3: # 如果 attention mask 是 3 维
            extended_attention_mask = attention_mask[:, None, :, :] # 扩展维度，使其可以广播到 heads
        elif attention_mask.dim() == 2: # 如果 attention mask 是 2 维 (通常是 padding mask)
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # 提供了维度为 [batch_size, seq_length] 的 padding mask
            # - 如果模型是解码器，除了 padding mask 外，还应用因果 mask
            # - 如果模型是编码器，则使 mask 可以广播到 [batch_size, num_heads, seq_length, seq_length]
            if is_decoder: # 如果是解码器
                batch_size, seq_length = input_shape # 获取 batch_size 和 seq_length

                seq_ids = torch.arange(seq_length, device=device) # 创建序列 ID 张量
                causal_mask = ( # 创建因果 mask
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1) # 形状为 (batch_size, seq_length, seq_length)
                    <= seq_ids[None, :, None] # 比较，生成上三角矩阵 (不包括对角线)
                )
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                # 如果使用了 past_key_values，我们需要在因果 mask 前面添加一个全 1 的 mask
                # 在 pytorch < 1.3 版本中，因果 mask 和 attention mask 必须具有相同的类型
                causal_mask = causal_mask.to(attention_mask.dtype) # 将因果 mask 转换为与 attention mask 相同的 dtype

                if causal_mask.shape[1] < attention_mask.shape[1]: # 如果因果 mask 的序列长度小于 attention mask 的序列长度 (通常是因为使用了 past_key_values)
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1] # 计算需要添加的前缀长度
                    causal_mask = torch.cat( # 拼接前缀全 1 mask 和因果 mask
                        [
                            torch.ones( # 创建前缀全 1 mask
                                (batch_size, seq_length, prefix_seq_len), # 形状为 (batch_size, seq_length, prefix_seq_len)
                                device=device, # 设备
                                dtype=causal_mask.dtype, # dtype
                            ),
                            causal_mask, # 原始因果 mask
                        ],
                        axis=-1, # 在最后一个维度上拼接
                    )

                extended_attention_mask = ( # 将因果 mask 和 padding mask 结合
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :] # 扩展维度并相乘
                )
            else: # 如果是编码器
                extended_attention_mask = attention_mask[:, None, None, :] # 扩展维度，使其可以广播到 heads 和序列长度
        else: # 如果 attention mask 的维度不是 2 或 3
            raise ValueError( # 抛出 ValueError 异常
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format( # 错误信息：input_ids 或 attention_mask 的形状错误
                    input_shape, attention_mask.shape # 格式化错误信息，包含实际的形状
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # 由于 attention_mask 对于我们想要关注的位置是 1.0，对于被 mask 的位置是 0.0，这个操作将创建一个张量，对于我们想要关注的位置是 0.0，对于被 mask 的位置是 -10000.0。
        # 由于我们在 Softmax 之前将其添加到原始分数中，这实际上与完全移除这些位置的效果相同。
        extended_attention_mask = extended_attention_mask.to( # 转换为与模型相同的 dtype
            dtype=self.dtype
        )  # fp16 compatibility # 兼容 fp16
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # 将 1.0 变为 0.0，0.0 变为 1.0，然后乘以 -10000.0
        return extended_attention_mask # 返回扩展 attention mask

    def forward( # 定义前向传播方法
        self,
        input_ids=None, # 输入的 token ID
        attention_mask=None, # attention mask
        position_ids=None, # 位置 ID
        head_mask=None, # head mask
        inputs_embeds=None, # 输入的嵌入表示
        encoder_embeds=None, # 编码器的嵌入表示 (用于交叉注意力)
        encoder_hidden_states=None, # 编码器的隐藏状态 (用于交叉注意力)
        encoder_attention_mask=None, # 编码器的 attention mask (用于交叉注意力)
        past_key_values=None, # 过去的 key 和 value (用于增量解码)
        use_cache=None, # 是否使用缓存 (用于增量解码)
        output_attentions=None, # 是否输出 attention map
        output_hidden_states=None, # 是否输出所有层的隐藏状态
        return_dict=None, # 是否以字典形式返回输出
        is_decoder=False, # 是否是解码器
        mode="multimodal", # 模式 (例如 "multimodal")
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
            编码器最后一层输出的隐藏状态序列。如果模型配置为解码器，则在交叉注意力中使用。
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            用于避免对编码器输入的 padding token 索引执行注意力的 mask。如果模型配置为解码器，则在交叉注意力中使用此 mask。mask 值选择在 ``[0, 1]`` 之间：
            - 1 表示**未被 mask** 的 token，
            - 0 表示**被 mask** 的 token。
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
            包含 attention 块预先计算的 key 和 value 隐藏状态。可用于加速解码。
            如果使用了 :obj:`past_key_values`，用户可以选择只输入形状为 :obj:`(batch_size, 1)` 的最后一个 :obj:`decoder_input_ids`（那些没有将其过去的 key value 状态提供给此模型的 token），而不是形状为 :obj:`(batch_size, sequence_length)` 的所有 :obj:`decoder_input_ids`。
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
            如果设置为 :obj:`True`，则返回 :obj:`past_key_values` key value 状态，可用于加速解码（参见 :obj:`past_key_values`）。
        """
        output_attentions = ( # 确定是否输出 attention
            output_attentions # 如果提供了 output_attentions，则使用它
            if output_attentions is not None # 检查 output_attentions 是否为 None
            else self.config.output_attentions # 如果为 None，则使用 config 中的设置
        )
        output_hidden_states = ( # 确定是否输出所有隐藏状态
            output_hidden_states # 如果提供了 output_hidden_states，则使用它
            if output_hidden_states is not None # 检查 output_hidden_states 是否为 None
            else self.config.output_hidden_states # 如果为 None，则使用 config 中的设置
        )
        return_dict = ( # 确定是否以字典形式返回
            return_dict # 如果提供了 return_dict，则使用它
            if return_dict is not None # 检查 return_dict 是否为 None
            else self.config.use_return_dict # 如果为 None，则使用 config 中的设置
        )

        if is_decoder: # 如果是解码器
            use_cache = use_cache if use_cache is not None else self.config.use_cache # 确定是否使用缓存
        else: # 如果不是解码器
            use_cache = False # 不使用缓存

        if input_ids is not None and inputs_embeds is not None: # 如果同时提供了 input_ids 和 inputs_embeds
            raise ValueError( # 抛出 ValueError 异常
                "You cannot specify both input_ids and inputs_embeds at the same time" # 错误信息：不能同时指定 input_ids 和 inputs_embeds
            )
        elif input_ids is not None: # 如果提供了 input_ids
            input_shape = input_ids.size() # 获取输入形状
            batch_size, seq_length = input_shape # 获取 batch_size 和 seq_length
            device = input_ids.device # 获取设备
        elif inputs_embeds is not None: # 如果提供了 inputs_embeds
            input_shape = inputs_embeds.size()[:-1] # 获取输入形状 (去除最后一个维度)
            batch_size, seq_length = input_shape # 获取 batch_size 和 seq_length
            device = inputs_embeds.device # 获取设备
        elif encoder_embeds is not None: # 如果提供了 encoder_embeds (用于交叉注意力)
            input_shape = encoder_embeds.size()[:-1] # 获取输入形状 (去除最后一个维度)
            batch_size, seq_length = input_shape # 获取 batch_size 和 seq_length
            device = encoder_embeds.device # 获取设备
        else: # 如果没有提供任何输入
            raise ValueError( # 抛出 ValueError 异常
                "You have to specify either input_ids or inputs_embeds or encoder_embeds" # 错误信息：必须指定 input_ids 或 inputs_embeds 或 encoder_embeds
            )

        # past_key_values_length
        # past_key_values 的长度
        past_key_values_length = ( # 计算 past_key_values 的序列长度
            past_key_values[0][0].shape[2] if past_key_values is not None else 0 # 如果 past_key_values 存在，获取其形状的第三个维度，否则为 0
        )

        if attention_mask is None: # 如果没有提供 attention mask
            attention_mask = torch.ones( # 创建一个全 1 的 attention mask
                ((batch_size, seq_length + past_key_values_length)), device=device # 形状为 (batch_size, seq_length + past_key_values_length)
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 我们可以自己提供一个维度为 [batch_size, from_seq_length, to_seq_length] 的自注意力 mask，在这种情况下，我们只需要使其可以广播到所有 heads。
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask( # 获取扩展 attention mask
            attention_mask, input_shape, device, is_decoder # 传入 attention mask, input_shape, device, is_decoder
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # 如果为交叉注意力提供了 2D 或 3D attention mask
        # 我们需要使其可以广播到 [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None: # 如果提供了编码器隐藏状态 (进行交叉注意力)
            if type(encoder_hidden_states) == list: # 如果编码器隐藏状态是列表
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[ # 获取第一个元素的形状
                    0
                ].size()
            else: # 如果编码器隐藏状态不是列表
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size() # 获取形状
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length) # 存储编码器隐藏状态的形状

            if type(encoder_attention_mask) == list: # 如果编码器 attention mask 是列表
                encoder_extended_attention_mask = [ # 对列表中的每个 mask 进行反转
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            elif encoder_attention_mask is None: # 如果编码器 attention mask 为 None
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device) # 创建一个全 1 的 mask
                encoder_extended_attention_mask = self.invert_attention_mask( # 反转 mask
                    encoder_attention_mask
                )
            else: # 如果编码器 attention mask 不是列表也不是 None
                encoder_extended_attention_mask = self.invert_attention_mask( # 反转 mask
                    encoder_attention_mask
                )
        else: # 如果没有提供编码器隐藏状态
            encoder_extended_attention_mask = None # 编码器扩展 attention mask 为 None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # 如果需要，准备 head mask
        # head_mask 中的 1.0 表示保留该 head
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入 head_mask 的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers) # 获取 head mask

        if encoder_embeds is None: # 如果没有提供编码器嵌入
            embedding_output = self.embeddings( # 通过 embeddings 模块获取嵌入表示
                input_ids=input_ids, # 输入 input_ids
                position_ids=position_ids, # 输入 position_ids
                inputs_embeds=inputs_embeds, # 输入 inputs_embeds
                past_key_values_length=past_key_values_length, # 输入 past_key_values_length
            )
        else: # 如果提供了编码器嵌入
            embedding_output = encoder_embeds # 使用编码器嵌入作为输入

        encoder_outputs = self.encoder( # 通过 encoder 模块进行前向传播
            embedding_output, # 输入嵌入表示
            attention_mask=extended_attention_mask, # 扩展 attention mask
            head_mask=head_mask, # head mask
            encoder_hidden_states=encoder_hidden_states, # 编码器隐藏状态
            encoder_attention_mask=encoder_extended_attention_mask, # 编码器扩展 attention mask
            past_key_values=past_key_values, # past_key_values
            use_cache=use_cache, # 是否使用缓存
            output_attentions=output_attentions, # 是否输出 attention
            output_hidden_states=output_hidden_states, # 是否输出所有隐藏状态
            return_dict=return_dict, # 是否以字典形式返回
            mode=mode, # 模式
        )
        sequence_output = encoder_outputs[0] # 获取编码器的序列输出 (最后一层的 hidden_states)
        pooled_output = ( # 获取池化输出
            self.pooler(sequence_output) if self.pooler is not None else None # 如果 pooler 存在，则对其进行前向传播，否则为 None
        )

        if not return_dict: # 如果不需要以字典形式返回
            return (sequence_output, pooled_output) + encoder_outputs[1:] # 返回一个元组，包含序列输出、池化输出和编码器的其他输出

        return BaseModelOutputWithPoolingAndCrossAttentions( # 以字典形式返回 BaseModelOutputWithPoolingAndCrossAttentions 对象
            last_hidden_state=sequence_output, # 最后一层的 hidden_states
            pooler_output=pooled_output, # 池化输出
            past_key_values=encoder_outputs.past_key_values, # past_key_values
            hidden_states=encoder_outputs.hidden_states, # 所有层的 hidden_states
            attentions=encoder_outputs.attentions, # 所有自注意力 map
            cross_attentions=encoder_outputs.cross_attentions, # 所有交叉注意力 map
        )