import math  # 导入数学库

import torch  # 导入PyTorch库
from torch.nn import functional as F  # 从PyTorch导入功能性模块
from torch import nn  # 从PyTorch导入神经网络模块

class PositionalEncoding(nn.Module):  # 定义位置编码类
    def __init__(self, emb_size, dropout, maxlen=5000):  # 初始化位置编码
        super(PositionalEncoding, self).__init__()  # 调用父类构造函数
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)  # 计算位置编码的分母
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)  # 创建位置索引
        pos_embedding = torch.zeros((maxlen, emb_size))  # 初始化位置编码矩阵
        pos_embedding[:, 0::2] = torch.sin(pos * den)  # 对偶数位置应用正弦函数
        pos_embedding[:, 1::2] = torch.cos(pos * den)  # 对奇数位置应用余弦函数
        pos_embedding = pos_embedding.unsqueeze(-2)  # 在最后一维增加维度

        self.dropout = nn.Dropout(dropout)  # 定义丢弃层
        self.register_buffer('pos_embedding', pos_embedding)  # 注册位置编码为缓冲区

    def forward(self, token_embedding):  # 前向传播函数
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])  # 返回加上位置编码的嵌入

class Translator(nn.Module):  # 定义翻译器类
    def __init__(self, num_encoder_layers, num_decoder_layers, embed_size, num_heads, src_vocab_size, tgt_vocab_size, dim_feedforward, dropout):  # 初始化翻译器
        super(Translator, self).__init__()  # 调用父类构造函数

        # Output of embedding must be equal (embed_size)
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)  # 定义源语言嵌入层
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)  # 定义目标语言嵌入层

        self.pos_enc = PositionalEncoding(embed_size, dropout)  # 定义位置编码

        self.transformer = nn.Transformer(  # 定义Transformer模型
            d_model=embed_size,  # 嵌入维度
            nhead=num_heads,  # 注意力头数
            num_encoder_layers=num_encoder_layers,  # 编码器层数
            num_decoder_layers=num_decoder_layers,  # 解码器层数
            dim_feedforward=dim_feedforward,  # 前馈网络维度
            dropout=dropout  # 丢弃率
        )

        self.ff = nn.Linear(embed_size, tgt_vocab_size)  # 定义线性层用于输出

        self._init_weights()  # 初始化权重

    def _init_weights(self):  # 权重初始化函数
        for p in self.parameters():  # 遍历模型参数
            if p.dim() > 1:  # 如果参数维度大于1
                nn.init.xavier_uniform_(p)  # 使用Xavier均匀分布初始化

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):  # 前向传播函数
        src_emb = self.pos_enc(self.src_embedding(src))  # 获取源语言嵌入并添加位置编码
        tgt_emb = self.pos_enc(self.tgt_embedding(trg))  # 获取目标语言嵌入并添加位置编码

        outs = self.transformer(  # 通过Transformer模型进行前向传播
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask
        )

        return self.ff(outs)  # 返回输出经过线性层处理的结果

    def encode(self, src, src_mask):  # 编码函数
        embed = self.src_embedding(src)  # 获取源语言嵌入
        pos_enc = self.pos_enc(embed)  # 添加位置编码
        return self.transformer.encoder(pos_enc, src_mask)  # 返回编码器的输出

    def decode(self, tgt, memory, tgt_mask):  # 解码函数
        embed = self.tgt_embedding(tgt)  # 获取目标语言嵌入
        pos_enc = self.pos_enc(embed)  # 添加位置编码
        return self.transformer.decoder(pos_enc, memory, tgt_mask)  # 返回解码器的输出