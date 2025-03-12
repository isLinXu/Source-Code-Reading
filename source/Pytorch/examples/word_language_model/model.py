import math  # 导入数学库
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.
    该类是一个容器模块，包含编码器、递归模块和解码器。"""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        """Initialize the RNN model.
        初始化 RNN 模型。

        Args:
            rnn_type: 类型（LSTM, GRU, RNN_TANH, RNN_RELU）
            ntoken: 词汇表大小
            ninp: 词嵌入维度
            nhid: 隐藏层单元数
            nlayers: 层数
            dropout: dropout 概率
            tie_weights: 是否绑定权重
        """
        super(RNNModel, self).__init__()  # 调用父类构造函数
        self.ntoken = ntoken  # 词汇表大小
        self.drop = nn.Dropout(dropout)  # dropout 层
        self.encoder = nn.Embedding(ntoken, ninp)  # 嵌入层
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)  # 选择 LSTM 或 GRU
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]  # 非线性激活函数
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e  # 抛出无效模型类型异常
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)  # 创建 RNN
        self.decoder = nn.Linear(nhid, ntoken)  # 解码器

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')  # 绑定权重时 nhid 必须等于 ninp
            self.decoder.weight = self.encoder.weight  # 绑定编码器和解码器的权重

        self.init_weights()  # 初始化权重

        self.rnn_type = rnn_type  # RNN 类型
        self.nhid = nhid  # 隐藏层单元数
        self.nlayers = nlayers  # 层数

    def init_weights(self):
        """Initialize weights.
        初始化权重。"""
        initrange = 0.1  # 权重初始化范围
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)  # 初始化编码器权重
        nn.init.zeros_(self.decoder.bias)  # 初始化解码器偏置为零
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)  # 初始化解码器权重

    def forward(self, input, hidden):
        """Forward pass.
        前向传播。

        Args:
            input: 输入序列
            hidden: 隐藏状态
        Returns:
            decoded: 解码后的输出
            hidden: 更新后的隐藏状态
        """
        emb = self.drop(self.encoder(input))  # 嵌入并应用 dropout
        output, hidden = self.rnn(emb, hidden)  # RNN 计算
        output = self.drop(output)  # 对输出应用 dropout
        decoded = self.decoder(output)  # 解码输出
        decoded = decoded.view(-1, self.ntoken)  # 调整输出形状
        return F.log_softmax(decoded, dim=1), hidden  # 返回 log softmax 输出和隐藏状态

    def init_hidden(self, bsz):
        """Initialize hidden state.
        初始化隐藏状态。

        Args:
            bsz: 批次大小
        Returns:
            隐藏状态
        """
        weight = next(self.parameters())  # 获取模型参数
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),  # LSTM 隐藏状态
                    weight.new_zeros(self.nlayers, bsz, self.nhid))  # LSTM 单元状态
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)  # RNN 隐藏状态

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """Initialize the positional encoding.
        初始化位置编码。

        Args:
            d_model: 嵌入维度
            dropout: dropout 概率
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()  # 调用父类构造函数
        self.dropout = nn.Dropout(p=dropout)  # dropout 层

        pe = torch.zeros(max_len, d_model)  # 创建位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算频率因子
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用 sin 函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用 cos 函数
        pe = pe.unsqueeze(0).transpose(0, 1)  # 调整维度
        self.register_buffer('pe', pe)  # 注册位置编码

    def forward(self, x):
        """Inputs of forward function
        前向函数的输入

        Args:
            x: 输入序列
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]  # 将位置编码添加到输入
        return self.dropout(x)  # 返回经过 dropout 的输出

class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.
    该类是一个容器模块，包含编码器、递归或变换模块和解码器。"""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        """Initialize the Transformer model.
        初始化变换模型。

        Args:
            ntoken: 词汇表大小
            ninp: 嵌入维度
            nhead: 注意力头数
            nhid: 隐藏层单元数
            nlayers: 层数
            dropout: dropout 概率
        """
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)  # 调用父类构造函数
        self.model_type = 'Transformer'  # 模型类型
        self.src_mask = None  # 源掩码
        self.pos_encoder = PositionalEncoding(ninp, dropout)  # 位置编码

        self.input_emb = nn.Embedding(ntoken, ninp)  # 输入嵌入层
        self.ninp = ninp  # 嵌入维度
        self.decoder = nn.Linear(ninp, ntoken)  # 解码器

        self.init_weights()  # 初始化权重

    def _generate_square_subsequent_mask(self, sz):
        """生成方形后续掩码
        Args:
            sz: 掩码大小
        Returns:
            掩码矩阵
        """
        return torch.log(torch.tril(torch.ones(sz, sz)))  # 生成下三角矩阵的对数

    def init_weights(self):
        """Initialize weights.
        初始化权重。"""
        initrange = 0.1  # 权重初始化范围
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)  # 初始化输入嵌入权重
        nn.init.zeros_(self.decoder.bias)  # 初始化解码器偏置为零
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)  # 初始化解码器权重

    def forward(self, src, has_mask=True):
        """Forward pass.
        前向传播。

        Args:
            src: 输入序列
            has_mask: 是否使用掩码
        Returns:
            输出经过 softmax 的结果
        """
        if has_mask:
            device = src.device  # 获取输入设备
            if self.src_mask is None or self.src_mask.size(0) != len(src):  # 如果掩码未生成或大小不匹配
                mask = self._generate_square_subsequent_mask(len(src)).to(device)  # 生成掩码
                self.src_mask = mask  # 设置源掩码
        else:
            self.src_mask = None  # 不使用掩码

        src = self.input_emb(src) * math.sqrt(self.ninp)  # 嵌入输入并缩放
        src = self.pos_encoder(src)  # 添加位置编码
        output = self.encoder(src, mask=self.src_mask)  # 编码器计算
        output = self.decoder(output)  # 解码
        return F.log_softmax(output, dim=-1)  # 返回经过 log softmax 的输出