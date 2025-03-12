import torch  # 导入PyTorch库
import torch.nn as nn  # 从PyTorch导入神经网络模块


class Bottle(nn.Module):  # 定义Bottle类，继承自nn.Module

    def forward(self, input):  # 前向传播函数
        if len(input.size()) <= 2:  # 如果输入的维度小于等于2
            return super(Bottle, self).forward(input)  # 调用父类的前向传播
        size = input.size()[:2]  # 获取输入的前两个维度
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))  # 将输入展平并调用父类的前向传播
        return out.view(size[0], size[1], -1)  # 将输出重塑为原来的形状


class Linear(Bottle, nn.Linear):  # 定义Linear类，继承自Bottle和nn.Linear
    pass  # 直接继承功能，无需额外实现


class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module

    def __init__(self, config):  # 初始化函数，接收配置参数
        super(Encoder, self).__init__()  # 调用父类构造函数
        self.config = config  # 保存配置
        input_size = config.d_proj if config.projection else config.d_embed  # 根据配置选择输入大小
        dropout = 0 if config.n_layers == 1 else config.dp_ratio  # 如果只有一层，则丢弃率为0
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,  # 定义LSTM层
                           num_layers=config.n_layers, dropout=dropout,
                           bidirectional=config.birnn)  # 根据配置设置双向LSTM

    def forward(self, inputs):  # 前向传播函数
        batch_size = inputs.size()[1]  # 获取批量大小
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden  # 定义状态形状
        h0 = c0 = inputs.new_zeros(state_shape)  # 初始化隐藏状态和细胞状态为零
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))  # 通过LSTM层进行前向传播
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)  # 返回最后的隐藏状态


class SNLIClassifier(nn.Module):  # 定义SNLIClassifier类，继承自nn.Module

    def __init__(self, config):  # 初始化函数，接收配置参数
        super(SNLIClassifier, self).__init__()  # 调用父类构造函数
        self.config = config  # 保存配置
        self.embed = nn.Embedding(config.n_embed, config.d_embed)  # 定义嵌入层
        self.projection = Linear(config.d_embed, config.d_proj)  # 定义线性投影层
        self.encoder = Encoder(config)  # 创建编码器实例
        self.dropout = nn.Dropout(p=config.dp_ratio)  # 定义丢弃层
        self.relu = nn.ReLU()  # 定义ReLU激活函数
        seq_in_size = 2 * config.d_hidden  # 初始化序列输入大小
        if self.config.birnn:  # 如果是双向RNN
            seq_in_size *= 2  # 序列输入大小翻倍
        lin_config = [seq_in_size] * 2  # 定义线性层配置
        self.out = nn.Sequential(  # 定义输出层
            Linear(*lin_config),  # 添加线性层
            self.relu,  # 添加ReLU激活
            self.dropout,  # 添加丢弃层
            Linear(*lin_config),  # 添加第二个线性层
            self.relu,  # 添加ReLU激活
            self.dropout,  # 添加丢弃层
            Linear(*lin_config),  # 添加第三个线性层
            self.relu,  # 添加ReLU激活
            self.dropout,  # 添加丢弃层
            Linear(seq_in_size, config.d_out))  # 最后一个线性层，输出维度为config.d_out

    def forward(self, batch):  # 前向传播函数
        prem_embed = self.embed(batch.premise)  # 获取前提句子的嵌入
        hypo_embed = self.embed(batch.hypothesis)  # 获取假设句子的嵌入
        if self.config.fix_emb:  # 如果配置中要求固定嵌入
            prem_embed = prem_embed.detach()  # 分离前提嵌入
            hypo_embed = hypo_embed.detach()  # 分离假设嵌入
        if self.config.projection:  # 如果配置中要求投影
            prem_embed = self.relu(self.projection(prem_embed))  # 通过线性投影层并应用ReLU
            hypo_embed = self.relu(self.projection(hypo_embed))  # 通过线性投影层并应用ReLU
        premise = self.encoder(prem_embed)  # 编码前提嵌入
        hypothesis = self.encoder(hypo_embed)  # 编码假设嵌入
        scores = self.out(torch.cat([premise, hypothesis], 1))  # 将前提和假设的编码结果连接并通过输出层
        return scores  # 返回分类得分