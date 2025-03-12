import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.distributed.rpc as rpc  # 导入分布式RPC模块
from torch.distributed.rpc import RRef  # 导入远程引用


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """  # 辅助函数，用于在给定的RRef上调用方法
    return method(rref.local_value(), *args, **kwargs)  # 调用方法并返回结果


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """  # 辅助函数，在RRef的拥有者上运行方法并通过RPC获取结果
    return rpc.rpc_sync(  # 使用RPC同步调用
        rref.owner(),  # 获取RRef的拥有者
        _call_method,  # 调用方法
        args=[method, rref] + list(args),  # 将方法和RRef添加到参数列表中
        kwargs=kwargs  # 传递关键字参数
    )


def _parameter_rrefs(module):
    r"""
    Create one RRef for each parameter in the given local module, and return a
    list of RRefs.
    """  # 为给定本地模块中的每个参数创建一个RRef，并返回RRef列表
    param_rrefs = []  # 初始化参数RRef列表
    for param in module.parameters():  # 遍历模块的所有参数
        param_rrefs.append(RRef(param))  # 将参数的RRef添加到列表中
    return param_rrefs  # 返回参数的远程引用


class EmbeddingTable(nn.Module):
    r"""
    Encoding layers of the RNNModel
    """  # RNN模型的编码层
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()  # 调用父类构造函数
        self.drop = nn.Dropout(dropout)  # 创建Dropout层
        self.encoder = nn.Embedding(ntoken, ninp)  # 创建嵌入层
        if torch.cuda.is_available():  # 如果可用CUDA
            self.encoder = self.encoder.cuda()  # 将嵌入层移动到GPU
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)  # 初始化嵌入层权重

    def forward(self, input):
        if torch.cuda.is_available():  # 如果可用CUDA
            input = input.cuda()  # 将输入移动到GPU
        return self.drop(self.encoder(input)).cpu()  # 应用Dropout并返回结果


class Decoder(nn.Module):
    r"""
    Decoding layers of the RNNModel
    """  # RNN模型的解码层
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()  # 调用父类构造函数
        self.drop = nn.Dropout(dropout)  # 创建Dropout层
        self.decoder = nn.Linear(nhid, ntoken)  # 创建线性解码层
        nn.init.zeros_(self.decoder.bias)  # 初始化偏置为0
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)  # 初始化权重

    def forward(self, output):
        return self.decoder(self.drop(output))  # 应用Dropout并返回解码结果


class RNNModel(nn.Module):
    r"""
    A distributed RNN model which puts embedding table and decoder parameters on
    a remote parameter server, and locally holds parameters for the LSTM module.
    The structure of the RNN model is borrowed from the word language model
    example. See https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """  # 一个分布式RNN模型，将嵌入表和解码器参数放置在远程参数服务器上，并在本地保持LSTM模块的参数
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()  # 调用父类构造函数

        # setup embedding table remotely
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))  # 设置远程嵌入表
        # setup LSTM locally
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)  # 设置本地LSTM
        # setup decoder remotely
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))  # 设置远程解码器

    def forward(self, input, hidden):
        # pass input to the remote embedding table and fetch emb tensor back
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)  # 将输入传递给远程嵌入表并获取嵌入张量
        output, hidden = self.rnn(emb, hidden)  # 通过LSTM进行前向传播
        # pass output to the remote decoder and get the decoded output back
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)  # 将输出传递给远程解码器并获取解码结果
        return decoded, hidden  # 返回解码结果和隐藏状态

    def parameter_rrefs(self):
        remote_params = []  # 初始化远程参数列表
        # get RRefs of embedding table
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))  # 获取嵌入表的RRef
        # create RRefs for local parameters
        remote_params.extend(_parameter_rrefs(self.rnn))  # 创建LSTM参数的RRef
        # get RRefs of decoder
        remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))  # 获取解码器的RRef
        return remote_params  # 返回所有远程参数引用