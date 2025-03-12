# coding: utf-8  # 指定文件编码为 UTF-8
import argparse  # 导入 argparse 库，用于处理命令行参数
import time  # 导入 time 库，用于时间相关的操作
import math  # 导入 math 库，用于数学计算
import os  # 导入操作系统库
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.onnx  # 导入 PyTorch 的 ONNX 模块，用于导出模型

import data  # 导入自定义数据模块
import model  # 导入自定义模型模块

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')  # 创建参数解析器，描述为 PyTorch Wikitext-2 语言模型
parser.add_argument('--data', type=str, default='./data/wikitext-2',  # 添加数据参数，指定数据集位置
                    help='location of the data corpus')  # 帮助信息：数据集位置
parser.add_argument('--model', type=str, default='LSTM',  # 添加模型类型参数，指定网络类型
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')  # 帮助信息：网络类型
parser.add_argument('--emsize', type=int, default=200,  # 添加词嵌入大小参数，指定词嵌入的维度
                    help='size of word embeddings')  # 帮助信息：词嵌入大小
parser.add_argument('--nhid', type=int, default=200,  # 添加隐藏单元数量参数，指定每层的隐藏单元数量
                    help='number of hidden units per layer')  # 帮助信息：每层的隐藏单元数量
parser.add_argument('--nlayers', type=int, default=2,  # 添加层数参数，指定网络的层数
                    help='number of layers')  # 帮助信息：层数
parser.add_argument('--lr', type=float, default=20,  # 添加初始学习率参数，指定学习率
                    help='initial learning rate')  # 帮助信息：初始学习率
parser.add_argument('--clip', type=float, default=0.25,  # 添加梯度裁剪参数，指定梯度裁剪的阈值
                    help='gradient clipping')  # 帮助信息：梯度裁剪
parser.add_argument('--epochs', type=int, default=40,  # 添加训练周期数参数，指定训练的周期数
                    help='upper epoch limit')  # 帮助信息：最大周期限制
parser.add_argument('--batch_size', type=int, default=20, metavar='N',  # 添加批次大小参数，指定每批次的样本数量
                    help='batch size')  # 帮助信息：批次大小
parser.add_argument('--bptt', type=int, default=35,  # 添加序列长度参数，指定每个序列的长度
                    help='sequence length')  # 帮助信息：序列长度
parser.add_argument('--dropout', type=float, default=0.2,  # 添加 dropout 参数，指定 dropout 的比例
                    help='dropout applied to layers (0 = no dropout)')  # 帮助信息：应用于层的 dropout
parser.add_argument('--tied', action='store_true',  # 添加参数，指定是否将词嵌入和 softmax 权重绑定
                    help='tie the word embedding and softmax weights')  # 帮助信息：绑定词嵌入和 softmax 权重
parser.add_argument('--seed', type=int, default=1111,  # 添加随机种子参数，指定随机种子
                    help='random seed')  # 帮助信息：随机种子
parser.add_argument('--cuda', action='store_true', default=False,  # 添加 CUDA 参数，指定是否使用 CUDA
                    help='use CUDA')  # 帮助信息：使用 CUDA
parser.add_argument('--mps', action='store_true', default=False,  # 添加 MPS 参数，指定是否启用 macOS GPU 训练
                        help='enables macOS GPU training')  # 帮助信息：启用 macOS GPU 训练
parser.add_argument('--log-interval', type=int, default=200, metavar='N',  # 添加日志间隔参数，指定报告间隔
                    help='report interval')  # 帮助信息：报告间隔
parser.add_argument('--save', type=str, default='model.pt',  # 添加保存路径参数，指定保存模型的路径
                    help='path to save the final model')  # 帮助信息：保存最终模型的路径
parser.add_argument('--onnx-export', type=str, default='',  # 添加 ONNX 导出路径参数，指定导出 ONNX 格式模型的路径
                    help='path to export the final model in onnx format')  # 帮助信息：导出最终模型为 ONNX 格式
parser.add_argument('--nhead', type=int, default=2,  # 添加 Transformer 模型头数参数，指定编码器/解码器的头数
                    help='the number of heads in the encoder/decoder of the transformer model')  # 帮助信息：Transformer 模型中的头数
parser.add_argument('--dry-run', action='store_true',  # 添加干运行参数，指定是否进行代码和模型验证
                    help='verify the code and the model')  # 帮助信息：验证代码和模型
args = parser.parse_args()  # 解析命令行参数

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)  # 手动设置随机种子以确保可重复性
if torch.cuda.is_available():  # 如果 CUDA 可用
    if not args.cuda:  # 如果未指定使用 CUDA
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")  # 警告：您有 CUDA 设备，因此应该使用 --cuda 运行
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # 如果 MPS 可用
    if not args.mps:  # 如果未指定使用 MPS
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")  # 警告：您有 MPS 设备，要启用 macOS GPU，请使用 --mps 运行

use_mps = args.mps and torch.backends.mps.is_available()  # 确定是否使用 MPS
if args.cuda:  # 如果指定使用 CUDA
    device = torch.device("cuda")  # 设置设备为 CUDA
elif use_mps:  # 如果使用 MPS
    device = torch.device("mps")  # 设置设备为 MPS
else:  # 否则
    device = torch.device("cpu")  # 设置设备为 CPU

###############################################################################
# Load data  # 加载数据
###############################################################################

corpus = data.Corpus(args.data)  # 创建 Corpus 实例，加载数据

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):  # 定义 batchify 函数，将数据集分成多个批次
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz  # 计算可以整除的批次数
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)  # 剪切数据，去掉不能整除的部分
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()  # 将数据分成 bsz 个批次，并转置
    return data.to(device)  # 将数据移动到指定设备

eval_batch_size = 10  # 设置评估批次大小
train_data = batchify(corpus.train, args.batch_size)  # 对训练数据进行批处理
val_data = batchify(corpus.valid, eval_batch_size)  # 对验证数据进行批处理
test_data = batchify(corpus.test, eval_batch_size)  # 对测试数据进行批处理

###############################################################################
# Build the model  # 构建模型
###############################################################################

ntokens = len(corpus.dictionary)  # 获取词典中的单词数量
if args.model == 'Transformer':  # 如果指定的模型是 Transformer
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)  # 创建 Transformer 模型
else:  # 否则
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)  # 创建 RNN 模型

criterion = nn.NLLLoss()  # 定义负对数似然损失

###############################################################################
# Training code  # 训练代码
###############################################################################

def repackage_hidden(h):  # 定义 repackage_hidden 函数
    """Wraps hidden states in new Tensors, to detach them from their history."""  # 将隐藏状态包装在新的张量中，以便从历史中分离

    if isinstance(h, torch.Tensor):  # 如果 h 是张量
        return h.detach()  # 返回分离的张量
    else:  # 否则
        return tuple(repackage_hidden(v) for v in h)  # 递归处理隐藏状态

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):  # 定义 get_batch 函数
    seq_len = min(args.bptt, len(source) - 1 - i)  # 计算序列长度
    data = source[i:i+seq_len]  # 获取数据片段
    target = source[i+1:i+1+seq_len].view(-1)  # 获取目标数据
    return data, target  # 返回数据和目标

def evaluate(data_source):  # 定义 evaluate 函数
    # Turn on evaluation mode which disables dropout.
    model.eval()  # 设置模型为评估模式
    total_loss = 0.  # 初始化总损失
    ntokens = len(corpus.dictionary)  # 获取词典中的单词数量
    if args.model != 'Transformer':  # 如果不是 Transformer 模型
        hidden = model.init_hidden(eval_batch_size)  # 初始化隐藏状态
    with torch.no_grad():  # 在不跟踪历史的情况下执行
        for i in range(0, data_source.size(0) - 1, args.bptt):  # 遍历数据源
            data, targets = get_batch(data_source, i)  # 获取数据和目标
            if args.model == 'Transformer':  # 如果是 Transformer 模型
                output = model(data)  # 获取模型输出
                output = output.view(-1, ntokens)  # 调整输出形状
            else:  # 否则
                output, hidden = model(data, hidden)  # 获取模型输出和隐藏状态
                hidden = repackage_hidden(hidden)  # 重新包装隐藏状态
            total_loss += len(data) * criterion(output, targets).item()  # 计算总损失
    return total_loss / (len(data_source) - 1)  # 返回平均损失

def train():  # 定义 train 函数
    # Turn on training mode which enables dropout.
    model.train()  # 设置模型为训练模式
    total_loss = 0.  # 初始化总损失
    start_time = time.time()  # 记录开始时间
    ntokens = len(corpus.dictionary)  # 获取词典中的单词数量
    if args.model != 'Transformer':  # 如果不是 Transformer 模型
        hidden = model.init_hidden(args.batch_size)  # 初始化隐藏状态
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):  # 遍历训练数据
        data, targets = get_batch(train_data, i)  # 获取数据和目标
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()  # 清除梯度
        if args.model == 'Transformer':  # 如果是 Transformer 模型
            output = model(data)  # 获取模型输出
            output = output.view(-1, ntokens)  # 调整输出形状
        else:  # 否则
            hidden = repackage_hidden(hidden)  # 重新包装隐藏状态
            output, hidden = model(data, hidden)  # 获取模型输出和隐藏状态
        loss = criterion(output, targets)  # 计算损失
        loss.backward()  # 反向传播计算梯度

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 裁剪梯度以防止梯度爆炸
        for p in model.parameters():  # 遍历模型参数
            p.data.add_(p.grad, alpha=-lr)  # 更新参数

        total_loss += loss.item()  # 累加总损失

        if batch % args.log_interval == 0 and batch > 0:  # 每 log_interval 批次打印一次信息
            cur_loss = total_loss / args.log_interval  # 计算当前损失
            elapsed = time.time() - start_time  # 计算经过的时间
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '  # 打印训练信息
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))  # 打印当前周期、批次、学习率、损失和困惑度
            total_loss = 0  # 重置总损失
            start_time = time.time()  # 重置开始时间
        if args.dry_run:  # 如果是干运行
            break  # 退出循环

def export_onnx(path, batch_size, seq_len):  # 定义导出 ONNX 模型的函数
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))  # 打印导出路径
    model.eval()  # 设置模型为评估模式
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)  # 创建虚拟输入
    hidden = model.init_hidden(batch_size)  # 初始化隐藏状态
    torch.onnx.export(model, (dummy_input, hidden), path)  # 导出模型为 ONNX 格式

# Loop over epochs.  # 遍历训练周期
lr = args.lr  # 获取学习率
best_val_loss = None  # 初始化最佳验证损失

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):  # 遍历每个训练周期
        epoch_start_time = time.time()  # 记录周期开始时间
        train()  # 训练模型
        val_loss = evaluate(val_data)  # 评估验证数据
        print('-' * 89)  # 打印分隔线
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '  # 打印周期结束信息
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),  # 打印周期、时间和验证损失
                                           val_loss, math.exp(val_loss)))  # 打印验证困惑度
        print('-' * 89)  # 打印分隔线
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:  # 如果当前验证损失是最佳
            with open(args.save, 'wb') as f:  # 以二进制模式打开保存文件
                torch.save(model, f)  # 保存模型
            best_val_loss = val_loss  # 更新最佳验证损失
        else:  # 否则
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0  # 如果没有改进，则降低学习率
except KeyboardInterrupt:  # 如果用户按下 Ctrl + C
    print('-' * 89)  # 打印分隔线
    print('Exiting from training early')  # 打印提前退出训练的信息

# Load the best saved model.  # 加载最佳保存的模型
with open(args.save, 'rb') as f:  # 以二进制模式打开保存的模型文件
    model = torch.load(f)  # 加载模型
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:  # 如果模型是 RNN 类型
        model.rnn.flatten_parameters()  # 使 RNN 参数连续，以加快前向传播

# Run on test data.  # 在测试数据上运行
test_loss = evaluate(test_data)  # 评估测试数据
print('=' * 89)  # 打印分隔线
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(  # 打印训练结束信息
    test_loss, math.exp(test_loss)))  # 打印测试损失和困惑度
print('=' * 89)  # 打印分隔线

if len(args.onnx_export) > 0:  # 如果指定了 ONNX 导出路径
    # Export the model in ONNX format.  # 导出模型为 ONNX 格式
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)  # 调用导出函数