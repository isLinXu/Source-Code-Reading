# Some part of the code was referenced from below.  # 部分代码参考自以下内容
# https://github.com/pytorch/examples/tree/master/word_language_model  # https://github.com/pytorch/examples/tree/master/word_language_model
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库
from torch.nn.utils import clip_grad_norm_  # 从 PyTorch 导入用于梯度裁剪的工具
from data_utils import Dictionary, Corpus  # 从 data_utils 模块导入 Dictionary 和 Corpus 类


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备配置：如果有可用的 GPU，则使用 GPU，否则使用 CPU

# Hyper-parameters
embed_size = 128  # 嵌入层的大小
hidden_size = 1024  # LSTM 隐藏层的大小
num_layers = 1  # LSTM 的层数
num_epochs = 5  # 训练的轮数
num_samples = 1000  # 要采样的单词数量
batch_size = 20  # 每个批次的样本数量
seq_length = 30  # 序列的长度
learning_rate = 0.002  # 学习率

# Load "Penn Treebank" dataset
corpus = Corpus()  # 创建语料库实例
ids = corpus.get_data('data/train.txt', batch_size)  # 从指定路径加载数据
vocab_size = len(corpus.dictionary)  # 获取词汇表的大小
num_batches = ids.size(1) // seq_length  # 计算批次数


# RNN based language model
class RNNLM(nn.Module):  # 定义基于 RNN 的语言模型类
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):  # 初始化方法
        super(RNNLM, self).__init__()  # 调用父类的初始化方法
        self.embed = nn.Embedding(vocab_size, embed_size)  # 嵌入层
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM 层
        self.linear = nn.Linear(hidden_size, vocab_size)  # 全连接层，输出特征为词汇表大小
        
    def forward(self, x, h):  # 前向传播方法
        # Embed word ids to vectors
        x = self.embed(x)  # 将单词 ID 转换为向量
        
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)  # 前向传播 LSTM，返回输出和隐藏状态
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))  # 将输出重塑为 (batch_size*sequence_length, hidden_size)
        
        # Decode hidden states of all time steps
        out = self.linear(out)  # 通过全连接层解码隐藏状态
        return out, (h, c)  # 返回输出和隐藏状态

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)  # 创建 RNN 模型并转移到指定设备

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器为 Adam

# Truncated backpropagation
def detach(states):  # 定义断开状态的方法
    return [state.detach() for state in states]  # 返回断开梯度的状态

# Train the model
for epoch in range(num_epochs):  # 遍历每个训练轮
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),  # 初始化隐藏状态
              torch.zeros(num_layers, batch_size, hidden_size).to(device))  # 初始化细胞状态
    
    for i in range(0, ids.size(1) - seq_length, seq_length):  # 按序列长度遍历数据
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i+seq_length].to(device)  # 获取小批量输入
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)  # 获取小批量目标
        
        # Forward pass
        states = detach(states)  # 断开状态
        outputs, states = model(inputs, states)  # 前向传播
        loss = criterion(outputs, targets.reshape(-1))  # 计算损失
        
        # Backward and optimize
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        clip_grad_norm_(model.parameters(), 0.5)  # 裁剪梯度
        optimizer.step()  # 更新参数

        step = (i+1) // seq_length  # 计算当前步骤
        if step % 100 == 0:  # 每 100 步打印一次信息
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'  # 打印当前轮次、步骤和损失值
                   .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))  # 打印当前轮次、步骤和损失值

# Test the model
with torch.no_grad():  # 在测试阶段，不需要计算梯度（节省内存）
    with open('sample.txt', 'w') as f:  # 打开文件以写入采样结果
        # Set initial hidden and cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),  # 初始化隐藏状态
                 torch.zeros(num_layers, 1, hidden_size).to(device))  # 初始化细胞状态

        # Select one word id randomly
        prob = torch.ones(vocab_size)  # 创建一个全为 1 的概率张量
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)  # 随机选择一个单词 ID

        for i in range(num_samples):  # 遍历要采样的单词数量
            # Forward propagate RNN 
            output, state = model(input, state)  # 前向传播 RNN 

            # Sample a word id
            prob = output.exp()  # 计算输出的指数以获取概率
            word_id = torch.multinomial(prob, num_samples=1).item()  # 采样一个单词 ID

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)  # 用采样的单词 ID 填充输入

            # File write
            word = corpus.dictionary.idx2word[word_id]  # 获取单词
            word = '\n' if word == '<eos>' else word + ' '  # 如果是结束符，则换行，否则加空格
            f.write(word)  # 写入文件

            if (i+1) % 100 == 0:  # 每 100 个单词打印一次信息
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))  # 打印已采样的单词数量

# Save the model checkpoints
torch.save(model.state_dict(), 'model.ckpt')  # 保存模型的状态字典