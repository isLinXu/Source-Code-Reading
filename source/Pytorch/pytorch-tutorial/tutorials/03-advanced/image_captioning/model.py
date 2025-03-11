import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torchvision.models as models  # 导入 torchvision 的模型模块
from torch.nn.utils.rnn import pack_padded_sequence  # 从 PyTorch 导入用于处理填充序列的工具


class EncoderCNN(nn.Module):  # 定义编码器类
    def __init__(self, embed_size):  # 初始化方法，接受嵌入层大小作为参数
        """Load the pretrained ResNet-152 and replace top fc layer."""  # 加载预训练的 ResNet-152 并替换顶部全连接层
        super(EncoderCNN, self).__init__()  # 调用父类的初始化方法
        resnet = models.resnet152(pretrained=True)  # 加载预训练的 ResNet-152 模型
        modules = list(resnet.children())[:-1]  # 删除最后的全连接层
        self.resnet = nn.Sequential(*modules)  # 创建一个新的顺序模型，包含 ResNet 的所有层（不包括最后一层）
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)  # 添加一个线性层，将特征维度映射到嵌入层大小
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)  # 添加批归一化层

    def forward(self, images):  # 前向传播方法
        """Extract feature vectors from input images."""  # 从输入图像中提取特征向量
        with torch.no_grad():  # 在不计算梯度的情况下进行推理
            features = self.resnet(images)  # 通过 ResNet 提取特征
        features = features.reshape(features.size(0), -1)  # 将特征重塑为二维张量
        features = self.bn(self.linear(features))  # 通过线性层和批归一化处理特征
        return features  # 返回特征向量


class DecoderRNN(nn.Module):  # 定义解码器类
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):  # 初始化方法
        """Set the hyper-parameters and build the layers."""  # 设置超参数并构建层
        super(DecoderRNN, self).__init__()  # 调用父类的初始化方法
        self.embed = nn.Embedding(vocab_size, embed_size)  # 嵌入层，用于将单词 ID 转换为嵌入向量
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM 层
        self.linear = nn.Linear(hidden_size, vocab_size)  # 全连接层，输出特征为词汇表大小
        self.max_seg_length = max_seq_length  # 最大序列长度
        
    def forward(self, features, captions, lengths):  # 前向传播方法
        """Decode image feature vectors and generates captions."""  # 解码图像特征向量并生成注释
        embeddings = self.embed(captions)  # 将注释转换为嵌入向量
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # 将特征向量和嵌入向量拼接
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # 将嵌入向量打包为填充序列
        hiddens, _ = self.lstm(packed)  # 前向传播 LSTM，获取隐藏状态
        outputs = self.linear(hiddens[0])  # 通过全连接层解码隐藏状态
        return outputs  # 返回输出
    
    def sample(self, features, states=None):  # 生成样本的方法
        """Generate captions for given image features using greedy search."""  # 使用贪婪搜索为给定图像特征生成注释
        sampled_ids = []  # 初始化采样 ID 列表
        inputs = features.unsqueeze(1)  # 将特征向量扩展为 3D 张量
        for i in range(self.max_seg_length):  # 遍历最大序列长度
            hiddens, states = self.lstm(inputs, states)  # 前向传播 LSTM，获取隐藏状态
            outputs = self.linear(hiddens.squeeze(1))  # 通过全连接层解码隐藏状态
            _, predicted = outputs.max(1)  # 获取预测的单词 ID
            sampled_ids.append(predicted)  # 将预测的 ID 添加到列表
            inputs = self.embed(predicted)  # 将预测的 ID 转换为嵌入向量
            inputs = inputs.unsqueeze(1)  # 扩展为 3D 张量
        sampled_ids = torch.stack(sampled_ids, 1)  # 将采样 ID 列表堆叠为 2D 张量
        return sampled_ids  # 返回采样的 ID