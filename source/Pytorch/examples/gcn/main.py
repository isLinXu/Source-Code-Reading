import os  # 导入os模块
import time  # 导入time模块
import requests  # 导入requests模块
import tarfile  # 导入tarfile模块
import numpy as np  # 导入numpy模块
import argparse  # 导入argparse模块

import torch  # 导入PyTorch库
from torch import nn  # 从torch导入nn模块
import torch.nn.functional as F  # 从torch.nn导入功能模块F
from torch.optim import Adam  # 从torch.optim导入Adam优化器


class GraphConv(nn.Module):  # 定义图卷积层类，继承自nn.Module
    """
        Graph Convolutional Layer described in "Semi-Supervised Classification with Graph Convolutional Networks".

        Given an input feature representation for each node in a graph, the Graph Convolutional Layer aims to aggregate
        information from the node's neighborhood to update its own representation. This is achieved by applying a graph
        convolutional operation that combines the features of a node with the features of its neighboring nodes.

        Mathematically, the Graph Convolutional Layer can be described as follows:

            H' = f(D^(-1/2) * A * D^(-1/2) * H * W)

        where:
            H: Input feature matrix with shape (N, F_in), where N is the number of nodes and F_in is the number of 
                input features per node.
            A: Adjacency matrix of the graph with shape (N, N), representing the relationships between nodes.
            W: Learnable weight matrix with shape (F_in, F_out), where F_out is the number of output features per node.
            D: The degree matrix.
    """
    def __init__(self, input_dim, output_dim, use_bias=False):  # 初始化图卷积层
        super(GraphConv, self).__init__()  # 调用父类的初始化方法

        # Initialize the weight matrix W (in this case called `kernel`)
        self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))  # 初始化权重矩阵
        nn.init.xavier_normal_(self.kernel)  # 使用Xavier初始化权重

        # Initialize the bias (if use_bias is True)
        self.bias = None  # 初始化偏置为None
        if use_bias:  # 如果使用偏置
            self.bias = nn.Parameter(torch.Tensor(output_dim))  # 初始化偏置
            nn.init.zeros_(self.bias)  # 将偏置初始化为零

    def forward(self, input_tensor, adj_mat):  # 前向传播方法
        """
        Performs a graph convolution operation.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Normalized adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """

        support = torch.mm(input_tensor, self.kernel)  # 输入与权重矩阵的矩阵乘法
        output = torch.spmm(adj_mat, support)  # 邻接矩阵与支持的稀疏矩阵乘法
        # Add the bias (if bias is not None)
        if self.bias is not None:  # 如果偏置不为None
            output = output + self.bias  # 将偏置添加到输出中

        return output  # 返回输出


class GCN(nn.Module):  # 定义图卷积网络类，继承自nn.Module
    """
    Graph Convolutional Network (GCN) as described in the paper `"Semi-Supervised Classification with Graph 
    Convolutional Networks" <https://arxiv.org/pdf/1609.02907.pdf>`.

    The Graph Convolutional Network is a deep learning architecture designed for semi-supervised node 
    classification tasks on graph-structured data. It leverages the graph structure to learn node representations 
    by propagating information through the graph using graph convolutional layers.

    The original implementation consists of two stacked graph convolutional layers. The ReLU activation function is 
    applied to the hidden representations, and the Softmax activation function is applied to the output representations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True, dropout_p=0.1):  # 初始化GCN模型
        """ Initializes the GAT model. 

        Args:
            input_dim (int): number of input features per node.
            hidden_dim (int): output size of the first Graph Attention Layer.
            output_dim (int): number of classes to predict for each node.
            use_bias (bool, optional): Whether to use bias term in convolutions (default: True).
            dropout_p (float, optional): dropout rate (default: 0.1).
        """

        super(GCN, self).__init__()  # 调用父类的初始化方法

        # Define the Graph Convolution layers
        self.gc1 = GraphConv(input_dim, hidden_dim, use_bias=use_bias)  # 第一层图卷积层
        self.gc2 = GraphConv(hidden_dim, output_dim, use_bias=use_bias)  # 第二层图卷积层

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_p)  # dropout层

    def forward(self, input_tensor, adj_mat):  # 前向传播方法
        """
        Performs forward pass of the Graph Convolutional Network (GCN).

        Args:
            input_tensor (torch.Tensor): Input node feature matrix with shape (N, input_dim), where N is the number of nodes
                and input_dim is the number of input features per node.
            adj_mat (torch.Tensor): Normalized adjacency matrix of the graph with shape (N, N), representing the relationships between
                nodes.

        Returns:
            torch.Tensor: Output tensor with shape (N, output_dim), representing the predicted class probabilities for each node.
        """

        # Perform the first graph convolutional layer
        x = self.gc1(input_tensor, adj_mat)  # 应用第一层图卷积
        x = F.relu(x)  # 对第一层的输出应用ReLU激活函数
        x = self.dropout(x)  # 应用dropout正则化

        # Perform the second graph convolutional layer
        x = self.gc2(x, adj_mat)  # 应用第二层图卷积

        return F.log_softmax(x, dim=1)  # 应用log-softmax激活函数进行分类


def load_cora(path='./cora', device='cpu'):  # 加载Cora数据集
    """
    The graph convolutional operation requires the normalized adjacency matrix: D^(-1/2) * A * D^(-1/2). This step 
    scales the adjacency matrix such that the features of neighboring nodes are weighted appropriately during 
    aggregation. The steps involved in the renormalization trick are as follows:
        - Compute the degree matrix.
        - Compute the inverse square root of the degree matrix.
        - Multiply the inverse square root of the degree matrix with the adjacency matrix.
    """

    # Set the paths to the data files
    content_path = os.path.join(path, 'cora.content')  # 设置内容文件路径
    cites_path = os.path.join(path, 'cora.cites')  # 设置引用文件路径

    # Load data from files
    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))  # 从文件加载数据
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)  # 从文件加载引用数据

    # Process features
    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32))  # 提取特征值
    scale_vector = torch.sum(features, dim=1)  # 计算每个节点特征的和
    scale_vector = 1 / scale_vector  # 计算和的倒数
    scale_vector[scale_vector == float('inf')] = 0  # 处理除以零的情况
    scale_vector = torch.diag(scale_vector).to_sparse()  # 将比例向量转换为稀疏对角矩阵
    features = scale_vector @ features  # 使用比例向量缩放特征

    # Process labels
    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True)  # 提取唯一类别并将标签映射到索引
    labels = torch.LongTensor(labels)  # 将标签转换为张量

    # Process adjacency matrix
    idx = content_tensor[:, 0].astype(np.int32)  # 提取节点索引
    idx_map = {id: pos for pos, id in enumerate(idx)}  # 创建一个字典，将索引映射到位置

    # Map node indices to positions in the adjacency matrix
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], 
            cites_tensor)), dtype=np.int32)  # 将节点索引映射到邻接矩阵中的位置

    V = len(idx)  # 节点数量
    E = edges.shape[0]  # 边的数量
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64)  # 创建初始邻接矩阵作为稀疏张量
    adj_mat = torch.eye(V) + adj_mat  # 在邻接矩阵中添加自环

    degree_mat = torch.sum(adj_mat, dim=1)  # 计算邻接矩阵中每一行的和（度矩阵）
    degree_mat = torch.sqrt(1 / degree_mat)  # 计算度的倒数平方根
    degree_mat[degree_mat == float('inf')] = 0  # 处理除以零的情况
    degree_mat = torch.diag(degree_mat).to_sparse()  # 将度矩阵转换为稀疏对角矩阵

    adj_mat = degree_mat @ adj_mat @ degree_mat  # 应用重归一化技巧

    return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)  # 返回特征、标签和邻接矩阵


def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):  # 训练迭代函数
    start_t = time.time()  # 记录开始时间
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清零梯度

    # Forward pass
    output = model(*input)  # 前向传播
    loss = criterion(output[mask_train], target[mask_train])  # 使用训练掩码计算损失

    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    # Evaluate the model performance on training and validation sets
    loss_train, acc_train = test(model, criterion, input, target, mask_train)  # 在训练集上评估模型性能
    loss_val, acc_val = test(model, criterion, input, target, mask_val)  # 在验证集上评估模型性能

    if epoch % print_every == 0:  # 每隔指定的epoch打印训练进度
        # Print the training progress at specified intervals
        print(f'Epoch: {epoch:04d} ({(time.time() - start_t):.4f}s) loss_train: {loss_train:.4f} acc_train: {acc_train:.4f} loss_val: {loss_val:.4f} acc_val: {acc_val:.4f}')


def test(model, criterion, input, target, mask):  # 测试函数
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在不计算梯度的情况下进行推理
        output = model(*input)  # 前向传播
        output, target = output[mask], target[mask]  # 根据掩码选择输出和目标

        loss = criterion(output, target)  # 计算损失
        acc = (output.argmax(dim=1) == target).float().sum() / len(target)  # 计算准确率
    return loss.item(), acc.item()  # 返回损失和准确率


if __name__ == '__main__':  # 主程序入口
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置设备为CUDA或CPU

    parser = argparse.ArgumentParser(description='PyTorch Graph Convolutional Network')  # 创建参数解析器
    parser.add_argument('--epochs', type=int, default=200,  # 训练的epoch数量（默认：200）
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,  # 学习率（默认：0.01）
                        help='learning rate (default: 0.01)')
    parser.add_argument('--l2', type=float, default=5e-4,  # 权重衰减（默认：5e-4）
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.5,  # dropout概率（默认：0.5）
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--hidden-dim', type=int, default=16,  # 隐藏表示的维度（默认：16）
                        help='dimension of the hidden representation (default: 16)')
    parser.add_argument('--val-every', type=int, default=20,  # 每隔多少个epoch打印一次训练和验证评估（默认：20）
                        help='epochs to wait for print training and validation evaluation (default: 20)')
    parser.add_argument('--include-bias', action='store_true', default=False,  # 是否在卷积中使用偏置项（默认：False）
                        help='use bias term in convolutions (default: False)')
    parser.add_argument('--no-cuda', action='store_true', default=False,  # 禁用CUDA训练
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,  # 禁用macOS GPU训练
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,  # 快速检查单次传递
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, metavar='S',  # 随机种子（默认：42）
                        help='random seed (default: 42)')
    args = parser.parse_args()  # 解析参数

    use_cuda = not args.no_cuda and torch.cuda.is_available()  # 检查是否可以使用CUDA
    use_mps = not args.no_mps and torch.backends.mps.is_available()  # 检查是否可以使用MPS

    torch.manual_seed(args.seed)  # 设置随机种子

    if use_cuda:  # 如果可以使用CUDA
        device = torch.device('cuda')  # 设置设备为CUDA
    elif use_mps:  # 如果可以使用MPS
        device = torch.device('mps')  # 设置设备为MPS
    else:  # 否则使用CPU
        device = torch.device('cpu')  # 设置设备为CPU
    print(f'Using {device} device')  # 打印所使用的设备

    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'  # Cora数据集的URL
    print('Downloading dataset...')  # 下载数据集
    with requests.get(cora_url, stream=True) as tgz_file:  # 发送请求获取数据集
        with tarfile.open(fileobj=tgz_file.raw, mode='r:gz') as tgz_object:  # 打开tar文件
            tgz_object.extractall()  # 解压缩数据集

    print('Loading dataset...')  # 加载数据集
    features, labels, adj_mat = load_cora(device=device)  # 加载Cora数据集
    idx = torch.randperm(len(labels)).to(device)  # 随机打乱标签索引
    idx_test, idx_val, idx_train = idx[:1000], idx[1000:1500], idx[1500:]  # 划分训练集、验证集和测试集

    gcn = GCN(features.shape[1], args.hidden_dim, labels.max().item() + 1, args.include_bias, args.dropout_p).to(device)  # 创建GCN模型
    optimizer = Adam(gcn.parameters(), lr=args.lr, weight_decay=args.l2)  # 配置优化器
    criterion = nn.NLLLoss()  # 配置损失函数

    for epoch in range(args.epochs):  # 进行训练
        train_iter(epoch + 1, gcn, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val, args.val_every)  # 训练迭代
        if args.dry_run:  # 如果是干运行
            break  # 退出循环

    loss_test, acc_test = test(gcn, criterion, (features, adj_mat), labels, idx_test)  # 在测试集上测试模型
    print(f'Test set results: loss {loss_test:.4f} accuracy {acc_test:.4f}')  # 打印测试集结果