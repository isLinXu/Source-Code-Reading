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


################################
###  GAT LAYER DEFINITION    ###
################################

class GraphAttentionLayer(nn.Module):  # 定义图注意力层类，继承自nn.Module
    """
    Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.
    
        This operation can be mathematically described as:
    
            e_ij = a(W h_i, W h_j)
            α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
            h_i' = σ(Σ_j(α_ij W h_j))
            
            where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
            a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()  # 调用父类的初始化方法

        self.n_heads = n_heads  # 注意力头的数量
        self.concat = concat  # 是否连接最终的注意力头
        self.dropout = dropout  # dropout比率

        if concat:  # 如果连接注意力头
            self.out_features = out_features  # 每个节点的输出特征数量
            assert out_features % n_heads == 0  # 确保输出特征数量是注意力头数量的倍数
            self.n_hidden = out_features // n_heads  # 每个头的隐藏特征数量
        else:  # 在注意力头上取平均（在主论文中使用）
            self.n_hidden = out_features  # 隐藏特征数量

        # 对每个节点应用共享线性变换，由权重矩阵W参数化
        # 初始化权重矩阵W 
        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))  # 权重矩阵

        # 初始化注意力权重a
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))  # 注意力权重

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)  # LeakyReLU激活函数
        self.softmax = nn.Softmax(dim=1)  # softmax激活函数用于注意力系数

        self.reset_parameters()  # 重置参数


    def reset_parameters(self):  # 重置可学习参数
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.W)  # 使用Xavier正态分布初始化W
        nn.init.xavier_normal_(self.a)  # 使用Xavier正态分布初始化a
    

    def _get_attention_scores(self, h_transformed: torch.Tensor):  # 计算注意力分数
        """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
        in vectorized parallel form. for each pair of source and target nodes (i, j),
        the attention score e_ij is computed as follows:

            e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

            where || denotes the concatenation operation, and a and W are the learnable parameters.

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])  # 计算源节点的分数
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])  # 计算目标节点的分数

        # 广播相加 
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        e = source_scores + target_scores.mT  # 计算注意力分数
        return self.leakyrelu(e)  # 返回经过LeakyReLU激活的分数

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):  # 前向传播方法
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        n_nodes = h.shape[0]  # 节点数量

        # 对节点特征应用线性变换 -> W h
        # 输出形状 (n_nodes, n_hidden * n_heads)
        h_transformed = torch.mm(h, self.W)  # 计算线性变换
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)  # 应用dropout

        # 通过重塑张量并将头维度放在前面来拆分头
        # 输出形状 (n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)  # 重塑张量

        # 获取注意力分数
        # 输出形状 (n_heads, n_nodes, n_nodes)
        e = self._get_attention_scores(h_transformed)  # 计算注意力分数

        # 将不存在的边的注意力分数设置为-9e15（屏蔽不存在的边）
        connectivity_mask = -9e16 * torch.ones_like(e)  # 创建连接掩码
        e = torch.where(adj_mat > 0, e, connectivity_mask)  # 屏蔽注意力分数
        
        # 注意力系数作为行的softmax计算
        # 对于注意力分数矩阵e中的每一列j
        attention = F.softmax(e, dim=-1)  # 计算注意力系数
        attention = F.dropout(attention, self.dropout, training=self.training)  # 应用dropout

        # 最终节点嵌入作为其邻居特征的加权平均计算
        h_prime = torch.matmul(attention, h_transformed)  # 计算节点嵌入

        # 连接/平均注意力头
        # 输出形状 (n_nodes, out_features)
        if self.concat:  # 如果连接注意力头
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)  # 连接头
        else:  # 否则取平均
            h_prime = h_prime.mean(dim=0)  # 取平均

        return h_prime  # 返回最终节点嵌入

################################
### MAIN GAT NETWORK MODULE  ###
################################

class GAT(nn.Module):  # 定义图注意力网络类，继承自nn.Module
    """
    Graph Attention Network (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.
    Consists of a 2-layer stack of Graph Attention Layers (GATs). The fist GAT Layer is followed by an ELU activation.
    And the second (final) layer is a GAT layer with a single attention head and softmax activation function. 
    """
    def __init__(self,
        in_features,
        n_hidden,
        n_heads,
        num_classes,
        concat=False,
        dropout=0.4,
        leaky_relu_slope=0.2):  # 初始化GAT模型
        """ Initializes the GAT model. 

        Args:
            in_features (int): number of input features per node.
            n_hidden (int): output size of the first Graph Attention Layer.
            n_heads (int): number of attention heads in the first Graph Attention Layer.
            num_classes (int): number of classes to predict for each node.
            concat (bool, optional): Wether to concatinate attention heads or take an average over them for the
                output of the first Graph Attention Layer. Defaults to False.
            dropout (float, optional): dropout rate. Defaults to 0.4.
            leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
        """

        super(GAT, self).__init__()  # 调用父类的初始化方法

        # 定义图注意力层
        self.gat1 = GraphAttentionLayer(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )  # 第一层GAT
        
        self.gat2 = GraphAttentionLayer(
            in_features=n_hidden, out_features=num_classes, n_heads=1,
            concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )  # 第二层GAT
        

    def forward(self, input_tensor: torch.Tensor , adj_mat: torch.Tensor):  # 前向传播方法
        """
        Performs a forward pass through the network.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """

        # Apply the first Graph Attention layer
        x = self.gat1(input_tensor, adj_mat)  # 应用第一层GAT
        x = F.elu(x)  # 对第一层的输出应用ELU激活函数

        # Apply the second Graph Attention layer
        x = self.gat2(x, adj_mat)  # 应用第二层GAT

        return F.log_softmax(x, dim=1)  # 应用log softmax激活函数并返回

################################
### LOADING THE CORA DATASET ###
################################

def load_cora(path='./cora', device='cpu'):  # 加载Cora数据集
    """
    Loads the Cora dataset. The dataset is downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz.

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

    # return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)
    return features.to(device), labels.to(device), adj_mat.to(device)  # 返回特征、标签和邻接矩阵

#################################
### TRAIN AND TEST FUNCTIONS  ###
#################################

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

    # Training settings
    # All defalut values are the same as in the config used in the main paper

    parser = argparse.ArgumentParser(description='PyTorch Graph Attention Network')  # 创建参数解析器
    parser.add_argument('--epochs', type=int, default=300,  # 训练的epoch数量（默认：300）
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.005,  # 学习率（默认：0.005）
                        help='learning rate (default: 0.005)')
    parser.add_argument('--l2', type=float, default=5e-4,  # 权重衰减（默认：6e-4）
                        help='weight decay (default: 6e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.6,  # dropout概率（默认：0.6）
                        help='dropout probability (default: 0.6)')
    parser.add_argument('--hidden-dim', type=int, default=64,  # 隐藏表示的维度（默认：64）
                        help='dimension of the hidden representation (default: 64)')
    parser.add_argument('--num-heads', type=int, default=8,  # 第一层GAT中的注意力头数量（默认：4）
                        help='number of the attention heads (default: 4)')
    parser.add_argument('--concat-heads', action='store_true', default=False,  # 是否连接注意力头，或对它们取平均（默认：False）
                        help='wether to concatinate attention heads, or average over them (default: False)')
    parser.add_argument('--val-every', type=int, default=20,  # 每隔多少个epoch打印一次训练和验证评估（默认：20）
                        help='epochs to wait for print training and validation evaluation (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,  # 禁用CUDA训练
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,  # 禁用macOS GPU训练
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,  # 快速检查单次传递
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=13, metavar='S',  # 随机种子（默认：13）
                        help='random seed (default: 13)')
    args = parser.parse_args()  # 解析参数

    torch.manual_seed(args.seed)  # 设置随机种子
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # 检查是否可以使用CUDA
    use_mps = not args.no_mps and torch.backends.mps.is_available()  # 检查是否可以使用MPS

    # Set the device to run on
    if use_cuda:  # 如果可以使用CUDA
        device = torch.device('cuda')  # 设置设备为CUDA
    elif use_mps:  # 如果可以使用MPS
        device = torch.device('mps')  # 设置设备为MPS
    else:  # 否则使用CPU
        device = torch.device('cpu')  # 设置设备为CPU
    print(f'Using {device} device')  # 打印所使用的设备

    # Load the dataset
    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'  # Cora数据集的URL
    path = './cora'  # 数据集存储路径

    if os.path.isfile(os.path.join(path, 'cora.content')) and os.path.isfile(os.path.join(path, 'cora.cites')):  # 检查数据集是否已下载
        print('Dataset already downloaded...')  # 数据集已下载
    else:  # 如果未下载
        print('Downloading dataset...')  # 下载数据集
        with requests.get(cora_url, stream=True) as tgz_file:  # 发送请求获取数据集
            with tarfile.open(fileobj=tgz_file.raw, mode='r:gz') as tgz_object:  # 打开tar文件
                tgz_object.extractall()  # 解压缩数据集

    print('Loading dataset...')  # 加载数据集
    # Load the dataset
    features, labels, adj_mat = load_cora(device=device)  # 加载Cora数据集
    # Split the dataset into training, validation, and test sets
    idx = torch.randperm(len(labels)).to(device)  # 随机打乱标签索引
    idx_test, idx_val, idx_train = idx[:1200], idx[1200:1600], idx[1600:]  # 划分训练集、验证集和测试集


    # Create the model
    # The model consists of a 2-layer stack of Graph Attention Layers (GATs).
    gat_net = GAT(
        in_features=features.shape[1],          # 每个节点的输入特征数量  
        n_hidden=args.hidden_dim,               # 第一层GAT的输出大小
        n_heads=args.num_heads,                 # 第一层GAT中的注意力头数量
        num_classes=labels.max().item() + 1,    # 每个节点预测的类别数量
        concat=args.concat_heads,               # 是否连接注意力头
        dropout=args.dropout_p,                 # dropout比率
        leaky_relu_slope=0.2                    # Leaky ReLU激活的alpha（斜率）
    ).to(device)  # 将模型移动到指定设备

    # configure the optimizer and loss function
    optimizer = Adam(gat_net.parameters(), lr=args.lr, weight_decay=args.l2)  # 配置优化器
    criterion = nn.NLLLoss()  # 配置损失函数

    # Train and evaluate the model
    for epoch in range(args.epochs):  # 进行训练
        train_iter(epoch + 1, gat_net, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val, args.val_every)  # 训练迭代
        if args.dry_run:  # 如果是干运行
            break  # 退出循环
    loss_test, acc_test = test(gat_net, criterion, (features, adj_mat), labels, idx_test)  # 在测试集上测试模型
    print(f'Test set results: loss {loss_test:.4f} accuracy {acc_test:.4f}')  # 打印测试集结果