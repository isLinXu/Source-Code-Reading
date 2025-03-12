from __future__ import print_function  # 为了兼容 Python 2 和 3 的 print 函数
import argparse, random, copy  # 导入 argparse 用于处理命令行参数，random 用于生成随机数，copy 用于对象复制
import numpy as np  # 导入 NumPy 库

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
import torchvision  # 导入 torchvision 库
from torch.utils.data import Dataset  # 导入数据集基类
from torchvision import datasets  # 导入数据集模块
from torchvision import transforms as T  # 导入转换模块
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        用于图像相似性估计的孪生网络。
        
        The network is composed of two identical networks, one for each input.
        网络由两个相同的子网络组成，每个输入一个。
        
        The output of each network is concatenated and passed to a linear layer. 
        每个网络的输出被连接并传递到一个线性层。
        
        The output of the linear layer passed through a sigmoid function.
        线性层的输出通过 sigmoid 函数。
        
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ 是一种孪生网络的变体。
        
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        该实现与 FaceNet 不同，因为我们使用 `ResNet-18` 模型作为特征提取器。
        
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
        此外，我们不使用 `TripletLoss`，因为 MNIST 数据集很简单，因此 `BCELoss` 足以处理。
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()  # 调用父类构造函数
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)  # 获取 ResNet-18 模型

        # over-write the first conv layer to be able to read MNIST images
        # 修改第一个卷积层以能够读取 MNIST 图像
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改卷积层以适应灰度图像
        self.fc_in_features = self.resnet.fc.in_features  # 获取 ResNet 的输入特征数量
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        # 移除 ResNet-18 的最后一层（在平均池化层之前的线性层）
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))  # 创建新的 ResNet 模型

        # add linear layers to compare between the features of the two images
        # 添加线性层以比较两幅图像的特征
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),  # 输入特征为两幅图像的特征
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Linear(256, 1),  # 输出一个值
        )

        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

        # initialize the weights
        self.resnet.apply(self.init_weights)  # 初始化 ResNet 权重
        self.fc.apply(self.init_weights)  # 初始化线性层权重
        
    def init_weights(self, m):
        """Initialize weights of the model.
        初始化模型的权重。
        """
        if isinstance(m, nn.Linear):  # 如果是线性层
            torch.nn.init.xavier_uniform_(m.weight)  # 使用 Xavier 均匀分布初始化权重
            m.bias.data.fill_(0.01)  # 将偏置初始化为 0.01

    def forward_once(self, x):
        """Forward pass for a single input.
        单个输入的前向传播。
        
        Args:
            x: 输入张量
        Returns:
            output: 特征输出
        """
        output = self.resnet(x)  # 通过 ResNet 进行前向传播
        output = output.view(output.size()[0], -1)  # 将输出展平
        return output

    def forward(self, input1, input2):
        """Forward pass for two inputs.
        两个输入的前向传播。
        
        Args:
            input1: 第一个输入张量
            input2: 第二个输入张量
        Returns:
            output: 相似度输出
        """
        # get two images' features
        output1 = self.forward_once(input1)  # 获取第一个图像的特征
        output2 = self.forward_once(input2)  # 获取第二个图像的特征

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)  # 连接两个图像的特征

        # pass the concatenation to the linear layers
        output = self.fc(output)  # 通过线性层

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)  # 通过 Sigmoid 层
        
        return output  # 返回相似度输出

class APP_MATCHER(Dataset):
    """Custom dataset class for MNIST image pairs.
    自定义数据集类，用于 MNIST 图像对。"""
    
    def __init__(self, root, train, download=False):
        super(APP_MATCHER, self).__init__()  # 调用父类构造函数

        # get MNIST dataset
        self.dataset = datasets.MNIST(root, train=train, download=download)  # 获取 MNIST 数据集
        
        # as `self.dataset.data`'s shape is (Nx28x28), where N is the number of
        # examples in MNIST dataset, a single example has the dimensions of
        # (28x28) for (WxH), where W and H are the width and the height of the image. 
        # However, every example should have (CxWxH) dimensions where C is the number 
        # of channels to be passed to the network. As MNIST contains gray-scale images, 
        # we add an additional dimension to corresponds to the number of channels.
        self.data = self.dataset.data.unsqueeze(1).clone()  # 将数据维度扩展以适应网络输入

        self.group_examples()  # 分组示例

    def group_examples(self):
        """Group examples based on class.
        根据类别分组示例。
        
        Every key in `grouped_examples` corresponds to a class in MNIST dataset.
        每个 `grouped_examples` 中的键对应 MNIST 数据集中的一个类别。
        For every key in `grouped_examples`, every value will conform to all of the indices for the MNIST 
        dataset examples that correspond to that key.
        对于 `grouped_examples` 中的每个键，每个值将对应于与该键相对应的 MNIST 数据集示例的所有索引。
        """
        # get the targets from MNIST dataset
        np_arr = np.array(self.dataset.targets.clone())  # 获取 MNIST 数据集的目标
        
        # group examples based on class
        self.grouped_examples = {}  # 初始化分组字典
        for i in range(0, 10):  # 遍历每个类别
            self.grouped_examples[i] = np.where((np_arr == i))[0]  # 根据类别分组示例
    
    def __len__(self):
        """Return the number of examples in the dataset.
        返回数据集中的示例数量。
        """
        return self.data.shape[0]  # 返回数据的数量
    
    def __getitem__(self, index):
        """Get a pair of images and their label.
        获取一对图像及其标签。
        
        For every example, we will select two images. There are two cases, 
        positive and negative examples. For positive examples, we will have two 
        images from the same class. For negative examples, we will have two images 
        from different classes.
        对于每个示例，我们将选择两幅图像。有两种情况，正例和负例。对于正例，我们将有两幅来自同一类别的图像。对于负例，我们将有两幅来自不同类别的图像。
        """
        # pick some random class for the first image
        selected_class = random.randint(0, 9)  # 随机选择一个类别

        # pick a random index for the first image in the grouped indices based on the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)  # 随机选择第一个图像的索引
        
        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]  # 获取第一个图像的索引

        # get the first image
        image_1 = self.data[index_1].clone().float()  # 获取第一个图像并转换为浮点数

        # same class
        if index % 2 == 0:  # 如果索引为偶数
            # pick a random index for the second image
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)  # 随机选择第二个图像的索引
            
            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:  # 确保第二个图像的索引与第一个不同
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)  # 重新选择索引
            
            # pick the index to get the second image
            index_2 = self.grouped_examples[selected_class][random_index_2]  # 获取第二个图像的索引

            # get the second image
            image_2 = self.data[index_2].clone().float()  # 获取第二个图像并转换为浮点数

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)  # 设置标签为正例（1）
        
        # different class
        else:  # 如果索引为奇数
            # pick a random class
            other_selected_class = random.randint(0, 9)  # 随机选择另一个类别

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:  # 确保第二个图像的类别与第一个不同
                other_selected_class = random.randint(0, 9)  # 重新选择类别
            
            # pick a random index for the second image in the grouped indices based on the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0] - 1)  # 随机选择第二个图像的索引

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_class][random_index_2]  # 获取第二个图像的索引

            # get the second image
            image_2 = self.data[index_2].clone().float()  # 获取第二个图像并转换为浮点数

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)  # 设置标签为负例（0）

        return image_1, image_2, target  # 返回图像对和标签


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # 设置模型为训练模式

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    # 我们不使用 `TripletLoss`，因为 MNIST 数据集很简单，因此 `BCELoss` 足以处理。
    criterion = nn.BCELoss()  # 定义二元交叉熵损失

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):  # 遍历训练数据加载器
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)  # 将数据移动到指定设备
        optimizer.zero_grad()  # 清除梯度
        outputs = model(images_1, images_2).squeeze()  # 前向传播得到输出
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        if batch_idx % args.log_interval == 0:  # 每 log_interval 批次打印一次信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))  # 打印训练信息
            if args.dry_run:  # 如果是干运行模式
                break  # 结束循环


def test(model, device, test_loader):
    model.eval()  # 设置模型为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测计数

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    # 我们不使用 `TripletLoss`，因为 MNIST 数据集很简单，因此 `BCELoss` 足以处理。
    criterion = nn.BCELoss()  # 定义二元交叉熵损失

    with torch.no_grad():  # 在不跟踪梯度的情况下执行
        for (images_1, images_2, targets) in test_loader:  # 遍历测试数据加载器
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)  # 将数据移动到指定设备
            outputs = model(images_1, images_2).squeeze()  # 前向传播得到输出
            test_loss += criterion(outputs, targets).sum().item()  # 累加测试损失
            pred = torch.where(outputs > 0.5, 1, 0)  # 获取预测结果
            correct += pred.eq(targets.view_as(pred)).sum().item()  # 计算正确预测数量

    test_loss /= len(test_loader.dataset)  # 计算平均测试损失

    # for the 1st epoch, the average loss is 0.0001 and the accuracy 97-98%
    # using default settings. After completing the 10th epoch, the average
    # loss is 0.0000 and the accuracy 99.5-100% using default settings.
    # 对于第一个周期，平均损失为 0.0001，准确率为 97-98%（使用默认设置）。完成第十个周期后，平均损失为 0.0000，准确率为 99.5-100%（使用默认设置）。
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))  # 打印测试结果


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')  # 创建参数解析器，描述为 PyTorch 孪生网络示例
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')  # 添加训练批次大小参数
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')  # 添加测试批次大小参数
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')  # 添加训练周期数参数
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')  # 添加学习率参数
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')  # 添加学习率衰减参数
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  # 添加禁用 CUDA 的参数
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')  # 添加禁用 macOS GPU 的参数
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')  # 添加干运行模式参数
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')  # 添加随机种子参数
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')  # 添加日志间隔参数
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')  # 添加保存模型的参数
    args = parser.parse_args()  # 解析命令行参数
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # 检查是否可以使用 CUDA
    use_mps = not args.no_mps and torch.backends.mps.is_available()  # 检查是否可以使用 MPS

    torch.manual_seed(args.seed)  # 设置随机种子

    if use_cuda:  # 如果可以使用 CUDA
        device = torch.device("cuda")  # 设置设备为 CUDA
    elif use_mps:  # 如果可以使用 MPS
        device = torch.device("mps")  # 设置设备为 MPS
    else:  # 否则
        device = torch.device("cpu")  # 设置设备为 CPU

    train_kwargs = {'batch_size': args.batch_size}  # 设置训练参数
    test_kwargs = {'batch_size': args.test_batch_size}  # 设置测试参数
    if use_cuda:  # 如果可以使用 CUDA
        cuda_kwargs = {'num_workers': 1,  # 设置工作线程数
                       'pin_memory': True,  # 针对 CUDA 的内存固定
                       'shuffle': True}  # 随机打乱数据
        train_kwargs.update(cuda_kwargs)  # 更新训练参数
        test_kwargs.update(cuda_kwargs)  # 更新测试参数

    train_dataset = APP_MATCHER('../data', train=True, download=True)  # 创建训练数据集
    test_dataset = APP_MATCHER('../data', train=False)  # 创建测试数据集
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)  # 创建训练数据加载器
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)  # 创建测试数据加载器

    model = SiameseNetwork().to(device)  # 实例化模型并移动到指定设备
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # 创建优化器

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # 创建学习率调度器
    for epoch in range(1, args.epochs + 1):  # 遍历每个训练周期
        train(args, model, device, train_loader, optimizer, epoch)  # 训练模型
        test(model, device, test_loader)  # 测试模型
        scheduler.step()  # 更新学习率

    if args.save_model:  # 如果需要保存模型
        torch.save(model.state_dict(), "siamese_network.pt")  # 保存模型状态


if __name__ == '__main__':
    main()  # 运行主函数