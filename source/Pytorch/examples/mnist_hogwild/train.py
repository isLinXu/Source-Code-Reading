import os  # 导入os模块，用于处理文件和目录
import torch  # 导入PyTorch库
import torch.optim as optim  # 从PyTorch导入优化器模块
import torch.nn.functional as F  # 从PyTorch导入功能性模块


def train(rank, args, model, device, dataset, dataloader_kwargs):  # 定义训练函数
    torch.manual_seed(args.seed + rank)  # 设置随机种子，确保每个进程的种子不同

    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)  # 创建训练数据加载器

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # 创建SGD优化器
    for epoch in range(1, args.epochs + 1):  # 遍历每个训练周期
        train_epoch(epoch, args, model, device, train_loader, optimizer)  # 调用训练周期函数


def test(args, model, device, dataset, dataloader_kwargs):  # 定义测试函数
    torch.manual_seed(args.seed)  # 设置随机种子

    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)  # 创建测试数据加载器

    test_epoch(model, device, test_loader)  # 调用测试周期函数


def train_epoch(epoch, args, model, device, data_loader, optimizer):  # 定义训练周期函数
    model.train()  # 将模型设置为训练模式
    pid = os.getpid()  # 获取当前进程ID
    for batch_idx, (data, target) in enumerate(data_loader):  # 遍历数据加载器
        optimizer.zero_grad()  # 清零梯度
        output = model(data.to(device))  # 前向传播，获取模型输出
        loss = F.nll_loss(output, target.to(device))  # 计算负对数似然损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新优化器
        if batch_idx % args.log_interval == 0:  # 每log_interval个批次打印一次日志
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  # 打印训练状态
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:  # 如果是干运行模式
                break  # 退出循环


def test_epoch(model, device, data_loader):  # 定义测试周期函数
    model.eval()  # 将模型设置为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测数量
    with torch.no_grad():  # 在不计算梯度的情况下进行测试
        for data, target in data_loader:  # 遍历数据加载器
            output = model(data.to(device))  # 前向传播，获取模型输出
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()  # 累加测试损失
            pred = output.max(1)[1]  # 获取最大对数概率的索引
            correct += pred.eq(target.to(device)).sum().item()  # 计算正确预测数量

    test_loss /= len(data_loader.dataset)  # 计算平均测试损失
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(  # 打印测试结果
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))