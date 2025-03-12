import os  # 导入os模块，用于处理文件和目录
import time  # 导入time模块，用于时间相关操作
import glob  # 导入glob模块，用于文件路径匹配

import torch  # 导入PyTorch库
import torch.optim as O  # 从PyTorch导入优化器模块，并命名为O
import torch.nn as nn  # 从PyTorch导入神经网络模块

from torchtext.legacy import data  # 从torchtext导入legacy版本的数据处理模块
from torchtext.legacy import datasets  # 从torchtext导入legacy版本的数据集模块

from model import SNLIClassifier  # 从model模块导入SNLIClassifier类
from util import get_args, makedirs  # 从util模块导入get_args和makedirs函数


args = get_args()  # 获取命令行参数
if torch.cuda.is_available():  # 如果CUDA可用
    torch.cuda.set_device(args.gpu)  # 设置GPU设备
    device = torch.device('cuda:{}'.format(args.gpu))  # 创建CUDA设备对象
elif torch.backends.mps.is_available():  # 如果MPS可用（适用于Mac）
    device = torch.device('mps')  # 创建MPS设备对象
else:  # 如果没有可用的GPU或MPS
    device = torch.device('cpu')  # 使用CPU设备

inputs = data.Field(lower=args.lower, tokenize='spacy')  # 定义输入字段，使用spacy进行分词
answers = data.Field(sequential=False)  # 定义答案字段，不是序列

train, dev, test = datasets.SNLI.splits(inputs, answers)  # 加载SNLI数据集并划分为训练、验证和测试集

inputs.build_vocab(train, dev, test)  # 构建输入词汇表
if args.word_vectors:  # 如果指定了词向量
    if os.path.isfile(args.vector_cache):  # 检查词向量缓存文件是否存在
        inputs.vocab.vectors = torch.load(args.vector_cache)  # 加载缓存的词向量
    else:  # 如果缓存文件不存在
        inputs.vocab.load_vectors(args.word_vectors)  # 加载指定的词向量
        makedirs(os.path.dirname(args.vector_cache))  # 创建缓存目录
        torch.save(inputs.vocab.vectors, args.vector_cache)  # 保存词向量到缓存文件
answers.build_vocab(train)  # 构建答案词汇表

train_iter, dev_iter, test_iter = data.BucketIterator.splits(  # 创建数据迭代器
            (train, dev, test), batch_size=args.batch_size, device=device)  # 设置批量大小和设备

config = args  # 将参数配置赋值给config
config.n_embed = len(inputs.vocab)  # 设置嵌入维度为输入词汇表大小
config.d_out = len(answers.vocab)  # 设置输出维度为答案词汇表大小
config.n_cells = config.n_layers  # 将细胞数量设置为层数

# double the number of cells for bidirectional networks
if config.birnn:  # 如果是双向RNN
    config.n_cells *= 2  # 细胞数量翻倍

if args.resume_snapshot:  # 如果指定了恢复快照
    model = torch.load(args.resume_snapshot, map_location=device)  # 从快照加载模型
else:  # 如果没有恢复快照
    model = SNLIClassifier(config)  # 创建新的SNLIClassifier模型
    if args.word_vectors:  # 如果指定了词向量
        model.embed.weight.data.copy_(inputs.vocab.vectors)  # 将词向量复制到模型嵌入层
        model.to(device)  # 将模型移动到指定设备

criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
opt = O.Adam(model.parameters(), lr=args.lr)  # 创建Adam优化器

iterations = 0  # 初始化迭代次数
start = time.time()  # 记录开始时间
best_dev_acc = -1  # 初始化最佳验证准确率
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'  # 日志头部信息
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))  # 验证日志模板
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))  # 训练日志模板
makedirs(args.save_path)  # 创建保存路径
print(header)  # 打印日志头部信息

for epoch in range(args.epochs):  # 遍历每个训练周期
    train_iter.init_epoch()  # 初始化训练迭代器
    n_correct, n_total = 0, 0  # 初始化正确预测和总预测数量
    for batch_idx, batch in enumerate(train_iter):  # 遍历训练数据批次

        # switch model to training mode, clear gradient accumulators
        model.train(); opt.zero_grad()  # 将模型设置为训练模式并清零梯度累积

        iterations += 1  # 迭代次数加1

        # forward pass
        answer = model(batch)  # 前向传播，获取模型输出

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()  # 计算当前批次的正确预测数量
        n_total += batch.batch_size  # 更新总预测数量
        train_acc = 100. * n_correct / n_total  # 计算训练准确率

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, batch.label)  # 计算损失

        # backpropagate and update optimizer learning rate
        loss.backward(); opt.step()  # 反向传播并更新优化器

        # checkpoint model periodically
        if iterations % args.save_every == 0:  # 如果达到保存间隔
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')  # 定义快照前缀
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(), iterations)  # 定义快照路径
            torch.save(model, snapshot_path)  # 保存模型快照
            for f in glob.glob(snapshot_prefix + '*'):  # 删除旧的快照文件
                if f != snapshot_path:  # 如果不是当前快照
                    os.remove(f)  # 删除文件

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:  # 如果达到验证间隔

            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch()  # 将模型设置为评估模式并初始化验证迭代器

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0  # 初始化验证集的正确预测数量和损失
            with torch.no_grad():  # 在不计算梯度的情况下进行验证
                for dev_batch_idx, dev_batch in enumerate(dev_iter):  # 遍历验证数据批次
                     answer = model(dev_batch)  # 前向传播，获取模型输出
                     n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()  # 计算验证集的正确预测数量
                     dev_loss = criterion(answer, dev_batch.label)  # 计算验证损失
            dev_acc = 100. * n_dev_correct / len(dev)  # 计算验证准确率

            print(dev_log_template.format(time.time() - start,  # 打印验证日志
                epoch, iterations, 1 + batch_idx, len(train_iter),
                100. * (1 + batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

            # update best validation set accuracy
            if dev_acc > best_dev_acc:  # 如果当前验证准确率优于最佳验证准确率

                # found a model with better validation set accuracy

                best_dev_acc = dev_acc  # 更新最佳验证准确率
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')  # 定义最佳快照前缀
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)  # 定义最佳快照路径

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)  # 保存最佳模型快照
                for f in glob.glob(snapshot_prefix + '*'):  # 删除旧的最佳快照文件
                    if f != snapshot_path:  # 如果不是当前最佳快照
                        os.remove(f)  # 删除文件

        elif iterations % args.log_every == 0:  # 如果达到日志间隔

            # print progress message
            print(log_template.format(time.time() - start,  # 打印训练日志
                epoch, iterations, 1 + batch_idx, len(train_iter),
                100. * (1 + batch_idx) / len(train_iter), loss.item(), ' ' * 8, n_correct / n_total * 100, ' ' * 12))
        if args.dry_run:  # 如果是干运行模式
            break  # 退出循环