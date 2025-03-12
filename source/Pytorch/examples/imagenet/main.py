import argparse  # 导入argparse模块，用于命令行参数解析
import os  # 导入os模块，用于与操作系统交互
import random  # 导入random模块，用于生成随机数
import shutil  # 导入shutil模块，用于文件操作
import time  # 导入time模块，用于时间操作
import warnings  # 导入warnings模块，用于发出警告
from enum import Enum  # 从enum模块导入Enum类，用于创建枚举类型

import torch  # 导入PyTorch库
import torch.backends.cudnn as cudnn  # 导入cudnn后端，用于加速卷积运算
import torch.distributed as dist  # 导入分布式训练模块
import torch.multiprocessing as mp  # 导入多进程模块
import torch.nn as nn  # 导入神经网络模块
import torch.nn.parallel  # 导入并行模块
import torch.optim  # 导入优化器模块
import torch.utils.data  # 导入数据处理模块
import torch.utils.data.distributed  # 导入分布式数据处理模块
import torchvision.datasets as datasets  # 导入torchvision数据集模块
import torchvision.models as models  # 导入torchvision模型模块
import torchvision.transforms as transforms  # 导入数据转换模块
from torch.optim.lr_scheduler import StepLR  # 从优化器模块导入学习率调度器
from torch.utils.data import Subset  # 从数据处理模块导入Subset类

# 获取所有可用模型名称并排序
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")  # 选择小写且不以双下划线开头的名称
    and callable(models.__dict__[name]))  # 确保名称对应的是可调用的模型

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',  # 数据集路径参数
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  # 模型架构参数
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +  # 可用模型架构列表
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',  # 数据加载工作线程数量
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',  # 总训练周期数
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  # 手动设置起始周期数
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,  # 批量大小
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,  # 初始学习率
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  # 动量
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,  # 权重衰减
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,  # 打印频率
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',  # 继续训练的检查点路径
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',  # 评估模型参数
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',  # 使用预训练模型
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,  # 分布式训练节点数量
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,  # 节点排名
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,  # 分布式训练的URL
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,  # 分布式后端
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,  # 初始化训练的随机种子
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,  # 使用的GPU ID
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',  # 使用多进程分布式训练
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")  # 使用假数据进行基准测试

best_acc1 = 0  # 初始化最佳准确率


def main():  # 主函数
    args = parser.parse_args()  # 解析命令行参数

    if args.seed is not None:  # 如果指定了随机种子
        random.seed(args.seed)  # 设置Python的随机种子
        torch.manual_seed(args.seed)  # 设置PyTorch的随机种子
        cudnn.deterministic = True  # 设置CUDNN为确定性模式
        cudnn.benchmark = False  # 禁用CUDNN的基准模式
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')  # 发出警告

    if args.gpu is not None:  # 如果指定了GPU
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')  # 发出警告

    if args.dist_url == "env://" and args.world_size == -1:  # 如果使用环境变量
        args.world_size = int(os.environ["WORLD_SIZE"])  # 从环境变量获取世界大小

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed  # 判断是否为分布式训练

    if torch.cuda.is_available():  # 如果CUDA可用
        ngpus_per_node = torch.cuda.device_count()  # 获取每个节点的GPU数量
        if ngpus_per_node == 1 and args.dist_backend == "nccl":  # 如果只有一个GPU且使用NCCL后端
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")  # 发出警告
    else:
        ngpus_per_node = 1  # 如果没有可用的GPU，设置为1

    if args.multiprocessing_distributed:  # 如果使用多进程分布式训练
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size  # 调整世界大小
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))  # 启动分布式进程
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)  # 调用主工作函数


def main_worker(gpu, ngpus_per_node, args):  # 主工作函数
    global best_acc1  # 声明全局变量
    args.gpu = gpu  # 设置当前GPU

    if args.gpu is not None:  # 如果指定了GPU
        print("Use GPU: {} for training".format(args.gpu))  # 打印使用的GPU

    if args.distributed:  # 如果是分布式训练
        if args.dist_url == "env://" and args.rank == -1:  # 如果使用环境变量
            args.rank = int(os.environ["RANK"])  # 从环境变量获取当前排名
        if args.multiprocessing_distributed:  # 如果使用多进程分布式训练
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu  # 计算全局排名
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,  # 初始化进程组
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:  # 如果使用预训练模型
        print("=> using pre-trained model '{}'".format(args.arch))  # 打印使用的预训练模型
        model = models.__dict__[args.arch](pretrained=True)  # 加载预训练模型
    else:
        print("=> creating model '{}'".format(args.arch))  # 打印创建的模型
        model = models.__dict__[args.arch]()  # 创建模型

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():  # 如果没有可用的CUDA和MPS
        print('using CPU, this will be slow')  # 打印使用CPU的警告
    elif args.distributed:  # 如果是分布式训练
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():  # 如果CUDA可用
            if args.gpu is not None:  # 如果指定了GPU
                torch.cuda.set_device(args.gpu)  # 设置当前GPU
                model.cuda(args.gpu)  # 将模型移动到指定GPU
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)  # 调整批量大小
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)  # 调整工作线程数量
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # 使用分布式数据并行
            else:
                model.cuda()  # 将模型移动到CUDA
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)  # 使用分布式数据并行
    elif args.gpu is not None and torch.cuda.is_available():  # 如果指定了GPU且CUDA可用
        torch.cuda.set_device(args.gpu)  # 设置当前GPU
        model = model.cuda(args.gpu)  # 将模型移动到指定GPU
    elif torch.backends.mps.is_available():  # 如果MPS可用
        device = torch.device("mps")  # 设置设备为MPS
        model = model.to(device)  # 将模型移动到MPS
    else:  # 如果没有可用的GPU
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):  # 如果使用AlexNet或VGG
            model.features = torch.nn.DataParallel(model.features)  # 使用数据并行
            model.cuda()  # 将模型移动到CUDA
        else:
            model = torch.nn.DataParallel(model).cuda()  # 使用数据并行并将模型移动到CUDA

    if torch.cuda.is_available():  # 如果CUDA可用
        if args.gpu:  # 如果指定了GPU
            device = torch.device('cuda:{}'.format(args.gpu))  # 设置设备为指定GPU
        else:
            device = torch.device("cuda")  # 设置设备为CUDA
    elif torch.backends.mps.is_available():  # 如果MPS可用
        device = torch.device("mps")  # 设置设备为MPS
    else:
        device = torch.device("cpu")  # 设置设备为CPU
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)  # 定义损失函数

    optimizer = torch.optim.SGD(model.parameters(), args.lr,  # 定义优化器
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 定义学习率调度器
    
    # optionally resume from a checkpoint
    if args.resume:  # 如果指定了恢复路径
        if os.path.isfile(args.resume):  # 如果恢复文件存在
            print("=> loading checkpoint '{}'".format(args.resume))  # 打印加载检查点
            if args.gpu is None:  # 如果没有指定GPU
                checkpoint = torch.load(args.resume)  # 加载检查点
            elif torch.cuda.is_available():  # 如果CUDA可用
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)  # 指定加载到的GPU
                checkpoint = torch.load(args.resume, map_location=loc)  # 加载检查点到指定GPU
            args.start_epoch = checkpoint['epoch']  # 设置起始周期
            best_acc1 = checkpoint['best_acc1']  # 加载最佳准确率
            if args.gpu is not None:  # 如果指定了GPU
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)  # 将最佳准确率移动到指定GPU
            model.load_state_dict(checkpoint['state_dict'])  # 加载模型状态字典
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态字典
            scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率调度器状态字典
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))  # 打印加载的检查点信息
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))  # 打印未找到检查点的警告


    # Data loading code
    if args.dummy:  # 如果使用假数据
        print("=> Dummy data is used!")  # 打印使用假数据的消息
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())  # 创建假数据集
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())  # 创建假验证集
    else:  # 如果使用真实数据
        traindir = os.path.join(args.data, 'train')  # 训练数据路径
        valdir = os.path.join(args.data, 'val')  # 验证数据路径
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 定义数据归一化参数
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(  # 创建训练数据集
            traindir,
            transforms.Compose([  # 定义数据预处理步骤
                transforms.RandomResizedCrop(224),  # 随机裁剪
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),  # 转换为张量
                normalize,  # 归一化
            ]))

        val_dataset = datasets.ImageFolder(  # 创建验证数据集
            valdir,
            transforms.Compose([  # 定义数据预处理步骤
                transforms.Resize(256),  # 调整大小
                transforms.CenterCrop(224),  # 中心裁剪
                transforms.ToTensor(),  # 转换为张量
                normalize,  # 归一化
            ]))

    if args.distributed:  # 如果是分布式训练
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  # 创建训练采样器
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)  # 创建验证采样器
    else:  # 如果不是分布式训练
        train_sampler = None  # 不使用采样器
        val_sampler = None  # 不使用采样器

    train_loader = torch.utils.data.DataLoader(  # 创建训练数据加载器
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),  # 设置批量大小和是否打乱
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)  # 设置工作线程数和采样器

    val_loader = torch.utils.data.DataLoader(  # 创建验证数据加载器
        val_dataset, batch_size=args.batch_size, shuffle=False,  # 设置批量大小
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)  # 设置工作线程数和采样器

    if args.evaluate:  # 如果指定了评估模式
        validate(val_loader, model, criterion, args)  # 验证模型
        return  # 结束主函数

    for epoch in range(args.start_epoch, args.epochs):  # 遍历每个周期
        if args.distributed:  # 如果是分布式训练
            train_sampler.set_epoch(epoch)  # 设置采样器的周期

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)  # 训练一个周期

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)  # 在验证集上评估模型
        
        scheduler.step()  # 更新学习率调度器
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1  # 判断当前准确率是否为最佳
        best_acc1 = max(acc1, best_acc1)  # 更新最佳准确率

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):  # 如果不是多进程分布式训练或当前进程为主进程
            save_checkpoint({  # 保存检查点
                'epoch': epoch + 1,  # 当前周期
                'arch': args.arch,  # 模型架构
                'state_dict': model.state_dict(),  # 模型状态字典
                'best_acc1': best_acc1,  # 最佳准确率
                'optimizer' : optimizer.state_dict(),  # 优化器状态字典
                'scheduler' : scheduler.state_dict()  # 学习率调度器状态字典
            }, is_best)  # 保存最佳检查点


def train(train_loader, model, criterion, optimizer, epoch, device, args):  # 训练函数
    batch_time = AverageMeter('Time', ':6.3f')  # 创建时间计量器
    data_time = AverageMeter('Data', ':6.3f')  # 创建数据计量器
    losses = AverageMeter('Loss', ':.4e')  # 创建损失计量器
    top1 = AverageMeter('Acc@1', ':6.2f')  # 创建准确率计量器
    top5 = AverageMeter('Acc@5', ':6.2f')  # 创建准确率计量器
    progress = ProgressMeter(  # 创建进度计量器
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))  # 设置前缀

    # switch to train mode
    model.train()  # 设置模型为训练模式

    end = time.time()  # 记录开始时间
    for i, (images, target) in enumerate(train_loader):  # 遍历训练数据
        # measure data loading time
        data_time.update(time.time() - end)  # 更新数据加载时间

        # move data to the same device as model
        images = images.to(device, non_blocking=True)  # 将图像数据移动到设备
        target = target.to(device, non_blocking=True)  # 将目标数据移动到设备

        # compute output
        output = model(images)  # 计算模型输出
        loss = criterion(output, target)  # 计算损失

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 计算准确率
        losses.update(loss.item(), images.size(0))  # 更新损失
        top1.update(acc1[0], images.size(0))  # 更新准确率
        top5.update(acc5[0], images.size(0))  # 更新准确率

        # compute gradient and do SGD step
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # measure elapsed time
        batch_time.update(time.time() - end)  # 更新批量时间
        end = time.time()  # 记录当前时间

        if i % args.print_freq == 0:  # 每隔指定频率打印进度
            progress.display(i + 1)  # 显示进度


def validate(val_loader, model, criterion, args):  # 验证函数

    def run_validate(loader, base_progress=0):  # 运行验证函数
        with torch.no_grad():  # 不计算梯度
            end = time.time()  # 记录开始时间
            for i, (images, target) in enumerate(loader):  # 遍历验证数据
                i = base_progress + i  # 更新索引
                if args.gpu is not None and torch.cuda.is_available():  # 如果指定了GPU且CUDA可用
                    images = images.cuda(args.gpu, non_blocking=True)  # 将图像数据移动到指定GPU
                if torch.backends.mps.is_available():  # 如果MPS可用
                    images = images.to('mps')  # 将图像数据移动到MPS
                    target = target.to('mps')  # 将目标数据移动到MPS
                if torch.cuda.is_available():  # 如果CUDA可用
                    target = target.cuda(args.gpu, non_blocking=True)  # 将目标数据移动到指定GPU

                # compute output
                output = model(images)  # 计算模型输出
                loss = criterion(output, target)  # 计算损失

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 计算准确率
                losses.update(loss.item(), images.size(0))  # 更新损失
                top1.update(acc1[0], images.size(0))  # 更新准确率
                top5.update(acc5[0], images.size(0))  # 更新准确率

                # measure elapsed time
                batch_time.update(time.time() - end)  # 更新批量时间
                end = time.time()  # 记录当前时间

                if i % args.print_freq == 0:  # 每隔指定频率打印进度
                    progress.display(i + 1)  # 显示进度

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)  # 创建时间计量器
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)  # 创建损失计量器
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)  # 创建准确率计量器
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)  # 创建准确率计量器
    progress = ProgressMeter(  # 创建进度计量器
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),  # 计算总样本数
        [batch_time, losses, top1, top5],
        prefix='Test: ')  # 设置前缀

    # switch to evaluate mode
    model.eval()  # 设置模型为评估模式

    run_validate(val_loader)  # 运行验证
    if args.distributed:  # 如果是分布式训练
        top1.all_reduce()  # 汇总准确率
        top5.all_reduce()  # 汇总准确率

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):  # 如果是分布式训练且样本数不足
        aux_val_dataset = Subset(val_loader.dataset,  # 创建辅助验证集
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(  # 创建辅助验证数据加载器
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))  # 运行辅助验证

    progress.display_summary()  # 显示进度汇总

    return top1.avg  # 返回平均准确率


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):  # 保存检查点函数
    torch.save(state, filename)  # 保存状态到文件
    if is_best:  # 如果是最佳检查点
        shutil.copyfile(filename, 'model_best.pth.tar')  # 复制最佳检查点


class Summary(Enum):  # 定义枚举类Summary
    NONE = 0  # 不汇总
    AVERAGE = 1  # 平均值
    SUM = 2  # 总和
    COUNT = 3  # 计数


class AverageMeter(object):  # 创建计量器类
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):  # 初始化计量器
        self.name = name  # 计量器名称
        self.fmt = fmt  # 格式
        self.summary_type = summary_type  # 汇总类型
        self.reset()  # 重置计量器

    def reset(self):  # 重置计量器
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):  # 更新计量器
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新总和
        self.count += n  # 更新计数
        self.avg = self.sum / self.count  # 计算平均值

    def all_reduce(self):  # 汇总所有进程的值
        if torch.cuda.is_available():  # 如果CUDA可用
            device = torch.device("cuda")  # 设置设备为CUDA
        elif torch.backends.mps.is_available():  # 如果MPS可用
            device = torch.device("mps")  # 设置设备为MPS
        else:
            device = torch.device("cpu")  # 设置设备为CPU
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)  # 创建总和和计数的张量
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)  # 汇总
        self.sum, self.count = total.tolist()  # 更新总和和计数
        self.avg = self.sum / self.count  # 计算平均值

    def __str__(self):  # 字符串表示
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'  # 格式化字符串
        return fmtstr.format(**self.__dict__)  # 返回格式化后的字符串
    
    def summary(self):  # 汇总信息
        fmtstr = ''  # 初始化格式化字符串
        if self.summary_type is Summary.NONE:  # 如果不汇总
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:  # 如果汇总平均值
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:  # 如果汇总总和
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:  # 如果汇总计数
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)  # 抛出错误
        
        return fmtstr.format(**self.__dict__)  # 返回格式化后的汇总字符串


class ProgressMeter(object):  # 创建进度计量器类
    def __init__(self, num_batches, meters, prefix=""):  # 初始化
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)  # 获取批次格式字符串
        self.meters = meters  # 计量器列表
        self.prefix = prefix  # 前缀

    def display(self, batch):  # 显示进度
        entries = [self.prefix + self.batch_fmtstr.format(batch)]  # 创建条目列表
        entries += [str(meter) for meter in self.meters]  # 添加计量器信息
        print('\t'.join(entries))  # 打印条目
        
    def display_summary(self):  # 显示汇总
        entries = [" *"]  # 初始化条目
        entries += [meter.summary() for meter in self.meters]  # 添加计量器汇总信息
        print(' '.join(entries))  # 打印汇总信息

    def _get_batch_fmtstr(self, num_batches):  # 获取批次格式字符串
        num_digits = len(str(num_batches // 1))  # 计算批次数量的位数
        fmt = '{:' + str(num_digits) + 'd}'  # 创建格式字符串
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'  # 返回格式化后的字符串

def accuracy(output, target, topk=(1,)):  # 计算准确率函数
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():  # 不计算梯度
        maxk = max(topk)  # 获取k的最大值
        batch_size = target.size(0)  # 获取批量大小

        _, pred = output.topk(maxk, 1, True, True)  # 获取top k预测
        pred = pred.t()  # 转置预测结果
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 计算正确预测

        res = []  # 初始化结果列表
        for k in topk:  # 遍历每个k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # 计算正确预测的数量
            res.append(correct_k.mul_(100.0 / batch_size))  # 计算准确率并添加到结果列表
        return res  # 返回结果


if __name__ == '__main__':  # 主程序入口
    main()  # 调用主函数