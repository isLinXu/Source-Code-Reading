# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset.

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 2022 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt, yolov5x-cls.pt
Torchvision models: --model resnet50, efficientnet_b0, etc. See https://pytorch.org/vision/stable/models.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
# 获取当前文件的绝对路径
ROOT = FILE.parents[1]  # YOLOv5 root directory
# 获取YOLOv5根目录（当前文件的父目录的父目录）
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    # 如果根目录不在系统路径中，则将其添加到系统路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# 将根目录转换为相对于当前工作目录的路径

from classify import val as validate
# 从classify模块导入验证函数
from models.experimental import attempt_load
# 从experimental模块导入尝试加载模型的函数
from models.yolo import ClassificationModel, DetectionModel
# 从yolo模块导入分类模型和检测模型
from utils.dataloaders import create_classification_dataloader
# 从dataloaders模块导入创建分类数据加载器的函数
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    WorkingDirectory,
    check_git_info,
    check_git_status,
    check_requirements,
    colorstr,
    download,
    increment_path,
    init_seeds,
    print_args,
    yaml_save,
)
# 从general模块导入各种实用工具函数和常量

from utils.loggers import GenericLogger
# 从loggers模块导入通用日志记录器
from utils.plots import imshow_cls
# 从plots模块导入显示分类图像的函数
from utils.torch_utils import (
    ModelEMA,
    de_parallel,
    model_info,
    reshape_classifier_output,
    select_device,
    smart_DDP,
    smart_optimizer,
    smartCrossEntropyLoss,
    torch_distributed_zero_first,
)
# 从torch_utils模块导入各种与PyTorch相关的工具函数和类

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
# 获取当前进程的本地排名，默认为-1
RANK = int(os.getenv("RANK", -1))
# 获取当前进程的全局排名，默认为-1
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
# 获取全局进程数量，默认为1
GIT_INFO = check_git_info()
# 检查Git信息

def train(opt, device):
    """Trains a YOLOv5 model, managing datasets, model optimization, logging, and saving checkpoints."""
    # 训练YOLOv5模型，管理数据集、模型优化、日志记录和保存检查点
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 初始化随机种子，确保可重复性
    save_dir, data, bs, epochs, nw, imgsz, pretrained = (
        opt.save_dir,
        Path(opt.data),
        opt.batch_size,
        opt.epochs,
        min(os.cpu_count() - 1, opt.workers),
        opt.imgsz,
        str(opt.pretrained).lower() == "true",
    )
    # 解包训练选项，包括保存目录、数据集路径、批处理大小、训练轮数、工作线程数、图像大小和是否使用预训练模型
    cuda = device.type != "cpu"
    # 判断当前设备是否为CUDA（GPU）

    # Directories
    wdir = save_dir / "weights"
    # 设置权重保存目录
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    # 创建权重保存目录，如果已存在则不报错
    last, best = wdir / "last.pt", wdir / "best.pt"
    # 定义最后一次和最佳模型的保存路径

    # Save run settings
    yaml_save(save_dir / "opt.yaml", vars(opt))
    # 将训练选项保存为YAML文件

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None
    # 初始化日志记录器，如果当前进程是主进程则启用日志记录

    # Download Dataset
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        # 在分布式训练中，确保只有一个进程下载数据集
        data_dir = data if data.is_dir() else (DATASETS_DIR / data)
        # 检查数据集路径，如果是目录则使用该路径，否则拼接为完整路径
        if not data_dir.is_dir():
            LOGGER.info(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")
            # 如果数据集目录不存在，记录信息并尝试下载数据集
            t = time.time()
            if str(data) == "imagenet":
                subprocess.run(["bash", str(ROOT / "data/scripts/get_imagenet.sh")], shell=True, check=True)
                # 如果数据集是imagenet，运行下载脚本
            else:
                url = f"https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip"
                download(url, dir=data_dir.parent)
                # 生成数据集下载链接并下载数据集
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)
            # 记录下载成功的信息，包括耗时和保存路径

    # Dataloaders
    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes
    # 计算训练数据集中类别的数量，通过检查训练目录下的子目录数量

    trainloader = create_classification_dataloader(
        path=data_dir / "train",
        imgsz=imgsz,
        batch_size=bs // WORLD_SIZE,
        augment=True,
        cache=opt.cache,
        rank=LOCAL_RANK,
        workers=nw,
    )
    # 创建训练数据加载器，设置数据路径、图像大小、批处理大小、数据增强、缓存选项、进程排名和工作线程数

    test_dir = data_dir / "test" if (data_dir / "test").exists() else data_dir / "val"  # data/test or data/val
    # 设置测试数据集路径，如果测试目录存在则使用测试目录，否则使用验证目录
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(
            path=test_dir,
            imgsz=imgsz,
            batch_size=bs // WORLD_SIZE * 2,
            augment=False,
            cache=opt.cache,
            rank=-1,
            workers=nw,
        )
        # 创建测试数据加载器，设置数据路径、图像大小、批处理大小（为训练时的两倍）、不进行数据增强、缓存选项、进程排名和工作线程数

    # Model
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        # 在分布式训练中，确保只有一个进程加载模型
        if Path(opt.model).is_file() or opt.model.endswith(".pt"):
            model = attempt_load(opt.model, device="cpu", fuse=False)
            # 如果模型路径是文件或以.pt结尾，则尝试加载该模型
        elif opt.model in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
            model = torchvision.models.__dict__[opt.model](weights="IMAGENET1K_V1" if pretrained else None)
            # 如果模型在TorchVision模型字典中，则加载相应的TorchVision模型
        else:
            m = hub.list("ultralytics/yolov5")  # + hub.list('pytorch/vision')  # models
            raise ModuleNotFoundError(f"--model {opt.model} not found. Available models are: \n" + "\n".join(m))
            # 如果模型不在已知模型列表中，则引发未找到模型的错误，并列出可用模型

        if isinstance(model, DetectionModel):
            LOGGER.warning("WARNING ⚠️ pass YOLOv5 classifier model with '-cls' suffix, i.e. '--model yolov5s-cls.pt'")
            # 如果模型是检测模型，发出警告，提示应使用带有'-cls'后缀的分类模型
            model = ClassificationModel(model=model, nc=nc, cutoff=opt.cutoff or 10)  # convert to classification model
            # 将检测模型转换为分类模型

        reshape_classifier_output(model, nc)  # update class count
        # 更新分类模型的类别数量

    for m in model.modules():
        if not pretrained and hasattr(m, "reset_parameters"):
            m.reset_parameters()
            # 如果模型不是预训练的，并且模块具有重置参数的方法，则重置模块的参数
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout
            # 如果模块是Dropout层，并且设置了dropout参数，则更新Dropout的概率

    for p in model.parameters():
        p.requires_grad = True  # for training
        # 设置模型参数为可训练

    model = model.to(device)
    # 将模型移动到指定设备（CPU或GPU）

    # Info
    if RANK in {-1, 0}:
        model.names = trainloader.dataset.classes  # attach class names
        # 将类别名称附加到模型中
        model.transforms = testloader.dataset.torch_transforms  # attach inference transforms
        # 将推理转换附加到模型中
        model_info(model)
        # 打印模型信息
        if opt.verbose:
            LOGGER.info(model)
            # 如果设置了详细输出，则记录模型信息
        images, labels = next(iter(trainloader))
        # 从训练加载器中获取下一批图像和标签
        file = imshow_cls(images[:25], labels[:25], names=model.names, f=save_dir / "train_images.jpg")
        # 显示前25张图像及其标签，并保存为train_images.jpg
        logger.log_images(file, name="Train Examples")
        # 记录训练示例图像
        logger.log_graph(model, imgsz)  # log model
        # 记录模型结构图

    # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)
    # 初始化优化器，使用自定义的智能优化器，设置优化器类型、初始学习率、动量和衰减率

    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0)
    # 设置最终学习率为初始学习率的1%

    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    # 定义余弦学习率调度函数（已注释掉）
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    # 定义线性学习率调度函数
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # 创建学习率调度器，使用定义的线性调度函数

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    # 如果当前进程是主进程，则初始化模型的指数移动平均（EMA）

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)
        # 如果使用CUDA并且不是单进程模式，则使用智能分布式数据并行（DDP）模型

    # Train
    t0 = time.time()
    # 记录训练开始的时间
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    # 定义损失函数，使用智能交叉熵损失，支持标签平滑
    best_fitness = 0.0
    # 初始化最佳适应度为0
    scaler = amp.GradScaler(enabled=cuda)
    # 初始化梯度缩放器，用于混合精度训练
    val = test_dir.stem  # 'val' or 'test'
    # 获取验证或测试目录的基本名称
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} test\n'
        f'Using {nw * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n'
        f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}"
    )
    # 记录训练的基本信息，包括图像大小、数据加载器工作线程数、保存路径、模型名称、数据集、类别数量和训练轮数

    for epoch in range(epochs):  # loop over the dataset multiple times
        # 遍历训练轮数
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        # 初始化训练损失、验证损失和适应度
        model.train()
        # 设置模型为训练模式
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
            # 如果不是单进程模式，则设置训练数据加载器的当前轮次
        pbar = enumerate(trainloader)
        # 初始化进度条
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format=TQDM_BAR_FORMAT)
            # 如果是主进程，则使用tqdm库显示进度条
        for i, (images, labels) in pbar:  # progress bar
            # 遍历训练数据加载器，获取图像和标签
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            # 将图像和标签移动到指定设备

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled
                # 使用自动混合精度进行前向传播
                loss = criterion(model(images), labels)
                # 计算损失

            # Backward
            scaler.scale(loss).backward()
            # 缩放损失并进行反向传播

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            # 反缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            # 裁剪梯度以防止爆炸
            scaler.step(optimizer)
            # 更新优化器
            scaler.update()
            # 更新缩放器
            optimizer.zero_grad()
            # 清零梯度
            if ema:
                ema.update(model)
                # 如果使用EMA，则更新EMA模型

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                # 更新平均训练损失
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                # 获取当前GPU内存使用情况
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + " " * 36
                # 更新进度条描述

                # Test
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss = validate.run(
                        model=ema.ema, dataloader=testloader, criterion=criterion, pbar=pbar
                    )  # test accuracy, loss
                    # 在最后一批次中进行验证，获取top1和top5准确率以及验证损失
                    fitness = top1  # define fitness as top1 accuracy
                    # 将适应度定义为top1准确率

        # Scheduler
        scheduler.step()
        # 更新学习率调度器

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness
                # 如果当前适应度超过最佳适应度，则更新最佳适应度

            # Log
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]["lr"],
            }  # learning rate
            # 记录当前训练损失、验证损失、top1和top5准确率以及学习率
            logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            # 检查是否为最后一轮
            if (not opt.nosave) or final_epoch:
                # 如果不禁止保存模型或者是最后一轮，则保存模型
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
                    "ema": None,  # deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": None,  # optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }
                # 创建检查点字典，包含当前轮次、最佳适应度、模型、EMA更新次数、优化器状态、选项和Git信息

                # Save last, best and delete
                torch.save(ckpt, last)
                # 保存最后一次模型
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                    # 如果当前适应度是最佳适应度，则保存最佳模型
                del ckpt
                # 删除检查点字典以释放内存

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(
            f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
            f"\nResults saved to {colorstr('bold', save_dir)}"
            f'\nPredict:         python classify/predict.py --weights {best} --source im.jpg'
            f'\nValidate:        python classify/val.py --weights {best} --data {data_dir}'
            f'\nExport:          python export.py --weights {best} --include onnx'
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{best}')"
            f'\nVisualize:       https://netron.app\n'
        )
        # 训练完成，记录训练时间和保存路径，并提供预测、验证和导出模型的命令

        # Plot examples
        images, labels = (x[:25] for x in next(iter(testloader)))  # first 25 images and labels
        # 获取测试加载器中的前25张图像和标签
        pred = torch.max(ema.ema(images.to(device)), 1)[1]
        # 使用EMA模型进行预测
        file = imshow_cls(images, labels, pred, de_parallel(model).names, verbose=False, f=save_dir / "test_images.jpg")
        # 显示图像及其真实标签和预测标签，并保存为test_images.jpg

        # Log results
        meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        # 创建元数据字典，包含训练轮数、最佳top1准确率和当前日期
        logger.log_images(file, name="Test Examples (true-predicted)", epoch=epoch)
        # 记录测试示例图像
        logger.log_model(best, epochs, metadata=meta)
        # 记录模型信息

def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    # 解析YOLOv5训练的命令行参数，包括模型路径、数据集、训练轮数等，并返回解析后的参数
    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器

    parser.add_argument("--model", type=str, default="yolov5s-cls.pt", help="initial weights path")
    # 添加模型路径参数，默认为yolov5s-cls.pt

    parser.add_argument("--data", type=str, default="imagenette160", help="cifar10, cifar100, mnist, imagenet, ...")
    # 添加数据集参数，默认为imagenette160

    parser.add_argument("--epochs", type=int, default=10, help="total training epochs")
    # 添加训练轮数参数，默认为10

    parser.add_argument("--batch-size", type=int, default=64, help="total batch size for all GPUs")
    # 添加批处理大小参数，默认为64，表示所有GPU的总批处理大小

    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    # 添加图像大小参数，默认为224像素

    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    # 添加不保存中间检查点的标志，如果设置则仅保存最终检查点

    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    # 添加缓存选项，支持在RAM或磁盘中缓存图像

    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 添加设备参数，指定使用的CUDA设备或CPU

    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    # 添加最大数据加载工作线程数参数，默认为8

    parser.add_argument("--project", default=ROOT / "runs/train-cls", help="save to project/name")
    # 添加项目保存路径参数，默认为runs/train-cls

    parser.add_argument("--name", default="exp", help="save to project/name")
    # 添加实验名称参数，默认为"exp"

    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 添加允许存在的项目名称标志，如果设置则不递增项目名称

    parser.add_argument("--pretrained", nargs="?", const=True, default=True, help="start from i.e. --pretrained False")
    # 添加预训练模型参数，默认为True

    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    # 添加优化器选择参数，默认为Adam

    parser.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    # 添加初始学习率参数，默认为0.001

    parser.add_argument("--decay", type=float, default=5e-5, help="weight decay")
    # 添加权重衰减参数，默认为5e-5

    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    # 添加标签平滑参数，默认为0.1

    parser.add_argument("--cutoff", type=int, default=None, help="Model layer cutoff index for Classify() head")
    # 添加模型层截止索引参数，用于分类头

    parser.add_argument("--dropout", type=float, default=None, help="Dropout (fraction)")
    # 添加Dropout参数，用于设置Dropout的比例

    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    # 添加详细输出模式的标志

    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    # 添加全局训练种子参数，默认为0

    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    # 添加自动DDP多GPU参数，默认为-1，不要修改

    return parser.parse_known_args()[0] if known else parser.parse_args()
    # 返回解析后的参数，如果known为True，则解析已知参数，否则解析所有参数


def main(opt):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    # 使用给定选项执行YOLOv5训练，处理设备设置和DDP模式，包括预训练检查
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # 如果是主进程，则打印参数
        check_git_status()
        # 检查Git状态
        check_requirements(ROOT / "requirements.txt")
        # 检查所需的依赖项

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 选择设备，设置为指定的CUDA设备或CPU
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, "AutoBatch is coming soon for classification, please pass a valid --batch-size"
        # 确保批处理大小有效
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        # 确保批处理大小是全局进程数量的倍数
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        # 确保有足够的CUDA设备用于DDP命令
        torch.cuda.set_device(LOCAL_RANK)
        # 设置当前CUDA设备
        device = torch.device("cuda", LOCAL_RANK)
        # 将设备设置为当前CUDA设备
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        # 初始化分布式进程组，使用NCCL作为后端（如果可用），否则使用Gloo

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    # 设置保存目录，确保目录名称唯一

    # Train
    train(opt, device)
    # 调用训练函数


def run(**kwargs):
    """
    Executes YOLOv5 model training or inference with specified parameters, returning updated options.

    Example: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    """
    # 执行YOLOv5模型训练或推理，返回更新后的选项
    opt = parse_opt(True)
    # 解析命令行参数
    for k, v in kwargs.items():
        setattr(opt, k, v)
        # 将传入的关键字参数设置到选项中
    main(opt)
    # 调用主函数
    return opt
    # 返回解析后的选项


if __name__ == "__main__":
    opt = parse_opt()
    # 解析命令行参数
    main(opt)
    # 调用主函数
