# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse  # 用于解析命令行参数
import math  # 数学函数库
import os  # 操作系统相关功能
import random  # 随机数生成库
import subprocess  # 用于运行子进程
import sys  # 系统相关功能
import time  # 时间相关功能
from copy import deepcopy  # 深拷贝
from datetime import datetime, timedelta  # 日期和时间处理
from pathlib import Path  # 路径处理

try:
    import comet_ml  # must be imported before torch (if installed)  # 如果安装了 comet_ml，必须在 torch 之前导入
except ImportError:
    comet_ml = None  # 如果没有安装 comet_ml，则将其设置为 None

import numpy as np  # 数值计算库
import torch  # PyTorch 库
import torch.distributed as dist  # PyTorch 分布式训练
import torch.nn as nn  # PyTorch 神经网络模块
import yaml  # YAML 解析库
from torch.optim import lr_scheduler  # 学习率调度器
from tqdm import tqdm  # 进度条库

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH  # 将根目录添加到系统路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative  # 获取相对路径

import val as validate  # for end-of-epoch mAP  # 导入验证模块，用于计算每个 epoch 结束时的 mAP
from models.experimental import attempt_load  # 尝试加载模型
from models.yolo import Model  # 导入 YOLO 模型
from utils.autoanchor import check_anchors  # 检查锚框
from utils.autobatch import check_train_batch_size  # 检查训练批次大小
from utils.callbacks import Callbacks  # 导入回调函数
from utils.dataloaders import create_dataloader  # 创建数据加载器
from utils.downloads import attempt_download, is_url  # 尝试下载文件和检查 URL
from utils.general import (
    LOGGER,  # 日志记录器
    TQDM_BAR_FORMAT,  # TQDM 进度条格式
    check_amp,  # 检查自动混合精度
    check_dataset,  # 检查数据集
    check_file,  # 检查文件
    check_git_info,  # 检查 git 信息
    check_git_status,  # 检查 git 状态
    check_img_size,  # 检查图像大小
    check_requirements,  # 检查依赖项
    check_suffix,  # 检查文件后缀
    check_yaml,  # 检查 YAML 文件
    colorstr,  # 颜色字符串处理
    get_latest_run,  # 获取最新的运行记录
    increment_path,  # 增加路径
    init_seeds,  # 初始化随机种子
    intersect_dicts,  # 字典交集
    labels_to_class_weights,  # 标签转为类别权重
    labels_to_image_weights,  # 标签转为图像权重
    methods,  # 方法集合
    one_cycle,  # 单周期学习率调度
    print_args,  # 打印参数
    print_mutation,  # 打印变异信息
    strip_optimizer,  # 清理优化器
    yaml_save,  # 保存 YAML 文件
)
from utils.loggers import LOGGERS, Loggers  # 导入日志记录器
from utils.loggers.comet.comet_utils import check_comet_resume  # 检查 comet 恢复
from utils.loss import ComputeLoss  # 计算损失
from utils.metrics import fitness  # 计算适应度
from utils.plots import plot_evolve  # 绘制进化图
from utils.torch_utils import (
    EarlyStopping,  # 提前停止
    ModelEMA,  # 模型指数移动平均
    de_parallel,  # 去并行化
    select_device,  # 选择设备
    smart_DDP,  # 智能分布式数据并行
    smart_optimizer,  # 智能优化器
    smart_resume,  # 智能恢复
    torch_distributed_zero_first,  # PyTorch 分布式零首选
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # 获取本地排名
RANK = int(os.getenv("RANK", -1))  # 获取全局排名
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))  # 获取世界大小
GIT_INFO = check_git_info()  # 检查 git 信息


def train(hyp, opt, device, callbacks):
    """
    Trains YOLOv5 model with given hyperparameters, options, and device, managing datasets, model architecture, loss
    computation, and optimizer steps.

    `hyp` argument is path/to/hyp.yaml or hyp dictionary.
    """
    # 使用给定的超参数、选项和设备训练 YOLOv5 模型，管理数据集、模型架构、损失计算和优化器步骤
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),  # 保存目录
        opt.epochs,  # 训练的轮数
        opt.batch_size,  # 批次大小
        opt.weights,  # 权重文件
        opt.single_cls,  # 是否为单类检测
        opt.evolve,  # 是否进化
        opt.data,  # 数据集配置
        opt.cfg,  # 模型配置
        opt.resume,  # 是否从上次中断处恢复
        opt.noval,  # 是否不验证
        opt.nosave,  # 是否不保存
        opt.workers,  # 数据加载的工作线程数
        opt.freeze,  # 冻结层数
    )
    callbacks.run("on_pretrain_routine_start")  # 运行预训练例程开始的回调

    # Directories
    w = save_dir / "weights"  # weights dir  # 权重目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir  # 创建目录
    last, best = w / "last.pt", w / "best.pt"  # 定义最后和最佳权重文件的路径

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict  # 加载超参数字典
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))  # 记录超参数信息
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints  # 将超参数复制到选项中以便保存

    # Save run settings
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)  # 保存超参数到 YAML 文件
        yaml_save(save_dir / "opt.yaml", vars(opt))  # 保存选项到 YAML 文件

    # Loggers
    data_dict = None  # 数据字典初始化为 None
    if RANK in {-1, 0}:  # 如果是主进程
        include_loggers = list(LOGGERS)  # 包含的日志记录器列表
        if getattr(opt, "ndjson_console", False):  # 如果需要控制台日志
            include_loggers.append("ndjson_console")  # 添加控制台日志记录器
        if getattr(opt, "ndjson_file", False):  # 如果需要文件日志
            include_loggers.append("ndjson_file")  # 添加文件日志记录器

        loggers = Loggers(
            save_dir=save_dir,  # 日志保存目录
            weights=weights,  # 权重文件
            opt=opt,  # 选项
            hyp=hyp,  # 超参数
            logger=LOGGER,  # 日志记录器
            include=tuple(include_loggers),  # 包含的日志记录器
        )

        # Register actions
        for k in methods(loggers):  # 遍历日志记录器的方法
            callbacks.register_action(k, callback=getattr(loggers, k))  # 注册回调

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset  # 获取远程数据集字典
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size  # 恢复时的参数

    # Config
    plots = not evolve and not opt.noplots  # create plots  # 创建绘图
    cuda = device.type != "cpu"  # 检查是否使用 CUDA
    init_seeds(opt.seed + 1 + RANK, deterministic=True)  # 初始化随机种子
    with torch_distributed_zero_first(LOCAL_RANK):  # 在分布式训练中确保第一个进程为零
        data_dict = data_dict or check_dataset(data)  # check if None  # 检查数据集是否为 None
    train_path, val_path = data_dict["train"], data_dict["val"]  # 获取训练和验证路径
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes  # 类别数量
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names  # 类别名称
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset  # 检查是否为 COCO 数据集

    # Model
    check_suffix(weights, ".pt")  # check weights  # 检查权重文件后缀
    pretrained = weights.endswith(".pt")  # 检查权重是否为预训练模型
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):  # 在分布式训练中确保第一个进程为零
            weights = attempt_download(weights)  # download if not found locally  # 如果未找到权重，则尝试下载
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak  # 将检查点加载到 CPU 以避免 CUDA 内存泄漏
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create  # 创建模型
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys  # 排除的键
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32  # 检查点的状态字典转换为 FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect  # 交集
        model.load_state_dict(csd, strict=False)  # load  # 加载模型状态字典
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report  # 报告转移的参数数量
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create  # 创建模型
    amp = check_amp(model)  # check AMP  # 检查自动混合精度

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze  # 冻结层
    for k, v in model.named_parameters():  # 遍历模型的所有参数
        v.requires_grad = True  # train all layers  # 所有层都进行训练
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)  # 将 NaN 转为 0（已注释，因训练结果不稳定）
        if any(x in k for x in freeze):  # 如果参数名称包含冻结的层
            LOGGER.info(f"freezing {k}")  # 记录冻结的层
            v.requires_grad = False  # 冻结该层的参数

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)  # 网格大小（最大步幅）
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple  # 验证图像大小是否为 gs 的倍数

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size  # 仅单 GPU，估算最佳批次大小
        batch_size = check_train_batch_size(model, imgsz, amp)  # 检查训练批次大小
        loggers.on_params_update({"batch_size": batch_size})  # 更新日志记录器中的批次大小

    # Optimizer
    nbs = 64  # nominal batch size  # 标称批次大小
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing  # 在优化之前累积损失
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay  # 按照批次大小缩放权重衰减
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])  # 创建智能优化器

    # Scheduler
    if opt.cos_lr:  # 如果使用余弦学习率
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']  # 余弦调度
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear  # 线性调度
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)  # 创建学习率调度器

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None  # 如果是主进程，则创建模型 EMA

    # Resume
    best_fitness, start_epoch = 0.0, 0  # 初始化最佳适应度和起始轮数
    if pretrained:  # 如果是预训练模型
        if resume:  # 如果从中断处恢复
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)  # 恢复最佳适应度、起始轮数和总轮数
        del ckpt, csd  # 删除检查点和状态字典

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:  # 如果使用 CUDA 且是单 GPU
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )  # 警告：不推荐使用数据并行，建议使用 torch.distributed.run 进行最佳 DDP 多 GPU 结果
        model = torch.nn.DataParallel(model)  # 使用数据并行

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:  # 如果需要同步批归一化
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)  # 转换为同步批归一化并移动到设备
        LOGGER.info("Using SyncBatchNorm()")  # 记录使用同步批归一化

    # # Trainloader
    # train_loader, dataset = create_dataloader(
    #     train_path,
    #     imgsz,
    #     batch_size // WORLD_SIZE,
    #     gs,
    #     single_cls,
    #     hyp=hyp,
    #     augment=True,
    #     cache=None if opt.cache == "val" else opt.cache,
    #     rect=opt.rect,
    #     rank=LOCAL_RANK,
    #     workers=workers,
    #     image_weights=opt.image_weights,
    #     quad=opt.quad,
    #     prefix=colorstr("train: "),
    #     shuffle=True,
    #     seed=opt.seed,
    # )
    # labels = np.concatenate(dataset.labels, 0)
    # mlc = int(labels[:, 0].max())  # max label class
    # assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # # Process 0
    # if RANK in {-1, 0}:
    #     val_loader = create_dataloader(
    #         val_path,
    #         imgsz,
    #         batch_size // WORLD_SIZE * 2,
    #         gs,
    #         single_cls,
    #         hyp=hyp,
    #         cache=None if noval else opt.cache,
    #         rect=True,
    #         rank=-1,
    #         workers=workers * 2,
    #         pad=0.5,
    #         prefix=colorstr("val: "),
    #     )[0]

    #     if not resume:
    #         if not opt.noautoanchor:
    #             check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
    #         model.half().float()  # pre-reduce anchor precision

    #     callbacks.run("on_pretrain_routine_end", labels, names)

    # # DDP mode
    # if cuda and RANK != -1:
    #     model = smart_DDP(model)

    # # Model attributes
    # nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # hyp["box"] *= 3 / nl  # scale to layers
    # hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    # hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    # hyp["label_smoothing"] = opt.label_smoothing
    # model.nc = nc  # attach number of classes to model
    # model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # model.names = names

    # # Start training
    # t0 = time.time()
    # nb = len(train_loader)  # number of batches
    # nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # last_opt_step = -1
    # maps = np.zeros(nc)  # mAP per class
    # results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # scheduler.last_epoch = start_epoch - 1  # do not move
    # scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # stopper, stop = EarlyStopping(patience=opt.patience), False
    # compute_loss = ComputeLoss(model)  # init loss class
    # callbacks.run("on_train_start")
    # LOGGER.info(
    #     f'Image sizes {imgsz} train, {imgsz} val\n'
    #     f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
    #     f"Logging results to {colorstr('bold', save_dir)}\n"
    #     f'Starting training for {epochs} epochs...'
    # )
    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,  # 训练数据路径
        imgsz,  # 图像大小
        batch_size // WORLD_SIZE,  # 每个进程的批次大小
        gs,  # 网格大小
        single_cls,  # 是否为单类检测
        hyp=hyp,  # 超参数
        augment=True,  # 是否进行数据增强
        cache=None if opt.cache == "val" else opt.cache,  # 缓存设置
        rect=opt.rect,  # 是否使用矩形训练
        rank=LOCAL_RANK,  # 当前进程的排名
        workers=workers,  # 数据加载的工作线程数
        image_weights=opt.image_weights,  # 图像权重
        quad=opt.quad,  # 是否使用四元组
        prefix=colorstr("train: "),  # 前缀
        shuffle=True,  # 是否打乱数据
        seed=opt.seed,  # 随机种子
    )
    labels = np.concatenate(dataset.labels, 0)  # 合并所有标签
    mlc = int(labels[:, 0].max())  # max label class  # 最大标签类别
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"  # 确保标签类别不超过类别数量

    # Process 0
    if RANK in {-1, 0}:  # 如果是主进程
        val_loader = create_dataloader(
            val_path,  # 验证数据路径
            imgsz,  # 图像大小
            batch_size // WORLD_SIZE * 2,  # 验证批次大小
            gs,  # 网格大小
            single_cls,  # 是否为单类检测
            hyp=hyp,  # 超参数
            cache=None if noval else opt.cache,  # 缓存设置
            rect=True,  # 是否使用矩形训练
            rank=-1,  # 排名设置为 -1 表示主进程
            workers=workers * 2,  # 数据加载的工作线程数加倍
            pad=0.5,  # 填充设置
            prefix=colorstr("val: "),  # 前缀
        )[0]  # 获取数据加载器

        if not resume:  # 如果不从中断处恢复
            if not opt.noautoanchor:  # 如果不禁用自动锚框
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor  # 运行自动锚框检查
            model.half().float()  # pre-reduce anchor precision  # 预先减少锚框精度

        callbacks.run("on_pretrain_routine_end", labels, names)  # 运行预训练例程结束的回调

    # DDP mode
    if cuda and RANK != -1:  # 如果使用 CUDA 且不是主进程
        model = smart_DDP(model)  # 使用智能分布式数据并行

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)  # 检测层的数量（用于缩放超参数）
    hyp["box"] *= 3 / nl  # scale to layers  # 按照层数缩放框的超参数
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers  # 按照类别和层数缩放分类超参数
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers  # 按照图像大小和层数缩放目标超参数
    hyp["label_smoothing"] = opt.label_smoothing  # 设置标签平滑
    model.nc = nc  # attach number of classes to model  # 将类别数量附加到模型
    model.hyp = hyp  # attach hyperparameters to model  # 将超参数附加到模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights  # 将类别权重附加到模型
    model.names = names  # 将类别名称附加到模型

    # Start training
    t0 = time.time()  # 记录开始时间
    nb = len(train_loader)  # number of batches  # 批次数量
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)  # 最大热身迭代次数
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training  # 限制热身到小于一半的训练
    last_opt_step = -1  # 上一次优化步骤
    maps = np.zeros(nc)  # mAP per class  # 每个类别的 mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)  # 初始化结果
    scheduler.last_epoch = start_epoch - 1  # do not move  # 不移动
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # 初始化梯度缩放器
    stopper, stop = EarlyStopping(patience=opt.patience), False  # 初始化提前停止
    compute_loss = ComputeLoss(model)  # init loss class  # 初始化损失类
    callbacks.run("on_train_start")  # 运行训练开始的回调
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'  # 记录训练和验证图像大小
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'  # 记录数据加载器的工作线程数
        f"Logging results to {colorstr('bold', save_dir)}\n"  # 记录结果保存目录
        f'Starting training for {epochs} epochs...'  # 开始训练的轮数
    )

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")  # 运行每个训练轮次开始的回调
        model.train()  # 设置模型为训练模式

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:  # 如果启用了图像权重
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights  # 计算类别权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights  # 计算图像权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx  # 根据权重随机选择索引

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)  # 更新马赛克边界（可选）
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders  # 设置马赛克边界

        mloss = torch.zeros(3, device=device)  # mean losses  # 初始化平均损失
        if RANK != -1:  # 如果不是主进程
            train_loader.sampler.set_epoch(epoch)  # 设置采样器的轮次
        pbar = enumerate(train_loader)  # 获取数据加载器的枚举对象
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))  # 记录日志标题
        if RANK in {-1, 0}:  # 如果是主进程
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar  # 显示进度条
        optimizer.zero_grad()  # 清零优化器的梯度
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")  # 运行每个训练批次开始的回调
            ni = i + nb * epoch  # number integrated batches (since train start)  # 计算自训练开始以来的总批次数
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0  # 将图像转换为浮点数并归一化

            # Warmup
            if ni <= nw:  # 如果当前批次小于或等于热身批次
                xi = [0, nw]  # x interp  # 线性插值的 x 值
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)  # 计算 IOU 损失比例（已注释）
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())  # 在优化之前累积损失
                for j, x in enumerate(optimizer.param_groups):  # 遍历优化器的参数组
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])  # 更新学习率
                    if "momentum" in x:  # 如果参数组中包含动量
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])  # 更新动量

            # Multi-scale
            if opt.multi_scale:  # 如果启用了多尺度训练
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size  # 随机选择图像大小
                sf = sz / max(imgs.shape[2:])  # scale factor  # 计算缩放因子
                if sf != 1:  # 如果缩放因子不为 1
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)  # 计算新的形状
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)  # 进行插值调整图像大小

            # Forward
            with torch.cuda.amp.autocast(amp):  # 使用自动混合精度
                pred = model(imgs)  # forward  # 前向传播
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size  # 计算损失
                if RANK != -1:  # 如果不是主进程
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode  # 在 DDP 模式下，损失按设备数量平均
                if opt.quad:  # 如果启用了四元组
                    loss *= 4.0  # 将损失乘以 4

            # Backward
            scaler.scale(loss).backward()  # 反向传播

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:  # 如果达到优化步数
                scaler.unscale_(optimizer)  # unscale gradients  # 反缩放梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients  # 裁剪梯度
                scaler.step(optimizer)  # optimizer.step  # 更新优化器
                scaler.update()  # 更新缩放器
                optimizer.zero_grad()  # 清零优化器的梯度
                if ema:  # 如果使用 EMA
                    ema.update(model)  # 更新 EMA
                last_opt_step = ni  # 更新最后优化步数

            # Log
            if RANK in {-1, 0}:  # 如果是主进程
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses  # 更新平均损失
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)  # 获取 GPU 内存使用情况
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)  # 设置进度条描述
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])  # 显示当前轮次、内存使用情况和损失
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))  # 运行每个训练批次结束的回调
                if callbacks.stop_training:  # 如果停止训练
                    return  # 退出训练
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers  # 获取当前学习率
        scheduler.step()  # 更新学习率调度器

        if RANK in {-1, 0}:  # 如果是主进程
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)  # 运行每个训练轮次结束的回调
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])  # 更新 EMA 属性
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop  # 检查是否为最后一个轮次或可能停止
            if not noval or final_epoch:  # Calculate mAP  # 如果不验证或是最后一个轮次
                results, maps, _ = validate.run(
                    data_dict,  # 数据字典
                    batch_size=batch_size // WORLD_SIZE * 2,  # 验证批次大小
                    imgsz=imgsz,  # 图像大小
                    half=amp,  # 是否使用半精度
                    model=ema.ema,  # 使用 EMA 模型
                    single_cls=single_cls,  # 是否为单类检测
                    dataloader=val_loader,  # 验证数据加载器
                    save_dir=save_dir,  # 保存目录
                    plots=False,  # 是否绘制图形
                    callbacks=callbacks,  # 回调函数
                    compute_loss=compute_loss,  # 计算损失函数
                )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]  # 计算加权适应度
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check  # 检查是否需要提前停止
            if fi > best_fitness:  # 如果当前适应度优于最佳适应度
                best_fitness = fi  # 更新最佳适应度
            log_vals = list(mloss) + list(results) + lr  # 记录损失、结果和学习率
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)  # 运行每个适应度轮次结束的回调

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save  # 如果需要保存模型
                ckpt = {
                    "epoch": epoch,  # 当前轮次
                    "best_fitness": best_fitness,  # 最佳适应度
                    "model": deepcopy(de_parallel(model)).half(),  # 模型副本，转换为半精度
                    "ema": deepcopy(ema.ema).half(),  # EMA 模型副本，转换为半精度
                    "updates": ema.updates,  # EMA 更新次数
                    "optimizer": optimizer.state_dict(),  # 优化器状态字典
                    "opt": vars(opt),  # 选项
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo  # git 信息
                    "date": datetime.now().isoformat(),  # 当前日期时间
                }

                # Save last, best and delete
                torch.save(ckpt, last)  # 保存最后的检查点
                if best_fitness == fi:  # 如果当前适应度是最佳适应度
                    torch.save(ckpt, best)  # 保存最佳检查点
                if opt.save_period > 0 and epoch % opt.save_period == 0:  # 根据保存周期保存
                    torch.save(ckpt, w / f"epoch{epoch}.pt")  # 保存当前轮次的检查点
                del ckpt  # 删除检查点
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)  # 运行模型保存的回调

        # EarlyStopping
        if RANK != -1:  # if DDP training  # 如果是 DDP 训练
            broadcast_list = [stop if RANK == 0 else None]  # 广播停止标志
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks  # 广播停止标志到所有进程
            if RANK != 0:  # 如果不是主进程
                stop = broadcast_list[0]  # 更新停止标志
        if stop:  # 如果需要停止
            break  # must break all DDP ranks  # 退出所有 DDP 进程


        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:  # 如果是主进程
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")  # 记录完成的轮次和耗时
        for f in last, best:  # 遍历最后和最佳检查点
            if f.exists():  # 如果文件存在
                strip_optimizer(f)  # strip optimizers  # 清理优化器
                if f is best:  # 如果是最佳检查点
                    LOGGER.info(f"\nValidating {f}...")  # 记录正在验证的最佳检查点
                    results, _, _ = validate.run(  # 验证最佳模型
                        data_dict,  # 数据字典
                        batch_size=batch_size // WORLD_SIZE * 2,  # 验证批次大小
                        imgsz=imgsz,  # 图像大小
                        model=attempt_load(f, device).half(),  # 加载最佳模型并转换为半精度
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65  # IOU 阈值
                        single_cls=single_cls,  # 是否为单类检测
                        dataloader=val_loader,  # 验证数据加载器
                        save_dir=save_dir,  # 保存目录
                        save_json=is_coco,  # 是否保存为 JSON（COCO 数据集）
                        verbose=True,  # 是否详细输出
                        plots=plots,  # 是否绘制图形
                        callbacks=callbacks,  # 回调函数
                        compute_loss=compute_loss,  # 计算损失函数
                    )  # val best model with plots  # 验证最佳模型并绘制图形
                    if is_coco:  # 如果是 COCO 数据集
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)  # 运行适应度轮次结束的回调

        callbacks.run("on_train_end", last, best, epoch, results)  # 运行训练结束的回调

    torch.cuda.empty_cache()  # 清空 CUDA 缓存
    return results  # 返回结果


def parse_opt(known=False):  # 定义解析命令行参数的函数，参数 known 表示是否解析已知参数
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""  # 解析 YOLOv5 训练、验证和测试的命令行参数
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")  # 初始权重路径
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")  # 模型配置文件路径
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")  # 数据集配置文件路径
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")  # 超参数文件路径
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")  # 总训练轮次
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")  # 所有 GPU 的总批次大小，-1 表示自动批次
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")  # 训练和验证的图像大小（像素）
    parser.add_argument("--rect", action="store_true", help="rectangular training")  # 矩形训练
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")  # 恢复最近的训练
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")  # 只保存最终检查点
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")  # 只验证最后一个轮次
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")  # 禁用自动锚框
    parser.add_argument("--noplots", action="store_true", help="save no plot files")  # 不保存绘图文件
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")  # 进化超参数的代数
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"  # 加载种群的位置
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")  # 从最后一代恢复进化
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")  # gsutil 存储桶
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")  # 图像缓存选项
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")  # 使用加权图像选择进行训练
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  # CUDA 设备，例如 0 或 0,1,2,3 或 cpu
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")  # 图像大小变化 +/- 50%
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")  # 将多类数据作为单类进行训练
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")  # 优化器选择
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")  # 使用同步批量归一化，仅在 DDP 模式下可用
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")  # 数据加载器的最大工作线程数（每个 DDP 进程）
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")  # 保存到项目名称
    parser.add_argument("--name", default="exp", help="save to project/name")  # 保存到项目名称
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")  # 允许存在的项目/名称，不递增
    parser.add_argument("--quad", action="store_true", help="quad dataloader")  # 使用四元组数据加载器
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")  # 使用余弦学习率调度器
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")  # 标签平滑的 epsilon 值
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")  # 提前停止的耐心值（没有改进的轮次）
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")  # 冻结层：主干=10，前3层=0 1 2
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")  # 每 x 轮保存检查点（如果 < 1 则禁用）
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")  # 全局训练种子
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")  # 自动 DDP 多 GPU 参数，不要修改

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")  # 实体
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')  # 上传数据，"val" 选项
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")  # 设置边界框图像日志记录间隔
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")  # 使用的数据集工件版本

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")  # 将 ndjson 日志记录到控制台
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")  # 将 ndjson 日志记录到文件

    return parser.parse_known_args()[0] if known else parser.parse_args()  # 返回解析的参数，如果已知则使用已知参数，否则解析所有参数


def main(opt, callbacks=Callbacks()):  # 定义主函数，接收选项和回调函数
    """Runs training or hyperparameter evolution with specified options and optional callbacks."""  # 使用指定选项和可选回调运行训练或超参数进化
    if RANK in {-1, 0}:  # 如果是主进程
        print_args(vars(opt))  # 打印选项参数
        check_git_status()  # 检查 Git 状态
        check_requirements(ROOT / "requirements.txt")  # 检查依赖项

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:  # 如果选择恢复训练且不使用 Comet 恢复且不在进化模式
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())  # 获取最近的检查点路径
        opt_yaml = last.parent.parent / "opt.yaml"  # 训练选项的 YAML 文件路径
        opt_data = opt.data  # 原始数据集
        if opt_yaml.is_file():  # 如果选项 YAML 文件存在
            with open(opt_yaml, errors="ignore") as f:  # 打开 YAML 文件
                d = yaml.safe_load(f)  # 加载 YAML 文件内容
        else:  # 如果 YAML 文件不存在
            d = torch.load(last, map_location="cpu")["opt"]  # 从最近的检查点加载选项
        opt = argparse.Namespace(**d)  # 替换当前选项
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # 重新设置配置文件、权重和恢复标志
        if is_url(opt_data):  # 如果数据集是 URL
            opt.data = check_file(opt_data)  # 检查文件以避免 HUB 恢复身份验证超时
    else:  # 如果不恢复训练
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (  # 检查数据集、配置文件、超参数、权重和项目路径
            check_file(opt.data),  # 检查数据集文件
            check_yaml(opt.cfg),  # 检查配置文件
            check_yaml(opt.hyp),  # 检查超参数文件
            str(opt.weights),  # 转换权重为字符串
            str(opt.project),  # 转换项目路径为字符串
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"  # 确保提供了配置文件或权重
        if opt.evolve:  # 如果在进化模式
            if opt.project == str(ROOT / "runs/train"):  # 如果项目名称为默认值
                opt.project = str(ROOT / "runs/evolve")  # 将项目名称更改为 runs/evolve
            opt.exist_ok, opt.resume = opt.resume, False  # 将恢复标志传递给 exist_ok，并禁用恢复
        if opt.name == "cfg":  # 如果名称为 "cfg"
            opt.name = Path(opt.cfg).stem  # 使用模型配置文件的名称作为名称
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 生成保存目录路径
    
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)  # 选择设备（CUDA或CPU），根据用户指定的设备和批次大小
    if LOCAL_RANK != -1:  # 如果在分布式数据并行（DDP）模式下
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"  # 错误消息
        assert not opt.image_weights, f"--image-weights {msg}"  # 确保不使用图像权重
        assert not opt.evolve, f"--evolve {msg}"  # 确保不在进化模式
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"  # 确保批次大小有效
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"  # 确保批次大小是 WORLD_SIZE 的倍数
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"  # 确保有足够的 CUDA 设备
        torch.cuda.set_device(LOCAL_RANK)  # 设置当前 CUDA 设备
        device = torch.device("cuda", LOCAL_RANK)  # 创建 CUDA 设备对象
        dist.init_process_group(  # 初始化进程组
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)  # 使用 NCCL 作为后端（如果可用），否则使用 Gloo
        )

    # Train
    if not opt.evolve:  # 如果不在进化模式
        train(opt.hyp, opt, device, callbacks)  # 调用训练函数，传入超参数、选项、设备和回调

    # Evolve hyperparameters (optional)
    else:  # 如果在进化模式
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        meta = {  # 超参数进化元数据，包括每个超参数的状态和范围
            "lr0": (False, 1e-5, 1e-1),  # 初始学习率 (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # 最终 OneCycleLR 学习率 (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD 动量/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # 优化器权重衰减
            "warmup_epochs": (False, 0.0, 5.0),  # 热身轮次 (可以是小数)
            "warmup_momentum": (False, 0.0, 0.95),  # 热身初始动量
            "warmup_bias_lr": (False, 0.0, 0.2),  # 热身初始偏置学习率
            "box": (False, 0.02, 0.2),  # 边界框损失增益
            "cls": (False, 0.2, 4.0),  # 类别损失增益
            "cls_pw": (False, 0.5, 2.0),  # 类别 BCELoss 正权重
            "obj": (False, 0.2, 4.0),  # 目标损失增益 (与像素规模)
            "obj_pw": (False, 0.5, 2.0),  # 目标 BCELoss 正权重
            "iou_t": (False, 0.1, 0.7),  # IoU 训练阈值
            "anchor_t": (False, 2.0, 8.0),  # 锚框倍数阈值
            "anchors": (False, 2.0, 10.0),  # 每个输出网格的锚框数量 (0 表示忽略)
            "fl_gamma": (False, 0.0, 2.0),  # 焦点损失的 gamma 值 (efficientDet 默认 gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # 图像 HSV-色调增强 (百分比)
            "hsv_s": (True, 0.0, 0.9),  # 图像 HSV-饱和度增强 (百分比)
            "hsv_v": (True, 0.0, 0.9),  # 图像 HSV-值增强 (百分比)
            "degrees": (True, 0.0, 45.0),  # 图像旋转 (+/- 度)
            "translate": (True, 0.0, 0.9),  # 图像平移 (+/- 百分比)
            "scale": (True, 0.0, 0.9),  # 图像缩放 (+/- 增益)
            "shear": (True, 0.0, 10.0),  # 图像剪切 (+/- 度)
            "perspective": (True, 0.0, 0.001),  # 图像透视 (+/- 百分比)，范围 0-0.001
            "flipud": (True, 0.0, 1.0),  # 图像上下翻转 (概率)
            "fliplr": (True, 0.0, 1.0),  # 图像左右翻转 (概率)
            "mosaic": (True, 0.0, 1.0),  # 图像混合 (概率)
            "mixup": (True, 0.0, 1.0),  # 图像混合 (概率)
            "copy_paste": (True, 0.0, 1.0),  # 复制粘贴 (概率)
        }  # 段复制粘贴 (概率)

        # GA configs
        pop_size = 50  # 种群大小
        mutation_rate_min = 0.01  # 最小变异率
        mutation_rate_max = 0.5  # 最大变异率
        crossover_rate_min = 0.5  # 最小交叉率
        crossover_rate_max = 1  # 最大交叉率
        min_elite_size = 2  # 最小精英大小
        max_elite_size = 5  # 最大精英大小
        tournament_size_min = 2  # 最小锦标赛大小
        tournament_size_max = 10  # 最大锦标赛大小

        with open(opt.hyp, errors="ignore") as f:  # 打开超参数文件
            hyp = yaml.safe_load(f)  # 加载超参数字典
            if "anchors" not in hyp:  # 如果超参数中没有锚框
                hyp["anchors"] = 3  # 设置默认锚框数量为 3
        if opt.noautoanchor:  # 如果禁用自动锚框
            del hyp["anchors"], meta["anchors"]  # 删除锚框设置
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # 只在最后一轮进行验证/保存
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # 可进化的索引
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"  # 设置进化 YAML 和 CSV 文件路径
        if opt.bucket:  # 如果指定了存储桶
            # download evolve.csv if exists
            subprocess.run(  # 下载 evolve.csv 文件（如果存在）
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",  # 从指定的存储桶复制文件
                    str(evolve_csv),  # 目标路径
                ]
            )

        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]  # 从 meta 字典中删除第一个值为 False 的项
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary  # 复制超参数字典

        for item in del_:  # 遍历需要删除的项
            del meta[item]  # Remove the item from meta dictionary  # 从 meta 字典中删除该项
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary  # 从 hyp_GA 字典中删除该项

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])  # 创建下限数组，存储搜索空间的下限
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])  # 创建上限数组，存储搜索空间的上限

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]  # 创建基因范围列表，存储每个基因的值范围

        # Initialize the population with initial_values or random values
        initial_values = []  # 初始化初始值列表

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:  # 如果指定了恢复进化的路径
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"  # 确保恢复路径是有效的文件
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:  # 打开恢复进化文件
                evolve_population = yaml.safe_load(f)  # 加载进化种群
                for value in evolve_population.values():  # 遍历进化种群的值
                    value = np.array([value[k] for k in hyp_GA.keys()])  # 将值转换为 NumPy 数组
                    initial_values.append(list(value))  # 将初始值添加到列表中

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:  # 如果不从之前的检查点恢复
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]  # 获取所有以 .yaml 结尾的文件
            for file_name in yaml_files:  # 遍历 YAML 文件
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:  # 打开 YAML 文件
                    value = yaml.safe_load(yaml_file)  # 加载 YAML 文件内容
                    value = np.array([value[k] for k in hyp_GA.keys()])  # 将值转换为 NumPy 数组
                    initial_values.append(list(value))  # 将初始值添加到列表中

        # Generate random values within the search space for the rest of the population
        if initial_values is None:  # 如果没有初始值
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]  # 在搜索空间内生成随机值，初始化种群
        elif pop_size > 1:  # 如果种群大小大于 1
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]  # 生成剩余的随机个体
            for initial_value in initial_values:  # 遍历初始值
                population = [initial_value] + population  # 将初始值添加到种群中



        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())  # 获取超参数字典的所有键
        for generation in range(opt.evolve):  # 遍历指定的进化代数
            if generation >= 1:  # 如果当前代数大于或等于 1
                save_dict = {}  # 初始化保存字典
                for i in range(len(population)):  # 遍历当前种群
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}  # 创建小字典，存储个体的超参数
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict  # 将小字典添加到保存字典中，以代数和个体编号为键

                with open(save_dir / "evolve_population.yaml", "w") as outfile:  # 打开文件以写入进化种群
                    yaml.dump(save_dict, outfile, default_flow_style=False)  # 将保存字典写入 YAML 文件

            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))  # 自适应精英大小

            # Evaluate the fitness of each individual in the population
            fitness_scores = []  # 初始化适应度分数列表
            for individual in population:  # 遍历当前种群中的每个个体
                for key, value in zip(hyp_GA.keys(), individual):  # 遍历超参数和个体的值
                    hyp_GA[key] = value  # 更新超参数字典
                hyp.update(hyp_GA)  # 更新超参数
                results = train(hyp.copy(), opt, device, callbacks)  # 调用训练函数，传入超参数、选项、设备和回调
                callbacks = Callbacks()  # 重置回调
                # Write mutation results
                keys = (  # 定义要记录的指标
                    "metrics/precision",  # 精确度
                    "metrics/recall",  # 召回率
                    "metrics/mAP_0.5",  # mAP @ 0.5
                    "metrics/mAP_0.5:0.95",  # mAP @ 0.5:0.95
                    "val/box_loss",  # 边界框损失
                    "val/obj_loss",  # 目标损失
                    "val/cls_loss",  # 类别损失
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)  # 打印变异结果
                fitness_scores.append(results[2])  # 将适应度分数添加到列表中

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []  # 初始化选择的索引列表
            for _ in range(pop_size - elite_size):  # 选择适应度最好的个体进行繁殖
                # Adaptive tournament size
                tournament_size = max(  # 自适应锦标赛大小
                    max(2, tournament_size_min),  # 至少为 2
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),  # 根据代数动态调整
                )
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)  # 随机选择锦标赛中的个体
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]  # 获取这些个体的适应度分数
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]  # 选择适应度最高的个体
                selected_indices.append(winner_index)  # 将赢家的索引添加到选择列表中

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]  # 获取适应度最高的精英个体索引
            selected_indices.extend(elite_indices)  # 将精英个体的索引添加到选择列表中

            # Create the next generation through crossover and mutation
            next_generation = []  # 初始化下一代个体列表
            for _ in range(pop_size):  # 为每个个体生成下一代
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]  # 随机选择父母 1
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]  # 随机选择父母 2
                # Adaptive crossover rate
                crossover_rate = max(  # 自适应交叉率
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))  # 根据代数动态调整
                )
                if random.uniform(0, 1) < crossover_rate:  # 根据交叉率决定是否进行交叉
                    crossover_point = random.randint(1, len(hyp_GA) - 1)  # 随机选择交叉点
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]  # 生成子代
                else:
                    child = population[parent1_index]  # 如果不交叉，则直接复制父母 1

                # Adaptive mutation rate
                mutation_rate = max(  # 自适应变异率
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))  # 根据代数动态调整
                )
                for j in range(len(hyp_GA)):  # 遍历每个超参数
                    if random.uniform(0, 1) < mutation_rate:  # 根据变异率决定是否进行变异
                        child[j] += random.uniform(-0.1, 0.1)  # 随机调整超参数
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])  # 确保超参数在范围内
                next_generation.append(child)  # 将子代添加到下一代列表

            # Replace the old population with the new generation
            population = next_generation  # 用新一代替换旧的种群

        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))  # 找到适应度最高的个体索引
        best_individual = population[best_index]  # 获取最佳个体
        print("Best solution found:", best_individual)  # 打印找到的最佳解决方案

        # Plot results
        plot_evolve(evolve_csv)  # 绘制进化结果
        LOGGER.info(  # 记录进化完成的信息
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'  # 使用示例
        )


def generate_individual(input_ranges, individual_length):  # 定义生成个体的函数，接收输入范围和个体长度
    """Generates a list of random values within specified input ranges for each gene in the individual."""  # 在指定的输入范围内为个体的每个基因生成随机值
    individual = []  # 初始化个体列表
    for i in range(individual_length):  # 遍历每个基因
        lower_bound, upper_bound = input_ranges[i]  # 获取当前基因的下限和上限
        individual.append(random.uniform(lower_bound, upper_bound))  # 在范围内生成随机值并添加到个体列表
    return individual  # 返回生成的个体

def run(**kwargs):  # 定义运行函数，接收任意关键字参数
    """
    Executes YOLOv5 training with given options, overriding with any kwargs provided.

    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """  # 执行 YOLOv5 训练，使用给定选项并覆盖提供的任何关键字参数
    opt = parse_opt(True)  # 解析命令行选项，传入 True 表示解析已知参数
    for k, v in kwargs.items():  # 遍历关键字参数
        setattr(opt, k, v)  # 将关键字参数设置到选项对象中
    main(opt)  # 调用主函数，传入选项
    return opt  # 返回选项对象

if __name__ == "__main__":  # 如果是主程序
    opt = parse_opt()  # 解析命令行选项
    main(opt)  # 调用主函数，传入选项