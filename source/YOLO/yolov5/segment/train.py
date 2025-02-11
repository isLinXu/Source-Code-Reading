# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 segment model on a segment dataset Models and datasets download automatically from the latest YOLOv5
release.

Usage - Single-GPU training:
    $ python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640  # from pretrained (recommended)
    $ python segment/train.py --data coco128-seg.yaml --weights '' --cfg yolov5s-seg.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse  # 导入argparse库，用于解析命令行参数
import math  # 导入math库，提供数学函数
import os  # 导入os库，提供与操作系统交互的功能
import random  # 导入random库，用于生成随机数
import subprocess  # 导入subprocess库，用于运行子进程
import sys  # 导入sys库，提供对Python解释器的访问
import time  # 导入time库，提供时间相关的功能
from copy import deepcopy  # 从copy库导入deepcopy，用于深拷贝对象
from datetime import datetime  # 从datetime库导入datetime，用于处理日期和时间
from pathlib import Path  # 从pathlib库导入Path，用于处理文件路径

import numpy as np  # 导入numpy库，支持大规模的多维数组和矩阵运算
import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入torch的分布式模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
import yaml  # 导入yaml库，用于处理YAML文件
from torch.optim import lr_scheduler  # 从torch.optim导入学习率调度器
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[1]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH 将根目录添加到系统路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 计算相对路径

import segment.val as validate  # for end-of-epoch mAP 导入验证模块，用于计算每个epoch结束时的mAP
from models.experimental import attempt_load  # 从experimental模块导入attempt_load函数
from models.yolo import SegmentationModel  # 从yolo模块导入SegmentationModel类
from utils.autoanchor import check_anchors  # 从autoanchor模块导入check_anchors函数
from utils.autobatch import check_train_batch_size  # 从autobatch模块导入check_train_batch_size函数
from utils.callbacks import Callbacks  # 从callbacks模块导入Callbacks类
from utils.downloads import attempt_download, is_url  # 从downloads模块导入attempt_download和is_url函数
from utils.general import (  # 导入general模块中的多个函数和变量
    LOGGER,  # 日志记录器
    TQDM_BAR_FORMAT,  # tqdm进度条格式
    check_amp,  # 检查自动混合精度
    check_dataset,  # 检查数据集
    check_file,  # 检查文件
    check_git_info,  # 检查Git信息
    check_git_status,  # 检查Git状态
    check_img_size,  # 检查图像大小
    check_requirements,  # 检查依赖项
    check_suffix,  # 检查文件后缀
    check_yaml,  # 检查YAML文件
    colorstr,  # 颜色字符串
    get_latest_run,  # 获取最新的运行
    increment_path,  # 增加路径
    init_seeds,  # 初始化随机种子
    intersect_dicts,  # 交集字典
    labels_to_class_weights,  # 标签转类权重
    labels_to_image_weights,  # 标签转图像权重
    one_cycle,  # 一周期学习率调度
    print_args,  # 打印参数
    print_mutation,  # 打印突变
    strip_optimizer,  # 清理优化器
    yaml_save,  # 保存YAML文件
)
from utils.loggers import GenericLogger  # 从loggers模块导入GenericLogger类
from utils.plots import plot_evolve, plot_labels  # 从plots模块导入plot_evolve和plot_labels函数
from utils.segment.dataloaders import create_dataloader  # 从dataloaders模块导入create_dataloader函数
from utils.segment.loss import ComputeLoss  # 从loss模块导入ComputeLoss类
from utils.segment.metrics import KEYS, fitness  # 从metrics模块导入KEYS和fitness函数
from utils.segment.plots import plot_images_and_masks, plot_results_with_masks  # 从plots模块导入绘图函数
from utils.torch_utils import (  # 导入torch_utils模块中的多个函数和类
    EarlyStopping,  # 提前停止
    ModelEMA,  # 模型指数移动平均
    de_parallel,  # 反向并行
    select_device,  # 选择设备
    smart_DDP,  # 智能DDP
    smart_optimizer,  # 智能优化器
    smart_resume,  # 智能恢复
    torch_distributed_zero_first,  # PyTorch分布式零优先
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # 获取本地进程的排名
RANK = int(os.getenv("RANK", -1))  # 获取全局进程的排名
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))  # 获取全局进程的总数
GIT_INFO = check_git_info()  # 检查Git信息


def train(hyp, opt, device, callbacks):
    """
    Trains the YOLOv5 model on a dataset, managing hyperparameters, model optimization, logging, and validation.
    训练YOLOv5模型，管理超参数、模型优化、日志记录和验证。

    `hyp` is path/to/hyp.yaml or hyp dictionary.
    `hyp`是路径/to/hyp.yaml或超参数字典。
    """
    (
        save_dir,  # 保存目录
        epochs,  # 训练轮数
        batch_size,  # 批次大小
        weights,  # 权重文件路径
        single_cls,  # 是否单类训练
        evolve,  # 是否进化训练
        data,  # 数据集路径
        cfg,  # 配置文件路径
        resume,  # 是否恢复训练
        noval,  # 是否不进行验证
        nosave,  # 是否不保存模型
        workers,  # 数据加载工作线程数
        freeze,  # 冻结层数
        mask_ratio,  # 掩码比例
    ) = (
        Path(opt.save_dir),  # 保存目录路径
        opt.epochs,  # 训练轮数
        opt.batch_size,  # 批次大小
        opt.weights,  # 权重文件路径
        opt.single_cls,  # 是否单类训练
        opt.evolve,  # 是否进化训练
        opt.data,  # 数据集路径
        opt.cfg,  # 配置文件路径
        opt.resume,  # 是否恢复训练
        opt.noval,  # 是否不进行验证
        opt.nosave,  # 是否不保存模型
        opt.workers,  # 数据加载工作线程数
        opt.freeze,  # 冻结层数
        opt.mask_ratio,  # 掩码比例
    )
    # callbacks.run('on_pretrain_routine_start')  # 在预训练例程开始时运行回调

    # Directories
    w = save_dir / "weights"  # 权重目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # 创建目录
    last, best = w / "last.pt", w / "best.pt"  # 定义最后和最佳权重文件路径

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # 加载超参数字典
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))  # 打印超参数信息
    opt.hyp = hyp.copy()  # 复制超参数以便保存到检查点

    # Save run settings
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)  # 保存超参数到YAML文件
        yaml_save(save_dir / "opt.yaml", vars(opt))  # 保存选项到YAML文件

    # Loggers
    data_dict = None  # 初始化数据字典
    if RANK in {-1, 0}:  # 如果是主进程
        logger = GenericLogger(opt=opt, console_logger=LOGGER)  # 创建日志记录器

    # Config
    plots = not evolve and not opt.noplots  # 是否创建绘图
    overlap = not opt.no_overlap  # 是否允许重叠
    cuda = device.type != "cpu"  # 检查是否使用CUDA
    init_seeds(opt.seed + 1 + RANK, deterministic=True)  # 初始化随机种子
    with torch_distributed_zero_first(LOCAL_RANK):  # 在分布式环境中进行零优先
        data_dict = data_dict or check_dataset(data)  # 检查数据集是否为空
    train_path, val_path = data_dict["train"], data_dict["val"]  # 获取训练和验证路径
    nc = 1 if single_cls else int(data_dict["nc"])  # 类别数量
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # 类别名称
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # 检查是否为COCO数据集

    # Model
    check_suffix(weights, ".pt")  # check weights 检查权重文件后缀是否为.pt
    pretrained = weights.endswith(".pt")  # 判断权重文件是否为预训练模型

    if pretrained:  # 如果是预训练模型
        with torch_distributed_zero_first(LOCAL_RANK):  # 在分布式环境中进行零优先
            weights = attempt_download(weights)  # download if not found locally 如果本地未找到，则下载权重文件
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak 将检查点加载到CPU以避免CUDA内存泄漏
        model = SegmentationModel(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # 创建分割模型
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys 排除的键
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32 将检查点的状态字典转换为FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect 交集
        model.load_state_dict(csd, strict=False)  # load 加载状态字典
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report 报告从权重文件转移的项数
    else:  # 如果不是预训练模型
        model = SegmentationModel(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create 创建模型
    amp = check_amp(model)  # check AMP 检查自动混合精度

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze 要冻结的层
    for k, v in model.named_parameters():  # 遍历模型的命名参数
        v.requires_grad = True  # train all layers 训练所有层
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results) 将NaN转换为0（注释掉以避免训练结果不稳定）
        if any(x in k for x in freeze):  # 如果参数名中包含要冻结的层
            LOGGER.info(f"freezing {k}")  # 记录冻结的层
            v.requires_grad = False  # 冻结该层

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride) 网格大小（最大步幅）
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple 验证图像大小是否为步幅的倍数

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size 仅单GPU时，估算最佳批次大小
        batch_size = check_train_batch_size(model, imgsz, amp)  # 检查训练批次大小
        logger.update_params({"batch_size": batch_size})  # 更新日志记录器中的批次大小
        # loggers.on_params_update({"batch_size": batch_size})  # 更新日志记录器的参数（注释掉）

    # Optimizer
    nbs = 64  # nominal batch size 名义批次大小
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing 在优化之前累积损失
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay 缩放权重衰减
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])  # 创建优化器

    # Scheduler
    if opt.cos_lr:  # 如果使用余弦学习率调度
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf'] 余弦调度
    else:  # 否则使用线性调度
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear 线性调度
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 创建学习率调度器

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None  # 如果是主进程，创建EMA模型

    # Resume
    best_fitness, start_epoch = 0.0, 0  # 初始化最佳适应度和起始轮数
    if pretrained:  # 如果是预训练模型
        if resume:  # 如果需要恢复训练
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)  # 恢复训练
        del ckpt, csd  # 删除检查点和状态字典

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:  # 如果使用CUDA且是单进程模式，并且有多个GPU
        LOGGER.warning(  # 记录警告信息
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)  # 使用DataParallel进行模型并行

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:  # 如果启用同步批量归一化且使用CUDA，并且不是主进程
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)  # 将模型转换为同步批量归一化并移动到设备上
        LOGGER.info("Using SyncBatchNorm()")  # 记录使用同步批量归一化的信息

    # Trainloader
    train_loader, dataset = create_dataloader(  # 创建数据加载器和数据集
        train_path,  # 训练数据路径
        imgsz,  # 图像大小
        batch_size // WORLD_SIZE,  # 每个进程的批次大小
        gs,  # 网格大小
        single_cls,  # 是否单类训练
        hyp=hyp,  # 超参数
        augment=True,  # 启用数据增强
        cache=None if opt.cache == "val" else opt.cache,  # 根据选项设置缓存
        rect=opt.rect,  # 是否使用矩形训练
        rank=LOCAL_RANK,  # 本地进程排名
        workers=workers,  # 数据加载工作线程数
        image_weights=opt.image_weights,  # 图像权重
        quad=opt.quad,  # 四分之一训练
        prefix=colorstr("train: "),  # 前缀
        shuffle=True,  # 是否打乱数据
        mask_downsample_ratio=mask_ratio,  # 掩码下采样比例
        overlap_mask=overlap,  # 重叠掩码
    )
    labels = np.concatenate(dataset.labels, 0)  # 将数据集中的标签连接成一个数组
    mlc = int(labels[:, 0].max())  # max label class 获取最大标签类别
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"  # 确保最大标签类别不超过类别数量
    
    # Process 0
    if RANK in {-1, 0}:  # 如果当前进程是主进程
        val_loader = create_dataloader(  # 创建验证数据加载器
            val_path,  # 验证数据路径
            imgsz,  # 图像大小
            batch_size // WORLD_SIZE * 2,  # 每个进程的批次大小，乘以2
            gs,  # 网格大小
            single_cls,  # 是否单类训练
            hyp=hyp,  # 超参数
            cache=None if noval else opt.cache,  # 根据选项设置缓存
            rect=True,  # 是否使用矩形训练
            rank=-1,  # 本地进程排名
            workers=workers * 2,  # 数据加载工作线程数，乘以2
            pad=0.5,  # 填充比例
            mask_downsample_ratio=mask_ratio,  # 掩码下采样比例
            overlap_mask=overlap,  # 重叠掩码
            prefix=colorstr("val: "),  # 前缀
        )[0]  # 只取返回的第一个值

        if not resume:  # 如果不需要恢复训练
            if not opt.noautoanchor:  # 如果没有禁用自动锚点
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor 运行自动锚点检查
            model.half().float()  # pre-reduce anchor precision 预先减少锚点精度

            if plots:  # 如果需要绘图
                plot_labels(labels, names, save_dir)  # 绘制标签并保存
            # callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:  # 如果使用CUDA并且不是主进程
        model = smart_DDP(model)  # 使用智能分布式数据并行模型

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps) 检测层的数量（用于缩放超参数）
    hyp["box"] *= 3 / nl  # scale to layers 缩放到层数
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers 缩放到类别和层数
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers 缩放到图像大小和层数
    hyp["label_smoothing"] = opt.label_smoothing  # 设置标签平滑
    model.nc = nc  # attach number of classes to model 将类别数量附加到模型
    model.hyp = hyp  # attach hyperparameters to model 将超参数附加到模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights 将类别权重附加到模型
    model.names = names  # 附加类别名称

    # Start training
    t0 = time.time()  # 记录开始时间
    nb = len(train_loader)  # number of batches 获取批次数量
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations) 预热迭代次数，最多100次
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training  限制预热次数小于训练的一半
    last_opt_step = -1  # 最后优化步骤
    maps = np.zeros(nc)  # mAP per class 每个类别的mAP
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls) 初始化结果
    scheduler.last_epoch = start_epoch - 1  # do not move 不移动
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # 初始化梯度缩放器
    stopper, stop = EarlyStopping(patience=opt.patience), False  # 初始化提前停止
    compute_loss = ComputeLoss(model, overlap=overlap)  # init loss class 初始化损失类
    # callbacks.run('on_train_start')  # 训练开始时运行回调
    LOGGER.info(  # 记录训练信息
        f'Image sizes {imgsz} train, {imgsz} val\n'  # 训练和验证图像大小
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'  # 使用的数据加载器工作线程数
        f"Logging results to {colorstr('bold', save_dir)}\n"  # 记录结果到指定目录
        f'Starting training for {epochs} epochs...'  # 开始训练指定轮数
    )

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------ 遍历每个训练轮次
        # callbacks.run('on_train_epoch_start')  # 在每个训练轮次开始时运行回调
        model.train()  # 设置模型为训练模式

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:  # 如果启用图像权重
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights 计算类别权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights 计算图像权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx 随机选择加权索引

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)  # 更新马赛克边界（可选）
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders 设置马赛克边界

        mloss = torch.zeros(4, device=device)  # mean losses 初始化平均损失
        if RANK != -1:  # 如果不是主进程
            train_loader.sampler.set_epoch(epoch)  # 设置训练数据加载器的epoch
        pbar = enumerate(train_loader)  # 遍历训练数据加载器
        LOGGER.info(  # 记录训练信息
            ("\n" + "%11s" * 8)  # 格式化输出
            % ("Epoch", "GPU_mem", "box_loss", "seg_loss", "obj_loss", "cls_loss", "Instances", "Size")  # 输出标题
        )
        if RANK in {-1, 0}:  # 如果是主进程
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar 显示进度条
            optimizer.zero_grad()  # 清零优化器的梯度
            for i, (imgs, targets, paths, _, masks) in pbar:  # batch ------------------------------------------------------
                # callbacks.run('on_train_batch_start')  # 在每个训练批次开始时运行回调
                ni = i + nb * epoch  # number integrated batches (since train start) 计算自训练开始以来的总批次数
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0 将图像转换为浮点数并归一化

                # Warmup
                if ni <= nw:  # 如果当前批次在预热阶段
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou) 计算IOU损失比例（注释掉）
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())  # 计算累积次数
                    for j, x in enumerate(optimizer.param_groups):  # 遍历优化器的参数组
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])  # 更新学习率
                        if "momentum" in x:  # 如果有动量
                            x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])  # 更新动量

                # Multi-scale
                if opt.multi_scale:  # 如果启用多尺度训练
                    sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size 随机选择图像大小
                    sf = sz / max(imgs.shape[2:])  # scale factor 计算缩放因子
                    if sf != 1:  # 如果缩放因子不为1
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple) 计算新的图像尺寸
                        imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)  # 重新调整图像大小

                # Forward
                with torch.cuda.amp.autocast(amp):  # 启用自动混合精度
                    pred = model(imgs)  # forward 前向传播
                    loss, loss_items = compute_loss(pred, targets.to(device), masks=masks.to(device).float())  # 计算损失
                    if RANK != -1:  # 如果不是主进程
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode 在DDP模式下对梯度进行平均
                    if opt.quad:  # 如果启用四分之一训练
                        loss *= 4.0  # 调整损失

                # Backward
                scaler.scale(loss).backward()  # 反向传播，缩放损失

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= accumulate:  # 如果达到累积次数
                    scaler.unscale_(optimizer)  # unscale gradients 反缩放梯度
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients 裁剪梯度
                    scaler.step(optimizer)  # optimizer.step 优化器更新
                    scaler.update()  # 更新缩放器
                    optimizer.zero_grad()  # 清零优化器的梯度
                    if ema:  # 如果使用EMA
                        ema.update(model)  # 更新EMA
                    last_opt_step = ni  # 更新最后优化步骤

                # Log
                if RANK in {-1, 0}:  # 如果是主进程
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses 更新平均损失
                    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB) 获取当前GPU内存使用情况
                    pbar.set_description(  # 更新进度条描述
                        ("%11s" * 2 + "%11.4g" * 6)  # 格式化输出
                        % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])  # 输出当前轮次、内存、损失等信息
                    )
                    # callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths)  # 在每个训练批次结束时运行回调
                    # if callbacks.stop_training:  # 如果回调请求停止训练
                    #    return

                    # Mosaic plots
                    if plots:  # 如果需要绘图
                        if ni < 3:  # 在前3个批次绘制图像
                            plot_images_and_masks(imgs, targets, masks, paths, save_dir / f"train_batch{ni}.jpg")  # 绘制图像和掩码
                        if ni == 10:  # 在第10个批次时记录马赛克图像
                            files = sorted(save_dir.glob("train*.jpg"))  # 获取保存的马赛克图像文件
                            logger.log_images(files, "Mosaics", epoch)  # 记录马赛克图像
                # end batch --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x["lr"] for x in optimizer.param_groups]  # for loggers 获取优化器每个参数组的学习率，用于记录
            scheduler.step()  # 更新学习率调度器

            if RANK in {-1, 0}:  # 如果是主进程
                # mAP
                # callbacks.run('on_train_epoch_end', epoch=epoch)  # 在每个训练轮次结束时运行回调
                ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])  # 更新EMA模型的属性
                final_epoch = (epoch + 1 == epochs) or stopper.possible_stop  # 检查是否为最后一个轮次或是否可以提前停止
                if not noval or final_epoch:  # 如果不进行验证或是最后一个轮次
                    results, maps, _ = validate.run(  # 计算mAP
                        data_dict,  # 数据字典
                        batch_size=batch_size // WORLD_SIZE * 2,  # 每个进程的批次大小，乘以2
                        imgsz=imgsz,  # 图像大小
                        half=amp,  # 是否使用半精度
                        model=ema.ema,  # 使用EMA模型
                        single_cls=single_cls,  # 是否单类训练
                        dataloader=val_loader,  # 验证数据加载器
                        save_dir=save_dir,  # 保存目录
                        plots=False,  # 不绘制图
                        callbacks=callbacks,  # 回调函数
                        compute_loss=compute_loss,  # 计算损失
                        mask_downsample_ratio=mask_ratio,  # 掩码下采样比例
                        overlap=overlap,  # 重叠掩码
                    )

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95] 计算加权组合的适应度
                stop = stopper(epoch=epoch, fitness=fi)  # early stop check 提前停止检查
                if fi > best_fitness:  # 如果当前适应度比最佳适应度高
                    best_fitness = fi  # 更新最佳适应度
                log_vals = list(mloss) + list(results) + lr  # 记录损失和学习率
                # callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)  # 在每个适应度计算结束时运行回调
                # Log val metrics and media
                metrics_dict = dict(zip(KEYS, log_vals))  # 将记录的值与键配对，生成字典
                logger.log_metrics(metrics_dict, epoch)  # 记录验证指标

                # Save model
                if (not nosave) or (final_epoch and not evolve):  # if save 如果需要保存模型
                    ckpt = {  # 创建检查点字典
                        "epoch": epoch,  # 当前轮次
                        "best_fitness": best_fitness,  # 最佳适应度
                        "model": deepcopy(de_parallel(model)).half(),  # 深拷贝模型并转换为半精度
                        "ema": deepcopy(ema.ema).half(),  # 深拷贝EMA模型并转换为半精度
                        "updates": ema.updates,  # EMA更新次数
                        "optimizer": optimizer.state_dict(),  # 优化器状态字典
                        "opt": vars(opt),  # 选项字典
                        "git": GIT_INFO,  # Git信息（{remote, branch, commit} 如果是Git仓库）
                        "date": datetime.now().isoformat(),  # 当前日期时间
                    }

                    # Save last, best and delete
                    torch.save(ckpt, last)  # 保存最后的检查点
                    if best_fitness == fi:  # 如果当前适应度是最佳适应度
                        torch.save(ckpt, best)  # 保存最佳检查点
                    if opt.save_period > 0 and epoch % opt.save_period == 0:  # 如果设置了保存周期
                        torch.save(ckpt, w / f"epoch{epoch}.pt")  # 保存当前轮次的检查点
                        logger.log_model(w / f"epoch{epoch}.pt")  # 记录模型
                    del ckpt  # 删除检查点字典
                    # callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)  # 在模型保存时运行回调

            # EarlyStopping
            if RANK != -1:  # if DDP training 如果是分布式数据并行训练
                broadcast_list = [stop if RANK == 0 else None]  # 创建广播列表，主进程发送停止信号
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks 将'stop'信号广播到所有进程
                if RANK != 0:  # 如果不是主进程
                    stop = broadcast_list[0]  # 接收停止信号
            if stop:  # 如果接收到停止信号
                break  # must break all DDP ranks 退出所有DDP进程

            # end epoch ----------------------------------------------------------------------------------------------------
            # 结束一个训练轮次
            # end training -----------------------------------------------------------------------------------------------------
            if RANK in {-1, 0}:  # 如果是主进程
                LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")  # 记录完成的轮次和耗时
                for f in last, best:  # 遍历最后和最佳检查点
                    if f.exists():  # 如果文件存在
                        strip_optimizer(f)  # strip optimizers 清理优化器信息
                        if f is best:  # 如果是最佳检查点
                            LOGGER.info(f"\nValidating {f}...")  # 记录正在验证的文件
                            results, _, _ = validate.run(  # 进行验证
                                data_dict,  # 数据字典
                                batch_size=batch_size // WORLD_SIZE * 2,  # 每个进程的批次大小，乘以2
                                imgsz=imgsz,  # 图像大小
                                model=attempt_load(f, device).half(),  # 加载最佳模型并转换为半精度
                                iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65 根据数据集类型设置IOU阈值
                                single_cls=single_cls,  # 是否单类训练
                                dataloader=val_loader,  # 验证数据加载器
                                save_dir=save_dir,  # 保存目录
                                save_json=is_coco,  # 是否保存为JSON格式
                                verbose=True,  # 是否详细输出
                                plots=plots,  # 是否绘制图
                                callbacks=callbacks,  # 回调函数
                                compute_loss=compute_loss,  # 计算损失
                                mask_downsample_ratio=mask_ratio,  # 掩码下采样比例
                                overlap=overlap,  # 重叠掩码
                            )  # val best model with plots 验证最佳模型并绘制图
                            if is_coco:  # 如果是COCO数据集
                                # callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)  # 在每个适应度计算结束时运行回调
                                metrics_dict = dict(zip(KEYS, list(mloss) + list(results) + lr))  # 将记录的值与键配对，生成字典
                                logger.log_metrics(metrics_dict, epoch)  # 记录指标

                # callbacks.run('on_train_end', last, best, epoch, results)  # 训练结束时运行回调
                # on train end callback using genericLogger 使用通用日志记录器的训练结束回调
                logger.log_metrics(dict(zip(KEYS[4:16], results)), epochs)  # 记录验证指标
                if not opt.evolve:  # 如果不进行进化训练
                    logger.log_model(best, epoch)  # 记录最佳模型
                if plots:  # 如果需要绘图
                    plot_results_with_masks(file=save_dir / "results.csv")  # save results.png 保存结果图
                    files = ["results.png", "confusion_matrix.png", *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R"))]  # 生成文件列表
                    files = [(save_dir / f) for f in files if (save_dir / f).exists()]  # filter 过滤存在的文件
                    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")  # 记录结果保存位置
                    logger.log_images(files, "Results", epoch + 1)  # 记录结果图像
                    logger.log_images(sorted(save_dir.glob("val*.jpg")), "Validation", epoch + 1)  # 记录验证图像
        torch.cuda.empty_cache()  # 清空CUDA缓存
        return results  # 返回结果


def parse_opt(known=False):  # 定义解析命令行参数的函数，参数known用于指示是否解析已知参数
    """
    Parses command line arguments for training configurations, returning parsed arguments.
    解析训练配置的命令行参数，并返回解析后的参数。

    Supports both known and unknown args.
    支持已知和未知参数。
    """
    parser = argparse.ArgumentParser()  # 创建一个命令行参数解析器对象
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s-seg.pt", help="initial weights path")  # 添加权重参数，默认值为yolov5s-seg.pt
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")  # 添加配置文件路径参数
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128-seg.yaml", help="dataset.yaml path")  # 添加数据集配置文件路径参数
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")  # 添加超参数路径参数
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")  # 添加训练轮次参数，默认值为100
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")  # 添加批次大小参数
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")  # 添加图像大小参数
    parser.add_argument("--rect", action="store_true", help="rectangular training")  # 添加矩形训练参数
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")  # 添加恢复训练参数
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")  # 添加不保存中间检查点参数
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")  # 添加仅验证最后一轮参数
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")  # 添加禁用自动锚点参数
    parser.add_argument("--noplots", action="store_true", help="save no plot files")  # 添加不保存绘图文件参数
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")  # 添加进化超参数参数
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")  # 添加Google云存储桶参数
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")  # 添加图像缓存参数
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")  # 添加使用加权图像选择参数
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  # 添加设备参数
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")  # 添加多尺度训练参数
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")  # 添加单类训练参数
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")  # 添加优化器选择参数
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")  # 添加使用同步批量归一化参数
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")  # 添加最大数据加载工作线程数参数
    parser.add_argument("--project", default=ROOT / "runs/train-seg", help="save to project/name")  # 添加保存项目路径参数
    parser.add_argument("--name", default="exp", help="save to project/name")  # 添加保存名称参数
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")  # 添加允许存在项目/名称参数
    parser.add_argument("--quad", action="store_true", help="quad dataloader")  # 添加四分之一数据加载器参数
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")  # 添加余弦学习率调度参数
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")  # 添加标签平滑参数
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")  # 添加提前停止耐心参数
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")  # 添加冻结层参数
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")  # 添加保存周期参数
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")  # 添加全局训练随机种子参数
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")  # 添加自动DDP多GPU参数

    # Instance Segmentation Args
    parser.add_argument("--mask-ratio", type=int, default=4, help="Downsample the truth masks to saving memory")  # 添加掩码比例参数
    parser.add_argument("--no-overlap", action="store_true", help="Overlap masks train faster at slightly less mAP")  # 添加重叠掩码参数

    return parser.parse_known_args()[0] if known else parser.parse_args()  # 返回解析后的参数，如果已知参数则返回已知参数，否则返回所有参数


def main(opt, callbacks=Callbacks()):  # 定义主函数，接收训练选项和回调函数
    """Initializes training or evolution of YOLOv5 models based on provided configuration and options."""
    # 根据提供的配置和选项初始化YOLOv5模型的训练或进化。

    if RANK in {-1, 0}:  # 如果当前进程是主进程
        print_args(vars(opt))  # 打印命令行参数
        check_git_status()  # 检查Git状态
        check_requirements(ROOT / "requirements.txt")  # 检查依赖项

    # Resume
    if opt.resume and not opt.evolve:  # 如果选择恢复训练且不进行进化
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())  # 获取最近的检查点路径
        opt_yaml = last.parent.parent / "opt.yaml"  # 获取训练选项的yaml文件路径
        opt_data = opt.data  # 原始数据集路径
        if opt_yaml.is_file():  # 如果选项文件存在
            with open(opt_yaml, errors="ignore") as f:  # 打开选项文件
                d = yaml.safe_load(f)  # 加载yaml内容
        else:
            d = torch.load(last, map_location="cpu")["opt"]  # 从检查点加载选项
        opt = argparse.Namespace(**d)  # 用加载的选项替换当前选项
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # 恢复配置和权重
        if is_url(opt_data):  # 如果数据集路径是URL
            opt.data = check_file(opt_data)  # 检查文件以避免HUB恢复认证超时
    else:  # 如果不是恢复训练
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),  # 检查数据集文件
            check_yaml(opt.cfg),  # 检查配置文件
            check_yaml(opt.hyp),  # 检查超参数文件
            str(opt.weights),  # 转换权重路径为字符串
            str(opt.project),  # 转换项目路径为字符串
        )  # 检查所有参数
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"  # 确保至少指定了配置或权重
        if opt.evolve:  # 如果选择进化超参数
            if opt.project == str(ROOT / "runs/train-seg"):  # 如果项目名称是默认值
                opt.project = str(ROOT / "runs/evolve-seg")  # 重命名项目路径
            opt.exist_ok, opt.resume = opt.resume, False  # 将恢复状态传递给exist_ok并禁用恢复
        if opt.name == "cfg":  # 如果名称是"cfg"
            opt.name = Path(opt.cfg).stem  # 使用模型配置文件名作为名称
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 生成保存目录

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)  # 选择设备
    if LOCAL_RANK != -1:  # 如果是分布式训练
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"  # 错误信息
        assert not opt.image_weights, f"--image-weights {msg}"  # 确保不使用图像权重
        assert not opt.evolve, f"--evolve {msg}"  # 确保不进行超参数进化
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"  # 确保批次大小有效
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"  # 确保批次大小是WORLD_SIZE的倍数
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"  # 检查CUDA设备数量
        torch.cuda.set_device(LOCAL_RANK)  # 设置当前CUDA设备
        device = torch.device("cuda", LOCAL_RANK)  # 创建CUDA设备对象
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")  # 初始化进程组

    # Train
    if not opt.evolve:  # 如果不进行超参数进化
        train(opt.hyp, opt, device, callbacks)  # 调用训练函数

    # Evolve hyperparameters (optional)
    else:  # 否则进行超参数进化
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)  # 初始学习率
            "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)  # 最终学习率
            "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1  # 动量
            "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay  # 优化器的权重衰减
            "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)  # 预热轮次
            "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum  # 预热初始动量
            "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr  # 预热初始偏置学习率
            "box": (1, 0.02, 0.2),  # box loss gain  # 边框损失增益
            "cls": (1, 0.2, 4.0),  # cls loss gain  # 类别损失增益
            "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight  # 类别BCELoss正权重
            "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)  # 目标损失增益
            "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight  # 目标BCELoss正权重
            "iou_t": (0, 0.1, 0.7),  # IoU training threshold  # IoU训练阈值
            "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold  # 锚点倍数阈值
            "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)  # 每个输出网格的锚点数量
            "fl_gamma": (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)  # 焦点损失的gamma值
            "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)  # 图像HSV色调增强
            "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)  # 图像HSV饱和度增强
            "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)  # 图像HSV值增强
            "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)  # 图像旋转
            "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)  # 图像平移
            "scale": (1, 0.0, 0.9),  # image scale (+/- gain)  # 图像缩放
            "shear": (1, 0.0, 10.0),  # image shear (+/- deg)  # 图像剪切
            "perspective": (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001  # 图像透视
            "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)  # 图像上下翻转
            "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)  # 图像左右翻转
            "mosaic": (1, 0.0, 1.0),  # image mixup (probability)  # 图像混合
            "mixup": (1, 0.0, 1.0),  # image mixup (probability)  # 图像混合
            "copy_paste": (1, 0.0, 1.0),  # 复制粘贴
        }  # segment copy-paste (probability)  # 分割复制粘贴的概率

        with open(opt.hyp, errors="ignore") as f:  # 打开超参数文件
            hyp = yaml.safe_load(f)  # 加载超参数字典
            if "anchors" not in hyp:  # 如果超参数中没有锚点
                hyp["anchors"] = 3  # 设置锚点数量为3
        if opt.noautoanchor:  # 如果禁用自动锚点
            del hyp["anchors"], meta["anchors"]  # 删除锚点相关的超参数
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # 仅验证/保存最后一轮
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices  # 可进化的索引
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"  # 进化超参数文件和CSV文件路径
        if opt.bucket:  # 如果指定了云存储桶
            # download evolve.csv if exists  # 如果存在，下载evolve.csv
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),  # 下载到指定路径
                ]
            )

        for _ in range(opt.evolve):  # 进行指定代数的进化
            if evolve_csv.exists():  # 如果evolve.csv存在
                # Select parent(s)  # 选择父代
                parent = "single"  # 父代选择方法：'single'或'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=",", skiprows=1)  # 从CSV文件加载数据
                n = min(5, len(x))  # 考虑的前n个结果
                x = x[np.argsort(-fitness(x))][:n]  # 选择最优的n个变异
                w = fitness(x) - fitness(x).min() + 1e-6  # 权重（确保和大于0）
                if parent == "single" or len(x) == 1:  # 如果选择单个父代或只有一个父代
                    # x = x[random.randint(0, n - 1)]  # random selection  # 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
                elif parent == "weighted":  # 如果选择加权组合
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

                # Mutate
                mp, s = 0.8, 0.2  # 变异概率和标准差
                npr = np.random  # 随机数生成器
                npr.seed(int(time.time()))  # 设置随机种子
                g = np.array([meta[k][0] for k in hyp.keys()])  # 获取增益
                ng = len(meta)  # 增益的数量
                v = np.ones(ng)  # 初始化变异值
                while all(v == 1):  # 变异直到发生变化（防止重复）
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)  # 生成变异值
                for i, k in enumerate(hyp.keys()):  # 更新超参数
                    hyp[k] = float(x[i + 12] * v[i])  # 变异

            # Constrain to limits
            for k, v in meta.items():  # 对超参数进行约束
                hyp[k] = max(hyp[k], v[1])  # 下限约束
                hyp[k] = min(hyp[k], v[2])  # 上限约束
                hyp[k] = round(hyp[k], 5)  # 保留有效数字

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)  # 训练变异后的超参数
            callbacks = Callbacks()  # 重置回调函数
            # Write mutation results
            print_mutation(KEYS[4:16], results, hyp.copy(), save_dir, opt.bucket)  # 打印变异结果

        # Plot results
        plot_evolve(evolve_csv)  # 绘制进化结果
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'  # 记录进化完成信息
            f"Results saved to {colorstr('bold', save_dir)}\n"  # 记录结果保存位置
            f'Usage example: $ python train.py --hyp {evolve_yaml}'  # 提供用法示例
        )


def run(**kwargs):  # 定义运行函数，接受可变关键字参数
    """
    Executes YOLOv5 training with given parameters, altering options programmatically; returns updated options.
    使用给定参数执行YOLOv5训练，程序化地修改选项；返回更新后的选项。

    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    示例：导入训练模块；调用train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    opt = parse_opt(True)  # 解析命令行参数，返回已知参数
    for k, v in kwargs.items():  # 遍历传入的关键字参数
        setattr(opt, k, v)  # 动态设置选项的属性值
    main(opt)  # 调用主函数进行训练
    return opt  # 返回更新后的选项


if __name__ == "__main__":  # 如果该脚本是主程序
    opt = parse_opt()  # 解析命令行参数
    main(opt)  # 调用主函数进行训练