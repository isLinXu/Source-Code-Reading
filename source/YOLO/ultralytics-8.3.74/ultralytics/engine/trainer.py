# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc  # 导入gc模块，用于垃圾回收
import math  # 导入math模块，用于数学计算
import os  # 导入os模块，用于与操作系统交互
import subprocess  # 导入subprocess模块，用于执行子进程
import time  # 导入time模块，用于时间相关操作
import warnings  # 导入warnings模块，用于发出警告
from copy import copy, deepcopy  # 从copy模块导入copy和deepcopy函数
from datetime import datetime, timedelta  # 从datetime模块导入datetime和timedelta类
from pathlib import Path  # 从pathlib模块导入Path类

import numpy as np  # 导入numpy库并命名为np，用于数组操作
import torch  # 导入PyTorch库
from torch import distributed as dist  # 从PyTorch导入分布式模块并命名为dist
from torch import nn, optim  # 从PyTorch导入神经网络模块和优化模块

from ultralytics.cfg import get_cfg, get_save_dir  # 从ultralytics.cfg模块导入get_cfg和get_save_dir函数
from ultralytics.data.utils import check_cls_dataset, check_det_dataset  # 从ultralytics.data.utils模块导入数据集检查函数
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights  # 从ultralytics.nn.tasks模块导入权重加载函数
from ultralytics.utils import (
    DEFAULT_CFG,  # 默认配置
    LOCAL_RANK,  # 本地进程的排名
    LOGGER,  # 日志记录器
    RANK,  # 全局进程的排名
    TQDM,  # 进度条显示
    __version__,  # 当前版本
    callbacks,  # 回调函数
    clean_url,  # 清理URL的函数
    colorstr,  # 颜色字符串处理函数
    emojis,  # 表情符号处理
    yaml_save,  # 保存YAML文件的函数
)
from ultralytics.utils.autobatch import check_train_batch_size  # 从ultralytics.utils.autobatch模块导入检查训练批次大小的函数
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args  # 从ultralytics.utils.checks模块导入检查函数
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command  # 从ultralytics.utils.dist模块导入DDP清理和命令生成函数
from ultralytics.utils.files import get_latest_run  # 从ultralytics.utils.files模块导入获取最新运行的函数
from ultralytics.utils.torch_utils import (
    TORCH_2_4,  # PyTorch版本检查
    EarlyStopping,  # 提前停止类
    ModelEMA,  # 指数移动平均模型
    autocast,  # 自动混合精度
    convert_optimizer_state_dict_to_fp16,  # 将优化器状态字典转换为fp16
    init_seeds,  # 初始化随机种子
    one_cycle,  # 一周期学习率调度
    select_device,  # 选择设备的函数
    strip_optimizer,  # 去除优化器的函数
    torch_distributed_zero_first,  # DDP的零进程优先
    unset_deterministic,  # 取消确定性设置
)

class BaseTrainer:
    """
    A base class for creating trainers.  # 创建训练器的基类

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.  # 训练器的配置
        validator (BaseValidator): Validator instance.  # 验证器实例
        model (nn.Module): Model instance.  # 模型实例
        callbacks (defaultdict): Dictionary of callbacks.  # 回调函数字典
        save_dir (Path): Directory to save results.  # 保存结果的目录
        wdir (Path): Directory to save weights.  # 保存权重的目录
        last (Path): Path to the last checkpoint.  # 最后检查点的路径
        best (Path): Path to the best checkpoint.  # 最佳检查点的路径
        save_period (int): Save checkpoint every x epochs (disabled if < 1).  # 每x个epoch保存检查点（如果<1则禁用）
        batch_size (int): Batch size for training.  # 训练的批次大小
        epochs (int): Number of epochs to train for.  # 训练的轮数
        start_epoch (int): Starting epoch for training.  # 开始训练的轮数
        device (torch.device): Device to use for training.  # 用于训练的设备
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).  # 启用自动混合精度的标志
        scaler (amp.GradScaler): Gradient scaler for AMP.  # 自动混合精度的梯度缩放器
        data (str): Path to data.  # 数据的路径
        trainset (torch.utils.data.Dataset): Training dataset.  # 训练数据集
        testset (torch.utils.data.Dataset): Testing dataset.  # 测试数据集
        ema (nn.Module): EMA (Exponential Moving Average) of the model.  # 模型的指数移动平均
        resume (bool): Resume training from a checkpoint.  # 从检查点恢复训练
        lf (nn.Module): Loss function.  # 损失函数
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.  # 学习率调度器
        best_fitness (float): The best fitness value achieved.  # 达到的最佳适应度值
        fitness (float): Current fitness value.  # 当前适应度值
        loss (float): Current loss value.  # 当前损失值
        tloss (float): Total loss value.  # 总损失值
        loss_names (list): List of loss names.  # 损失名称列表
        csv (Path): Path to results CSV file.  # 结果CSV文件的路径
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.  # 初始化BaseTrainer类

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.  # 配置文件的路径，默认为DEFAULT_CFG
            overrides (dict, optional): Configuration overrides. Defaults to None.  # 配置覆盖，默认为None
        """
        self.args = get_cfg(cfg, overrides)  # 获取配置
        self.check_resume(overrides)  # 检查是否从检查点恢复
        self.device = select_device(self.args.device, self.args.batch)  # 选择设备
        self.validator = None  # 验证器初始化为None
        self.metrics = None  # 指标初始化为None
        self.plots = {}  # 初始化绘图字典
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)  # 初始化随机种子

        # Dirs  # 目录设置
        self.save_dir = get_save_dir(self.args)  # 获取保存目录
        self.args.name = self.save_dir.name  # update name for loggers  # 更新日志记录器的名称
        self.wdir = self.save_dir / "weights"  # weights dir  # 权重目录
        if RANK in {-1, 0}:  # 如果是主进程
            self.wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
            self.args.save_dir = str(self.save_dir)  # 保存目录
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # 保存运行参数
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # 检查点路径
        self.save_period = self.args.save_period  # 保存周期

        self.batch_size = self.args.batch  # 批次大小
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training  # 如果用户意外传递epochs=None，默认为100
        self.start_epoch = 0  # 起始轮数
        if RANK == -1:  # 如果是主进程
            print_args(vars(self.args))  # 打印参数

        # Device  # 设备设置
        if self.device.type in {"cpu", "mps"}:  # 如果设备是CPU或MPS
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading  # 更快的CPU训练，因为时间主要由推理而不是数据加载主导

        # Model and Dataset  # 模型和数据集设置
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt  # 检查模型文件
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times  # 避免多次自动下载数据集
            self.trainset, self.testset = self.get_dataset()  # 获取训练集和测试集
        self.ema = None  # 指数移动平均初始化为None

        # Optimization utils init  # 优化工具初始化
        self.lf = None  # 损失函数初始化为None
        self.scheduler = None  # 学习率调度器初始化为None

        # Epoch level metrics  # 轮次级别的指标
        self.best_fitness = None  # 最佳适应度初始化为None
        self.fitness = None  # 当前适应度初始化为None
        self.loss = None  # 当前损失初始化为None
        self.tloss = None  # 总损失初始化为None
        self.loss_names = ["Loss"]  # 损失名称列表
        self.csv = self.save_dir / "results.csv"  # 结果CSV文件路径
        self.plot_idx = [0, 1, 2]  # 绘图索引

        # HUB  # HUB设置
        self.hub_session = None  # HUB会话初始化为None

        # Callbacks  # 回调函数设置
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 获取回调函数
        if RANK in {-1, 0}:  # 如果是主进程
            callbacks.add_integration_callbacks(self)  # 添加集成回调

    def add_callback(self, event: str, callback):
        """Appends the given callback.  # 添加给定的回调函数"""
        self.callbacks[event].append(callback)  # 将回调函数添加到指定事件

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback.  # 用给定的回调函数覆盖现有回调函数"""
        self.callbacks[event] = [callback]  # 设置指定事件的回调函数

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event.  # 运行与特定事件相关的所有现有回调函数"""
        for callback in self.callbacks.get(event, []):  # 遍历指定事件的回调函数
            callback(self)  # 执行回调函数

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0.  # 允许在多GPU系统上将device=''或device=None默认为device=0"""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'  # 如果设备是字符串且长度大于0
            world_size = len(self.args.device.split(","))  # 获取世界大小
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)  # 如果设备是元组或列表
            world_size = len(self.args.device)  # 获取世界大小
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'  # 如果设备是CPU或MPS
            world_size = 0  # 世界大小为0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number  # 如果CUDA可用
            world_size = 1  # 默认为设备0
        else:  # i.e. device=None or device=''  # 否则
            world_size = 0  # 世界大小为0

        # Run subprocess if DDP training, else train normally  # 如果是DDP训练则运行子进程，否则正常训练
        if world_size > 1 and "LOCAL_RANK" not in os.environ:  # 如果世界大小大于1且LOCAL_RANK不在环境变量中
            # Argument checks  # 参数检查
            if self.args.rect:  # 如果使用矩形训练
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")  # 发出警告
                self.args.rect = False  # 将矩形设置为False
            if self.args.batch < 1.0:  # 如果批次小于1.0
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"  # 发出警告
                )
                self.args.batch = 16  # 将批次设置为16

            # Command  # 命令设置
            cmd, file = generate_ddp_command(world_size, self)  # 生成DDP命令
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")  # 记录DDP调试命令
                subprocess.run(cmd, check=True)  # 运行命令
            except Exception as e:  # 捕获异常
                raise e  # 抛出异常
            finally:
                ddp_cleanup(self, str(file))  # 清理DDP

        else:
            self._do_train(world_size)  # 正常训练

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler.  # 初始化训练学习率调度器"""
        if self.args.cos_lr:  # 如果使用余弦学习率
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']  # 设置余弦学习率
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear  # 设置线性学习率
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)  # 初始化学习率调度器

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training.  # 初始化并设置分布式数据并行训练参数"""
        torch.cuda.set_device(RANK)  # 设置当前CUDA设备
        self.device = torch.device("cuda", RANK)  # 设置设备为CUDA
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')  # 记录DDP信息
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout  # 设置以强制超时
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",  # 初始化进程组
            timeout=timedelta(seconds=10800),  # 3 hours  # 设置超时时间为3小时
            rank=RANK,  # 当前进程的排名
            world_size=world_size,  # 世界大小
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process.  # 在正确的排名进程上构建数据加载器和优化器"""
        # Model  # 模型设置
        self.run_callbacks("on_pretrain_routine_start")  # 运行预训练例程开始的回调
        ckpt = self.setup_model()  # 设置模型
        self.model = self.model.to(self.device)  # 将模型移动到设备
        self.set_model_attributes()  # 设置模型属性

        # Freeze layers  # 冻结层设置
        freeze_list = (
            self.args.freeze  # 冻结层列表
            if isinstance(self.args.freeze, list)  # 如果是列表
            else range(self.args.freeze)  # 否则使用范围
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers  # 始终冻结这些层
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names  # 冻结层名称列表
        for k, v in self.model.named_parameters():  # 遍历模型参数
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)  # NaN转为0（注释掉以避免训练结果不稳定）
            if any(x in k for x in freeze_layer_names):  # 如果参数名称在冻结层名称中
                LOGGER.info(f"Freezing layer '{k}'")  # 记录冻结层
                v.requires_grad = False  # 不计算梯度
            elif not v.requires_grad and v.dtype.is_floating_point:  # 仅浮点型Tensor可以计算梯度
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."  # 发出警告
                )
                v.requires_grad = True  # 设置为计算梯度

        # Check AMP  # 检查自动混合精度
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True或False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP  # 单GPU和DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them  # 备份回调函数
            self.amp = torch.tensor(check_amp(self.model), device=self.device)  # 检查AMP
            callbacks.default_callbacks = callbacks_backup  # 恢复回调函数
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)  # 从rank 0广播张量到所有其他rank
        self.amp = bool(self.amp)  # as boolean  # 转换为布尔值
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)  # 初始化梯度缩放器
        )
        if world_size > 1:  # 如果世界大小大于1
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)  # 设置分布式数据并行

        # Check imgsz  # 检查图像大小
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)  # 网格大小（最大步幅）
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)  # 检查图像大小
        self.stride = gs  # for multiscale training  # 用于多尺度训练

        # Batch size  # 批次大小设置
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size  # 仅单GPU，估计最佳批次大小
            self.args.batch = self.batch_size = self.auto_batch()  # 自动批次大小

        # Dataloaders  # 数据加载器设置
        batch_size = self.batch_size // max(world_size, 1)  # 计算每个进程的批次大小
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")  # 获取训练数据加载器
        if RANK in {-1, 0}:  # 如果是主进程
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.  # 注意：训练DOTA数据集时，双倍批次大小可能导致内存不足
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"  # 获取测试数据加载器
            )
            self.validator = self.get_validator()  # 获取验证器
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")  # 获取指标键
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # 初始化指标字典
            self.ema = ModelEMA(self.model)  # 初始化EMA模型
            if self.args.plots:  # 如果需要绘图
                self.plot_training_labels()  # 绘制训练标签

        # Optimizer  # 优化器设置
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing  # 在优化之前累积损失
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay  # 缩放权重衰减
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs  # 计算迭代次数
        self.optimizer = self.build_optimizer(
            model=self.model,  # 模型
            name=self.args.optimizer,  # 优化器名称
            lr=self.args.lr0,  # 初始学习率
            momentum=self.args.momentum,  # 动量
            decay=weight_decay,  # 权重衰减
            iterations=iterations,  # 迭代次数
        )
        # Scheduler  # 学习率调度器设置
        self._setup_scheduler()  # 初始化学习率调度器
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False  # 初始化提前停止
        self.resume_training(ckpt)  # 从检查点恢复训练
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move  # 不移动调度器的最后轮数
        self.run_callbacks("on_pretrain_routine_end")  # 运行预训练例程结束的回调

        
    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        # 训练完成后，如果参数指定，则进行评估和绘图。
        if world_size > 1:
            self._setup_ddp(world_size)  # 如果世界大小大于1，设置分布式数据并行（DDP）。
        self._setup_train(world_size)  # 设置训练环境。
    
        nb = len(self.train_loader)  # number of batches
        # 批次数量
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        # 计算预热迭代次数
        last_opt_step = -1  # 上一次优化步骤
        self.epoch_time = None  # 记录每个epoch的时间
        self.epoch_time_start = time.time()  # 记录当前epoch开始时间
        self.train_time_start = time.time()  # 记录训练开始时间
        self.run_callbacks("on_train_start")  # 运行训练开始的回调函数
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        # 记录图像大小、数据加载器工作线程数量和训练开始信息
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb  # 计算基础索引
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])  # 更新绘图索引
        epoch = self.start_epoch  # 从起始epoch开始
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        # 将任何恢复的梯度归零，以确保训练开始时的稳定性
        while True:
            self.epoch = epoch  # 更新当前epoch
            self.run_callbacks("on_train_epoch_start")  # 运行每个epoch开始的回调函数
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                # 抑制警告：在优化器步骤之前检测到学习率调度器步骤
                self.scheduler.step()  # 更新学习率调度器
    
            self.model.train()  # 将模型设置为训练模式
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)  # 设置数据加载器的采样器epoch
            pbar = enumerate(self.train_loader)  # 遍历训练数据加载器
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()  # 关闭数据加载器的马赛克
                self.train_loader.reset()  # 重置数据加载器
    
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())  # 记录进度信息
                pbar = TQDM(enumerate(self.train_loader), total=nb)  # 使用TQDM显示进度条
            self.tloss = None  # 初始化总损失
            for i, batch in pbar:  # 遍历每个批次
                self.run_callbacks("on_train_batch_start")  # 运行每个批次开始的回调函数
                # Warmup
                ni = i + nb * epoch  # 当前迭代次数
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    # 计算累计步骤
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        # 更新学习率
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                            # 更新动量
    
                # Forward
                with autocast(self.amp):  # 使用自动混合精度
                    batch = self.preprocess_batch(batch)  # 预处理批次
                    self.loss, self.loss_items = self.model(batch)  # 计算损失
                    if RANK != -1:
                        self.loss *= world_size  # 如果是分布式训练，损失乘以世界大小
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                    # 更新总损失
    
                # Backward
                self.scaler.scale(self.loss).backward()  # 反向传播
    
                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()  # 执行优化步骤
                    last_opt_step = ni  # 更新上一次优化步骤
    
                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)  # 判断是否超时
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            # 将停止标志广播到所有进程
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break  # 超过训练时间，退出循环
    
                # Log
                if RANK in {-1, 0}:  # 记录日志
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1  # 总损失的长度
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",  # 当前epoch/总epoch
                            f"{self._get_memory():.3g}G",  # (GB) GPU内存使用
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # 损失
                            batch["cls"].shape[0],  # 批次大小，例如8
                            batch["img"].shape[-1],  # 图像大小，例如640
                        )
                    )
                    self.run_callbacks("on_batch_end")  # 运行每个批次结束的回调函数
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)  # 绘制训练样本
    
                self.run_callbacks("on_train_batch_end")  # 运行每个训练批次结束的回调函数
    
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            # 记录学习率
            self.run_callbacks("on_train_epoch_end")  # 运行每个epoch结束的回调函数
            if RANK in {-1, 0}:  # 进行验证
                final_epoch = epoch + 1 >= self.epochs  # 判断是否为最后一个epoch
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                # 更新模型的指数移动平均
    
                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()  # 进行验证
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})  # 保存指标
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch  # 判断是否停止训练
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)  # 超过时间停止
    
                # Save model
                if self.args.save or final_epoch:
                    self.save_model()  # 保存模型
                    self.run_callbacks("on_model_save")  # 运行模型保存的回调函数
    
            # Scheduler
            t = time.time()  # 获取当前时间
            self.epoch_time = t - self.epoch_time_start  # 计算epoch时间
            self.epoch_time_start = t  # 更新epoch开始时间
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)  # 计算平均epoch时间
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)  # 更新总epoch数
                self._setup_scheduler()  # 设置学习率调度器
                self.scheduler.last_epoch = self.epoch  # 不移动
                self.stop |= epoch >= self.epochs  # 超过总epoch数停止
            self.run_callbacks("on_fit_epoch_end")  # 运行每个fit epoch结束的回调函数
            self._clear_memory()  # 清除内存
    
            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                # 将停止标志广播到所有进程
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
                # 必须退出所有DDP进程
            epoch += 1  # 增加epoch计数
    
        if RANK in {-1, 0}:  # 如果是主进程
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start  # 计算训练总时间
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            # 记录完成的epoch数和总时间
            self.final_eval()  # 最终评估
            if self.args.plots:
                self.plot_metrics()  # 绘制指标
            self.run_callbacks("on_train_end")  # 运行训练结束的回调函数
        self._clear_memory()  # 清除内存
        unset_deterministic()  # 取消确定性设置
        self.run_callbacks("teardown")  # 运行清理的回调函数

    def auto_batch(self, max_num_obj=0):
        """Get batch size by calculating memory occupation of model."""
        # 通过计算模型的内存占用来获取批次大小。
        return check_train_batch_size(
            model=self.model,  # 传入模型
            imgsz=self.args.imgsz,  # 传入图像大小
            amp=self.amp,  # 传入自动混合精度设置
            batch=self.batch_size,  # 传入当前批次大小
            max_num_obj=max_num_obj,  # 传入最大对象数
        )  # returns batch size

    def _get_memory(self):
        """Get accelerator memory utilization in GB."""
        # 获取加速器内存使用情况（单位：GB）。
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()  # 如果设备是MPS，获取分配的内存
        elif self.device.type == "cpu":
            memory = 0  # 如果设备是CPU，内存为0
        else:
            memory = torch.cuda.memory_reserved()  # 否则，获取CUDA保留的内存
        return memory / 1e9  # 将内存转换为GB并返回

    def _clear_memory(self):
        """Clear accelerator memory on different platforms."""
        # 在不同平台上清除加速器内存。
        gc.collect()  # 垃圾回收，清理未使用的内存
        if self.device.type == "mps":
            torch.mps.empty_cache()  # 如果设备是MPS，清空缓存
        elif self.device.type == "cpu":
            return  # 如果设备是CPU，不进行操作
        else:
            torch.cuda.empty_cache()  # 否则，清空CUDA缓存

    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        # 使用pandas读取results.csv并转换为字典。
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.read_csv(self.csv).to_dict(orient="list")  # 读取CSV文件并返回字典格式

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        # 保存模型训练检查点及附加元数据。
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        # 将检查点序列化到字节缓冲区中（比重复调用torch.save()更快）
        buffer = io.BytesIO()  # 创建字节缓冲区
        torch.save(
            {
                "epoch": self.epoch,  # 当前epoch
                "best_fitness": self.best_fitness,  # 最佳适应度
                "model": None,  # resume and final checkpoints derive from EMA
                # 恢复和最终检查点来自EMA
                "ema": deepcopy(self.ema.ema).half(),  # 深拷贝EMA并转换为半精度
                "updates": self.ema.updates,  # 更新次数
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),  # 将优化器状态字典转换为FP16
                "train_args": vars(self.args),  # 将训练参数保存为字典
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},  # 保存训练指标，包括适应度
                "train_results": self.read_results_csv(),  # 读取训练结果CSV
                "date": datetime.now().isoformat(),  # 当前日期时间
                "version": __version__,  # 当前版本
                "license": "AGPL-3.0 (https://ultralytics.com/license)",  # 许可证信息
                "docs": "https://docs.ultralytics.com",  # 文档链接
            },
            buffer,  # 将数据写入缓冲区
        )
        serialized_ckpt = buffer.getvalue()  # 获取序列化的内容以保存

        # Save checkpoints
        # 保存检查点
        self.last.write_bytes(serialized_ckpt)  # 保存last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # 保存best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # 保存当前epoch，例如'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # 保存马赛克检查点
    
    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.
    
        Returns None if data format is not recognized.
        """
        # 从数据字典中获取训练和验证路径（如果存在）。
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)  # 检查分类数据集
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)  # 检查检测数据集
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # 用于验证'yolo train data=url.zip'的用法
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
            # 如果发生异常，抛出运行时错误并显示数据集错误信息
        self.data = data  # 将数据集赋值给实例变量
        return data["train"], data.get("val") or data.get("test")  # 返回训练和验证（或测试）数据路径
    
    def setup_model(self):
        """Load/create/download model for any task."""
        # 加载/创建/下载模型以适应任何任务。
        if isinstance(self.model, torch.nn.Module):  # 如果模型已加载，则无需设置
            return
    
        cfg, weights = self.model, None  # 初始化配置和权重
        ckpt = None  # 初始化检查点
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)  # 尝试加载权重
            cfg = weights.yaml  # 获取配置
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)  # 加载预训练权重
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # 调用Model(cfg, weights)
        return ckpt  # 返回检查点
    
    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        # 执行一次训练优化器步骤，包括梯度裁剪和EMA更新。
        self.scaler.unscale_(self.optimizer)  # 反缩放梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # 裁剪梯度
        self.scaler.step(self.optimizer)  # 执行优化器步骤
        self.scaler.update()  # 更新缩放器
        self.optimizer.zero_grad()  # 将梯度归零
        if self.ema:
            self.ema.update(self.model)  # 更新EMA
    
    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        # 根据任务类型允许自定义预处理模型输入和真实值。
        return batch  # 返回未修改的批次
    
    def validate(self):
        """
        Runs validation on test set using self.validator.
    
        The returned dict is expected to contain "fitness" key.
        """
        # 使用self.validator对测试集进行验证。
        metrics = self.validator(self)  # 运行验证器
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # 如果未找到，则使用损失作为适应度度量
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness  # 更新最佳适应度
        return metrics, fitness  # 返回指标和适应度
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        # 获取模型，如果加载cfg文件，则引发NotImplementedError。
        raise NotImplementedError("This task trainer doesn't support loading cfg files")
    
    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        # 当调用get_validator函数时返回NotImplementedError。
        raise NotImplementedError("get_validator function not implemented in trainer")
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        # 返回从torch.data.Dataloader派生的数据加载器。
        raise NotImplementedError("get_dataloader function not implemented in trainer")
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        # 构建数据集。
        raise NotImplementedError("build_dataset function not implemented in trainer")
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.
    
        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        # 返回带有标记的训练损失项张量的损失字典。
        return {"loss": loss_items} if loss_items is not None else ["loss"]  # 如果提供了损失项，则返回字典，否则返回损失列表
    
    def set_model_attributes(self):
        """To set or update model parameters before training."""
        # 在训练之前设置或更新模型参数。
        self.model.names = self.data["names"]  # 设置模型的类别名称
    
    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        # 为训练YOLO模型构建目标张量。
        pass  # 具体实现待定
    
    def progress_string(self):
        """Returns a string describing training progress."""
        # 返回描述训练进度的字符串。
        return ""  # 返回空字符串
    
    
    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        # 在YOLO训练期间绘制训练样本。
        pass  # 具体实现待定
    
    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        # 绘制YOLO模型的训练标签。
        pass  # 具体实现待定
    
    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        # 将训练指标保存到CSV文件中。
        keys, vals = list(metrics.keys()), list(metrics.values())  # 获取指标的键和值
        n = len(metrics) + 2  # 列数
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # 表头
        t = time.time() - self.train_time_start  # 计算训练时间
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")  # 写入CSV文件
    
    def plot_metrics(self):
        """Plot and display metrics visually."""
        # 可视化绘制和显示指标。
        pass  # 具体实现待定
    
    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        # 注册绘图（例如，在回调中使用）。
        path = Path(name)  # 将名称转换为路径对象
        self.plots[path] = {"data": data, "timestamp": time.time()}  # 存储绘图数据和时间戳
    
    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        # 对YOLO模型进行最终评估和验证。
        ckpt = {}  # 初始化检查点字典
        for f in self.last, self.best:  # 遍历最后和最佳检查点
            if f.exists():  # 如果检查点存在
                if f is self.last:
                    ckpt = strip_optimizer(f)  # 从最后检查点中去除优化器信息
                elif f is self.best:
                    k = "train_results"  # 从last.pt更新best.pt的训练指标
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)  # 更新最佳检查点
                    LOGGER.info(f"\nValidating {f}...")  # 记录正在验证的检查点
                    self.validator.args.plots = self.args.plots  # 设置验证器的绘图参数
                    self.metrics = self.validator(model=f)  # 运行验证器并获取指标
                    self.metrics.pop("fitness", None)  # 移除适应度指标
                    self.run_callbacks("on_fit_epoch_end")  # 运行每个fit epoch结束的回调函数
    
    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        # 检查恢复检查点是否存在，并相应更新参数。
        resume = self.args.resume  # 获取恢复参数
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()  # 检查恢复路径是否存在
                last = Path(check_file(resume) if exists else get_latest_run())  # 获取最后的检查点路径
    
                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                # 检查恢复数据的YAML文件是否存在，否则强制重新下载数据集
                ckpt_args = attempt_load_weights(last).args  # 尝试加载最后检查点的参数
                if not Path(ckpt_args["data"]).exists():  # 检查数据路径是否存在
                    ckpt_args["data"] = self.args.data  # 如果不存在，则使用当前数据路径
    
                resume = True  # 设置恢复标志为True
                self.args = get_cfg(ckpt_args)  # 更新参数
                self.args.model = self.args.resume = str(last)  # 恢复模型路径
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # 允许参数更新以减少内存或在恢复时更新设备
                    if k in overrides:
                        setattr(self.args, k, overrides[k])  # 更新参数
    
            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e  # 如果发生异常，抛出文件未找到错误
        self.resume = resume  # 更新恢复标志
    
    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        # 从给定的epoch和最佳适应度恢复YOLO训练。
        if ckpt is None or not self.resume:  # 如果检查点为空或不需要恢复
            return
        best_fitness = 0.0  # 初始化最佳适应度
        start_epoch = ckpt.get("epoch", -1) + 1  # 获取起始epoch
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # 加载优化器状态
            best_fitness = ckpt["best_fitness"]  # 获取最佳适应度
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # 加载EMA状态
            self.ema.updates = ckpt["updates"]  # 更新EMA次数
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )  # 确保起始epoch大于0
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        # 记录恢复训练的信息
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )  # 记录已训练的epoch数
            self.epochs += ckpt["epoch"]  # 增加总epoch数
        self.best_fitness = best_fitness  # 更新最佳适应度
        self.start_epoch = start_epoch  # 设置起始epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()  # 关闭数据加载器的马赛克增强
    
    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        # 更新数据加载器以停止使用马赛克增强。
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False  # 关闭马赛克增强
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")  # 记录关闭马赛克的操作
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))  # 关闭马赛克
    
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.
    
        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.
    
        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        # 根据指定的优化器名称、学习率、动量、权重衰减和迭代次数构造给定模型的优化器。
        g = [], [], []  # 优化器参数组
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 归一化层，例如BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )  # 记录自动选择优化器的信息
            nc = self.data.get("nc", 10)  # 类别数量
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0拟合方程，保留6位小数
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)  # 根据迭代次数选择优化器
            self.args.warmup_bias_lr = 0.0  # 对于Adam，偏置不高于0.01
    
        for module_name, module in model.named_modules():  # 遍历模型的所有模块
            for param_name, param in module.named_parameters(recurse=False):  # 遍历模块的参数
                fullname = f"{module_name}.{param_name}" if module_name else param_name  # 完整参数名
                if "bias" in fullname:  # 偏置（不衰减）
                    g[2].append(param)  # 添加到偏置组
                elif isinstance(module, bn):  # 权重（不衰减）
                    g[1].append(param)  # 添加到归一化层权重组
                else:  # 权重（衰减）
                    g[0].append(param)  # 添加到权重组
    
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}  # 可用的优化器列表
        name = {x.lower(): x for x in optimizers}.get(name.lower())  # 将优化器名称转换为小写
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)  # 创建Adam系列优化器
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)  # 创建RMSProp优化器
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)  # 创建SGD优化器
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )  # 如果未找到优化器，则引发NotImplementedError
    
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # 添加g0组（带权重衰减）
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # 添加g1组（BatchNorm2d权重，不衰减）
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )  # 记录优化器信息
        return optimizer  # 返回构造的优化器