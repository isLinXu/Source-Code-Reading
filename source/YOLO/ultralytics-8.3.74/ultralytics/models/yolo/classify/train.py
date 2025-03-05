# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first


class ClassificationTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationTrainer

        args = dict(model="yolo11n-cls.pt", data="imagenet10", epochs=3)
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()
        ```
    """
    # 继承自BaseTrainer的分类模型训练器类
    # 支持YOLO和Torchvision的分类模型训练

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a ClassificationTrainer object with optional configuration overrides and callbacks."""
        # 初始化分类训练器，设置默认配置和回调函数
        
        # 如果未提供覆盖配置，创建空字典
        if overrides is None:
            overrides = {}
        
        # 强制设置任务类型为分类
        overrides["task"] = "classify"
        
        # 如果未指定图像大小，默认设置为224x224
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        
        # 调用父类初始化方法，传入配置、覆盖参数和回调函数
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        # 从加载的数据集中设置模型的类别名称
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        # 创建并配置用于训练的分类模型
        
        # 使用ClassificationModel创建模型，传入配置和类别数量
        # verbose参数控制是否打印详细信息（仅在主进程时）
        model = ClassificationModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        
        # 如果提供了权重，加载预训练权重
        if weights:
            model.load(weights)

        # 遍历模型的所有模块进行配置
        for m in model.modules():
            # 如果不是预训练模型，重置模块参数
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            
            # 如果模块是Dropout且配置了dropout率，设置dropout概率
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        
        # 确保所有参数都可以进行梯度更新（训练模式）
        for p in model.parameters():
            p.requires_grad = True  # for training
        
        # 返回配置好的模型
        return model


    def setup_model(self):
        """Load, create or download model for any task."""
        # 加载、创建或下载模型的方法
        import torchvision  # scope for faster 'import ultralytics'
        # 导入torchvision，优化导入ultralytics的性能

        # 检查模型是否是torchvision内置模型
        if str(self.model) in torchvision.models.__dict__:
            # 如果是torchvision模型，根据是否使用预训练权重加载
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None  # 不需要检查点
        else:
            # 如果不是torchvision模型，调用父类的模型设置方法
            ckpt = super().setup_model()
        
        # 根据数据集的类别数重新调整模型输出层
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path, mode="train", batch=None):
        """Creates a ClassificationDataset instance given an image path, and mode (train/test etc.)."""
        # 根据图像路径和模式创建分类数据集实例
        # mode决定是否启用数据增强（训练模式下启用）
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
        # 创建并返回带有图像预处理变换的PyTorch数据加载器
        
        # 使用分布式训练的零号进程首先初始化数据集缓存
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode)

        # 构建数据加载器，支持分布式训练
        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
        
        # 为推理模式附加图像变换
        if mode != "train":
            # 处理并行模型和单一模型的情况
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        
        return loader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        # 预处理一批图像和类别标签
        # 将图像和类别标签移动到指定设备（GPU/CPU）
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        # 返回格式化的训练进度字符串
        # 包括轮次、GPU内存、损失名称、实例数和图像大小
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",       # 训练轮次
            "GPU_mem",     # GPU内存使用
            *self.loss_names,  # 损失函数名称
            "Instances",   # 训练实例数
            "Size",        # 图像大小
        )
        
    def get_validator(self):
        """Returns an instance of ClassificationValidator for validation."""
        # 返回分类验证器实例
        # 设置损失名称为"loss"
        self.loss_names = ["loss"]
        
        # 创建并返回ClassificationValidator实例
        # 使用测试加载器、保存目录、参数副本和回调函数
        return yolo.classify.ClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.
    
        Not needed for classification but necessary for segmentation & detection
        """
        # 返回带标签的训练损失项字典
        # 对于分类任务不是必需的，但对于分割和检测任务很重要
        
        # 为损失名称添加前缀（默认为"train"）
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        
        # 如果没有提供损失项，返回键列表
        if loss_items is None:
            return keys
        
        # 将损失项转换为四舍五入的浮点数
        loss_items = [round(float(loss_items), 5)]
        
        # 使用键和损失项创建字典
        return dict(zip(keys, loss_items))
    
    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        # 从CSV文件绘制指标
        # 使用plot_results函数，特别针对分类任务
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)  # save results.png
    
    def final_eval(self):
        """Evaluate trained model and save validation results."""
        # 评估训练好的模型并保存验证结果
        
        # 遍历最后一个和最佳模型检查点
        for f in self.last, self.best:
            # 如果检查点文件存在
            if f.exists():
                # 剥离优化器状态
                strip_optimizer(f)  # strip optimizers
                
                # 对最佳模型进行额外处理
                if f is self.best:
                    # 记录日志
                    LOGGER.info(f"\nValidating {f}...")
                    
                    # 设置验证器参数
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    
                    # 使用模型文件进行验证
                    self.metrics = self.validator(model=f)
                    
                    # 移除适应度指标（如果存在）
                    self.metrics.pop("fitness", None)
                    
                    # 运行回调函数
                    self.run_callbacks("on_fit_epoch_end")
    
    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        # 绘制带注释的训练样本
        plot_images(
            images=batch["img"],  # 输入图像批次
            batch_idx=torch.arange(len(batch["img"])),  # 批次索引
            cls=batch["cls"].view(-1),  # 类别标签
            # 警告：对于分类模型，使用.view()而不是.squeeze()
            fname=self.save_dir / f"train_batch{ni}.jpg",  # 保存文件名
            on_plot=self.on_plot,  # 绘图回调函数
        )
