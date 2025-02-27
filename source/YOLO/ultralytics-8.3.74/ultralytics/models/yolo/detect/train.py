# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """
    # 继承自BaseTrainer的目标检测模型训练器类
    # 专门用于YOLO目标检测模型的训练

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): [train](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/classify/train.py:316:4-326:9) mode or [val](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/classify/train.py:287:4-314:58) mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # 构建YOLO数据集
        
        # 计算模型的最大步长，确保最小为32
        gs = max(
            int(de_parallel(self.model).stride.max() if self.model else 0), 
            32
        )
        
        # 使用build_yolo_dataset构建数据集
        return build_yolo_dataset(
            self.args,        # 参数配置
            img_path,         # 图像路径
            batch,            # 批次大小
            self.data,        # 数据配置
            mode=mode,        # 模式（训练/验证）
            rect=mode == "val",  # 是否使用矩形训练（验证模式）
            stride=gs         # 步长
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        # 构建并返回数据加载器
        
        # 确保模式只能是训练或验证
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        
        # 使用分布式训练的零号进程首次初始化数据集缓存
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        # 确定是否需要打乱数据
        shuffle = mode == "train"
        
        # 如果使用矩形训练且需要打乱，发出警告并禁用打乱
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        
        # 设置工作进程数（验证模式下增加工作进程）
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        
        # 构建并返回数据加载器
        return build_dataloader(
            dataset,     # 数据集
            batch_size,  # 批次大小
            workers,     # 工作进程数
            shuffle,     # 是否打乱
            rank         # 分布式训练等级
        )

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        # 预处理图像批次
        
        # 将图像移动到指定设备并归一化（除以255）
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        
        # 如果启用多尺度训练
        if self.args.multi_scale:
            imgs = batch["img"]
            
            # 随机生成新的图像尺寸
            sz = (
                random.randrange(
                    int(self.args.imgsz * 0.5),      # 最小尺寸
                    int(self.args.imgsz * 1.5 + self.stride)  # 最大尺寸
                ) // self.stride * self.stride       # 确保尺寸是步长的倍数
            )
            
            # 计算缩放因子
            sf = sz / max(imgs.shape[2:])
            
            # 如果需要缩放
            if sf != 1:
                # 计算新的图像尺寸（确保是步长的倍数）
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride 
                    for x in imgs.shape[2:]
                ]
                
                # 使用双线性插值调整图像大小
                imgs = nn.functional.interpolate(
                    imgs, 
                    size=ns, 
                    mode="bilinear", 
                    align_corners=False
                )
            
            # 更新批次中的图像
            batch["img"] = imgs
        
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # 设置模型属性的方法
        # 注释部分是关于缩放超参数的可能实现（当前未启用）
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        
        # 设置模型的类别数量
        self.model.nc = self.data["nc"]
        
        # 设置模型的类别名称
        self.model.names = self.data["names"]
        
        # 将训练参数附加到模型
        self.model.args = self.args
        
        # TODO: 未来可能实现的类别权重计算
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        # 创建并返回YOLO检测模型
        
        # 使用DetectionModel创建模型
        # 传入配置、类别数量和详细信息标志
        model = DetectionModel(
            cfg, 
            nc=self.data["nc"],  # 类别数量
            verbose=verbose and RANK == -1  # 是否显示详细信息
        )
        
        # 如果提供权重，加载预训练权重
        if weights:
            model.load(weights)
        
        return model
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        # 返回YOLO模型验证器
        
        # 设置损失名称（边界框损失、类别损失、分布损失）
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        
        # 创建并返回检测验证器
        return yolo.detect.DetectionValidator(
            self.test_loader,        # 测试数据加载器
            save_dir=self.save_dir,  # 保存目录
            args=copy(self.args),    # 参数副本
            _callbacks=self.callbacks  # 回调函数
        )
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.
    
        Not needed for classification but necessary for segmentation & detection
        """
        # 为损失项添加标签
        # 对分类不需要，但对分割和检测任务很重要
        
        # 为损失名称添加前缀
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        
        # 如果提供损失项
        if loss_items is not None:
            # 将损失项转换为5位小数的浮点数
            loss_items = [round(float(x), 5) for x in loss_items]
            # 创建损失字典
            return dict(zip(keys, loss_items))
        else:
            # 如果没有损失项，返回键列表
            return keys
    
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        # 返回训练进度的格式化字符串
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",        # 训练轮次
            "GPU_mem",      # GPU内存
            *self.loss_names,  # 损失名称
            "Instances",    # 训练实例数
            "Size",         # 图像大小
        )
    
    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        # 绘制带注释的训练样本
        plot_images(
            images=batch["img"],             # 输入图像
            batch_idx=batch["batch_idx"],    # 批次索引
            cls=batch["cls"].squeeze(-1),    # 类别标签
            bboxes=batch["bboxes"],          # 边界框
            paths=batch["im_file"],          # 图像文件路径
            fname=self.save_dir / f"train_batch{ni}.jpg",  # 保存文件名
            on_plot=self.on_plot,            # 绘图回调函数
        )
    
    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        # 从CSV文件绘制指标
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png
    
    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        # 创建YOLO模型的标签训练图
        
        # 收集所有边界框
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        
        # 收集所有类别
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        
        # 绘制标签分布图
        plot_labels(
            boxes, 
            cls.squeeze(), 
            names=self.data["names"],    # 类别名称
            save_dir=self.save_dir,      # 保存目录
            on_plot=self.on_plot         # 绘图回调函数
        )
    
    def auto_batch(self):
        """Get batch size by calculating memory occupation of model."""
        # 通过计算模型内存占用来获取批次大小
        
        # 构建训练数据集
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        
        # 计算最大对象数（考虑马赛克增强）
        # 4 for mosaic augmentation
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        
        # 调用父类方法自动确定批次大小
        return super().auto_batch(max_num_obj)