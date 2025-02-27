# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationValidator

        args = dict(model="yolo11n-cls.pt", data="imagenet10")
        validator = ClassificationValidator(args=args)
        validator()
        ```
    """
    # 继承自BaseValidator的分类模型验证器类
    # 支持YOLO和Torchvision的分类模型验证

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        # 初始化分类验证器实例
        # 调用父类初始化方法
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        # 初始化预测和目标为None
        self.targets = None
        self.pred = None
        
        # 强制设置任务类型为分类
        self.args.task = "classify"
        
        # 初始化分类指标
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        # 返回格式化的分类指标摘要字符串
        # 显示类别数、Top-1准确率和Top-5准确率
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""
        # 初始化评估指标
        
        # 获取模型的类别名称
        self.names = model.names
        
        # 获取类别数量
        self.nc = len(model.names)
        
        # 创建混淆矩阵
        self.confusion_matrix = ConfusionMatrix(
            nc=self.nc,           # 类别数量
            conf=self.args.conf,  # 置信度阈值
            task="classify"       # 任务类型
        )
        
        # 初始化预测和目标列表
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        # 预处理输入批次
        
        # 将图像移动到指定设备（非阻塞）
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        
        # 根据参数选择半精度或全精度
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        
        # 将类别标签移动到指定设备
        batch["cls"] = batch["cls"].to(self.device)
        
        return batch

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        # 使用模型预测和批次目标更新运行指标
        
        # 选择前5个类别（不超过总类别数）
        n5 = min(len(self.names), 5)
        
        # 记录预测结果（按降序排列的前n5个类别索引）
        self.pred.append(
            preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu()
        )
        
        # 记录目标类别
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        # 完成模型指标的最终计算
        
        # 处理分类预测结果
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        
        # 如果启用绘图
        if self.args.plots:
            # 绘制归一化和非归一化的混淆矩阵
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,       # 保存目录
                    names=self.names.values(),    # 类别名称
                    normalize=normalize,          # 是否归一化
                    on_plot=self.on_plot          # 绘图回调函数
                )
        
        # 设置指标的其他属性
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir


    def postprocess(self, preds):
        """Preprocesses the classification predictions."""
        # 后处理分类预测结果
        # 如果预测结果是列表或元组，返回第一个元素；否则直接返回
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        # 获取通过处理目标和预测结果计算的指标字典
        # 使用metrics对象处理目标和预测结果
        self.metrics.process(self.targets, self.pred)
        
        # 返回结果字典
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        """Creates and returns a ClassificationDataset instance using given image path and preprocessing parameters."""
        # 使用给定的图像路径和预处理参数创建分类数据集实例
        return ClassificationDataset(
            root=img_path,           # 图像根路径
            args=self.args,          # 参数配置
            augment=False,           # 禁用数据增强（验证阶段）
            prefix=self.args.split   # 数据集拆分前缀（如train/val/test）
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        # 构建并返回分类任务的数据加载器
        
        # 创建数据集
        dataset = self.build_dataset(dataset_path)
        
        # 使用build_dataloader构建数据加载器
        return build_dataloader(
            dataset,                 # 数据集
            batch_size,              # 批次大小
            self.args.workers,       # 工作进程数
            rank=-1                  # 分布式训练等级（-1表示非分布式）
        )

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""
        # 打印评估指标
        
        # 定义打印格式：22个字符宽度的字符串 + 11位小数的指标
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        
        # 使用日志记录器打印所有样本的Top-1和Top-5准确率
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        # 绘制验证图像样本
        plot_images(
            images=batch["img"],                 # 输入图像批次
            batch_idx=torch.arange(len(batch["img"])),  # 批次索引
            cls=batch["cls"].view(-1),           # 类别标签（使用.view()避免维度问题）
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",  # 保存文件名
            names=self.names,                    # 类别名称
            on_plot=self.on_plot,                # 绘图回调函数
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        # 绘制预测结果
        plot_images(
            batch["img"],                        # 输入图像批次
            batch_idx=torch.arange(len(batch["img"])),  # 批次索引
            cls=torch.argmax(preds, dim=1),      # 预测的最高概率类别
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",  # 保存文件名
            names=self.names,                    # 类别名称
            on_plot=self.on_plot,                # 绘图回调函数
        )
