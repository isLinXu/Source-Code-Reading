# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results

class PoseTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a pose model.
    一个继承自DetectionTrainer类的训练器，专门用于基于姿态估计模型的训练。

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseTrainer

        args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        trainer = PoseTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        # 使用指定的配置和覆盖参数初始化PoseTrainer对象
        
        # 如果未提供覆盖参数，则初始化为空字典
        if overrides is None:
            overrides = {}
        
        # 强制设置任务为"pose"（姿态估计）
        overrides["task"] = "pose"
        
        # 调用父类初始化方法
        super().__init__(cfg, overrides, _callbacks)

        # 检查是否使用Apple MPS设备
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            # 对于姿态估计模型，警告使用MPS可能存在已知问题
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        # 获取具有指定配置和权重的姿态估计模型
        
        # 创建PoseModel实例
        # - cfg: 模型配置
        # - ch: 输入通道数（默认3，RGB图像）
        # - nc: 类别数（从数据配置中获取）
        # - data_kpt_shape: 关键点形状（从数据配置中获取）
        # - verbose: 是否打印详细信息
        model = PoseModel(
            cfg, 
            ch=3, 
            nc=self.data["nc"], 
            data_kpt_shape=self.data["kpt_shape"], 
            verbose=verbose
        )
        
        # 如果提供了权重，加载预训练权重
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        # 设置PoseModel的关键点形状属性
        
        # 调用父类的模型属性设置方法
        super().set_model_attributes()
        
        # 设置模型的关键点形状
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        # 返回用于验证的PoseValidator实例
        
        # 设置损失函数名称（针对姿态估计的特定损失）
        # - box_loss: 边界框回归损失
        # - pose_loss: 姿态估计损失
        # - kobj_loss: 关键点目标损失
        # - cls_loss: 类别分类损失
        # - dfl_loss: 分布式边界框损失
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        
        # 创建并返回PoseValidator实例
        # - self.test_loader: 测试数据加载器
        # - save_dir: 保存结果的目录
        # - args: 训练参数（使用深拷贝避免意外修改）
        # - _callbacks: 回调函数
        return yolo.pose.PoseValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args), 
            _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        # 绘制带有注释类别标签、边界框和关键点的训练样本批次
        
        # 提取批次中的图像
        images = batch["img"]
        
        # 提取关键点
        kpts = batch["keypoints"]
        
        # 提取类别标签
        cls = batch["cls"].squeeze(-1)
        
        # 提取边界框
        bboxes = batch["bboxes"]
        
        # 提取图像文件路径
        paths = batch["im_file"]
        
        # 提取批次索引
        batch_idx = batch["batch_idx"]
        
        # 绘制图像
        plot_images(
            images,             # 图像
            batch_idx,          # 批次索引
            cls,                # 类别标签
            bboxes,             # 边界框
            kpts=kpts,          # 关键点
            paths=paths,        # 图像文件路径
            fname=self.save_dir / f"train_batch{ni}.jpg",  # 保存文件名
            on_plot=self.on_plot  # 绘图回调函数
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        # 绘制训练/验证指标
        
        # 使用plot_results函数绘制结果
        # - file: CSV文件路径
        # - pose: 表示绘制姿态估计相关指标
        # - on_plot: 绘图回调函数
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)
