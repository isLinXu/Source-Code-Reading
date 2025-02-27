# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.
    一个继承自DetectionTrainer类的训练器，专门用于基于带方向的边界框（OBB）模型的训练。

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBTrainer

        args = dict(model="yolo11n-obb.pt", data="dota8.yaml", epochs=3)
        trainer = OBBTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        # 使用给定参数初始化OBBTrainer对象
        
        # 如果未提供覆盖参数，则初始化为空字典
        if overrides is None:
            overrides = {}
        
        # 强制设置任务为"obb"（带方向边界框）
        overrides["task"] = "obb"
        
        # 调用父类初始化方法
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        # 返回使用指定配置和权重初始化的OBBModel
        
        # 创建OBBModel实例
        # - cfg: 模型配置
        # - ch: 输入通道数（默认3，RGB图像）
        # - nc: 类别数（从数据配置中获取）
        # - verbose: 是否打印详细信息（仅在主进程时）
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        
        # 如果提供了权重，加载预训练权重
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        # 返回用于YOLO模型验证的OBBValidator实例
        
        # 设置损失函数名称（针对带方向边界框的特定损失）
        # - box_loss: 边界框回归损失
        # - cls_loss: 类别分类损失
        # - dfl_loss: 分布式边界框损失
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        
        # 创建并返回OBBValidator实例
        # - self.test_loader: 测试数据加载器
        # - save_dir: 保存结果的目录
        # - args: 训练参数（使用深拷贝避免意外修改）
        # - _callbacks: 回调函数
        return yolo.obb.OBBValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args), 
            _callbacks=self.callbacks
        )