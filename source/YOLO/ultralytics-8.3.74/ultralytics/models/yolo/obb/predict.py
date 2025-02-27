# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.
    一个继承自DetectionPredictor类的预测器，专门用于基于带方向边界框（OBB）模型的预测。

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model="yolo11n-obb.pt", source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes OBBPredictor with optional model and data configuration overrides."""
        # 使用可选的模型和数据配置覆盖参数初始化OBBPredictor
        
        # 调用父类初始化方法
        super().__init__(cfg, overrides, _callbacks)
        
        # 强制设置任务为"obb"（带方向边界框）
        self.args.task = "obb"

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Constructs the result object from the prediction.
        从预测结果构建结果对象。

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and rotation angles.
                                 预测的边界框、置信度和旋转角度
            img (torch.Tensor): The image after preprocessing.
                                预处理后的图像
            orig_img (np.ndarray): The original image before preprocessing.
                                   预处理前的原始图像
            img_path (str): The path to the original image.
                            原始图像的路径

        Returns:
            (Results): The result object containing the original image, image path, class names, and oriented bounding boxes.
                       包含原始图像、图像路径、类别名称和带方向边界框的结果对象
        """
        # 规范化旋转边界框
        # 1. 从预测结果中提取边界框坐标（前4列）和旋转角度（最后一列）
        # 2. 使用regularize_rboxes函数标准化边界框
        rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
        
        # 缩放边界框到原始图像尺寸
        # - img.shape[2:]: 预处理后图像的高度和宽度
        # - rboxes[:, :4]: 规范化后的边界框坐标
        # - orig_img.shape: 原始图像尺寸
        # - xywh=True: 使用XYWH（中心x、中心y、宽度、高度）格式
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        
        # 组合旋转边界框和置信度、类别信息
        # - rboxes: 缩放后的边界框
        # - pred[:, 4:6]: 置信度和类别信息
        # 使用torch.cat在最后一个维度上拼接张量
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        
        # 创建并返回Results对象
        # - orig_img: 原始图像
        # - path: 图像路径
        # - names: 类别名称
        # - obb: 带方向的边界框结果
        return Results(orig_img, path=img_path, names=self.model.names, obb=obb)
