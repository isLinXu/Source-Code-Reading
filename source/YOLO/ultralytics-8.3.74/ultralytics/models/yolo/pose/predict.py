# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops

class PosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a pose model.
    一个继承自DetectionPredictor类的预测器，专门用于基于姿态估计模型的预测。

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import PosePredictor

        args = dict(model="yolo11n-pose.pt", source=ASSETS)
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes PosePredictor, sets task to 'pose' and logs a warning for using 'mps' as device."""
        # 初始化PosePredictor，设置任务为'pose'并针对使用'mps'设备记录警告
        
        # 调用父类初始化方法
        super().__init__(cfg, overrides, _callbacks)
        
        # 强制设置任务为"pose"（姿态估计）
        self.args.task = "pose"
        
        # 检查是否使用Apple MPS设备
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            # 对于姿态估计模型，警告使用MPS可能存在已知问题
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Constructs the result object from the prediction.
        从预测结果构建结果对象。

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints.
                                 预测的边界框、置信度和关键点
            img (torch.Tensor): The image after preprocessing.
                                预处理后的图像
            orig_img (np.ndarray): The original image before preprocessing.
                                   预处理前的原始图像
            img_path (str): The path to the original image.
                            原始图像的路径

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and keypoints.
                       包含原始图像、图像路径、类别名称、边界框和关键点的结果对象
        """
        # 调用父类的结果构建方法，获取基本的检测结果
        result = super().construct_result(pred, img, orig_img, img_path)
        
        # 提取关键点预测
        # - 如果存在预测结果，从第6列开始提取关键点
        # - 使用模型的关键点形状重塑张量
        # - 如果没有预测结果，保持原样
        pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
        
        # 将关键点坐标缩放到原始图像尺寸
        # - img.shape[2:]: 预处理后图像的高度和宽度
        # - pred_kpts: 预测的关键点坐标
        # - orig_img.shape: 原始图像尺寸
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        
        # 更新结果对象，添加关键点信息
        result.update(keypoints=pred_kpts)
        
        # 返回包含关键点的结果对象
        return result