# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    # 继承自BasePredictor的目标检测模型预测器类
    # 专门用于YOLO目标检测模型的预测

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-processes predictions and returns a list of Results objects."""
        # 后处理预测结果并返回Results对象列表
        
        # 使用非极大值抑制（NMS）处理预测结果
        preds = ops.non_max_suppression(
            preds,                      # 原始预测结果
            self.args.conf,             # 置信度阈值
            self.args.iou,              # IoU阈值
            self.args.classes,          # 选定的类别
            self.args.agnostic_nms,     # 是否使用类别无关的NMS
            max_det=self.args.max_det,  # 每张图最大检测数
            nc=len(self.model.names),   # 类别数量
            end2end=getattr(self.model, "end2end", False),  # 是否为端到端模型
            rotated=self.args.task == "obb"  # 是否处理旋转边界框
        )

        # 如果原始图像不是列表，转换为NumPy批次
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # 构建并返回结果
        return self.construct_results(preds, img, orig_imgs, **kwargs)

    def construct_results(self, preds, img, orig_imgs):
        """
        Constructs a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list): List of result objects containing the original images, image paths, class names, and bounding boxes.
        """
        # 从预测结果构建结果对象列表
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Constructs the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes and scores.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.

        Returns:
            (Results): The result object containing the original image, image path, class names, and bounding boxes.
        """
        # 构建单个图像的结果对象
        
        # 将预测的边界框坐标从预处理图像尺寸缩放到原始图像尺寸
        pred[:, :4] = ops.scale_boxes(
            img.shape[2:],       # 预处理图像尺寸
            pred[:, :4],         # 预测的边界框坐标
            orig_img.shape       # 原始图像尺寸
        )
        
        # 创建并返回Results对象
        return Results(
            orig_img,                # 原始图像
            path=img_path,           # 图像路径
            names=self.model.names,  # 类别名称
            boxes=pred[:, :6]        # 边界框（前6列：坐标、置信度、类别）
        )