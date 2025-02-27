# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.
    一个扩展了 DetectionPredictor 类的类，用于基于分割模型的预测。

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolo11n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks.
        使用提供的配置、覆盖和回调初始化 SegmentationPredictor。"""
        super().__init__(cfg, overrides, _callbacks)  # 调用父类的初始化方法
        self.args.task = "segment"  # 设置任务类型为分割

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch.
        对输入批次中的每个图像应用非极大值抑制并处理检测结果。"""
        # tuple if PyTorch model or array if exported
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # 根据 preds 的类型获取原型
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)  # 调用父类的后处理方法

    def construct_results(self, preds, img, orig_imgs, protos):
        """
        Constructs a list of result objects from the predictions.
        从预测结果构建结果对象的列表。

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            preds（List[torch.Tensor]）：预测的边界框、分数和掩码的列表。
            img (torch.Tensor): The image after preprocessing.
            img（torch.Tensor）：经过预处理的图像。
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.
            orig_imgs（List[np.ndarray]）：预处理前的原始图像列表。
            protos (List[torch.Tensor]): List of prototype masks.
            protos（List[torch.Tensor]）：原型掩码的列表。

        Returns:
            (list): List of result objects containing the original images, image paths, class names, bounding boxes, and masks.
            (list)：包含原始图像、图像路径、类名、边界框和掩码的结果对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)  # 构建每个结果对象
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)  # 遍历所有预测结果
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """
        Constructs the result object from the prediction.
        从预测构建结果对象。

        Args:
            pred (np.ndarray): The predicted bounding boxes, scores, and masks.
            pred（np.ndarray）：预测的边界框、分数和掩码。
            img (torch.Tensor): The image after preprocessing.
            img（torch.Tensor）：经过预处理的图像。
            orig_img (np.ndarray): The original image before preprocessing.
            orig_img（np.ndarray）：预处理前的原始图像。
            img_path (str): The path to the original image.
            img_path（str）：原始图像的路径。
            proto (torch.Tensor): The prototype masks.
            proto（torch.Tensor）：原型掩码。

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and masks.
            (Results)：包含原始图像、图像路径、类名、边界框和掩码的结果对象。
        """
        if not len(pred):  # save empty boxes
            masks = None  # 如果没有预测结果，掩码设置为 None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)  # 缩放边界框
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)  # 缩放边界框
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)  # 返回结果对象