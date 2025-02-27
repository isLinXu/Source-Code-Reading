# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import torch
from PIL import Image

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.classify import ClassificationPredictor

        args = dict(model="yolo11n-cls.pt", source=ASSETS)
        predictor = ClassificationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    # 继承自BasePredictor的分类模型预测器类
    # 支持YOLO和Torchvision的分类模型预测

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes ClassificationPredictor setting the task to 'classify'."""
        # 初始化分类预测器
        # 调用父类初始化方法
        super().__init__(cfg, overrides, _callbacks)
        
        # 强制设置任务类型为分类
        self.args.task = "classify"
        
        # 设置遗留转换名称（用于兼容旧版本的图像转换）
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        # 将输入图像转换为模型兼容的数据类型
        
        # 如果输入不是张量，需要进行预处理
        if not isinstance(img, torch.Tensor):
            # 检查是否使用遗留的图像转换方法
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            
            if is_legacy_transform:  # 处理遗留转换
                # 直接应用transforms到输入图像
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                # 将图像从BGR转换到RGB，然后应用transforms
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
        
        # 确保图像是张量并移动到模型设备
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        
        # 根据模型是否使用半精度浮点数进行类型转换
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        # 后处理预测结果，返回Results对象
        
        # 如果原始图像不是列表，转换为NumPy批次
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # 处理预测结果（确保是单个张量）
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        
        # 为每个图像创建Results对象
        return [
            Results(
                orig_img,               # 原始图像
                path=img_path,          # 图像路径
                names=self.model.names,  # 类别名称
                probs=pred              # 预测概率
            )
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
