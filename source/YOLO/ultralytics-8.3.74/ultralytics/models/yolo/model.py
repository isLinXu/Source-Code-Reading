# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import ROOT, yaml_load


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""
    # YOLO（你只看一次）目标检测模型

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        # 初始化YOLO模型，如果模型文件名包含'-world'则切换到YOLOWorld
        path = Path(model)  # 将模型路径转换为Path对象
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            # 如果是YOLOWorld PyTorch模型
            new_instance = YOLOWorld(path, verbose=verbose)  # 创建YOLOWorld的实例
            self.__class__ = type(new_instance)  # 将当前类的类型更改为new_instance的类型
            self.__dict__ = new_instance.__dict__  # 复制new_instance的属性字典
        else:
            # Continue with default YOLO initialization
            # 继续进行默认的YOLO初始化
            super().__init__(model=model, task=task, verbose=verbose)  # 调用父类的初始化方法

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        # 将头映射到模型、训练器、验证器和预测器类
        return {
            "classify": {
                "model": ClassificationModel,  # 分类模型
                "trainer": yolo.classify.ClassificationTrainer,  # 分类训练器
                "validator": yolo.classify.ClassificationValidator,  # 分类验证器
                "predictor": yolo.classify.ClassificationPredictor,  # 分类预测器
            },
            "detect": {
                "model": DetectionModel,  # 检测模型
                "trainer": yolo.detect.DetectionTrainer,  # 检测训练器
                "validator": yolo.detect.DetectionValidator,  # 检测验证器
                "predictor": yolo.detect.DetectionPredictor,  # 检测预测器
            },
            "segment": {
                "model": SegmentationModel,  # 分割模型
                "trainer": yolo.segment.SegmentationTrainer,  # 分割训练器
                "validator": yolo.segment.SegmentationValidator,  # 分割验证器
                "predictor": yolo.segment.SegmentationPredictor,  # 分割预测器
            },
            "pose": {
                "model": PoseModel,  # 姿态模型
                "trainer": yolo.pose.PoseTrainer,  # 姿态训练器
                "validator": yolo.pose.PoseValidator,  # 姿态验证器
                "predictor": yolo.pose.PosePredictor,  # 姿态预测器
            },
            "obb": {
                "model": OBBModel,  # 方向边界框模型
                "trainer": yolo.obb.OBBTrainer,  # 方向边界框训练器
                "validator": yolo.obb.OBBValidator,  # 方向边界框验证器
                "predictor": yolo.obb.OBBPredictor,  # 方向边界框预测器
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""
    # YOLO-World目标检测模型

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        # 使用预训练模型文件初始化YOLOv8-World模型
        super().__init__(model=model, task="detect", verbose=verbose)  # 调用父类的初始化方法

        # Assign default COCO class names when there are no custom names
        # 当没有自定义名称时，分配默认的COCO类名称
        if not hasattr(self.model, "names"):
            # 如果模型没有类名属性
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")  # 从yaml文件加载类名

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        # 将头映射到模型、验证器和预测器类
        return {
            "detect": {
                "model": WorldModel,  # 世界模型
                "validator": yolo.detect.DetectionValidator,  # 检测验证器
                "predictor": yolo.detect.DetectionPredictor,  # 检测预测器
                "trainer": yolo.world.WorldTrainer,  # 世界训练器
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        # 设置类
        self.model.set_classes(classes)  # 设置模型的类
        # Remove background if it's given
        # 如果给定了背景，则移除背景
        background = " "
        if background in classes:
            classes.remove(background)  # 从类列表中移除背景
        self.model.names = classes  # 更新模型的类名称

        # Reset method class names
        # 重置方法的类名称
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes  # 更新预测器的类名称