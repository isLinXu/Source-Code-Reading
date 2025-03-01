# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel


class WorldTrainerFromScratch(WorldTrainer):
    """
    A class extending the WorldTrainer class for training a world model from scratch on open-set dataset.
    一个扩展 WorldTrainer 类的类，用于在开放集数据集上从头开始训练世界模型。

    Example:
        ```python
        from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        from ultralytics import YOLOWorld

        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="../datasets/flickr30k/images",
                        json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
                    ),
                    dict(
                        img_path="../datasets/GQA/images",
                        json_file="../datasets/GQA/final_mixed_train_no_coco.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )

        model = YOLOWorld("yolov8s-worldv2.yaml")
        model.train(data=data, trainer=WorldTrainerFromScratch)
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments.
        用给定参数初始化 WorldTrainer 对象。"""
        if overrides is None:
            overrides = {}  # 如果没有提供 overrides，则初始化为空字典
        super().__init__(cfg, overrides, _callbacks)  # 调用父类的初始化方法

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.
        构建 YOLO 数据集。

        Args:
            img_path (List[str] | str): Path to the folder containing images.
            mode (str): [train](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/segment/train.py:56:4-68:9) mode or [val](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/segment/val.py:296:4-324:42) mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)  # 获取模型的最大步幅
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)  # 如果不是训练模式，构建数据集并返回
        dataset = [
            build_yolo_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)  # 构建 YOLO 数据集
            if isinstance(im_path, str)
            else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)  # 如果是 grounding 数据，构建相应的数据集
            for im_path in img_path
        ]
        return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]  # 如果有多个数据集，合并返回；否则返回单个数据集

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.
        如果数据字典中存在，获取训练和验证路径。

        Returns None if data format is not recognized.
        如果数据格式无法识别，则返回 None。
        """
        final_data = {}  # 初始化最终数据字典
        data_yaml = self.args.data  # 获取数据配置
        assert data_yaml.get("train", False), "train dataset not found"  # 确保训练数据集存在
        assert data_yaml.get("val", False), "validation dataset not found"  # 确保验证数据集存在
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}  # 检查并获取数据集
        assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."  # 确保只支持一个验证数据集
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"  # 根据数据集类型选择验证分割
        for d in data["val"]:
            if d.get("minival") is None:  # 对于 lvis 数据集
                continue
            d["minival"] = str(d["path"] / d["minival"])  # 设置最小验证集路径
        for s in ["train", "val"]:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]  # 获取训练和验证数据
            # save grounding data if there's one
            grounding_data = data_yaml[s].get("grounding_data")  # 获取 grounding 数据
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]  # 确保 grounding 数据为列表
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"  # 确保 grounding 数据为字典格式
            final_data[s] += grounding_data  # 将 grounding 数据添加到最终数据中
        # NOTE: to make training work properly, set [nc](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/segment/val.py:271:8-276:22) and `names`
        final_data["nc"] = data["val"][0]["nc"]  # 设置类别数量
        final_data["names"] = data["val"][0]["names"]  # 设置类别名称
        self.data = final_data  # 将最终数据赋值给 self.data
        return final_data["train"], final_data["val"][0]  # 返回训练和验证数据

    def plot_training_labels(self):
        """DO NOT plot labels.
        不要绘制标签。"""
        pass  # 不执行任何操作

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO-World model.
        对 YOLO-World 模型进行最终评估和验证。"""
        val = self.args.data["val"]["yolo_data"][0]  # 获取验证数据
        self.validator.args.data = val  # 设置验证器的数据
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"  # 设置验证分割
        return super().final_eval()  # 调用父类的最终评估方法
