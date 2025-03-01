# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import itertools

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import DEFAULT_CFG, RANK, checks
from ultralytics.utils.torch_utils import de_parallel


def on_pretrain_routine_end(trainer):
    """Callback.
    回调函数。"""
    if RANK in {-1, 0}:  # 如果当前进程是主进程
        # NOTE: for evaluation
        names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]  # 获取类别名称
        de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)  # 设置类别并禁用缓存
    device = next(trainer.model.parameters()).device  # 获取模型参数所在的设备
    trainer.text_model, _ = trainer.clip.load("ViT-B/32", device=device)  # 加载文本模型
    for p in trainer.text_model.parameters():
        p.requires_grad_(False)  # 冻结文本模型的参数

class WorldTrainer(yolo.detect.DetectionTrainer):
    """
    A class to fine-tune a world model on a close-set dataset.
    一个用于在闭集数据集上微调世界模型的类。

    Example:
        ```python
        from ultralytics.models.yolo.world import WorldModel

        args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
        trainer = WorldTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments.
        用给定参数初始化 WorldTrainer 对象。"""
        if overrides is None:
            overrides = {}  # 如果没有提供 overrides，则初始化为空字典
        super().__init__(cfg, overrides, _callbacks)  # 调用父类的初始化方法

        # Import and assign clip
        try:
            import clip  # 尝试导入 clip 库
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")  # 检查 clip 的依赖
            import clip  # 再次导入 clip 库
        self.clip = clip  # 将 clip 赋值给实例变量

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return WorldModel initialized with specified config and weights.
        返回使用指定配置和权重初始化的 WorldModel。"""
        # NOTE: This [nc](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/segment/val.py:271:8-276:22) here is the max number of different text samples in one image, rather than the actual [nc](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/segment/val.py:271:8-276:22).
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,  # 获取配置文件
            ch=3,  # 通道数
            nc=min(self.data["nc"], 80),  # 类别数量，最多为 80
            verbose=verbose and RANK == -1,  # 如果是主进程则输出详细信息
        )
        if weights:
            model.load(weights)  # 如果提供了权重，则加载权重
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)  # 添加回调函数

        return model  # 返回模型

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.
        构建 YOLO 数据集。

        Args:
            img_path (str): Path to the folder containing images.  # 图像所在文件夹的路径
            mode (str): [train](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/world/train_world.py:104:4-107:37) mode or [val](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/segment/val.py:296:4-324:42) mode, users are able to customize different augmentations for each mode.  # 训练模式或验证模式，用户可以为每种模式自定义不同的增强
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.  # 批次大小，适用于矩形模式
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)  # 获取模型的最大步幅
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"  # 构建 YOLO 数据集并返回
        )

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed.
        预处理 YOLOWorld 训练的图像批次，调整格式和尺寸。"""
        batch = super().preprocess_batch(batch)  # 调用父类的预处理方法

        # NOTE: add text features
        texts = list(itertools.chain(*batch["texts"]))  # 将文本特征展平
        text_token = self.clip.tokenize(texts).to(batch["img"].device)  # 对文本进行编码并移动到图像所在设备
        txt_feats = self.text_model.encode_text(text_token).to(dtype=batch["img"].dtype)  # 编码文本特征
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)  # 归一化文本特征
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])  # 将文本特征重塑为适当的形状
        return batch  # 返回处理后的批次
