# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.
    重用工作线程的数据加载器。

    Uses same syntax as vanilla DataLoader.
    使用与普通DataLoader相同的语法。
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))  # 设置批次采样器为重复采样器
        self.iterator = super().__iter__()  # 创建迭代器

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)  # 返回批次采样器的长度

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):  # 无限循环
            yield next(self.iterator)  # 返回下一个元素

    def __del__(self):
        """Ensure that workers are terminated."""
        try:
            if not hasattr(self.iterator, "_workers"):  # 检查迭代器是否有_workers属性
                return
            for w in self.iterator._workers:  # force terminate 强制终止
                if w.is_alive():  # 如果工作线程仍在运行
                    w.terminate()  # 终止工作线程
            self.iterator._shutdown_workers()  # 清理工作线程
        except Exception:
            pass  # 忽略异常

    def reset(self):
        """
        Reset iterator.
        重置迭代器。

        This is useful when we want to modify settings of dataset while training.
        当我们想在训练过程中修改数据集的设置时，这很有用。
        """
        self.iterator = self._get_iterator()  # 获取新的迭代器


class _RepeatSampler:
    """
    Sampler that repeats forever.
    永久重复的采样器。

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler  # 设置要重复的采样器

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:  # 无限循环
            yield from iter(self.sampler)  # 从采样器中迭代并返回内容


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32  # 设置工作线程的种子
    np.random.seed(worker_seed)  # 设置numpy随机数种子
    random.seed(worker_seed)  # 设置Python随机数种子


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset  # 根据是否多模态选择数据集
    return dataset(
        img_path=img_path,  # 图像路径
        imgsz=cfg.imgsz,  # 图像大小
        batch_size=batch,  # 批次大小
        augment=mode == "train",  # augmentation 数据增强
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches 矩形批次
        cache=cfg.cache or None,  # 缓存设置
        single_cls=cfg.single_cls or False,  # 单类训练设置
        stride=int(stride),  # 步幅
        pad=0.0 if mode == "train" else 0.5,  # 填充设置
        prefix=colorstr(f"{mode}: "),  # 日志前缀
        task=cfg.task,  # 任务类型
        classes=cfg.classes,  # 类别
        data=data,  # 数据
        fraction=cfg.fraction if mode == "train" else 1.0,  # 使用的数据集比例
    )


def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return GroundingDataset(
        img_path=img_path,  # 图像路径
        json_file=json_file,  # JSON文件路径
        imgsz=cfg.imgsz,  # 图像大小
        batch_size=batch,  # 批次大小
        augment=mode == "train",  # 数据增强
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # 矩形批次
        cache=cfg.cache or None,  # 缓存设置
        single_cls=cfg.single_cls or False,  # 单类训练设置
        stride=int(stride),  # 步幅
        pad=0.0 if mode == "train" else 0.5,  # 填充设置
        prefix=colorstr(f"{mode}: "),  # 日志前缀
        task=cfg.task,  # 任务类型
        classes=cfg.classes,  # 类别
        fraction=cfg.fraction if mode == "train" else 1.0,  # 使用的数据集比例
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))  # 设置批次大小
    nd = torch.cuda.device_count()  # number of CUDA devices 获取CUDA设备数量
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers 获取工作线程数量
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)  # 设置采样器
    generator = torch.Generator()  # 创建随机数生成器
    generator.manual_seed(6148914691236517205 + RANK)  # 设置生成器种子
    return InfiniteDataLoader(
        dataset=dataset,  # 数据集
        batch_size=batch,  # 批次大小
        shuffle=shuffle and sampler is None,  # 是否打乱数据
        num_workers=nw,  # 工作线程数量
        sampler=sampler,  # 采样器
        pin_memory=PIN_MEMORY,  # 是否将数据固定在内存中
        collate_fn=getattr(dataset, "collate_fn", None),  # 获取合并函数
        worker_init_fn=seed_worker,  # 工作线程初始化函数
        generator=generator,  # 随机数生成器
    )


def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False  # 初始化标志
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)  # 转换为字符串
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)  # 检查是否为文件
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))  # 检查是否为URL
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)  # 检查是否为网络摄像头
        screenshot = source.lower() == "screen"  # 检查是否为屏幕截图
        if is_url and is_file:
            source = check_file(source)  # download 下载文件
    elif isinstance(source, LOADERS):
        in_memory = True  # 如果是LOADERS类型，设置为内存加载
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays 将列表中的所有元素转换为PIL或numpy数组
        from_img = True  # 设置为图像来源
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True  # 设置为图像来源
    elif isinstance(source, torch.Tensor):
        tensor = True  # 设置为张量来源
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")  # 抛出不支持的类型错误

    return source, webcam, screenshot, from_img, in_memory, tensor  # 返回源和标志


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.
    加载用于目标检测的推理源并应用必要的变换。

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        source (str, Path, Tensor, PIL.Image, np.ndarray): 用于推理的输入源。
        batch (int, optional): Batch size for dataloaders. Default is 1.
        batch (int, optional): 数据加载器的批次大小。默认为1。
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        vid_stride (int, optional): 视频源的帧间隔。默认为1。
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.
        buffer (bool, optional): 确定流帧是否会被缓冲。默认为False。

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
        dataset (Dataset): 指定输入源的数据集对象。
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)  # 检查源类型
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)  # 获取源类型

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)  # 如果是张量，加载张量数据集
    elif in_memory:
        dataset = source  # 如果在内存中，直接使用源
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)  # 加载流数据集
    elif screenshot:
        dataset = LoadScreenshots(source)  # 加载屏幕截图数据集
    elif from_img:
        dataset = LoadPilAndNumpy(source)  # 加载PIL和numpy数据集
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)  # 加载图像和视频数据集

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)  # 将源类型附加到数据集

    return dataset  # 返回数据集