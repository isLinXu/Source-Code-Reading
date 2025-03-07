# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.
    基础数据集类，用于加载和处理图像数据。

    Args:
        img_path (str): Path to the folder containing images.
        img_path (str): 包含图像的文件夹路径。
        imgsz (int, optional): Image size. Defaults to 640.
        imgsz (int, optional): 图像大小。默认为640。
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        cache (bool, optional): 在训练期间将图像缓存到RAM或磁盘。默认为False。
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        augment (bool, optional): 如果为True，则应用数据增强。默认为True。
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        hyp (dict, optional): 应用数据增强的超参数。默认为None。
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        prefix (str, optional): 在日志消息中打印的前缀。默认为''。
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        rect (bool, optional): 如果为True，则使用矩形训练。默认为False。
        batch_size (int, optional): Size of batches. Defaults to None.
        batch_size (int, optional): 批次大小。默认为None。
        stride (int, optional): Stride. Defaults to 32.
        stride (int, optional): 步幅。默认为32。
        pad (float, optional): Padding. Defaults to 0.0.
        pad (float, optional): 填充。默认为0.0。
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        single_cls (bool, optional): 如果为True，则使用单类训练。默认为False。
        classes (list): List of included classes. Default is None.
        classes (list): 包含的类别列表。默认为None。
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).
        fraction (float): 要使用的数据集的比例。默认为1.0（使用所有数据）。

    Attributes:
        im_files (list): List of image file paths.
        im_files (list): 图像文件路径列表。
        labels (list): List of label data dictionaries.
        labels (list): 标签数据字典列表。
        ni (int): Number of images in the dataset.
        ni (int): 数据集中图像的数量。
        ims (list): List of loaded images.
        ims (list): 加载的图像列表。
        npy_files (list): List of numpy file paths.
        npy_files (list): numpy文件路径列表。
        transforms (callable): Image transformation function.
        transforms (callable): 图像变换函数。
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.img_path = img_path  # 设置图像路径
        self.imgsz = imgsz  # 设置图像大小
        self.augment = augment  # 设置是否应用数据增强
        self.single_cls = single_cls  # 设置是否使用单类训练
        self.prefix = prefix  # 设置日志前缀
        self.fraction = fraction  # 设置数据集使用比例
        self.im_files = self.get_img_files(self.img_path)  # 获取图像文件路径
        self.labels = self.get_labels()  # 获取标签数据
        self.update_labels(include_class=classes)  # 更新标签以包含指定类别（单类和包含类别）
        self.ni = len(self.labels)  # 图像数量
        self.rect = rect  # 设置是否使用矩形训练
        self.batch_size = batch_size  # 设置批次大小
        self.stride = stride  # 设置步幅
        self.pad = pad  # 设置填充
        if self.rect:
            assert self.batch_size is not None  # 确保批次大小不为None
            self.set_rectangle()  # 设置矩形训练

        # Buffer thread for mosaic images
        self.buffer = []  # 缓冲区大小 = 批次大小
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0  # 最大缓冲区长度

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni  # 初始化图像缓存
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]  # 获取numpy文件路径
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None  # 设置缓存类型
        if self.cache == "ram" and self.check_cache_ram():  # 检查RAM缓存
            if hyp.deterministic:
                LOGGER.warning(
                    "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )  # 警告：使用RAM缓存可能导致非确定性训练结果
            self.cache_images()  # 缓存图像到RAM
        elif self.cache == "disk" and self.check_cache_disk():  # 检查磁盘缓存
            self.cache_images()  # 缓存图像到磁盘

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)  # 构建图像变换

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)  # 递归获取文件
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()  # 读取文件内容
                        parent = str(p.parent) + os.sep  # 获取父目录
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # 本地路径转换为全局路径
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")  # 文件不存在错误
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)  # 获取有效图像文件
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"  # 确保找到图像
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e  # 数据加载错误
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # 保留数据集的一部分
        return im_files  # 返回图像文件列表

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)  # 转换为数组
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]  # 获取类别
                bboxes = self.labels[i]["bboxes"]  # 获取边界框
                segments = self.labels[i]["segments"]  # 获取分段
                keypoints = self.labels[i]["keypoints"]  # 获取关键点
                j = (cls == include_class_array).any(1)  # 检查类别是否在包含类中
                self.labels[i]["cls"] = cls[j]  # 更新类别
                self.labels[i]["bboxes"] = bboxes[j]  # 更新边界框
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]  # 更新分段
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]  # 更新关键点
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0  # 单类训练时将类设置为0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]  # 获取图像和文件路径
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)  # 加载numpy文件
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")  # 警告：移除损坏的npy文件
                    Path(fn).unlink(missing_ok=True)  # 删除文件
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")  # 图像未找到错误

            h0, w0 = im.shape[:2]  # orig hw 获取原始高度和宽度
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio 计算比例
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))  # 计算调整后的宽高
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)  # 调整图像大小
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)  # 调整图像大小为正方形

            # Add to buffer if training with augmentations
            if self.augment:  # 如果使用增强
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)  # 将索引添加到缓冲区
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer 防止缓冲区为空
                    j = self.buffer.pop(0)  # 移除第一个元素
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None  # 清空缓存

            return im, (h0, w0), im.shape[:2]  # 返回图像和尺寸

        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # 返回缓存的图像和尺寸

    def cache_images(self):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")  # 设置缓存函数和存储类型
        with ThreadPool(NUM_THREADS) as pool:  # 使用线程池
            results = pool.imap(fcn, range(self.ni))  # 并行加载图像
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)  # 进度条
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size  # 更新缓存大小
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes  # 更新缓存大小
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"  # 更新进度条描述
            pbar.close()  # 关闭进度条

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]  # 获取numpy文件路径
        if not f.exists():  # 如果文件不存在
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)  # 保存图像为npy文件

    def check_cache_disk(self, safety_margin=0.5):
        """Check image caching requirements vs available disk space."""
        import shutil  # 导入shutil模块

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)  # 随机选择图像文件
            im = cv2.imread(im_file)  # 读取图像
            if im is None:
                continue  # 如果图像为空，跳过
            b += im.nbytes  # 更新缓存大小
            if not os.access(Path(im_file).parent, os.W_OK):  # 检查目录是否可写
                self.cache = None
                LOGGER.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")  # 警告：目录不可写
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)  # 获取磁盘使用情况
        if disk_required > free:  # 如果所需磁盘空间超过可用空间
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
            )  # 警告：磁盘空间不足
            return False
        return True  # 磁盘检查通过

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue  # 如果图像为空，跳过
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio 计算比例
            b += im.nbytes * ratio**2  # 更新缓存大小
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()  # 获取内存使用情况
        if mem_required > mem.available:  # 如果所需内存超过可用内存
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )  # 警告：内存不足
            return False
        return True  # 内存检查通过

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio 计算纵横比
        irect = ar.argsort()  # 获取排序索引
        self.im_files = [self.im_files[i] for i in irect]  # 根据排序索引更新图像文件
        self.labels = [self.labels[i] for i in irect]  # 根据排序索引更新标签
        ar = ar[irect]  # 更新纵横比

        # Set training image shapes
        shapes = [[1, 1]] * nb  # 初始化训练图像形状
        for i in range(nb):
            ari = ar[bi == i]  # 获取当前批次的纵横比
            mini, maxi = ari.min(), ari.max()  # 获取最小和最大纵横比
            if maxi < 1:
                shapes[i] = [maxi, 1]  # 设置形状
            elif mini > 1:
                shapes[i] = [1, 1 / mini]  # 设置形状

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride  # 计算批次形状
        self.batch = bi  # 设置图像的批次索引

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))  # 返回变换后的标签信息

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)  # 获取图像和尺寸
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation 用于评估
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]  # 设置矩形形状
        return self.update_labels_info(label)  # 返回更新后的标签信息

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)  # 返回标签列表的长度

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label  # 返回标签

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.
        用户可以在此处自定义增强。

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError  # 抛出未实现错误

    def get_labels(self):
        """
        Users can customize their own format here.
        用户可以在此处自定义自己的格式。

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError  # 抛出未实现错误
