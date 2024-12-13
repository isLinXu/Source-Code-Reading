#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
from io import UnsupportedOperation
import os
import os.path as osp
import random
import json
import time
import hashlib
from pathlib import Path

from multiprocessing.pool import Pool

import cv2
import numpy as np
from tqdm import tqdm
from PIL import ExifTags, Image, ImageOps

import torch
from torch.utils.data import Dataset
import torch.distributed as dist

from .data_augment import (
    augment_hsv,
    letterbox,
    mixup,
    random_affine,
    mosaic_augmentation,
)
from yolov6.utils.events import LOGGER
import copy
import psutil
from multiprocessing.pool import ThreadPool


# Parameters
# 定义支持的图像格式
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
# 定义支持的视频格式
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
# 将图像格式转换为大写并添加到列表中
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
# 将视频格式转换为大写并添加到列表中
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])

# Get orientation exif tag
# 获取方向的exif标签
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k # 如果标签值为"Orientation"，则将其键赋值给ORIENTATION
        break

# 定义一个函数，将图像路径转换为标签路径
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    # 根据图像路径定义标签路径
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings 子字符串
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths] # 返回标签路径列表

# 定义训练和验证数据集类
class TrainValDataset(Dataset):
    '''YOLOv6 train_loader/val_loader, loads images and labels for training and validation.'''
    def __init__(
        self,
        img_dir,  # 图像目录
        img_size=640,  # 图像大小，默认640
        batch_size=16,  # 批次大小，默认16
        augment=False,  # 是否进行数据增强，默认不增强
        hyp=None,  # 超参数，默认为None
        rect=False,  # 是否使用矩形图像，默认不使用
        check_images=False,  # 是否检查图像，默认不检查
        check_labels=False,  # 是否检查标签，默认不检查
        stride=32,  # 步幅，默认32
        pad=0.0,  # 填充，默认0.0
        rank=-1,  # 当前进程的排名，默认-1
        data_dict=None,  # 数据字典，默认为None
        task="train",  # 任务类型，默认是训练
        specific_shape=False,  # 是否使用特定形状，默认不使用
        height=1088,  # 目标高度，默认1088
        width=1920,  # 目标宽度，默认1920
        cache_ram=False  # 是否缓存图像到内存，默认不缓存
    ):
        # 确保任务类型是支持的类型
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        tik = time.time()  # 记录初始化开始时间
        self.__dict__.update(locals())  # 更新实例字典，存储所有参数
        self.main_process = self.rank in (-1, 0)  # 判断是否为主进程
        self.task = self.task.capitalize()  # 将任务名称首字母大写
        self.class_names = data_dict["names"]  # 获取类名
        self.img_paths, self.labels = self.get_imgs_labels(self.img_dir)  # 获取图像路径和标签
        self.rect = rect  # 设置矩形标志
        self.specific_shape = specific_shape  # 设置特定形状标志
        self.target_height = height  # 设置目标高度
        self.target_width = width  # 设置目标宽度
        self.cache_ram = cache_ram  # 设置内存缓存标志

        if self.rect:  # 如果使用矩形图像
            shapes = [self.img_info[p]["shape"] for p in self.img_paths]  # 获取所有图像的形状
            self.shapes = np.array(shapes, dtype=np.float64)  # 将形状转换为NumPy数组
            if dist.is_initialized():  # 如果分布式训练已初始化
                # 在DDP模式下，我们需要确保每个batch_size * gpu_num内的所有图像
                # 都被调整大小并填充为相同的形状。
                sample_batch_size = self.batch_size * dist.get_world_size()  # 计算样本批次大小
            else:
                sample_batch_size = self.batch_size  # 否则使用默认批次大小
            self.batch_indices = np.floor(
                np.arange(len(shapes)) / sample_batch_size
            ).astype(
                np.int_
            )  # 计算每个图像的批次索引

            self.sort_files_shapes()  # 对文件和形状进行排序
 
        if self.cache_ram:  # 如果启用了内存缓存
            self.num_imgs = len(self.img_paths)  # 获取图像数量
            self.imgs, self.imgs_hw0, self.imgs_hw = [None] * self.num_imgs, [None] * self.num_imgs, [None] * self.num_imgs  # 初始化图像缓存
            self.cache_images(num_imgs=self.num_imgs)  # 缓存图像

        tok = time.time()  # 记录初始化结束时间

        if self.main_process:  # 如果是主进程
            LOGGER.info(f"%.1fs for dataset initialization." % (tok - tik))  # 记录数据集初始化耗时
    
# 缓存图像的方法
def cache_images(self, num_imgs=None):
    # 确保指定图像数量
    assert num_imgs is not None, "num_imgs must be specified as the size of the dataset"  

    # 获取虚拟内存信息
    mem = psutil.virtual_memory()  
    # 计算所需内存
    mem_required = self.cal_cache_occupy(num_imgs)  
    # 1GB的字节数
    gb = 1 << 30  

    # 如果所需内存大于可用内存
    if mem_required > mem.available:  
        self.cache_ram = False  # 禁用内存缓存
        # 记录警告信息
        LOGGER.warning("Not enough RAM to cache images, caching is disabled.")  
    else:
        # 记录内存使用情况
        LOGGER.warning(
            f"{mem_required / gb:.1f}GB RAM required, "
            f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB RAM available, "
            f"Since the first thing we do is cache, "
            f"there is no guarantee that the remaining memory space is sufficient"
        )  

    # 打印缓存图像数量
    print(f"self.imgs: {len(self.imgs)}")  
    # 记录使用内存缓存的日志
    LOGGER.info("You are using cached images in RAM to accelerate training!")  
    # 记录缓存图像的日志
    LOGGER.info(
        "Caching images...\n"
        "This might take some time for your dataset"
    )  
    # 设置线程数量
    num_threads = min(16, max(1, os.cpu_count() - 1))  
    # 使用线程池加载图像
    load_imgs = ThreadPool(num_threads).imap(self.load_image, range(num_imgs))  
    # 创建进度条
    pbar = tqdm(enumerate(load_imgs), total=num_imgs, disable=self.rank > 0)  
    # 遍历加载的图像
    for i, (x, (h0, w0), shape) in pbar:
        # 将加载的图像和其尺寸信息存储到缓存中
        self.imgs[i], self.imgs_hw0[i], self.imgs_hw[i] = x, (h0, w0), shape  

    def __del__(self):
        if self.cache_ram:
            del self.imgs  # 删除缓存的图像数据，以释放内存

    def cal_cache_occupy(self, num_imgs):
        '''estimate the memory required to cache images in RAM.'''
        # 估算在RAM中缓存图像所需的内存
        cache_bytes = 0  # 初始化缓存字节数
        num_imgs = len(self.img_paths)  # 获取图像路径的数量
        num_samples = min(num_imgs, 32)  # 取样本数，最多为32
        for _ in range(num_samples):
            img, _, _ = self.load_image(index=random.randint(0, len(self.img_paths) - 1))  # 随机加载图像
            cache_bytes += img.nbytes  # 累加图像的字节数
        mem_required = cache_bytes * num_imgs / num_samples  # 计算所需内存
        return mem_required  # 返回估算的内存需求

    def __len__(self):
        """Get the length of dataset"""
        # 获取数据集的长度
        return len(self.img_paths)  # 返回图像路径的数量

    def __getitem__(self, index):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        # 根据给定的索引获取数据样本。训练时应用马赛克和混合增强，验证时应用信箱增强
        target_shape = (
            (self.target_height, self.target_width) if self.specific_shape else
            self.batch_shapes[self.batch_indices[index]] if self.rect
            else self.img_size
        )  # 确定目标形状

        # Mosaic Augmentation
        if self.augment and random.random() < self.hyp["mosaic"]:
            img, labels = self.get_mosaic(index, target_shape)  # 获取马赛克增强的图像和标签
            shapes = None

            # MixUp augmentation
            if random.random() < self.hyp["mixup"]:
                img_other, labels_other = self.get_mosaic(
                    random.randint(0, len(self.img_paths) - 1), target_shape
                )  # 随机获取另一张马赛克增强的图像和标签
                img, labels = mixup(img, labels, img_other, labels_other)  # 应用混合增强

        else:
            # Load image
            if self.hyp and "shrink_size" in self.hyp:
                img, (h0, w0), (h, w) = self.load_image(index, self.hyp["shrink_size"])  # 加载图像，考虑缩小尺寸
            else:
                img, (h0, w0), (h, w) = self.load_image(index)  # 加载图像

            # letterbox
            img, ratio, pad = letterbox(img, target_shape, auto=False, scaleup=self.augment)  # 应用信箱增强
            shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # 用于COCO mAP重缩放的形状

            labels = self.labels[index].copy()  # 复制标签
            if labels.size:
                w *= ratio  # 更新宽度
                h *= ratio  # 更新高度
                # new boxes
                boxes = np.copy(labels[:, 1:])  # 复制边界框
                boxes[:, 0] = (
                    w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
                )  # 左上角x
                boxes[:, 1] = (
                    h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
                )  # 左上角y
                boxes[:, 2] = (
                    w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
                )  # 右下角x
                boxes[:, 3] = (
                    h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
                )  # 右下角y
                labels[:, 1:] = boxes  # 更新标签中的边界框

            if self.augment:
                img, labels = random_affine(
                    img,
                    labels,
                    degrees=self.hyp["degrees"],
                    translate=self.hyp["translate"],
                    scale=self.hyp["scale"],
                    shear=self.hyp["shear"],
                    new_shape=target_shape,
                )  # 应用随机仿射变换

        if len(labels):
            h, w = img.shape[:2]  # 获取图像的高度和宽度

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2 限制在图像宽度范围内
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2 限制在图像高度范围内

            boxes = np.copy(labels[:, 1:])  # 复制标签中的边界框
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x中心
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y中心
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # 宽度
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # 高度
            labels[:, 1:] = boxes  # 更新标签中的边界框

        if self.augment:
            img, labels = self.general_augment(img, labels)  # 应用通用增强

        labels_out = torch.zeros((len(labels), 6))  # 初始化输出标签
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)  # 将标签转换为张量

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC转为CHW，BGR转为RGB
        img = np.ascontiguousarray(img)  # 确保图像是连续的

        return torch.from_numpy(img), labels_out, self.img_paths[index], shapes  # 返回图像、标签、图像路径和形状
 
    def load_image(self, index, shrink_size=None):
        """Load image. 
        该函数通过cv2加载图像，调整原始图像的大小以符合目标形状(img_size)，同时保持比例。

        Returns:
            Image, original shape of image, resized image shape
        返回:
            图像，原始图像形状，调整后的图像形状
        """
        path = self.img_paths[index]  # 获取图像路径

        if self.cache_ram and self.imgs[index] is not None:  # 如果启用了内存缓存并且图像已缓存
            im = self.imgs[index]  # 从缓存中获取图像
            # im = copy.deepcopy(im)  # 深拷贝图像（注释掉的代码）
            return self.imgs[index], self.imgs_hw0[index], self.imgs_hw[index]  # 返回缓存的图像及其原始和调整后的形状
        else:
            try:
                im = cv2.imread(path)  # 尝试使用cv2读取图像
                assert im is not None, f"opencv cannot read image correctly or {path} not exists"  # 确保图像成功读取
            except Exception as e:  # 如果读取失败
                print(e)  # 打印异常信息
                im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)  # 使用PIL加载图像并转换颜色格式
                assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"  # 确保图像存在

            h0, w0 = im.shape[:2]  # 获取原始图像的高度和宽度
            if self.specific_shape:  # 如果需要特定形状
                # keep ratio resize
                ratio = min(self.target_width / w0, self.target_height / h0)  # 计算保持比例的缩放比例

            elif shrink_size:  # 如果提供了缩小尺寸
                ratio = (self.img_size - shrink_size) / max(h0, w0)  # 计算缩放比例

            else:
                ratio = self.img_size / max(h0, w0)  # 计算默认的缩放比例

            if ratio != 1:  # 如果缩放比例不为1
                im = cv2.resize(  # 调整图像大小
                    im,
                    (int(w0 * ratio), int(h0 * ratio)),  # 新的宽度和高度
                    interpolation=cv2.INTER_AREA  # 使用区域插值法
                    if ratio < 1 and not self.augment  # 如果缩小且不进行增强
                    else cv2.INTER_LINEAR,  # 否则使用线性插值法
                )
            return im, (h0, w0), im.shape[:2]  # 返回调整后的图像、原始形状和调整后形状
        
    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        # 合并一组样本以形成一个小批量的Tensor

        img, label, path, shapes = zip(*batch)  # 解压batch中的每个样本，分别获取图像、标签、路径和形状
        for i, l in enumerate(label):  # 遍历标签列表及其索引
            l[:, 0] = i  # add target image index for build_targets()  # 为build_targets()添加目标图像索引
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes  # 返回堆叠后的图像、拼接后的标签、路径和形状

    def get_imgs_labels(self, img_dirs):
        if not isinstance(img_dirs, list):  # 检查img_dirs是否为列表
            img_dirs = [img_dirs]  # 如果不是，则将其转换为列表
    
        # we store the cache img file in the first directory of img_dirs
        # 我们将缓存的图像文件存储在img_dirs的第一个目录中
        valid_img_record = osp.join(
            osp.dirname(img_dirs[0]), "." + osp.basename(img_dirs[0]) + "_cache.json"
        )  # 构建缓存信息的文件路径
        NUM_THREADS = min(8, os.cpu_count())  # 设置线程数，最多8个线程，取决于CPU核心数
        img_paths = []  # 初始化图像路径列表
    
        for img_dir in img_dirs:  # 遍历每个图像目录
            assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"  # 确保目录存在
            img_paths += glob.glob(osp.join(img_dir, "**/*"), recursive=True)  # 获取目录下所有文件的路径
    
        img_paths = sorted(  # 对图像路径进行排序
            p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p)
        )  # 只保留有效的图像文件
    
        assert img_paths, f"No images found in {img_dir}."  # 确保找到图像
        img_hash = self.get_hash(img_paths)  # 计算图像路径的哈希值
        LOGGER.info(f'img record infomation path is:{valid_img_record}')  # 记录缓存信息路径
        if osp.exists(valid_img_record):  # 如果缓存信息文件存在
            with open(valid_img_record, "r") as f:  # 打开缓存信息文件
                cache_info = json.load(f)  # 加载缓存信息
                if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:  # 检查哈希值是否匹配
                    img_info = cache_info["information"]  # 如果匹配，获取图像信息
                else:
                    self.check_images = True  # 如果不匹配，则需要检查图像
        else:
            self.check_images = True  # 如果文件不存在，则需要检查图像
    
        # check images
        # 检查图像
        if self.check_images and self.main_process:  # 如果需要检查图像并且是主进程
            img_info = {}  # 初始化图像信息字典
            nc, msgs = 0, []  # 记录损坏图像数量和消息
            LOGGER.info(
                f"{self.task}: Checking formats of images with {NUM_THREADS} process(es): "
            )  # 记录检查图像格式的信息
            with Pool(NUM_THREADS) as pool:  # 创建线程池
                pbar = tqdm(  # 初始化进度条
                    pool.imap(TrainValDataset.check_image, img_paths),
                    total=len(img_paths),
                )
                for img_path, shape_per_img, nc_per_img, msg in pbar:  # 遍历每个图像路径
                    if nc_per_img == 0:  # 如果图像没有损坏
                        img_info[img_path] = {"shape": shape_per_img}  # 记录图像形状
                    nc += nc_per_img  # 累加损坏图像数量
                    if msg:  # 如果有消息
                        msgs.append(msg)  # 记录消息
                    pbar.desc = f"{nc} image(s) corrupted"  # 更新进度条描述
            pbar.close()  # 关闭进度条
            if msgs:  # 如果有消息
                LOGGER.info("\n".join(msgs))  # 记录所有消息
    
            cache_info = {"information": img_info, "image_hash": img_hash}  # 构建缓存信息
            # save valid image paths.
            # 保存有效的图像路径
            with open(valid_img_record, "w") as f:  # 打开缓存信息文件进行写入
                json.dump(cache_info, f)  # 将缓存信息写入文件
    
        # check and load anns
        # 检查并加载标签
        img_paths = list(img_info.keys())  # 获取有效图像的路径
        label_paths = img2label_paths(img_paths)  # 获取对应的标签路径
        assert label_paths, f"No labels found."  # 确保找到标签
        label_hash = self.get_hash(label_paths)  # 计算标签路径的哈希值
        if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:  # 检查标签哈希值
            self.check_labels = True  # 如果不匹配，则需要检查标签
    
        if self.check_labels:  # 如果需要检查标签
            cache_info["label_hash"] = label_hash  # 更新标签哈希值
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 记录损坏标签数量和消息
            LOGGER.info(
                f"{self.task}: Checking formats of labels with {NUM_THREADS} process(es): "
            )  # 记录检查标签格式的信息
            with Pool(NUM_THREADS) as pool:  # 创建线程池
                pbar = pool.imap(
                    TrainValDataset.check_label_files, zip(img_paths, label_paths)
                )  # 检查标签文件
                pbar = tqdm(pbar, total=len(label_paths)) if self.main_process else pbar  # 如果是主进程，初始化进度条
                for (
                    img_path,
                    labels_per_file,
                    nc_per_file,
                    nm_per_file,
                    nf_per_file,
                    ne_per_file,
                    msg,
                ) in pbar:  # 遍历每个标签文件
                    if nc_per_file == 0:  # 如果标签文件没有损坏
                        img_info[img_path]["labels"] = labels_per_file  # 记录标签
                    else:
                        img_info.pop(img_path)  # 如果损坏，则移除图像信息
                    nc += nc_per_file  # 累加损坏标签数量
                    nm += nm_per_file  # 累加缺失标签数量
                    nf += nf_per_file  # 累加无效标签数量
                    ne += ne_per_file  # 累加空标签数量
                    if msg:  # 如果有消息
                        msgs.append(msg)  # 记录消息
                    if self.main_process:  # 如果是主进程
                        pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"  # 更新进度条描述
                if self.main_process:  # 如果是主进程
                    pbar.close()  # 关闭进度条
                    with open(valid_img_record, "w") as f:  # 打开缓存信息文件进行写入
                        json.dump(cache_info, f)  # 将缓存信息写入文件
                if msgs:  # 如果有消息
                    LOGGER.info("\n".join(msgs))  # 记录所有消息
                if nf == 0:  # 如果没有找到标签
                    LOGGER.warning(
                        f"WARNING: No labels found in {osp.dirname(img_paths[0])}. "
                    )  # 记录警告信息
    
        if self.task.lower() == "val":  # 如果任务是验证
            if self.data_dict.get("is_coco", False):  # 如果使用COCO数据集
                # use original json file when evaluating on coco dataset.
                # 在COCO数据集上评估时使用原始json文件
                assert osp.exists(self.data_dict["anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"  # 确保注释文件路径有效
            else:
                assert (
                    self.class_names
                ), "Class names is required when converting labels to coco format for evaluating."  # 确保提供类名
                save_dir = osp.join(osp.dirname(osp.dirname(img_dirs[0])), "annotations")  # 构建保存标签的目录
                if not osp.exists(save_dir):  # 如果目录不存在
                    os.mkdir(save_dir)  # 创建目录
                save_path = osp.join(
                    save_dir, "instances_" + osp.basename(img_dirs[0]) + ".json"
                )  # 构建保存标签的文件路径
                TrainValDataset.generate_coco_format_labels(  # 生成COCO格式的标签
                    img_info, self.class_names, save_path
                )
    
        img_paths, labels = list(  # 获取图像路径和标签
            zip(
                *[
                    (
                        img_path,
                        np.array(info["labels"], dtype=np.float32)  # 将标签转换为浮点型数组
                        if info["labels"]
                        else np.zeros((0, 5), dtype=np.float32),  # 如果没有标签，则返回空数组
                    )
                    for img_path, info in img_info.items()  # 遍历图像信息
                ]
            )
        )
        self.img_info = img_info  # 保存图像信息
        LOGGER.info(
            f"{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. "
        )  # 记录有效图像和标签的数量
        return img_paths, labels  # 返回图像路径和标签

    def get_mosaic(self, index, shape):
        """Gets images and labels after mosaic augments"""
        # 获取经过马赛克增强后的图像和标签

        indices = [index] + random.choices(  # 生成一个包含当前索引和3个随机索引的列表
            range(0, len(self.img_paths)), k=3  # 3 additional image indices  # 从图像路径中随机选择3个额外的图像索引
        )
        random.shuffle(indices)  # 打乱索引顺序
        imgs, hs, ws, labels = [], [], [], []  # 初始化图像、原始高度、原始宽度和标签的列表
        for index in indices:  # 遍历所有选择的索引
            img, _, (h, w) = self.load_image(index)  # 加载图像及其形状
            labels_per_img = self.labels[index]  # 获取对应图像的标签
            imgs.append(img)  # 将图像添加到图像列表中
            hs.append(h)  # 将原始高度添加到高度列表中
            ws.append(w)  # 将原始宽度添加到宽度列表中
            labels.append(labels_per_img)  # 将标签添加到标签列表中

        img, labels = mosaic_augmentation(shape, imgs, hs, ws, labels, self.hyp, self.specific_shape, self.target_height, self.target_width)
        # 调用mosaic_augmentation函数进行马赛克增强，返回增强后的图像和标签
        return img, labels  # 返回增强后的图像和标签

def general_augment(self, img, labels):
    """Gets images and labels after general augment
    This function applies hsv, random ud-flip and random lr-flips augments.
    """
    # 获取经过一般增强后的图像和标签
    # 该函数应用HSV、随机上下翻转和随机左右翻转增强

    nl = len(labels)  # 获取标签数量

    # HSV color-space
    # HSV颜色空间
    augment_hsv(
        img,
        hgain=self.hyp["hsv_h"],  # 色调增益
        sgain=self.hyp["hsv_s"],  # 饱和度增益
        vgain=self.hyp["hsv_v"],  # 明度增益
    )

    # Flip up-down
    # 上下翻转
    if random.random() < self.hyp["flipud"]:  # 根据设置的概率决定是否翻转
        img = np.flipud(img)  # 进行上下翻转
        if nl:  # 如果有标签
            labels[:, 2] = 1 - labels[:, 2]  # 更新标签的y坐标

    # Flip left-right
    # 左右翻转
    if random.random() < self.hyp["fliplr"]:  # 根据设置的概率决定是否翻转
        img = np.fliplr(img)  # 进行左右翻转
        if nl:  # 如果有标签
            labels[:, 1] = 1 - labels[:, 1]  # 更新标签的x坐标

    return img, labels  # 返回增强后的图像和标签

def sort_files_shapes(self):
    '''Sort by aspect ratio.'''
    # 按宽高比排序
    batch_num = self.batch_indices[-1] + 1  # 获取批次数量
    s = self.shapes  # [height, width]  # 获取形状信息
    ar = s[:, 1] / s[:, 0]  # 计算宽高比
    irect = ar.argsort()  # 获取排序后的索引
    self.img_paths = [self.img_paths[i] for i in irect]  # 根据排序索引重新排列图像路径
    self.labels = [self.labels[i] for i in irect]  # 根据排序索引重新排列标签
    self.shapes = s[irect]  # wh  # 根据排序索引重新排列形状
    ar = ar[irect]  # 根据排序索引重新排列宽高比

    # Set training image shapes
    # 设置训练图像的形状
    shapes = [[1, 1]] * batch_num  # 初始化形状列表
    for i in range(batch_num):  # 遍历每个批次
        ari = ar[self.batch_indices == i]  # 获取当前批次的宽高比
        mini, maxi = ari.min(), ari.max()  # 获取最小和最大宽高比
        if maxi < 1:  # 如果最大宽高比小于1
            shapes[i] = [1, maxi]  # 设置形状为[1, 最大宽高比]
        elif mini > 1:  # 如果最小宽高比大于1
            shapes[i] = [1 / mini, 1]  # 设置形状为[1/最小宽高比, 1]
    self.batch_shapes = (  # 计算批次形状
        np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(
            np.int_
        )  # 将形状调整为合适的尺寸
        * self.stride  # 乘以步长
    )

    @staticmethod
    def check_image(im_file):
        '''Verify an image.'''
        # 验证图像
        nc, msg = 0, ""  # 初始化损坏图像计数和消息
        try:
            im = Image.open(im_file)  # 打开图像文件
            im.verify()  # PIL verify  # 验证图像完整性
            im = Image.open(im_file)  # need to reload the image after using verify()  # 验证后需要重新加载图像
            shape = (im.height, im.width)  # (height, width)  # 获取图像的高度和宽度
            try:
                im_exif = im._getexif()  # 获取图像的EXIF信息
                if im_exif and ORIENTATION in im_exif:  # 如果存在EXIF信息并且包含方向信息
                    rotation = im_exif[ORIENTATION]  # 获取旋转角度
                    if rotation in (6, 8):  # 如果旋转角度为6或8
                        shape = (shape[1], shape[0])  # 交换宽高
            except:
                im_exif = None  # 如果获取EXIF信息失败，则设置为None

            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"  # 确保图像尺寸大于10像素
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"  # 确保图像格式有效
            if im.format.lower() in ("jpg", "jpeg"):  # 如果图像格式为JPEG
                with open(im_file, "rb") as f:  # 以二进制方式打开图像文件
                    f.seek(-2, 2)  # 定位到文件末尾前两个字节
                    if f.read() != b"\xff\xd9":  # 检查JPEG文件是否损坏
                        ImageOps.exif_transpose(Image.open(im_file)).save(  # 修复损坏的JPEG
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"  # 记录修复消息
            return im_file, shape, nc, msg  # 返回图像文件路径、形状、损坏计数和消息
        except Exception as e:
            nc = 1  # 如果发生异常，设置损坏计数为1
            msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"  # 记录忽略损坏图像的消息
            return im_file, None, nc, msg  # 返回图像文件路径、None、损坏计数和消息


    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args  # 解压参数，获取图像路径和标签路径
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        # 初始化缺失、找到、空标签和消息计数
        try:
            if osp.exists(lb_path):  # 如果标签文件存在
                nf = 1  # label found  # 找到标签
                with open(lb_path, "r") as f:  # 打开标签文件
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]  # 读取标签并按行分割
                    labels = np.array(labels, dtype=np.float32)  # 转换为浮点型数组
                if len(labels):  # 如果标签不为空
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."  # 确保每个标签包含5个元素
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"  # 确保标签值大于0
                    assert (
                        labels[:, 1:] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"  # 确保坐标值在0到1之间

                    _, indices = np.unique(labels, axis=0, return_index=True)  # 找到唯一的标签
                    if len(indices) < len(labels):  # duplicate row check  # 检查重复行
                        labels = labels[indices]  # remove duplicates  # 移除重复标签
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"  # 记录重复标签移除的消息
                    labels = labels.tolist()  # 转换为列表
                else:
                    ne = 1  # label empty  # 标签为空
                    labels = []  # 设置标签为空
            else:
                nm = 1  # label missing  # 标签缺失
                labels = []  # 设置标签为空

            return img_path, labels, nc, nm, nf, ne, msg  # 返回图像路径、标签、损坏计数、缺失计数、找到计数、空标签计数和消息
        except Exception as e:
            nc = 1  # 如果发生异常，设置损坏计数为1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"  # 记录忽略无效标签的消息
            return img_path, None, nc, nm, nf, ne, msg  # 返回图像路径、None、损坏计数、缺失计数、找到计数、空标签计数和消息

    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        # 用于与pycocotools进行评估
        dataset = {"categories": [], "annotations": [], "images": []}  # 初始化COCO格式的数据集字典
        for i, class_name in enumerate(class_names):  # 遍历类名及其索引
            dataset["categories"].append(  # 添加类别信息
                {"id": i, "name": class_name, "supercategory": ""}  # 类别ID、名称和超类别
            )

        ann_id = 0  # 初始化注释ID
        LOGGER.info(f"Convert to COCO format")  # 记录转换为COCO格式的信息
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):  # 遍历图像信息
            labels = info["labels"] if info["labels"] else []  # 获取标签，如果没有则为空列表
            img_id = osp.splitext(osp.basename(img_path))[0]  # 获取图像ID
            img_h, img_w = info["shape"]  # 获取图像的高度和宽度
            dataset["images"].append(  # 添加图像信息
                {
                    "file_name": os.path.basename(img_path),  # 图像文件名
                    "id": img_id,  # 图像ID
                    "width": img_w,  # 图像宽度
                    "height": img_h,  # 图像高度
                }
            )
            if labels:  # 如果有标签
                for label in labels:  # 遍历每个标签
                    c, x, y, w, h = label[:5]  # 解压标签信息
                    # convert x,y,w,h to x1,y1,x2,y2
                    # 将中心坐标和宽高转换为左上角和右下角坐标
                    x1 = (x - w / 2) * img_w  # 计算左上角x坐标
                    y1 = (y - h / 2) * img_h  # 计算左上角y坐标
                    x2 = (x + w / 2) * img_w  # 计算右下角x坐标
                    y2 = (y + h / 2) * img_h  # 计算右下角y坐标
                    # cls_id starts from 0
                    # 类别ID从0开始
                    cls_id = int(c)  # 将类别转换为整数
                    w = max(0, x2 - x1)  # 计算宽度，确保不小于0
                    h = max(0, y2 - y1)  # 计算高度，确保不小于0
                    dataset["annotations"].append(  # 添加注释信息
                        {
                            "area": h * w,  # 计算区域
                            "bbox": [x1, y1, w, h],  # 边界框信息
                            "category_id": cls_id,  # 类别ID
                            "id": ann_id,  # 注释ID
                            "image_id": img_id,  # 图像ID
                            "iscrowd": 0,  # 是否为拥挤对象
                            # mask
                            "segmentation": [],  # 分割信息
                        }
                    )
                    ann_id += 1  # 增加注释ID

        with open(save_path, "w") as f:  # 打开保存路径以写入
            json.dump(dataset, f)  # 将数据集写入文件
            LOGGER.info(  # 记录转换完成的信息
                f"Convert to COCO format finished. Results saved in {save_path}"
            )

@staticmethod
def get_hash(paths):
    """Get the hash value of paths"""
    # 获取路径的哈希值
    assert isinstance(paths, list), "Only support list currently."  # 确保输入为列表
    h = hashlib.md5("".join(paths).encode())  # 计算MD5哈希值
    return h.hexdigest()  # 返回哈希值的十六进制表示


class LoadData:
    def __init__(self, path, webcam, webcam_addr):
        self.webcam = webcam  # 保存是否使用网络摄像头的标志
        self.webcam_addr = webcam_addr  # 保存网络摄像头地址
        if webcam:  # if use web camera  # 如果使用网络摄像头
            imgp = []  # 初始化图像路径列表
            vidp = [int(webcam_addr) if webcam_addr.isdigit() else webcam_addr]  # 将摄像头地址转换为整数或保留为字符串
        else:
            p = str(Path(path).resolve())  # os-agnostic absolute path  # 获取绝对路径
            if os.path.isdir(p):  # 如果路径是目录
                files = sorted(glob.glob(os.path.join(p, '**/*.*'), recursive=True))  # dir  # 获取目录下所有文件的路径
            elif os.path.isfile(p):  # 如果路径是文件
                files = [p]  # files  # 将文件路径添加到列表
            else:
                raise FileNotFoundError(f'Invalid path {p}')  # 如果路径无效，抛出异常
            imgp = [i for i in files if i.split('.')[-1] in IMG_FORMATS]  # 筛选出图像文件
            vidp = [v for v in files if v.split('.')[-1] in VID_FORMATS]  # 筛选出视频文件
        self.files = imgp + vidp  # 合并图像和视频文件列表
        self.nf = len(self.files)  # 计算文件数量
        self.type = 'image'  # 默认类型为图像
        if len(vidp) > 0:  # 如果有视频文件
            self.add_video(vidp[0])  # new video  # 添加第一个视频
        else:
            self.cap = None  # 如果没有视频，设置为None

    # @staticmethod
    def checkext(self, path):
        if self.webcam:  # 如果使用网络摄像头
            file_type = 'video'  # 文件类型为视频
        else:
            file_type = 'image' if path.split('.')[-1].lower() in IMG_FORMATS else 'video'  # 根据文件扩展名判断文件类型
        return file_type  # 返回文件类型

    def __iter__(self):
        self.count = 0  # 初始化计数器
        return self  # 返回自身以支持迭代

    def __next__(self):
        if self.count == self.nf:  # 如果计数器等于文件数量
            raise StopIteration  # 停止迭代
        path = self.files[self.count]  # 获取当前文件路径
        if self.checkext(path) == 'video':  # 如果当前文件是视频
            self.type = 'video'  # 设置类型为视频
            ret_val, img = self.cap.read()  # 读取视频帧
            while not ret_val:  # 如果未成功读取
                self.count += 1  # 增加计数器
                self.cap.release()  # 释放视频捕获对象
                if self.count == self.nf:  # last video  # 如果是最后一个视频
                    raise StopIteration  # 停止迭代
                path = self.files[self.count]  # 获取下一个文件路径
                self.add_video(path)  # 添加新视频
                ret_val, img = self.cap.read()  # 读取视频帧
        else:
            # Read image
            self.count += 1  # 增加计数器
            img = cv2.imread(path)  # 读取图像（BGR格式）
        return img, path, self.cap  # 返回图像、路径和视频捕获对象

    def add_video(self, path):
        self.frame = 0  # 初始化帧计数器
        self.cap = cv2.VideoCapture(path)  # 创建视频捕获对象
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

    def __len__(self):
        return self.nf  # number of files  # 返回文件数量