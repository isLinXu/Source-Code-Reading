# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
# Ultralytics数据集*.cache版本，>= 1.0.0用于YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.
    用于加载YOLO格式的目标检测和/或分割标签的数据集类。

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        data (dict, optional): 数据集YAML字典。默认为None。
        task (str): An explicit arg to point current task, Defaults to 'detect'.
        task (str): 指定当前任务的显式参数，默认为'detect'。

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
        (torch.utils.data.Dataset): 可以用于训练目标检测模型的PyTorch数据集对象。
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints.
        使用可选的段和关键点配置初始化YOLODataset。
        """
        self.use_segments = task == "segment"  # Check if the task is segmentation
        # 检查任务是否为分割
        self.use_keypoints = task == "pose"  # Check if the task is pose estimation
        # 检查任务是否为姿态估计
        self.use_obb = task == "obb"  # Check if the task is oriented bounding box
        # 检查任务是否为定向边界框
        self.data = data  # Store the dataset information
        # 存储数据集信息
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        # 确保不能同时使用段和关键点
        super().__init__(*args, **kwargs)  # Initialize the base class


    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.
        缓存数据集标签，检查图像并读取形状。

        Args:
            path (Path): Path where to save the cache file. Default is Path("./labels.cache").
            path (Path): 缓存文件保存路径。默认是Path("./labels.cache")。

        Returns:
            (dict): labels.
        """
        x = {"labels": []}  # Initialize a dictionary to hold labels
        # 初始化一个字典以保存标签
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        # 缺失、找到、空、损坏的数量和消息
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)  # Total number of images
        # 图像总数
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))  # Get keypoint shape
        # 获取关键点形状
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        # 检查关键点形状是否正确
        with ThreadPool(NUM_THREADS) as pool:  # Create a thread pool for concurrent processing
            # 创建线程池以进行并发处理
            results = pool.imap(
                func=verify_image_label,  # Function to verify image and label
                iterable=zip(
                    self.im_files,  # Image files
                    self.label_files,  # Label files
                    repeat(self.prefix),  # Repeat prefix for logging
                    repeat(self.use_keypoints),  # Repeat use_keypoints flag
                    repeat(len(self.data["names"])),  # Repeat number of classes
                    repeat(nkpt),  # Repeat number of keypoints
                    repeat(ndim),  # Repeat dimension of keypoints
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)  # Progress bar for processing
            # 处理的进度条
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f  # Update number of missing files
                nf += nf_f  # Update number of found files
                ne += ne_f  # Update number of empty files
                nc += nc_f  # Update number of corrupt files
                if im_file:  # If the image file is valid
                    x["labels"].append(
                        {
                            "im_file": im_file,  # Add image file path
                            "shape": shape,  # Add image shape
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,  # Add segments
                            "keypoints": keypoint,  # Add keypoints
                            "normalized": True,  # Indicate that the data is normalized
                            "bbox_format": "xywh",  # Bounding box format
                        }
                    )
                if msg:  # If there are any messages
                    msgs.append(msg)  # Append messages to the list
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"  # Update progress bar description
                # 更新进度条描述
            pbar.close()  # Close the progress bar

        if msgs:  # If there are any messages
            LOGGER.info("\n".join(msgs))  # Log all messages
        if nf == 0:  # If no labels found
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
            # 记录没有找到标签的警告信息
        x["hash"] = get_hash(self.label_files + self.im_files)  # Get hash of the dataset
        # 获取数据集的哈希值
        x["results"] = nf, nm, ne, nc, len(self.im_files)  # Store results in the dictionary
        # 将结果存储在字典中
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)  # Save the cache file
        # 保存缓存文件
        return x  # Return the labels dictionary


    def get_labels(self):
        """Returns dictionary of labels for YOLO training.
        返回YOLO训练的标签字典。
        """
        self.label_files = img2label_paths(self.im_files)  # Get label files corresponding to the image files
        # 获取与图像文件对应的标签文件
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")  # Define cache path
        # 定义缓存路径
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            # 尝试加载*.cache文件
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            # 确保版本匹配
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
            # 确保哈希值相同
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops
            # 运行缓存操作

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        # 找到、缺失、空、损坏、总数
        if exists and LOCAL_RANK in {-1, 0}:  # If cache exists and is the main process
            # 如果缓存存在且是主进程
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            # 显示结果
            if cache["msgs"]:  # If there are messages in cache
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
                # 显示警告

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        # 移除项目
        labels = cache["labels"]  # Get labels from cache
        # 从缓存中获取标签
        if not labels:  # If no labels found
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
            # 记录没有找到图像的警告信息
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files
        # 更新im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)  # Get lengths of cls, bboxes, segments
        # 获取cls、bboxes、segments的长度
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))  # Sum lengths
        # 计算长度的总和
        if len_segments and len_boxes != len_segments:  # If segments exist but don't match box count
            # 如果存在段但与框计数不匹配
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            # 记录框和段计数不匹配的警告信息
            for lb in labels:  # Remove segments from labels
                lb["segments"] = []  # 清空标签中的段
        if len_cls == 0:  # If no class labels found
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
            # 记录没有找到标签的警告信息
        return labels  # Return the labels


    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list.
        构建并附加变换到列表。
        """
        if self.augment:  # If augmentations are enabled
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0  # Set mosaic ratio
            # 设置马赛克比率
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0  # Set mixup ratio
            # 设置混合比率
            transforms = v8_transforms(self, self.imgsz, hyp)  # Build transforms using v8_transforms
            # 使用v8_transforms构建变换
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])  # Default transformation
            # 默认变换
        transforms.append(
            Format(
                bbox_format="xywh",  # Set bounding box format
                normalize=True,  # Normalize the data
                return_mask=self.use_segments,  # Return mask if using segments
                return_keypoint=self.use_keypoints,  # Return keypoints if using keypoints
                return_obb=self.use_obb,  # Return oriented bounding box if using OBB
                batch_idx=True,  # Include batch index
                mask_ratio=hyp.mask_ratio,  # Set mask ratio
                mask_overlap=hyp.overlap_mask,  # Set mask overlap
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
                # 仅影响训练
            )
        )
        return transforms  # Return the list of transformations
        # 返回变换列表


    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations.
        将马赛克、copy_paste和混合选项设置为0.0并构建变换。
        """
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        # 设置马赛克比率为0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        # 保持与之前v8 close-mosaic相同的行为
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        # 保持与之前v8 close-mosaic相同的行为
        self.transforms = self.build_transforms(hyp)  # Build transformations with updated hyperparameters
        # 使用更新的超参数构建变换


    def update_labels_info(self, label):
        """
        Custom your label format here.
        在这里自定义您的标签格式。

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        注意：
            cls现在不与边界框一起，分类和语义分割需要独立的cls标签
            也可以通过添加或删除字典键来支持分类和语义分割。
        """
        bboxes = label.pop("bboxes")  # Remove and get bounding boxes from the label
        # 从标签中移除并获取边界框
        segments = label.pop("segments", [])  # Remove and get segments from the label, default to empty list
        # 从标签中移除并获取段，默认为空列表
        keypoints = label.pop("keypoints", None)  # Remove and get keypoints from the label
        # 从标签中移除并获取关键点
        bbox_format = label.pop("bbox_format")  # Remove and get bounding box format
        # 从标签中移除并获取边界框格式
        normalized = label.pop("normalized")  # Remove and get normalization status
        # 从标签中移除并获取归一化状态

        # NOTE: do NOT resample oriented boxes
        # 注意：不要重新采样定向框
        segment_resamples = 100 if self.use_obb else 1000  # Set number of resamples for segments
        # 设置段的重新采样数量
        if len(segments) > 0:  # If segments exist
            # 如果存在段
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            # 确保如果原始长度大于段重新采样数量，则段插值正确
            max_len = max(len(s) for s in segments)  # Get the maximum length of segments
            # 获取段的最大长度
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # 如果重新采样数量小于最大长度，则更新重新采样数量
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)  # Resample segments
            # 重新采样段
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)  # Create an empty array for segments
            # 为段创建一个空数组
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)  # Create instances and add to label
        # 创建实例并添加到标签中
        return label  # Return the updated label
        # 返回更新后的标签


    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches.
        将数据样本合并为批次。
        """
        new_batch = {}  # Initialize a new batch dictionary
        # 初始化新的批次字典
        keys = batch[0].keys()  # Get keys from the first sample
        # 从第一个样本中获取键
        values = list(zip(*[list(b.values()) for b in batch]))  # Zip values from all samples
        # 从所有样本中压缩值
        for i, k in enumerate(keys):  # Iterate through keys
            # 遍历键
            value = values[i]  # Get corresponding values
            # 获取相应的值
            if k == "img":  # If the key is 'img'
                value = torch.stack(value, 0)  # Stack images into a tensor
                # 将图像堆叠到一个张量中
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:  # If key is one of the specified types
                value = torch.cat(value, 0)  # Concatenate values into a single tensor
                # 将值连接到一个张量中
            new_batch[k] = value  # Add the processed value to the new batch
            # 将处理后的值添加到新批次中
        new_batch["batch_idx"] = list(new_batch["batch_idx"])  # Convert batch index to a list
        # 将批次索引转换为列表
        for i in range(len(new_batch["batch_idx"])):  # Iterate through batch indices
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
            # 为build_targets()添加目标图像索引
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)  # Concatenate batch indices into a tensor
        # 将批次索引连接到一个张量中
        return new_batch  # Return the collated batch
        # 返回合并的批次


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True))
        return transforms


class GroundingDataset(YOLODataset):
    """Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format.
    通过加载指定JSON文件中的注释来处理目标检测任务，支持YOLO格式。
    """

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """Initializes a GroundingDataset for object detection, loading annotations from a specified JSON file.
        初始化一个GroundingDataset用于目标检测，从指定的JSON文件加载注释。
        """
        assert task == "detect", "[GroundingDataset](cci:2://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/data/dataset.py:382:0-454:25) only support `detect` task for now!"
        # 确保任务是检测任务，GroundingDataset目前仅支持检测任务
        self.json_file = json_file  # Store the path to the JSON file
        # 存储JSON文件的路径
        super().__init__(*args, task=task, data={}, **kwargs)  # Initialize the base class


    def get_img_files(self, img_path):
        """The image files would be read in [get_labels](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/data/dataset.py:395:4-446:21) function, return empty list here.
        图像文件将在[get_labels](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/data/dataset.py:395:4-446:21)函数中读取，此处返回空列表。
        """
        return []  # Return an empty list


    def get_labels(self):
        """Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image.
        从JSON文件加载注释，过滤并归一化每个图像的边界框。
        """
        labels = []  # Initialize a list to hold labels
        # 初始化一个列表以保存标签
        LOGGER.info("Loading annotation file...")  # Log the loading process
        # 记录加载过程
        with open(self.json_file) as f:  # Open the JSON file
            annotations = json.load(f)  # Load annotations from the file
            # 从文件中加载注释
        images = {f"{x['id']:d}": x for x in annotations["images"]}  # Create a dictionary of images
        # 创建图像字典
        img_to_anns = defaultdict(list)  # Create a default dictionary to hold annotations for each image
        # 创建一个默认字典以保存每个图像的注释
        for ann in annotations["annotations"]:  # Iterate through annotations
            # 遍历注释
            img_to_anns[ann["image_id"]].append(ann)  # Append annotation to the corresponding image
            # 将注释附加到相应的图像

        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            # 遍历图像ID和注释，显示进度条
            img = images[f"{img_id:d}"]  # Get the image information
            # 获取图像信息
            h, w, f = img["height"], img["width"], img["file_name"]  # Get height, width, and filename
            # 获取高度、宽度和文件名
            im_file = Path(self.img_path) / f  # Define the full path to the image file
            # 定义图像文件的完整路径
            if not im_file.exists():  # Check if the image file exists
                # 检查图像文件是否存在
                continue  # Skip if the image file does not exist
                # 如果图像文件不存在，则跳过
            self.im_files.append(str(im_file))  # Add the image file path to the list
            # 将图像文件路径添加到列表中
            bboxes = []  # Initialize a list to hold bounding boxes
            # 初始化一个列表以保存边界框
            cat2id = {}  # Initialize a dictionary to map category names to IDs
            # 初始化一个字典以将类别名称映射到ID
            texts = []  # Initialize a list to hold category names
            # 初始化一个列表以保存类别名称
            for ann in anns:  # Iterate through annotations for the current image
                # 遍历当前图像的注释
                if ann["iscrowd"]:  # Skip crowd annotations
                    # 跳过人群注释
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)  # Get the bounding box
                # 获取边界框
                box[:2] += box[2:] / 2  # Convert to center format
                # 转换为中心格式
                box[[0, 2]] /= float(w)  # Normalize x-coordinates
                # 归一化x坐标
                box[[1, 3]] /= float(h)  # Normalize y-coordinates
                # 归一化y坐标
                if box[2] <= 0 or box[3] <= 0:  # Skip invalid boxes
                    # 跳过无效的边界框
                    continue

                caption = img["caption"]  # Get the caption for the image
                # 获取图像的标题
                cat_name = " ".join([caption[t[0]:t[1]] for t in ann["tokens_positive"]])  # Get category name from tokens
                # 从tokens中获取类别名称
                if cat_name not in cat2id:  # If category name is not in the dictionary
                    # 如果类别名称不在字典中
                    cat2id[cat_name] = len(cat2id)  # Assign a new ID to the category
                    # 将新ID分配给类别
                    texts.append([cat_name])  # Add the category name to the texts list
                cls = cat2id[cat_name]  # Get the class ID
                # 获取类ID
                box = [cls] + box.tolist()  # Combine class ID with bounding box
                # 将类ID与边界框组合
                if box not in bboxes:  # Avoid duplicates
                    # 避免重复
                    bboxes.append(box)  # Add bounding box to the list
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)  # Convert to array
            # 转换为数组
            labels.append(
                {
                    "im_file": im_file,  # Add image file path
                    "shape": (h, w),  # Add image shape
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "normalized": True,  # Indicate that the data is normalized
                    "bbox_format": "xywh",  # Bounding box format
                    "texts": texts,  # Add category names
                }
            )
        return labels  # Return the list of labels


    def build_transforms(self, hyp=None):
        """Configures augmentations for training with optional text loading; `hyp` adjusts augmentation intensity.
        配置用于训练的增强，支持可选的文本加载；`hyp`调整增强强度。
        """
        transforms = super().build_transforms(hyp)  # Build base transforms
        # 构建基础变换
        if self.augment:  # If augmentations are enabled
            # 如果启用了增强
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=80, padding=True))  # Insert text loading transform
            # 插入文本加载变换
        return transforms  # Return the list of transformations
        # 返回变换列表


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.
    作为多个数据集的连接的数据集。

    This class is useful to assemble different existing datasets.
    此类对于组合不同的现有数据集非常有用。
    """

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches.
        将数据样本合并为批次。
        """
        return YOLODataset.collate_fn(batch)  # Use the collate function from YOLODataset
        # 使用YOLODataset中的合并函数


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.
    语义分割数据集。

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.
    该类负责处理用于语义分割任务的数据集。它继承自BaseDataset类的功能。

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    注意：
        该类目前是一个占位符，需要填充方法和属性以支持语义分割任务。
    """

    def __init__(self):
        """Initialize a SemanticDataset object.
        初始化一个SemanticDataset对象。
        """
        super().__init__()  # Initialize the base class


class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.
    扩展torchvision的ImageFolder以支持YOLO分类任务，提供图像增强、缓存和验证等功能。它旨在高效处理大型数据集以训练深度学习模型，支持可选的图像变换和缓存机制以加速训练。

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.
    该类允许使用torchvision和Albumentations库进行增强，并支持将图像缓存到RAM或磁盘以减少训练过程中的IO开销。此外，它实现了强大的验证过程，以确保数据的完整性和一致性。

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_ram (bool): 指示是否启用RAM中的缓存。
        cache_disk (bool): Indicates if caching on disk is enabled.
        cache_disk (bool): 指示是否启用磁盘上的缓存。
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        samples (list): 一个元组列表，每个元组包含图像路径、类索引、.npy缓存文件的路径（如果在磁盘上缓存）以及可选的加载图像数组（如果在RAM中缓存）。
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        torch_transforms (callable): 应用于图像的PyTorch变换。
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.
        使用根目录、图像大小、增强和缓存设置初始化YOLO对象。

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            root (str): 数据集目录的路径，图像存储在特定类别的文件夹结构中。
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings. It includes attributes like `imgsz` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, [cache](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/data/dataset.py:78:4-157:48) (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            args (Namespace): 配置包含与数据集相关的设置，如图像大小、增强参数和缓存设置。它包括属性，如`imgsz`（图像大小）、`fraction`（使用的数据比例）、`scale`、`fliplr`、`flipud`、[cache](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/data/dataset.py:78:4-157:48)（用于更快训练的磁盘或RAM缓存）、`auto_augment`、`hsv_h`、`hsv_s`、`hsv_v`和`crop_fraction`。
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            augment (bool, optional): 是否对数据集应用增强。默认值为False。
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification and
                debugging. Default is an empty string.
            prefix (str, optional): 日志和缓存文件名的前缀，有助于数据集的识别和调试。默认值为空字符串。
        """
        import torchvision  # scope for faster 'import ultralytics'
        # 为了更快地导入'ultralytics'而引入torchvision

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        # 基类作为属性分配，而不是用作基类，以允许作用域慢速torchvision导入
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)  # Allow empty folders
            # 允许空文件夹
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)  # Create ImageFolder dataset
            # 创建ImageFolder数据集
        self.samples = self.base.samples  # Get samples from the base dataset
        # 从基类数据集中获取样本
        self.root = self.base.root  # Store root directory of the dataset
        # 存储数据集的根目录

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            # 如果增强和args.fraction小于1.0
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]  # Reduce samples based on fraction
            # 根据比例减少样本
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""  # Set prefix for logging
        # 设置日志的前缀
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        # 将图像缓存到RAM
        if self.cache_ram:  # If caching in RAM
            # 如果在RAM中缓存
            LOGGER.warning(
                "WARNING ⚠️ Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            # 记录有关RAM缓存训练已知内存泄漏的警告信息
            self.cache_ram = False  # Disable RAM caching due to memory leak
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        # 将图像缓存到硬盘作为未压缩的*.npy文件
        self.samples = self.verify_images()  # filter out bad images
        # 过滤掉坏图像
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        # 文件、索引、npy、图像
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)  # Set scale range for augmentations
        # 设置增强的缩放范围
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,  # Set image size for augmentations
                scale=scale,  # Set scale for augmentations
                hflip=args.fliplr,  # Set horizontal flip augmentation
                vflip=args.flipud,  # Set vertical flip augmentation
                erasing=args.erasing,  # Set erasing augmentation
                auto_augment=args.auto_augment,  # Set auto augment flag
                hsv_h=args.hsv_h,  # Set hue augmentation
                hsv_s=args.hsv_s,  # Set saturation augmentation
                hsv_v=args.hsv_v,  # Set value augmentation
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)  # Default transformations
            # 默认变换
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices.
        返回与给定索引对应的数据和目标的子集。
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        # 文件名、索引、文件名带后缀为'.npy'、图像
        if self.cache_ram:  # If caching in RAM
            # 如果在RAM中缓存
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                # 警告：这里需要两个单独的if语句，不要将其与前一行合并
                im = self.samples[i][3] = cv2.imread(f)  # Read image if not already loaded
                # 如果未加载，则读取图像
        elif self.cache_disk:  # If caching on disk
            # 如果在磁盘上缓存
            if not fn.exists():  # load npy
                # 加载npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)  # Save image as .npy file
                # 将图像保存为.npy文件
            im = np.load(fn)  # Load image from .npy file
            # 从.npy文件加载图像
        else:  # read image
            im = cv2.imread(f)  # BGR  # Read image from file
            # 从文件中读取图像
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # Convert image to RGB format
        # 将图像转换为RGB格式
        sample = self.torch_transforms(im)  # Apply transformations to the image
        # 对图像应用变换
        return {"img": sample, "cls": j}  # Return dictionary with image and class index
        # 返回包含图像和类索引的字典

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.
        返回数据集中的样本总数。
        """
        return len(self.samples)  # Return the length of samples
        # 返回样本的长度

    def verify_images(self):
        """Verify all images in dataset.
        验证数据集中的所有图像。
        """
        desc = f"{self.prefix}Scanning {self.root}..."  # Description for progress bar
        # 进度条描述
        path = Path(self.root).with_suffix(".cache")  # *.cache file path
        # *.cache文件路径

        try:
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            # 尝试加载*.cache文件
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            # 确保版本匹配
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            # 确保哈希值相同
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            # 找到、缺失、空、损坏、总数
            if LOCAL_RANK in {-1, 0}:  # If cache exists and is the main process
                # 如果缓存存在且是主进程
                d = f"{desc} {nf} images, {nc} corrupt"  # Update description with found and corrupt counts
                # 使用找到和损坏的计数更新描述
                TQDM(None, desc=d, total=n, initial=n)  # display results
                # 显示结果
                if cache["msgs"]:  # If there are messages in cache
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
                    # 显示警告
            return samples  # Return the verified samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            # 如果*.cache检索失败，则运行扫描
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:  # Create a thread pool for concurrent processing
                # 创建线程池以进行并发处理
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))  # Verify images
                # 验证图像
                pbar = TQDM(results, desc=desc, total=len(self.samples))  # Progress bar for verification
                # 验证的进度条
                for sample, nf_f, nc_f, msg in pbar:  # Iterate through verification results
                    # 遍历验证结果
                    if nf_f:  # If the image is found
                        samples.append(sample)  # Add sample to the list
                        # 将样本添加到列表中
                    if msg:  # If there are any messages
                        msgs.append(msg)  # Append messages to the list
                    nf += nf_f  # Update number of found images
                    nc += nc_f  # Update number of corrupt images
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"  # Update progress bar description
                    # 更新进度条描述
                pbar.close()  # Close the progress bar
            if msgs:  # If there are any messages
                LOGGER.info("\n".join(msgs))  # Log all messages
                # 记录所有消息
            x["hash"] = get_hash([x[0] for x in self.samples])  # Get hash of the dataset
            # 获取数据集的哈希值
            x["results"] = nf, nc, len(samples), samples  # Store results in the dictionary
            # 将结果存储在字典中
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)  # Save the cache file
            # 保存缓存文件
            return samples  # Return the verified samples
            # 返回验证后的样本
