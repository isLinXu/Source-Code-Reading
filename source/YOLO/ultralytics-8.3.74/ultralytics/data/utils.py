# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import hashlib
import json
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    ROOT,
    SETTINGS_FILE,
    TQDM,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
    yaml_load,
    yaml_save,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes

HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."  # 查看数据集格式指导的链接
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes  # 图像后缀
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes  # 视频后缀
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders  # 全局 pin_memory 用于数据加载器
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"  # 支持的格式消息

def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""  # 根据图像路径定义标签路径
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings  # /images/ 和 /labels/ 子字符串
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]  # 返回标签路径列表

def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""  # 返回路径列表（文件或目录）的单个哈希值
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes  # 计算路径的总大小
    h = hashlib.sha256(str(size).encode())  # hash sizes  # 对大小进行哈希
    h.update("".join(paths).encode())  # hash paths  # 对路径进行哈希
    return h.hexdigest()  # return hash  # 返回哈希值

def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""  # 返回经过 EXIF 校正的 PIL 大小
    s = img.size  # (width, height)  # 获取图像的尺寸（宽度，高度）
    if img.format == "JPEG":  # only support JPEG images  # 仅支持 JPEG 图像
        try:
            if exif := img.getexif():  # 获取图像的 EXIF 信息
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274  # EXIF 方向标签的键为 274
                if rotation in {6, 8}:  # rotation 270 or 90  # 旋转 270 或 90 度
                    s = s[1], s[0]  # 交换宽高
        except Exception:
            pass
    return s  # 返回校正后的尺寸

def verify_image(args):
    """Verify one image."""  # 验证一张图像
    (im_file, cls), prefix = args  # 解包参数
    # Number (found, corrupt), message  # 数量（找到，损坏），消息
    nf, nc, msg = 0, 0, ""  # 初始化找到和损坏的计数，以及消息
    try:
        im = Image.open(im_file)  # 打开图像文件
        im.verify()  # PIL verify  # 使用 PIL 验证图像
        shape = exif_size(im)  # image size  # 获取图像尺寸
        shape = (shape[1], shape[0])  # hw  # 交换宽高
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"  # 确保尺寸大于 10 像素
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"  # 确保格式有效
        if im.format.lower() in {"jpg", "jpeg"}:  # 如果是 JPEG 格式
            with open(im_file, "rb") as f:  # 以二进制模式打开图像文件
                f.seek(-2, 2)  # 移动到文件末尾前两个字节
                if f.read() != b"\xff\xd9":  # corrupt JPEG  # 检查是否为损坏的 JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)  # 恢复并保存损坏的 JPEG
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"  # 记录恢复消息
        nf = 1  # 找到图像
    except Exception as e:
        nc = 1  # 记录损坏的图像
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"  # 记录忽略的消息
    return (im_file, cls), nf, nc, msg  # 返回结果

def verify_image_label(args):
    """Verify one image-label pair."""  # 验证一对图像-标签
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args  # 解包参数
    # Number (missing, found, empty, corrupt), message, segments, keypoints  # 数量（缺失，找到，空，损坏），消息，段，关键点
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None  # 初始化计数和消息
    try:
        # Verify images  # 验证图像
        im = Image.open(im_file)  # 打开图像文件
        im.verify()  # PIL verify  # 使用 PIL 验证图像
        shape = exif_size(im)  # image size  # 获取图像尺寸
        shape = (shape[1], shape[0])  # hw  # 交换宽高
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"  # 确保尺寸大于 10 像素
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"  # 确保格式有效
        if im.format.lower() in {"jpg", "jpeg"}:  # 如果是 JPEG 格式
            with open(im_file, "rb") as f:  # 以二进制模式打开图像文件
                f.seek(-2, 2)  # 移动到文件末尾前两个字节
                if f.read() != b"\xff\xd9":  # corrupt JPEG  # 检查是否为损坏的 JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)  # 恢复并保存损坏的 JPEG
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"  # 记录恢复消息

        # Verify labels  # 验证标签
        if os.path.isfile(lb_file):  # 如果标签文件存在
            nf = 1  # label found  # 找到标签
            with open(lb_file) as f:  # 打开标签文件
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]  # 读取标签并分割
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment  # 如果标签超过 6 列且不是关键点
                    classes = np.array([x[0] for x in lb], dtype=np.float32)  # 获取类别
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)  # 获取段
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)  # 合并类别和段
                lb = np.array(lb, dtype=np.float32)  # 转换为 numpy 数组
            if nl := len(lb):  # 如果标签数量大于 0
                if keypoint:  # 如果是关键点
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"  # 检查列数
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]  # 获取关键点
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"  # 检查列数
                    points = lb[:, 1:]  # 获取标签点
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"  # 检查坐标是否超出范围
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"  # 检查标签值是否为负

                # All labels  # 所有标签
                max_cls = lb[:, 0].max()  # max label count  # 获取最大标签数量
                assert max_cls < num_cls, (  # 确保最大标签小于数据集类别数量
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)  # 获取唯一标签及其索引
                if len(i) < nl:  # duplicate row check  # 检查重复行
                    lb = lb[i]  # remove duplicates  # 移除重复标签
                    if segments:  # 如果有段
                        segments = [segments[x] for x in i]  # 根据索引保留段
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"  # 记录重复标签移除消息
            else:
                ne = 1  # label empty  # 标签为空
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)  # 创建空标签
        else:
            nm = 1  # label missing  # 标签缺失
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)  # 创建空标签
        if keypoint:  # 如果是关键点
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)  # 获取关键点
            if ndim == 2:  # 如果维度为 2
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)  # 创建关键点掩码
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)  # 合并关键点和掩码
        lb = lb[:, :5]  # 保留标签的前 5 列
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg  # 返回结果
    except Exception as e:
        nc = 1  # 记录损坏的标签
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"  # 记录忽略的消息
        return [None, None, None, None, None, nm, nf, ne, nc, msg]  # 返回结果


def visualize_image_annotations(image_path, txt_path, label_map):
    """
    Visualizes YOLO annotations (bounding boxes and class labels) on an image.  # 在图像上可视化 YOLO 注释（边界框和类别标签）

    This function reads an image and its corresponding annotation file in YOLO format, then  # 此函数读取图像及其对应的 YOLO 格式注释文件，然后
    draws bounding boxes around detected objects and labels them with their respective class names.  # 在检测到的物体周围绘制边界框，并用相应的类别名称标记它们。
    The bounding box colors are assigned based on the class ID, and the text color is dynamically  # 边界框的颜色根据类别 ID 分配，文本颜色根据背景颜色的亮度动态调整
    adjusted for readability, depending on the background color's luminance.  # 以提高可读性，取决于背景颜色的亮度。

    Args:
        image_path (str): The path to the image file to annotate, and it can be in formats supported by PIL (e.g., .jpg, .png).  # 图像文件的路径，可以是 PIL 支持的格式（例如 .jpg，.png）。
        txt_path (str): The path to the annotation file in YOLO format, that should contain one line per object with:  # YOLO 格式注释文件的路径，每个对象应包含一行：
                        - class_id (int): The class index.  # 类别 ID（整数）：类别索引。
                        - x_center (float): The X center of the bounding box (relative to image width).  # x_center（浮点数）：边界框的 X 中心（相对于图像宽度）。
                        - y_center (float): The Y center of the bounding box (relative to image height).  # y_center（浮点数）：边界框的 Y 中心（相对于图像高度）。
                        - width (float): The width of the bounding box (relative to image width).  # width（浮点数）：边界框的宽度（相对于图像宽度）。
                        - height (float): The height of the bounding box (relative to image height).  # height（浮点数）：边界框的高度（相对于图像高度）。
        label_map (dict): A dictionary that maps class IDs (integers) to class labels (strings).  # 将类别 ID（整数）映射到类别标签（字符串）的字典。

    Example:
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # It should include all annotated classes details  # 应包括所有注释类的详细信息
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)  # 调用示例
    """
    import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库用于绘图

    from ultralytics.utils.plotting import colors  # 从 ultralytics.utils.plotting 导入颜色函数

    img = np.array(Image.open(image_path))  # 读取图像并转换为 numpy 数组
    img_height, img_width = img.shape[:2]  # 获取图像的高度和宽度
    annotations = []  # 初始化注释列表
    with open(txt_path) as file:  # 打开注释文件
        for line in file:  # 遍历每一行
            class_id, x_center, y_center, width, height = map(float, line.split())  # 将行内容拆分并转换为浮点数
            x = (x_center - width / 2) * img_width  # 计算边界框左上角的 x 坐标
            y = (y_center - height / 2) * img_height  # 计算边界框左上角的 y 坐标
            w = width * img_width  # 计算边界框的实际宽度
            h = height * img_height  # 计算边界框的实际高度
            annotations.append((x, y, w, h, int(class_id)))  # 将边界框信息添加到注释列表
    fig, ax = plt.subplots(1)  # 创建一个图形和坐标轴，用于绘制图像和注释
    for x, y, w, h, label in annotations:  # 遍历所有注释
        color = tuple(c / 255 for c in colors(label, True))  # 获取并归一化 RGB 颜色
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")  # 创建矩形边界框
        ax.add_patch(rect)  # 将矩形添加到坐标轴
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # 计算颜色的亮度
        ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)  # 在边界框上方添加文本标签
    ax.imshow(img)  # 显示图像
    plt.show()  # 展示图形


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Convert a list of polygons to a binary mask of the specified image size.  # 将多边形列表转换为指定图像大小的二进制掩码

    Args:
        imgsz (tuple): The size of the image as (height, width).  # 图像的大小，格式为（高度，宽度）。
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where  # 多边形列表。每个多边形是一个形状为 [N, M] 的数组，
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.  # N 是多边形的数量，M 是点的数量，且 M % 2 = 0。
        color (int, optional): The color value to fill in the polygons on the mask. Defaults to 1.  # 在掩码上填充多边形的颜色值。默认为 1。
        downsample_ratio (int, optional): Factor by which to downsample the mask. Defaults to 1.  # 降采样掩码的因子。默认为 1。

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.  # 返回填充多边形的指定图像大小的二进制掩码。
    """
    mask = np.zeros(imgsz, dtype=np.uint8)  # 创建一个全零的掩码
    polygons = np.asarray(polygons, dtype=np.int32)  # 将多边形转换为 numpy 数组
    polygons = polygons.reshape((polygons.shape[0], -1, 2))  # 重塑多边形数组形状
    cv2.fillPoly(mask, polygons, color=color)  # 在掩码上填充多边形
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)  # 计算降采样后的高度和宽度
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1  # 注意：先填充多边形然后再调整大小是为了保持在 mask-ratio=1 时相同的损失计算方法
    return cv2.resize(mask, (nw, nh))  # 返回调整大小后的掩码


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Convert a list of polygons to a set of binary masks of the specified image size.  # 将多边形列表转换为指定图像大小的一组二进制掩码

    Args:
        imgsz (tuple): The size of the image as (height, width).  # 图像的大小，格式为（高度，宽度）。
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where  # 多边形列表。每个多边形是一个形状为 [N, M] 的数组，
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.  # N 是多边形的数量，M 是点的数量，且 M % 2 = 0。
        color (int): The color value to fill in the polygons on the masks.  # 在掩码上填充多边形的颜色值。
        downsample_ratio (int, optional): Factor by which to downsample each mask. Defaults to 1.  # 降采样每个掩码的因子。默认为 1。

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.  # 返回填充多边形的指定图像大小的一组二进制掩码。
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])  # 对每个多边形调用 polygon2mask 函数并返回结果数组


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""  # 返回 (640, 640) 的重叠掩码
    masks = np.zeros(  # 创建一个全零的掩码
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),  # 根据降采样因子调整大小
        dtype=np.int32 if len(segments) > 255 else np.uint8,  # 根据段的数量选择数据类型
    )
    areas = []  # 初始化区域列表
    ms = []  # 初始化掩码列表
    for si in range(len(segments)):  # 遍历每个段
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)  # 创建掩码
        ms.append(mask.astype(masks.dtype))  # 将掩码添加到列表，并转换为相应的数据类型
        areas.append(mask.sum())  # 计算并添加区域总和
    areas = np.asarray(areas)  # 转换区域列表为 numpy 数组
    index = np.argsort(-areas)  # 按区域大小排序索引
    ms = np.array(ms)[index]  # 根据排序索引重新排列掩码
    for i in range(len(segments)):  # 遍历每个段
        mask = ms[i] * (i + 1)  # 将掩码乘以索引值
        masks = masks + mask  # 更新重叠掩码
        masks = np.clip(masks, a_min=0, a_max=i + 1)  # 限制掩码值在有效范围内
    return masks, index  # 返回重叠掩码和索引


def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.  # 查找并返回与检测、分割或姿态数据集相关的 YAML 文件

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it  # 此函数首先在提供的目录的根级别搜索 YAML 文件，如果未找到，
    performs a recursive search. It prefers YAML files that have the same stem as the provided path. An AssertionError  # 进行递归搜索。它优先选择与提供路径具有相同主干的 YAML 文件。如果未找到 YAML 文件或找到多个 YAML 文件，则引发 AssertionError。
    is raised if no YAML file is found or if multiple YAML files are found.

    Args:
        path (Path): The directory path to search for the YAML file.  # 要搜索 YAML 文件的目录路径。

    Returns:
        (Path): The path of the found YAML file.  # 返回找到的 YAML 文件的路径。
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive  # 首先尝试根级别，然后进行递归搜索
    assert files, f"No YAML file found in '{path.resolve()}'"  # 如果未找到文件，则引发 AssertionError
    if len(files) > 1:  # 如果找到多个文件
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match  # 优先选择与提供路径主干匹配的 *.yaml 文件
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"  # 确保只找到一个 YAML 文件
    return files[0]  # 返回找到的 YAML 文件


def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.  # 如果在本地未找到数据集，则下载、验证和/或解压数据集

    This function checks the availability of a specified dataset, and if not found, it has the option to download and  # 此函数检查指定数据集的可用性，如果未找到，则可以选择下载和
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also  # 解压数据集。然后读取和解析随附的 YAML 数据，确保满足关键要求，并解析与数据集相关的路径。
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).  # 数据集的路径或数据集描述符（如 YAML 文件）。
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.  # 如果未找到，是否自动下载数据集。默认为 True。

    Returns:
        (dict): Parsed dataset information and paths.  # 返回解析后的数据集信息和路径。
    """
    file = check_file(dataset)  # 检查数据集文件

    # Download (optional)  # 下载（可选）
    extract_dir = ""  # 初始化提取目录
    if zipfile.is_zipfile(file) or is_tarfile(file):  # 如果文件是 zip 或 tar 格式
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)  # 安全下载并解压文件
        file = find_dataset_yaml(DATASETS_DIR / new_dir)  # 查找解压后的 YAML 文件
        extract_dir, autodownload = file.parent, False  # 更新提取目录并设置自动下载为 False

    # Read YAML  # 读取 YAML
    data = yaml_load(file, append_filename=True)  # dictionary  # 加载 YAML 数据为字典

    # Checks  # 检查
    for k in "train", "val":  # 遍历训练和验证键
        if k not in data:  # 如果键不在数据中
            if k != "val" or "validation" not in data:  # 如果不是验证且验证键不在数据中
                raise SyntaxError(  # 引发语法错误
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")  # 错误消息
                )
            LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")  # 记录警告信息
            data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key  # 将 'validation' 键替换为 'val' 键
    if "names" not in data and "nc" not in data:  # 如果数据中没有 'names' 和 'nc'
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))  # 引发语法错误
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:  # 如果同时存在 'names' 和 'nc'，且长度不匹配
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))  # 引发语法错误
    if "names" not in data:  # 如果数据中没有 'names'
        data["names"] = [f"class_{i}" for i in range(data["nc"])]  # 根据 'nc' 创建默认类名
    else:
        data["nc"] = len(data["names"])  # 更新 'nc' 为 'names' 的长度

    data["names"] = check_class_names(data["names"])  # 检查类名的有效性

    # Resolve paths  # 解析路径
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root  # 数据集根目录
    if not path.is_absolute():  # 如果路径不是绝对路径
        path = (DATASETS_DIR / path).resolve()  # 解析为绝对路径

    # Set paths  # 设置路径
    data["path"] = path  # download scripts  # 下载脚本
    for k in "train", "val", "test", "minival":  # 遍历训练、验证、测试和小验证键
        if data.get(k):  # 如果键存在
            if isinstance(data[k], str):  # 如果是字符串
                x = (path / data[k]).resolve()  # 解析为绝对路径
                if not x.exists() and data[k].startswith("../"):  # 如果路径不存在且以 "../" 开头
                    x = (path / data[k][3:]).resolve()  # 解析为绝对路径
                data[k] = str(x)  # 更新为字符串路径
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]  # 更新为解析后的路径列表

    # Parse YAML  # 解析 YAML
    val, s = (data.get(x) for x in ("val", "download"))  # 获取验证和下载路径
    if val:  # 如果验证路径存在
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path  # 解析验证路径
        if not all(x.exists() for x in val):  # 如果验证路径中有不存在的路径
            name = clean_url(dataset)  # dataset name with URL auth stripped  # 清理数据集名称
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"  # 错误消息
            if s and autodownload:  # 如果下载链接存在且允许自动下载
                LOGGER.warning(m)  # 记录警告信息
            else:  # 如果不允许自动下载
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_FILE}'"  # 提示下载目录
                raise FileNotFoundError(m)  # 引发文件未找到错误
            t = time.time()  # 记录当前时间
            r = None  # success  # 初始化成功标志
            if s.startswith("http") and s.endswith(".zip"):  # 如果是 URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)  # 安全下载数据集
            elif s.startswith("bash "):  # 如果是 bash 脚本
                LOGGER.info(f"Running {s} ...")  # 记录正在运行的脚本信息
                r = os.system(s)  # 执行 bash 脚本
            else:  # python script  # 如果是 Python 脚本
                exec(s, {"yaml": data})  # 执行 Python 脚本
            dt = f"({round(time.time() - t, 1)}s)"  # 计算下载时间
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"failure {dt} ❌"  # 下载结果消息
            LOGGER.info(f"Dataset download {s}\n")  # 记录下载结果
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # download fonts  # 检查字体

    return data  # dictionary  # 返回数据集信息字典

def check_cls_dataset(dataset, split=""):
    """
    Checks a classification dataset such as Imagenet.  # 检查分类数据集，例如 Imagenet

    This function accepts a [dataset](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/data/utils.py:387:0-462:91) name and attempts to retrieve the corresponding dataset information.  # 此函数接受一个 [dataset](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/data/utils.py:387:0-462:91) 名称，并尝试检索相应的数据集信息。
    If the dataset is not found locally, it attempts to download the dataset from the internet and save it locally.  # 如果在本地未找到数据集，它将尝试从互联网下载数据集并将其保存在本地。

    Args:
        dataset (str | Path): The name of the dataset.  # 数据集的名称。
        split (str, optional): The split of the dataset. Either 'val', 'test', or ''. Defaults to ''.  # 数据集的划分。可以是 'val'、'test' 或 ''。默认为 ''。

    Returns:
        (dict): A dictionary containing the following keys:  # 返回一个字典，包含以下键：
            - 'train' (Path): The directory path containing the training set of the dataset.  # 'train'（Path）：包含数据集训练集的目录路径。
            - 'val' (Path): The directory path containing the validation set of the dataset.  # 'val'（Path）：包含数据集验证集的目录路径。
            - 'test' (Path): The directory path containing the test set of the dataset.  # 'test'（Path）：包含数据集测试集的目录路径。
            - 'nc' (int): The number of classes in the dataset.  # 'nc'（int）：数据集中的类别数量。
            - 'names' (dict): A dictionary of class names in the dataset.  # 'names'（dict）：数据集中的类别名称字典。
    """
    # Download (optional if dataset=https://file.zip is passed directly)  # 下载（如果直接传递 dataset=https://file.zip，则为可选）
    if str(dataset).startswith(("http:/", "https:/")):  # 如果数据集是 URL
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)  # 安全下载数据集
    elif Path(dataset).suffix in {".zip", ".tar", ".gz"}:  # 如果数据集文件是压缩格式
        file = check_file(dataset)  # 检查文件有效性
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)  # 安全下载并解压文件

    dataset = Path(dataset)  # 将数据集转换为 Path 对象
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()  # 解析数据集目录
    if not data_dir.is_dir():  # 如果数据目录不存在
        LOGGER.warning(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")  # 记录警告信息，尝试下载
        t = time.time()  # 记录当前时间
        if str(dataset) == "imagenet":  # 如果数据集是 imagenet
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)  # 运行下载脚本
        else:  # 其他数据集
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip"  # 构建下载 URL
            download(url, dir=data_dir.parent)  # 下载数据集
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"  # 下载成功消息
        LOGGER.info(s)  # 记录下载成功信息
    train_set = data_dir / "train"  # 获取训练集路径
    val_set = (  # 获取验证集路径
        data_dir / "val"  # 优先检查 'val'
        if (data_dir / "val").exists()  # 如果 'val' 存在
        else data_dir / "validation"  # 否则检查 'validation'
        if (data_dir / "validation").exists()  # 如果 'validation' 存在
        else None  # 否则返回 None
    )  # data/test or data/val
    test_set = data_dir / "test" if (data_dir / "test").exists() else None  # 获取测试集路径
    if split == "val" and not val_set:  # 如果请求验证集但未找到
        LOGGER.warning("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")  # 记录警告信息，使用测试集
    elif split == "test" and not test_set:  # 如果请求测试集但未找到
        LOGGER.warning("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")  # 记录警告信息，使用验证集

    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # 计算类别数量
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  # 获取类别名称列表
    names = dict(enumerate(sorted(names)))  # 将类别名称转换为字典

    # Print to console  # 打印到控制台
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():  # 遍历训练、验证和测试集
        prefix = f"{colorstr(f'{k}:')} {v}..."  # 构建前缀
        if v is None:  # 如果路径为 None
            LOGGER.info(prefix)  # 记录信息
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]  # 获取图像文件
            nf = len(files)  # 计算文件数量
            nd = len({file.parent for file in files})  # 计算目录数量
            if nf == 0:  # 如果没有文件
                if k == "train":  # 如果是训练集
                    raise FileNotFoundError(emojis(f"{dataset} '{k}:' no training images found ❌ "))  # 引发文件未找到错误
                else:
                    LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: WARNING ⚠️ no images found")  # 记录警告信息
            elif nd != nc:  # 如果目录数量与类别数量不匹配
                LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: ERROR ❌️ requires {nc} classes, not {nd}")  # 记录错误信息
            else:
                LOGGER.info(f"{prefix} found {nf} images in {nd} classes ✅ ")  # 记录成功信息

    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names}  # 返回数据集信息字典


class HUBDatasetStats:
    """
    A class for generating HUB dataset JSON and `-hub` dataset directory.  # 用于生成 HUB 数据集 JSON 和 `-hub` 数据集目录的类

    Args:
        path (str): Path to data.yaml or data.zip (with data.yaml inside data.zip). Default is 'coco8.yaml'.  # data.yaml 或 data.zip 的路径（其中 data.yaml 在 data.zip 内）。默认为 'coco8.yaml'。
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify'. Default is 'detect'.  # 数据集任务。选项为 'detect'、'segment'、'pose'、'classify'。默认为 'detect'。
        autodownload (bool): Attempt to download dataset if not found locally. Default is False.  # 如果未找到数据集，是否尝试下载。默认为 False。

    Example:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets  # 从 https://github.com/ultralytics/hub/tree/main/example_datasets 下载 *.zip 文件
            i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.  # 例如，coco8.zip 的链接。
        ```python
        from ultralytics.data.utils import HUBDatasetStats

        stats = HUBDatasetStats("path/to/coco8.zip", task="detect")  # detect dataset  # 检测数据集
        stats = HUBDatasetStats("path/to/coco8-seg.zip", task="segment")  # segment dataset  # 分割数据集
        stats = HUBDatasetStats("path/to/coco8-pose.zip", task="pose")  # pose dataset  # 姿态数据集
        stats = HUBDatasetStats("path/to/dota8.zip", task="obb")  # OBB dataset  # OBB 数据集
        stats = HUBDatasetStats("path/to/imagenet10.zip", task="classify")  # classification dataset  # 分类数据集

        stats.get_json(save=True)  # 获取 JSON
        stats.process_images()  # 处理图像
        ```
    """

    def __init__(self, path="coco8.yaml", task="detect", autodownload=False):
        """Initialize class."""  # 初始化类
        path = Path(path).resolve()  # 解析路径
        LOGGER.info(f"Starting HUB dataset checks for {path}....")  # 记录开始检查数据集的信息

        self.task = task  # detect, segment, pose, classify, obb  # 任务类型
        if self.task == "classify":  # 如果任务是分类
            unzip_dir = unzip_file(path)  # 解压文件
            data = check_cls_dataset(unzip_dir)  # 检查分类数据集
            data["path"] = unzip_dir  # 设置数据路径
        else:  # detect, segment, pose, obb  # 对于其他任务
            _, data_dir, yaml_path = self._unzip(Path(path))  # 解压并获取数据目录和 YAML 路径
            try:
                # Load YAML with checks  # 加载 YAML 并进行检查
                data = yaml_load(yaml_path)  # 加载 YAML 数据
                data["path"] = ""  # strip path since YAML should be in dataset root for all HUB datasets  # 清除路径，因为 YAML 应在所有 HUB 数据集的根目录中
                yaml_save(yaml_path, data)  # 保存 YAML 数据
                data = check_det_dataset(yaml_path, autodownload)  # dict  # 检查检测数据集
                data["path"] = data_dir  # YAML 路径应设置为 ''（相对）或父目录（绝对）
            except Exception as e:
                raise Exception("error/HUB/dataset_stats/init") from e  # 引发异常

        self.hub_dir = Path(f"{data['path']}-hub")  # 设置 HUB 目录
        self.im_dir = self.hub_dir / "images"  # 设置图像目录
        self.stats = {"nc": len(data["names"]), "names": list(data["names"].values())}  # statistics dictionary  # 统计信息字典
        self.data = data  # 保存数据

    @staticmethod
    def _unzip(path):
        """Unzip data.zip."""  # 解压 data.zip
        if not str(path).endswith(".zip"):  # path is data.yaml  # 如果路径不是 ZIP 文件
            return False, None, path  # 返回 False 和路径
        unzip_dir = unzip_file(path, path=path.parent)  # 解压文件
        assert unzip_dir.is_dir(), (  # 确保解压后的目录存在
            f"Error unzipping {path}, {unzip_dir} not found. path/to/abc.zip MUST unzip to path/to/abc/"  # 错误消息
        )
        return True, str(unzip_dir), find_dataset_yaml(unzip_dir)  # zipped, data_dir, yaml_path  # 返回 True、解压目录和 YAML 路径

    def _hub_ops(self, f):
        """Saves a compressed image for HUB previews."""  # 保存压缩图像以供 HUB 预览
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub  # 保存到 dataset-hub

    def get_json(self, save=False, verbose=False):
        """Return dataset JSON for Ultralytics HUB."""  # 返回 Ultralytics HUB 的数据集 JSON

        def _round(labels):
            """Update labels to integer class and 4 decimal place floats."""  # 更新标签为整数类和 4 位小数浮点数
            if self.task == "detect":  # 如果任务是检测
                coordinates = labels["bboxes"]  # 获取边界框坐标
            elif self.task in {"segment", "obb"}:  # Segment and OBB use segments. OBB segments are normalized xyxyxyxy  # 分割和 OBB 使用段。OBB 段是标准化的 xyxyxyxy
                coordinates = [x.flatten() for x in labels["segments"]]  # 获取段坐标
            elif self.task == "pose":  # 如果任务是姿态
                n, nk, nd = labels["keypoints"].shape  # 获取关键点的形状
                coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, nk * nd)), 1)  # 合并边界框和关键点
            else:
                raise ValueError(f"Undefined dataset task={self.task}.")  # 引发未定义任务的错误
            zipped = zip(labels["cls"], coordinates)  # 将类别和坐标压缩在一起
            return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]  # 返回更新后的标签

        for split in "train", "val", "test":  # 遍历训练、验证和测试集
            self.stats[split] = None  # predefine  # 预定义
            path = self.data.get(split)  # 获取当前划分的路径

            # Check split  # 检查划分
            if path is None:  # no split  # 如果没有划分
                continue  # 跳过
            files = [f for f in Path(path).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]  # 获取划分中的图像文件
            if not files:  # no images  # 如果没有图像
                continue  # 跳过

            # Get dataset statistics  # 获取数据集统计信息
            if self.task == "classify":  # 如果任务是分类
                from torchvision.datasets import ImageFolder  # scope for faster 'import ultralytics'  # 为了更快的 'import ultralytics'

                dataset = ImageFolder(self.data[split])  # 创建 ImageFolder 数据集

                x = np.zeros(len(dataset.classes)).astype(int)  # 初始化类别计数
                for im in dataset.imgs:  # 遍历图像
                    x[im[1]] += 1  # 更新类别计数

                self.stats[split] = {  # 更新统计信息
                    "instance_stats": {"total": len(dataset), "per_class": x.tolist()},  # 实例统计
                    "image_stats": {"total": len(dataset), "unlabelled": 0, "per_class": x.tolist()},  # 图像统计
                    "labels": [{Path(k).name: v} for k, v in dataset.imgs],  # 标签信息
                }
            else:  # detect, segment, pose, obb  # 对于其他任务
                from ultralytics.data import YOLODataset  # 导入 YOLODataset

                dataset = YOLODataset(img_path=self.data[split], data=self.data, task=self.task)  # 创建 YOLO 数据集
                x = np.array(  # 获取类别计数
                    [
                        np.bincount(label["cls"].astype(int).flatten(), minlength=self.data["nc"])  # 计算类别计数
                        for label in TQDM(dataset.labels, total=len(dataset), desc="Statistics")  # 显示进度条
                    ]
                )  # shape(128x80)
                self.stats[split] = {  # 更新统计信息
                    "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},  # 实例统计
                    "image_stats": {  # 图像统计
                        "total": len(dataset),
                        "unlabelled": int(np.all(x == 0, 1).sum()),  # 统计未标记的图像
                        "per_class": (x > 0).sum(0).tolist(),  # 每个类的图像数量
                    },
                    "labels": [{Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)],  # 标签信息
                }

        # Save, print and return  # 保存、打印并返回
        if save:  # 如果需要保存
            self.hub_dir.mkdir(parents=True, exist_ok=True)  # 创建 dataset-hub/
            stats_path = self.hub_dir / "stats.json"  # 设置统计信息路径
            LOGGER.info(f"Saving {stats_path.resolve()}...")  # 记录保存信息
            with open(stats_path, "w") as f:  # 打开文件以写入
                json.dump(self.stats, f)  # 保存 stats.json
        if verbose:  # 如果需要详细信息
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))  # 记录统计信息
        return self.stats  # 返回统计信息

    def process_images(self):
        """Compress images for Ultralytics HUB."""  # 压缩图像以供 Ultralytics HUB 使用
        from ultralytics.data import YOLODataset  # ClassificationDataset  # 分类数据集

        self.im_dir.mkdir(parents=True, exist_ok=True)  # 创建 dataset-hub/images/
        for split in "train", "val", "test":  # 遍历训练、验证和测试集
            if self.data.get(split) is None:  # 如果没有数据
                continue  # 跳过
            dataset = YOLODataset(img_path=self.data[split], data=self.data)  # 创建 YOLO 数据集
            with ThreadPool(NUM_THREADS) as pool:  # 使用线程池
                for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f"{split} images"):  # 处理图像
                    pass
        LOGGER.info(f"Done. All images saved to {self.im_dir}")  # 记录完成信息
        return self.im_dir  # 返回图像目录


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the Python  # 压缩单个图像文件以减少大小，同时使用 Python Imaging Library (PIL) 或 OpenCV 库保持其纵横比和质量。
    Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will not be  # 如果输入图像小于最大尺寸，则不会调整大小。
    resized.

    Args:
        f (str): The path to the input image file.  # 输入图像文件的路径。
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.  # 输出图像文件的路径。如果未指定，则输入文件将被覆盖。
        max_dim (int, optional): The maximum dimension (width or height) of the output image. Default is 1920 pixels.  # 输出图像的最大尺寸（宽度或高度）。默认为 1920 像素。
        quality (int, optional): The image compression quality as a percentage. Default is 50%.  # 图像压缩质量的百分比。默认为 50%。

    Example:
        ```python
        from pathlib import Path
        from ultralytics.data.utils import compress_one_image

        for f in Path("path/to/dataset").rglob("*.jpg"):  # 遍历数据集中所有的 JPG 文件
            compress_one_image(f)  # 压缩图像
        ```
    """
    try:  # use PIL  # 使用 PIL
        im = Image.open(f)  # 打开图像文件
        r = max_dim / max(im.height, im.width)  # ratio  # 计算比例
        if r < 1.0:  # image too large  # 如果图像太大
            im = im.resize((int(im.width * r), int(im.height * r)))  # 调整图像大小
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # 保存图像
    except Exception as e:  # use OpenCV  # 使用 OpenCV
        LOGGER.info(f"WARNING ⚠️ HUB ops PIL failure {f}: {e}")  # 记录警告信息
        im = cv2.imread(f)  # 读取图像
        im_height, im_width = im.shape[:2]  # 获取图像高度和宽度
        r = max_dim / max(im_height, im_width)  # ratio  # 计算比例
        if r < 1.0:  # image too large  # 如果图像太大
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)  # 调整图像大小
        cv2.imwrite(str(f_new or f), im)  # 保存图像


def autosplit(path=DATASETS_DIR / "coco8/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.  # 自动将数据集拆分为 train/val/test 划分，并将结果保存到 autosplit_*.txt 文件中。

    Args:
        path (Path, optional): Path to images directory. Defaults to DATASETS_DIR / 'coco8/images'.  # 图像目录的路径。默认为 DATASETS_DIR / 'coco8/images'。
        weights (list | tuple, optional): Train, validation, and test split fractions. Defaults to (0.9, 0.1, 0.0).  # 训练、验证和测试划分的比例。默认为 (0.9, 0.1, 0.0)。
        annotated_only (bool, optional): If True, only images with an associated txt file are used. Defaults to False.  # 如果为 True，则仅使用与 txt 文件关联的图像。默认为 False。

    Example:
        ```python
        from ultralytics.data.utils import autosplit  # 导入 autosplit 函数

        autosplit()  # 调用 autosplit 函数
        ```
    """
    path = Path(path)  # images dir  # 图像目录
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only  # 仅获取图像文件
    n = len(files)  # number of files  # 文件数量
    random.seed(0)  # for reproducibility  # 为可重复性设置随机种子
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split  # 为每个图像分配划分

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files  # 3 个 txt 文件
    for x in txt:  # 遍历 txt 文件
        if (path.parent / x).exists():  # 如果文件已存在
            (path.parent / x).unlink()  # remove existing  # 删除现有文件

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)  # 记录自动拆分信息
    for i, img in TQDM(zip(indices, files), total=n):  # 遍历图像和索引
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label  # 检查标签
            with open(path.parent / txt[i], "a") as f:  # 打开相应的 txt 文件以追加
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # 将图像路径添加到 txt 文件


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""  # 从路径加载 Ultralytics *.cache 字典
    import gc  # 导入垃圾回收模块

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585  # 禁用垃圾回收以减少加载时间
    cache = np.load(str(path), allow_pickle=True).item()  # load dict  # 加载字典
    gc.enable()  # 启用垃圾回收
    return cache  # 返回缓存


def save_dataset_cache_file(prefix, path, x, version):
    """Save an Ultralytics dataset *.cache dictionary x to path."""  # 将 Ultralytics 数据集 *.cache 字典 x 保存到路径
    x["version"] = version  # add cache version  # 添加缓存版本
    if is_dir_writeable(path.parent):  # 检查目录是否可写
        if path.exists():  # 如果路径已存在
            path.unlink()  # remove *.cache file if exists  # 删除现有的 *.cache 文件
        np.save(str(path), x)  # save cache for next time  # 保存缓存以供下次使用
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix  # 删除 .npy 后缀
        LOGGER.info(f"{prefix}New cache created: {path}")  # 记录新缓存创建信息
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")  # 记录警告信息，缓存未保存