# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import itertools  # 导入itertools库，用于创建迭代器
from glob import glob  # 从glob模块导入glob函数，用于文件路径匹配
from math import ceil  # 从math模块导入ceil函数，用于向上取整
from pathlib import Path  # 从pathlib模块导入Path类，用于处理文件路径

import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入Numpy库，用于数值计算
from PIL import Image  # 从PIL库导入Image类，用于处理图像

from ultralytics.data.utils import exif_size, img2label_paths  # 导入自定义的工具函数
from ultralytics.utils import TQDM  # 导入TQDM类，用于显示进度条
from ultralytics.utils.checks import check_requirements  # 导入检查依赖的函数


def bbox_iof(polygon1, bbox2, eps=1e-6):
    """
    Calculate Intersection over Foreground (IoF) between polygons and bounding boxes.
    计算多边形和边界框之间的前景交集（IoF）。

    Args:
        polygon1 (np.ndarray): Polygon coordinates, shape (n, 8).
        polygon1 (np.ndarray): 多边形坐标，形状为(n, 8)。
        bbox2 (np.ndarray): Bounding boxes, shape (n, 4).
        bbox2 (np.ndarray): 边界框，形状为(n, 4)。
        eps (float, optional): Small value to prevent division by zero. Defaults to 1e-6.
        eps (float, 可选): 防止除以零的小值。默认为1e-6。

    Returns:
        (np.ndarray): IoF scores, shape (n, 1) or (n, m) if bbox2 is (m, 4).
        (np.ndarray): IoF得分，形状为(n, 1)或(n, m)如果bbox2为(m, 4)。

    Note:
        Polygon format: [x1, y1, x2, y2, x3, y3, x4, y4].
        多边形格式: [x1, y1, x2, y2, x3, y3, x4, y4]。
        Bounding box format: [x_min, y_min, x_max, y_max].
        边界框格式: [x_min, y_min, x_max, y_max]。
    """
    check_requirements("shapely")  # 检查是否安装了shapely库
    from shapely.geometry import Polygon  # 从shapely库导入Polygon类

    polygon1 = polygon1.reshape(-1, 4, 2)  # 将多边形坐标重塑为(n, 4, 2)的形状
    lt_point = np.min(polygon1, axis=-2)  # left-top  # 获取左上角点
    rb_point = np.max(polygon1, axis=-2)  # right-bottom  # 获取右下角点
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)  # 创建边界框

    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])  # 计算左上角的最大值
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])  # 计算右下角的最小值
    wh = np.clip(rb - lt, 0, np.inf)  # 计算宽高并限制在非负范围内
    h_overlaps = wh[..., 0] * wh[..., 1]  # 计算重叠面积

    left, top, right, bottom = (bbox2[..., i] for i in range(4))  # 获取边界框的四个边
    polygon2 = np.stack([left, top, right, top, right, bottom, left, bottom], axis=-1).reshape(-1, 4, 2)  # 创建第二个多边形

    sg_polys1 = [Polygon(p) for p in polygon1]  # 将多边形1转换为shapely多边形对象
    sg_polys2 = [Polygon(p) for p in polygon2]  # 将多边形2转换为shapely多边形对象
    overlaps = np.zeros(h_overlaps.shape)  # 初始化重叠面积数组
    for p in zip(*np.nonzero(h_overlaps)):  # 遍历重叠面积不为零的索引
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area  # 计算多边形的交集面积
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)  # 计算多边形的面积
    unions = unions[..., None]  # 扩展维度以便于广播

    unions = np.clip(unions, eps, np.inf)  # 限制面积在eps和无穷大之间
    outputs = overlaps / unions  # 计算IoF得分
    if outputs.ndim == 1:
        outputs = outputs[..., None]  # 如果输出是一维，则扩展维度
    return outputs  # 返回IoF得分


def load_yolo_dota(data_root, split="train"):
    """
    Load DOTA dataset.
    加载DOTA数据集。

    Args:
        data_root (str): Data root.
        data_root (str): 数据根目录。
        split (str): The split data set, could be `train` or `val`.
        split (str): 数据集的划分，可以是`train`或`val`。

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."  # 确保划分是'train'或'val'
    im_dir = Path(data_root) / "images" / split  # 构建图像目录路径
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."  # 确保图像目录存在
    im_files = glob(str(Path(data_root) / "images" / split / "*"))  # 获取图像文件列表
    lb_files = img2label_paths(im_files)  # 获取标签文件路径
    annos = []  # 初始化注释列表
    for im_file, lb_file in zip(im_files, lb_files):  # 遍历图像和标签文件
        w, h = exif_size(Image.open(im_file))  # 获取图像的原始尺寸
        with open(lb_file) as f:  # 打开标签文件
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]  # 读取标签数据
            lb = np.array(lb, dtype=np.float32)  # 转换为Numpy数组
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))  # 将注释信息添加到列表
    return annos  # 返回注释列表


def get_windows(im_size, crop_sizes=(1024,), gaps=(200,), im_rate_thr=0.6, eps=0.01):
    """
    Get the coordinates of windows.
    获取窗口的坐标。

    Args:
        im_size (tuple): Original image size, (h, w).
        im_size (tuple): 原始图像大小，(h, w)。
        crop_sizes (List(int)): Crop size of windows.
        crop_sizes (List(int)): 窗口的裁剪大小。
        gaps (List(int)): Gap between crops.
        gaps (List(int)): 裁剪之间的间隙。
        im_rate_thr (float): Threshold of windows areas divided by image areas.
        im_rate_thr (float): 窗口面积与图像面积之比的阈值。
        eps (float): Epsilon value for math operations.
        eps (float): 数学运算的epsilon值。
    """
    h, w = im_size  # 获取图像的高度和宽度
    windows = []  # 初始化窗口列表
    for crop_size, gap in zip(crop_sizes, gaps):  # 遍历裁剪大小和间隙
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"  # 确保裁剪大小大于间隙
        step = crop_size - gap  # 计算步幅

        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)  # 计算在宽度方向上可以放置的窗口数量
        xs = [step * i for i in range(xn)]  # 计算窗口的起始位置
        if len(xs) > 1 and xs[-1] + crop_size > w:  # 如果最后一个窗口超出图像宽度
            xs[-1] = w - crop_size  # 调整最后一个窗口的起始位置

        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)  # 计算在高度方向上可以放置的窗口数量
        ys = [step * i for i in range(yn)]  # 计算窗口的起始位置
        if len(ys) > 1 and ys[-1] + crop_size > h:  # 如果最后一个窗口超出图像高度
            ys[-1] = h - crop_size  # 调整最后一个窗口的起始位置

        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)  # 计算窗口的起始坐标
        stop = start + crop_size  # 计算窗口的结束坐标
        windows.append(np.concatenate([start, stop], axis=1))  # 将起始和结束坐标合并并添加到窗口列表
    windows = np.concatenate(windows, axis=0)  # 合并所有窗口坐标

    im_in_wins = windows.copy()  # 复制窗口坐标
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)  # 限制窗口的左边界在图像宽度范围内
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)  # 限制窗口的上边界在图像高度范围内
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])  # 计算窗口面积
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])  # 计算窗口的原始面积
    im_rates = im_areas / win_areas  # 计算图像面积与窗口面积之比
    if not (im_rates > im_rate_thr).any():  # 如果没有窗口满足面积比阈值
        max_rate = im_rates.max()  # 获取最大的面积比
        im_rates[abs(im_rates - max_rate) < eps] = 1  # 将接近最大比率的窗口设置为1
    return windows[im_rates > im_rate_thr]  # 返回满足面积比阈值的窗口


def get_window_obj(anno, windows, iof_thr=0.7):
    """Get objects for each window."""
    # 获取每个窗口的对象。
    h, w = anno["ori_size"]  # 获取原始图像的高度和宽度
    label = anno["label"]  # 获取标签信息
    if len(label):  # 如果有标签
        label[:, 1::2] *= w  # 将标签的x坐标乘以图像宽度
        label[:, 2::2] *= h  # 将标签的y坐标乘以图像高度
        iofs = bbox_iof(label[:, 1:], windows)  # 计算每个标签与窗口的IoF
        # Unnormalized and misaligned coordinates
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # 返回满足IoF阈值的窗口注释
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # 返回空的窗口注释


def crop_and_save(anno, windows, window_objs, im_dir, lb_dir, allow_background_images=True):
    """
    Crop images and save new labels.
    裁剪图像并保存新标签。

    Args:
        anno (dict): Annotation dict, including `filepath`, `label`, `ori_size` as its keys.
        anno (dict): 注释字典，包括`filepath`、`label`、`ori_size`作为其键。
        windows (list): A list of windows coordinates.
        windows (list): 窗口坐标的列表。
        window_objs (list): A list of labels inside each window.
        window_objs (list): 每个窗口内标签的列表。
        im_dir (str): The output directory path of images.
        im_dir (str): 图像的输出目录路径。
        lb_dir (str): The output directory path of labels.
        lb_dir (str): 标签的输出目录路径。
        allow_background_images (bool): Whether to include background images without labels.
        allow_background_images (bool): 是否包含没有标签的背景图像。

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    im = cv2.imread(anno["filepath"])  # 读取原始图像
    name = Path(anno["filepath"]).stem  # 获取图像文件名（不带扩展名）
    for i, window in enumerate(windows):  # 遍历每个窗口
        x_start, y_start, x_stop, y_stop = window.tolist()  # 获取窗口的坐标
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"  # 生成新文件名
        patch_im = im[y_start:y_stop, x_start:x_stop]  # 裁剪图像
        ph, pw = patch_im.shape[:2]  # 获取裁剪图像的高度和宽度

        label = window_objs[i]  # 获取当前窗口的标签
        if len(label) or allow_background_images:  # 如果有标签或允许背景图像
            cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)  # 保存裁剪后的图像
        if len(label):  # 如果有标签
            label[:, 1::2] -= x_start  # 将标签的x坐标减去窗口的左坐标
            label[:, 2::2] -= y_start  # 将标签的y坐标减去窗口的上坐标
            label[:, 1::2] /= pw  # 将标签的x坐标归一化
            label[:, 2::2] /= ph  # 将标签的y坐标归一化

            with open(Path(lb_dir) / f"{new_name}.txt", "w") as f:  # 打开标签文件以写入
                for lb in label:  # 遍历标签
                    formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]  # 格式化坐标
                    f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")  # 写入标签信息


def split_images_and_labels(data_root, save_dir, split="train", crop_sizes=(1024,), gaps=(200,)):
    """
    Split both images and labels.
    同时拆分图像和标签。

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - split
                - labels
                    - split
        and the output directory structure is:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / "images" / split  # 构建输出图像目录路径
    im_dir.mkdir(parents=True, exist_ok=True)  # 创建图像目录
    lb_dir = Path(save_dir) / "labels" / split  # 构建输出标签目录路径
    lb_dir.mkdir(parents=True, exist_ok=True)  # 创建标签目录

    annos = load_yolo_dota(data_root, split=split)  # 加载DOTA数据集
    for anno in TQDM(annos, total=len(annos), desc=split):  # 遍历注释
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)  # 获取窗口坐标
        window_objs = get_window_obj(anno, windows)  # 获取每个窗口的对象
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))  # 裁剪并保存图像和标签


def split_trainval(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    Split train and val set of DOTA.
    拆分DOTA的训练和验证集。

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        and the output directory structure is:
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    crop_sizes, gaps = [], []  # 初始化裁剪大小和间隙列表
    for r in rates:  # 遍历比例
        crop_sizes.append(int(crop_size / r))  # 计算裁剪大小
        gaps.append(int(gap / r))  # 计算间隙
    for split in ["train", "val"]:  # 遍历训练和验证集
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)  # 拆分图像和标签


def split_test(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    Split test set of DOTA, labels are not included within this set.
    拆分DOTA的测试集，标签不包含在此集中。

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - test
        and the output directory structure is:
            - save_dir
                - images
                    - test
    """
    crop_sizes, gaps = [], []  # 初始化裁剪大小和间隙列表
    for r in rates:  # 遍历比例
        crop_sizes.append(int(crop_size / r))  # 计算裁剪大小
        gaps.append(int(gap / r))  # 计算间隙
    save_dir = Path(save_dir) / "images" / "test"  # 构建输出测试图像目录路径
    save_dir.mkdir(parents=True, exist_ok=True)  # 创建测试图像目录

    im_dir = Path(data_root) / "images" / "test"  # 构建输入测试图像目录路径
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."  # 确保输入目录存在
    im_files = glob(str(im_dir / "*"))  # 获取测试图像文件列表
    for im_file in TQDM(im_files, total=len(im_files), desc="test"):  # 遍历测试图像
        w, h = exif_size(Image.open(im_file))  # 获取图像的原始尺寸
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)  # 获取窗口坐标
        im = cv2.imread(im_file)  # 读取图像
        name = Path(im_file).stem  # 获取图像文件名（不带扩展名）
        for window in windows:  # 遍历每个窗口
            x_start, y_start, x_stop, y_stop = window.tolist()  # 获取窗口的坐标
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"  # 生成新文件名
            patch_im = im[y_start:y_stop, x_start:x_stop]  # 裁剪图像
            cv2.imwrite(str(save_dir / f"{new_name}.jpg"), patch_im)  # 保存裁剪后的图像


if __name__ == "__main__":
    split_trainval(data_root="DOTAv2", save_dir="DOTAv2-split")  # 拆分训练和验证集
    split_test(data_root="DOTAv2", save_dir="DOTAv2-split")  # 拆分测试集