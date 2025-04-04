# Copyright (c) Facebook, Inc. and its affiliates.
# 导入所需的Python标准库
import functools  # 导入functools模块，用于高阶函数和操作可调用对象
import json      # 导入json模块，用于JSON数据的编码和解码
import logging   # 导入logging模块，用于日志记录
import multiprocessing as mp  # 导入多进程模块，用于并行处理
import numpy as np  # 导入numpy库，用于数值计算
import os        # 导入os模块，用于操作系统相关功能
from itertools import chain  # 导入chain函数，用于展平嵌套序列
import pycocotools.mask as mask_util  # 导入COCO工具库中的mask工具
from PIL import Image  # 导入PIL库的Image模块，用于图像处理

# 导入detectron2相关模块
from detectron2.structures import BoxMode  # 导入边界框模式定义
from detectron2.utils.comm import get_world_size  # 导入获取分布式训练世界大小的函数
from detectron2.utils.file_io import PathManager  # 导入路径管理器
from detectron2.utils.logger import setup_logger  # 导入日志设置函数

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    # OpenCV是一个可选依赖，不是必需的
    pass


logger = logging.getLogger(__name__)  # 获取当前模块的logger


def _get_cityscapes_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    # 扫描目录获取所有的数据
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")  # 记录找到的城市数量
    # 遍历每个目录
    for city in cities:
        # 构建图像目录和标注目录的路径
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        # 遍历目录下的所有图像文件
        for basename in PathManager.ls(city_img_dir):
            # 构建完整的图像文件路径
            image_file = os.path.join(city_img_dir, basename)

            # 检查文件后缀名是否正确
            suffix = "leftImg8bit.png"
            assert basename.endswith(suffix), basename
            # 去除文件后缀，获取基础文件名
            basename = basename[: -len(suffix)]

            # 构建对应的实例标注、语义标注和多边形标注文件路径
            instance_file = os.path.join(city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")

            # 将所有相关文件路径添加到列表中
            files.append((image_file, instance_file, label_file, json_file))
    # 确保找到了图像文件
    assert len(files), "No images found in {}".format(image_dir)
    # 验证第一组文件的所有路径都是有效的
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_cityscapes_instances(image_dir, gt_dir, from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        # 原始数据集的路径，例如："~/cityscapes/leftImg8bit/train"
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        # 原始标注的路径，例如："~/cityscapes/gtFine/train"
        from_json (bool): whether to read annotations from the raw json file or the png files.
        # 是否从原始json文件读取标注，而不是从png文件读取
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).
        # 是否将分割表示为多边形（COCO格式）而不是掩码（Cityscapes格式）

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
        # 返回Detectron2标准格式的字典列表（参见`使用自定义数据集</tutorials/datasets.html>`_）
    """
    if from_json:
        assert to_polygons, (
            "Cityscapes's json annotations are in polygon format. "
            "Converting to mask format is not supported now."
            # Cityscapes的json标注是多边形格式的，目前不支持转换为掩码格式
        )
    files = _get_cityscapes_files(image_dir, gt_dir)

    logger.info("Preprocessing cityscapes annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    # 这个处理过程仍然不够快：所有工作进程都会执行重复的工作，在8GPU服务器上可能需要10分钟
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))  # 创建进程池进行并行处理

    ret = pool.map(
        functools.partial(_cityscapes_files_to_dict, from_json=from_json, to_polygons=to_polygons),
        files,
    )  # 使用进程池并行处理所有文件
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))  # 记录加载的图像数量

    # Map cityscape ids to contiguous ids
    # 将Cityscapes的ID映射为连续的ID
    from cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]  # 筛选有实例且不忽略的标签
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}  # 创建ID映射字典
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]  # 更新类别ID为连续ID
    return ret


def load_cityscapes_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        # 原始数据集的路径，例如："~/cityscapes/leftImg8bit/train"
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        # 原始标注的路径，例如："~/cityscapes/gtFine/train"

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
        # 返回字典列表，每个字典包含图像文件名和语义分割标注文件名
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    # gt_dir包含许多小文件，先获取本地路径更有效率
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, _, label_file, json_file in _get_cityscapes_files(image_dir, gt_dir):
        # 将labelIds替换为labelTrainIds，获取训练用的标签文件
        label_file = label_file.replace("labelIds", "labelTrainIds")

        # 读取JSON文件获取图像尺寸信息
        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        # 构建返回字典，包含图像路径、分割标注路径和图像尺寸
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": jsonobj["imgHeight"],
                "width": jsonobj["imgWidth"],
            }
        )
    # 确保找到了图像文件
    assert len(ret), f"No images found in {image_dir}!"
    # 确保语义分割标注文件存在
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret


def _cityscapes_files_to_dict(files, from_json, to_polygons):
    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.
    # 将Cityscapes标注文件解析为实例分割数据集字典

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        # 文件元组，包含（图像文件、实例ID文件、标签ID文件、JSON文件）
        from_json (bool): whether to read annotations from the raw json file or the png files.
        # 是否从原始json文件读取标注，而不是从png文件读取
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).
        # 是否将分割表示为多边形（COCO格式）而不是掩码（Cityscapes格式）

    Returns:
        A dict in Detectron2 Dataset format.
        # 返回Detectron2数据集格式的字典
    """
    # 导入Cityscapes标签辅助工具
    from cityscapesscripts.helpers.labels import id2label, name2label

    # 解析文件路径元组
    image_file, instance_id_file, _, json_file = files

    # 初始化标注列表
    annos = []

    if from_json:
        # 导入shapely库用于多边形处理
        from shapely.geometry import MultiPolygon, Polygon

        # 读取JSON文件
        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        # 构建基本的返回字典，包含图像信息
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": jsonobj["imgHeight"],
            "width": jsonobj["imgWidth"],
        }

        # `polygons_union` contains the union of all valid polygons.
        # `polygons_union`包含所有有效多边形的并集
        polygons_union = Polygon()

        # CityscapesScripts draw the polygons in sequential order
        # and each polygon *overwrites* existing ones. See
        # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
        # We use reverse order, and each polygon *avoids* early ones.
        # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
        # CityscapesScripts按顺序绘制多边形，每个多边形会覆盖已存在的多边形
        # 我们使用相反的顺序，每个多边形会避开之前的多边形
        # 这样可以以与CityscapesScripts相同的方式解决多边形重叠问题
        # 反向遍历所有对象以处理重叠问题
        for obj in jsonobj["objects"][::-1]:
            if "deleted" in obj:  # cityscapes data format specific
                continue
            label_name = obj["label"]

            # 获取标签对象，处理特殊情况（如群组标注）
            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    # 处理群组标注，去掉"group"后缀
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

            # Cityscapes's raw annotations uses integer coordinates
            # Therefore +0.5 here
            # Cityscapes的原始标注使用整数坐标，这里加0.5进行调整
            poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
            # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
            # polygons for evaluation. This function operates in integer space
            # and draws each pixel whose center falls into the polygon.
            # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
            # We therefore dilate the input polygon by 0.5 as our input.
            # CityscapesScript使用PIL.ImageDraw.polygon进行光栅化
            # 该函数在整数空间中操作，绘制中心落在多边形内的每个像素
            # 因此绘制的多边形预期会比实际大0.5个像素
            # 所以我们将输入多边形膨胀0.5作为输入
            poly = Polygon(poly_coord).buffer(0.5, resolution=4)

            if not label.hasInstances or label.ignoreInEval:
                # even if we won't store the polygon it still contributes to overlaps resolution
                # 即使我们不存储这个多边形，它仍然参与重叠解析
                polygons_union = polygons_union.union(poly)
                continue

            # Take non-overlapping part of the polygon
            # 获取多边形的非重叠部分
            poly_wo_overlaps = poly.difference(polygons_union)
            if poly_wo_overlaps.is_empty:
                continue
            polygons_union = polygons_union.union(poly)

            # 创建标注字典
            anno = {}
            anno["iscrowd"] = label_name.endswith("group")  # 判断是否为群组标注
            anno["category_id"] = label.id  # 设置类别ID

            # 处理不同类型的多边形对象
            if isinstance(poly_wo_overlaps, Polygon):
                poly_list = [poly_wo_overlaps]  # 单个多边形直接放入列表
            elif isinstance(poly_wo_overlaps, MultiPolygon):
                poly_list = poly_wo_overlaps.geoms  # 多个多边形获取所有几何对象
            else:
                raise NotImplementedError("Unknown geometric structure {}".format(poly_wo_overlaps))

            # 转换多边形坐标为COCO格式
            poly_coord = []
            for poly_el in poly_list:
                # COCO API can work only with exterior boundaries now, hence we store only them.
                # TODO: store both exterior and interior boundaries once other parts of the
                # codebase support holes in polygons.
                # COCO API目前只能处理外部边界，因此我们只存储外部边界
                # TODO：一旦代码库支持多边形中的孔洞，就同时存储外部和内部边界
                poly_coord.append(list(chain(*poly_el.exterior.coords)))
            anno["segmentation"] = poly_coord  # 设置分割标注
            # 计算边界框坐标
            (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds

            # 设置边界框和坐标模式
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS  # 使用绝对坐标模式

            annos.append(anno)  # 添加标注到列表
    else:
        # See also the official annotation parsing scripts at
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
        # 从PNG文件读取实例ID图像
        with PathManager.open(instance_id_file, "rb") as f:
            inst_image = np.asarray(Image.open(f), order="F")
        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        # ID < 24的是stuff标签（先过滤掉可以提高5%的速度）
        flattened_ids = np.unique(inst_image[inst_image >= 24])

        # 构建基本的返回字典
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        # 处理每个实例ID
        for instance_id in flattened_ids:
            # For non-crowd annotations, instance_id // 1000 is the label_id
            # Crowd annotations have <1000 instance ids
            # 对于非群组标注，instance_id // 1000得到label_id
            # 群组标注的instance_id < 1000
            label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
            label = id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue

            # 创建标注字典
            anno = {}
            anno["iscrowd"] = instance_id < 1000  # 判断是否为群组标注
            anno["category_id"] = label.id  # 设置类别ID

            # 创建实例掩码
            mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:  # 跳过无效的边界框
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS  # 使用绝对坐标模式
            if to_polygons:
                # This conversion comes from D4809743 and D5171122,
                # when Mask-RCNN was first developed.
                # 这个转换来自D4809743和D5171122，是在Mask-RCNN首次开发时实现的
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                    -2
                ]  # 使用OpenCV查找轮廓
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]  # 将轮廓转换为多边形
                # opencv's can produce invalid polygons
                # OpenCV可能产生无效的多边形
                if len(polygons) == 0:
                    continue
                anno["segmentation"] = polygons  # 设置多边形分割
            else:
                anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]  # 使用RLE编码掩码
            annos.append(anno)  # 添加标注到列表
    ret["annotations"] = annos  # 将所有标注添加到返回字典
    return ret  # 返回处理后的数据字典


if __name__ == "__main__":
    """
    Test the cityscapes dataset loader.
    测试Cityscapes数据集加载器。

    Usage:
        python -m detectron2.data.datasets.cityscapes \
            cityscapes/leftImg8bit/train cityscapes/gtFine/train
    使用方法：
        通过命令行运行此脚本，需要提供图像目录和标注目录的路径
    """
    import argparse  # 导入命令行参数解析模块

    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("image_dir")  # 添加图像目录参数
    parser.add_argument("gt_dir")  # 添加标注目录参数
    parser.add_argument("--type", choices=["instance", "semantic"], default="instance")  # 添加数据类型参数，可选实例分割或语义分割
    args = parser.parse_args()  # 解析命令行参数
    from detectron2.data.catalog import Metadata  # 导入元数据相关模块
    from detectron2.utils.visualizer import Visualizer  # 导入可视化工具
    from cityscapesscripts.helpers.labels import labels  # 导入Cityscapes标签定义

    logger = setup_logger(name=__name__)  # 设置日志记录器

    dirname = "cityscapes-data-vis"  # 设置可视化结果保存目录
    os.makedirs(dirname, exist_ok=True)  # 创建保存目录，如果已存在则不报错

    if args.type == "instance":  # 如果是实例分割类型
        dicts = load_cityscapes_instances(  # 加载Cityscapes实例分割数据
            args.image_dir, args.gt_dir, from_json=True, to_polygons=True  # 从JSON文件加载，并将分割掩码转换为多边形格式
        )
        logger.info("Done loading {} samples.".format(len(dicts)))  # 记录加载的样本数量

        thing_classes = [k.name for k in labels if k.hasInstances and not k.ignoreInEval]  # 获取所有可实例化且不忽略的类别名称
        meta = Metadata().set(thing_classes=thing_classes)  # 设置元数据中的类别信息

    else:  # 如果是语义分割类型
        dicts = load_cityscapes_semantic(args.image_dir, args.gt_dir)  # 加载Cityscapes语义分割数据
        logger.info("Done loading {} samples.".format(len(dicts)))  # 记录加载的样本数量

        stuff_classes = [k.name for k in labels if k.trainId != 255]  # 获取所有有效的语义类别名称（trainId不为255的类别）
        stuff_colors = [k.color for k in labels if k.trainId != 255]  # 获取对应的类别颜色
        meta = Metadata().set(stuff_classes=stuff_classes, stuff_colors=stuff_colors)  # 设置元数据中的类别和颜色信息

    for d in dicts:  # 遍历所有加载的数据
        img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))  # 读取图像文件
        visualizer = Visualizer(img, metadata=meta)  # 创建可视化器实例
        vis = visualizer.draw_dataset_dict(d)  # 在图像上绘制数据集信息（如边界框、分割掩码等）
        # cv2.imshow("a", vis.get_image()[:, :, ::-1])  # OpenCV显示可视化结果（已注释）
        # cv2.waitKey()  # 等待按键（已注释）
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))  # 构建保存文件路径
        vis.save(fpath)  # 保存可视化结果
