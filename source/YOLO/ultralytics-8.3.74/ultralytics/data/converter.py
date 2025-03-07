# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics.utils import DATASETS_DIR, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.downloads import download
from ultralytics.utils.files import increment_path


def coco91_to_coco80_class():
    """
    Converts 91-index COCO class IDs to 80-index COCO class IDs.
    将91索引的COCO类ID转换为80索引的COCO类ID。

    Returns:
        (list): A list of 91 class IDs where the index represents the 80-index class ID and the value is the
            corresponding 91-index class ID.
        (list): 一个包含91个类ID的列表，其中索引表示80索引类ID，值为对应的91索引类ID。
    """
    return [
        0,  # 类别0
        1,  # 类别1
        2,  # 类别2
        3,  # 类别3
        4,  # 类别4
        5,  # 类别5
        6,  # 类别6
        7,  # 类别7
        8,  # 类别8
        9,  # 类别9
        10,  # 类别10
        None,  # 类别11（无）
        11,  # 类别12
        12,  # 类别13
        13,  # 类别14
        14,  # 类别15
        15,  # 类别16
        16,  # 类别17
        17,  # 类别18
        18,  # 类别19
        19,  # 类别20
        20,  # 类别21
        21,  # 类别22
        22,  # 类别23
        23,  # 类别24
        None,  # 类别25（无）
        24,  # 类别26
        25,  # 类别27
        None,  # 类别28（无）
        None,  # 类别29（无）
        26,  # 类别30
        27,  # 类别31
        28,  # 类别32
        29,  # 类别33
        30,  # 类别34
        31,  # 类别35
        32,  # 类别36
        33,  # 类别37
        34,  # 类别38
        35,  # 类别39
        36,  # 类别40
        37,  # 类别41
        38,  # 类别42
        39,  # 类别43
        None,  # 类别44（无）
        40,  # 类别45
        41,  # 类别46
        42,  # 类别47
        43,  # 类别48
        44,  # 类别49
        45,  # 类别50
        46,  # 类别51
        47,  # 类别52
        48,  # 类别53
        49,  # 类别54
        50,  # 类别55
        51,  # 类别56
        52,  # 类别57
        53,  # 类别58
        54,  # 类别59
        55,  # 类别60
        56,  # 类别61
        57,  # 类别62
        58,  # 类别63
        59,  # 类别64
        None,  # 类别65（无）
        60,  # 类别66
        None,  # 类别67（无）
        None,  # 类别68（无）
        61,  # 类别69
        None,  # 类别70（无）
        62,  # 类别71
        63,  # 类别72
        64,  # 类别73
        65,  # 类别74
        66,  # 类别75
        67,  # 类别76
        68,  # 类别77
        69,  # 类别78
        70,  # 类别79
        71,  # 类别80
        72,  # 类别81
        None,  # 类别82（无）
        73,  # 类别83
        74,  # 类别84
        75,  # 类别85
        76,  # 类别86
        77,  # 类别87
        78,  # 类别88
        79,  # 类别89
        None,  # 类别90（无）
    ]


def coco80_to_coco91_class():
    r"""
    Converts 80-index (val2014) to 91-index (paper).
    将80索引（val2014）转换为91索引（论文）。

    For details see https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/.
    有关详细信息，请参见 https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/。

    Example:
        ```python
        import numpy as np

        a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")  # 加载COCO类名称
        b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")  # 加载COCO论文类名称
        x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco darknet到coco
        x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet coco到darknet
        ```
    """
    return [
        1,  # 类别1
        2,  # 类别2
        3,  # 类别3
        4,  # 类别4
        5,  # 类别5
        6,  # 类别6
        7,  # 类别7
        8,  # 类别8
        9,  # 类别9
        10,  # 类别10
        11,  # 类别11
        13,  # 类别12
        14,  # 类别13
        15,  # 类别14
        16,  # 类别15
        17,  # 类别16
        18,  # 类别17
        19,  # 类别18
        20,  # 类别19
        21,  # 类别20
        22,  # 类别21
        23,  # 类别22
        24,  # 类别23
        25,  # 类别24
        27,  # 类别25
        28,  # 类别26
        31,  # 类别27
        32,  # 类别28
        33,  # 类别29
        34,  # 类别30
        35,  # 类别31
        36,  # 类别32
        37,  # 类别33
        38,  # 类别34
        39,  # 类别35
        40,  # 类别36
        41,  # 类别37
        42,  # 类别38
        43,  # 类别39
        44,  # 类别40
        46,  # 类别41
        47,  # 类别42
        48,  # 类别43
        49,  # 类别44
        50,  # 类别45
        51,  # 类别46
        52,  # 类别47
        53,  # 类别48
        54,  # 类别49
        55,  # 类别50
        56,  # 类别51
        57,  # 类别52
        58,  # 类别53
        59,  # 类别54
        60,  # 类别55
        61,  # 类别56
        62,  # 类别57
        63,  # 类别58
        64,  # 类别59
        65,  # 类别60
        67,  # 类别61
        70,  # 类别62
        72,  # 类别63
        73,  # 类别64
        74,  # 类别65
        75,  # 类别66
        76,  # 类别67
        77,  # 类别68
        78,  # 类别69
        79,  # 类别70
        80,  # 类别71
        81,  # 类别72
        82,  # 类别73
        84,  # 类别74
        85,  # 类别75
        86,  # 类别76
        87,  # 类别77
        88,  # 类别78
        89,  # 类别79
        90,  # 类别80
    ]


def convert_coco(
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False,
):
    """
    Converts COCO dataset annotations to a YOLO annotation format suitable for training YOLO models.
    将COCO数据集注释转换为适合训练YOLO模型的YOLO注释格式。

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        labels_dir (str, optional): 包含COCO数据集注释文件的目录路径。
        save_dir (str, optional): Path to directory to save results to.
        save_dir (str, optional): 保存结果的目录路径。
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_segments (bool, optional): 是否在输出中包含分割掩码。
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        use_keypoints (bool, optional): 是否在输出中包含关键点注释。
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
        cls91to80 (bool, optional): 是否将91个COCO类ID映射到相应的80个COCO类ID。
        lvis (bool, optional): Whether to convert data in lvis dataset way.
        lvis (bool, optional): 是否以lvis数据集的方式转换数据。

    Example:
        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco("../datasets/coco/annotations/", use_segments=True, use_keypoints=False, cls91to80=False)
        convert_coco(
            "../datasets/lvis/annotations/", use_segments=True, use_keypoints=False, cls91to80=False, lvis=True
        )
        ```

    Output:
        Generates output files in the specified output directory.
        生成指定输出目录中的输出文件。
    """
    # Create dataset directory
    save_dir = increment_path(save_dir)  # increment if save directory already exists
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # 创建目录

    # Convert classes
    coco80 = coco91_to_coco80_class()  # 获取80类的映射

    # Import json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):  # 遍历所有JSON文件
        lname = "" if lvis else json_file.stem.replace("instances_", "")  # 获取文件名
        fn = Path(save_dir) / "labels" / lname  # 文件夹名称
        fn.mkdir(parents=True, exist_ok=True)  # 创建文件夹
        if lvis:
            # NOTE: create folders for both train and val in advance,
            # since LVIS val set contains images from COCO 2017 train in addition to the COCO 2017 val split.
            (fn / "train2017").mkdir(parents=True, exist_ok=True)  # 创建训练集文件夹
            (fn / "val2017").mkdir(parents=True, exist_ok=True)  # 创建验证集文件夹
        with open(json_file, encoding="utf-8") as f:  # 打开JSON文件
            data = json.load(f)  # 读取JSON数据

        # Create image dict
        images = {f"{x['id']:d}": x for x in data["images"]}  # 创建图像字典
        # Create image-annotations dict
        imgToAnns = defaultdict(list)  # 创建图像与注释的字典
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)  # 将注释添加到对应的图像中

        image_txt = []
        # Write labels file
        for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):  # 遍历图像及其注释
            img = images[f"{img_id:d}"]  # 获取图像信息
            h, w = img["height"], img["width"]  # 获取图像高度和宽度
            f = str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org")) if lvis else img["file_name"]  # 获取文件名
            if lvis:
                image_txt.append(str(Path("./images") / f))  # 如果是LVIS，添加到图像文本列表

            bboxes = []  # 边界框列表
            segments = []  # 分割列表
            keypoints = []  # 关键点列表
            for ann in anns:
                if ann.get("iscrowd", False):  # 如果是拥挤的注释，跳过
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)  # 获取边界框
                box[:2] += box[2:] / 2  # xy top-left corner to center  将左上角坐标转换为中心坐标
                box[[0, 2]] /= w  # normalize x  归一化x坐标
                box[[1, 3]] /= h  # normalize y  归一化y坐标
                if box[2] <= 0 or box[3] <= 0:  # 如果宽度或高度小于等于0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class 获取类别ID
                box = [cls] + box.tolist()  # 创建边界框列表
                if box not in bboxes:  # 如果边界框不在列表中
                    bboxes.append(box)  # 添加边界框
                    if use_segments and ann.get("segmentation") is not None:  # 如果使用分割并且存在分割数据
                        if len(ann["segmentation"]) == 0:  # 如果分割为空
                            segments.append([])  # 添加空分割
                            continue
                        elif len(ann["segmentation"]) > 1:  # 如果有多个分割
                            s = merge_multi_segment(ann["segmentation"])  # 合并多个分割
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()  # 归一化分割
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated  所有分割合并
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()  # 归一化分割
                        s = [cls] + s  # 添加类别ID
                        segments.append(s)  # 添加分割
                    if use_keypoints and ann.get("keypoints") is not None:  # 如果使用关键点并且存在关键点数据
                        keypoints.append(
                            box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()  # 归一化关键点
                        )

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:  # 打开标签文件
                for i in range(len(bboxes)):
                    if use_keypoints:  # 如果使用关键点
                        line = (*(keypoints[i]),)  # cls, box, keypoints  类别，边界框，关键点
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),  # cls, box or segments  类别，边界框或分割
                        )
                    file.write(("%g " * len(line)).rstrip() % line + "\n")  # 写入文件

        if lvis:
            with open((Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")), "a") as f:  # 打开LVIS文件
                f.writelines(f"{line}\n" for line in image_txt)  # 写入图像文本

    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")  # 日志输出转换成功信息

def convert_segment_masks_to_yolo_seg(masks_dir, output_dir, classes):
    """
    Converts a dataset of segmentation mask images to the YOLO segmentation format.
    将分割掩码图像数据集转换为YOLO分割格式。

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation format.
    此函数接受包含二进制格式掩码图像的目录，并将其转换为YOLO分割格式。
    The converted masks are saved in the specified output directory.
    转换后的掩码保存在指定的输出目录中。

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        masks_dir (str): 所有掩码图像（png，jpg）存储的目录路径。
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        output_dir (str): 转换后的YOLO分割掩码将存储的目录路径。
        classes (int): Total classes in the dataset i.e. for COCO classes=80
        classes (int): 数据集中总的类别数，即COCO数据集的类别数为80。

    Example:
        ```python
        from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        # The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
        # 这里的classes是数据集中的总类别数，对于COCO数据集，我们有80个类别
        convert_segment_masks_to_yolo_seg("path/to/masks_directory", "path/to/output/directory", classes=80)
        ```

    Notes:
        The expected directory structure for the masks is:
        掩码的预期目录结构是：

            - masks
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        After execution, the labels will be organized in the following structure:
        执行后，标签将按照以下结构组织：

            - output_dir
                ├─ mask_yolo_01.txt
                ├─ mask_yolo_02.txt
                ├─ mask_yolo_03.txt
                └─ mask_yolo_04.txt
    """
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}  # Create a mapping from pixel values to class indices
    # 创建从像素值到类索引的映射
    for mask_path in Path(masks_dir).iterdir():  # Iterate through each mask image in the directory
        # 遍历目录中的每个掩码图像
        if mask_path.suffix in {".png", ".jpg"}:  # Check if the file is a PNG or JPG image
            # 检查文件是否为PNG或JPG图像
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
            # 以灰度模式读取掩码图像
            img_height, img_width = mask.shape  # Get image dimensions
            # 获取图像尺寸
            LOGGER.info(f"Processing {mask_path} imgsz = {img_height} x {img_width}")  # Log the processing info
            # 记录处理信息

            unique_values = np.unique(mask)  # Get unique pixel values representing different classes
            # 获取表示不同类的唯一像素值
            yolo_format_data = []  # Prepare a list to hold YOLO format data

            for value in unique_values:  # Iterate through each unique pixel value
                # 遍历每个唯一的像素值
                if value == 0:
                    continue  # Skip background
                    # 跳过背景
                class_index = pixel_to_class_mapping.get(value, -1)  # Get the class index from the mapping
                # 从映射中获取类索引
                if class_index == -1:
                    LOGGER.warning(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    # 记录警告信息：未知类
                    continue

                # Create a binary mask for the current class and find contours
                # 为当前类创建二进制掩码并找到轮廓
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # Find contours
                # 找到轮廓

                for contour in contours:  # Iterate through each contour found
                    # 遍历找到的每个轮廓
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        # YOLO要求有效分割至少有3个点
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        # 去掉单维条目
                        yolo_format = [class_index]  # Start the YOLO format with the class index
                        # 用类索引开始YOLO格式
                        for point in contour:  # Iterate through each point in the contour
                            # 遍历轮廓中的每个点
                            # Normalize the coordinates
                            yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                            # 归一化坐标，四舍五入到小数点后6位
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)  # Append the formatted data to the list
            # Save Ultralytics YOLO format data to file
            # 将Ultralytics YOLO格式数据保存到文件
            output_path = Path(output_dir) / f"{mask_path.stem}.txt"  # Define the output path for the YOLO file
            # 定义YOLO文件的输出路径
            with open(output_path, "w") as file:  # Open the output file for writing
                # 打开输出文件以进行写入
                for item in yolo_format_data:  # Iterate through the YOLO format data
                    # 遍历YOLO格式数据
                    line = " ".join(map(str, item))  # Create a line from the YOLO format data
                    # 从YOLO格式数据创建一行
                    file.write(line + "\n")  # Write the line to the file
                    # 将行写入文件
            LOGGER.info(f"Processed and stored at {output_path} imgsz = {img_height} x {img_width}")  # Log completion
            # 记录完成信息

def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    Converts DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.
    将DOTA数据集注释转换为YOLO OBB（定向边界框）格式。

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.
    此函数处理DOTA数据集的“train”和“val”文件夹中的图像。对于每个图像，它从原始标签目录读取相关标签，并将新的YOLO OBB格式标签写入新目录。

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.
        dota_root_path (str): DOTA数据集的根目录路径。

    Example:
        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb("path/to/DOTA")
        ```

    Notes:
        The directory structure assumed for the DOTA dataset:
        假设DOTA数据集的目录结构：

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:
        执行后，该函数将把标签组织成：

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)  # Convert the root path to a Path object

    # Class names to indices mapping
    # 类名到索引的映射
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
        "airport": 16,
        "helipad": 17,
    }

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory.
        将单个图像的DOTA注释转换为YOLO OBB格式，并将其保存到指定目录。
        """
        orig_label_path = orig_label_dir / f"{image_name}.txt"  # Define the original label path
        # 定义原始标签路径
        save_path = save_dir / f"{image_name}.txt"  # Define the save path for the new label
        # 定义新标签的保存路径

        with orig_label_path.open("r") as f, save_path.open("w") as g:  # Open original and save paths
            # 打开原始路径和保存路径
            lines = f.readlines()  # Read all lines from the original label file
            # 从原始标签文件中读取所有行
            for line in lines:  # Iterate through each line
                # 遍历每一行
                parts = line.strip().split()  # Split the line into parts
                # 将行分割成部分
                if len(parts) < 9:  # Check if the line has enough parts
                    # 检查行是否有足够的部分
                    continue
                class_name = parts[8]  # Get the class name from the parts
                # 从部分中获取类名
                class_idx = class_mapping[class_name]  # Get the class index from the mapping
                # 从映射中获取类索引
                coords = [float(p) for p in parts[:8]]  # Convert coordinates to float
                # 将坐标转换为浮点数
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]  # Normalize the coordinates
                # 归一化坐标
                formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]  # Format the coordinates
                # 格式化坐标
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")  # Write the class index and coordinates to the new file
                # 将类索引和坐标写入新文件

    for phase in ["train", "val"]:  # Iterate through train and val phases
        # 遍历训练和验证阶段
        image_dir = dota_root_path / "images" / phase  # Define the image directory for the phase
        # 定义该阶段的图像目录
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"  # Define the original label directory
        # 定义原始标签目录
        save_dir = dota_root_path / "labels" / phase  # Define the save directory for the new labels
        # 定义新标签的保存目录

        save_dir.mkdir(parents=True, exist_ok=True)  # Create the save directory if it doesn't exist
        # 如果保存目录不存在，则创建

        image_paths = list(image_dir.iterdir())  # Get a list of image paths in the directory
        # 获取目录中图像路径的列表
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):  # Iterate through images with a progress bar
            # 遍历图像并显示进度条
            if image_path.suffix != ".png":  # Check if the image is a PNG
                # 检查图像是否为PNG
                continue
            image_name_without_ext = image_path.stem  # Get the image name without extension
            # 获取不带扩展名的图像名称
            img = cv2.imread(str(image_path))  # Read the image
            # 读取图像
            h, w = img.shape[:2]  # Get the height and width of the image
            # 获取图像的高度和宽度
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)  # Convert the label for the image
            # 转换图像的标签

def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.
    找到两个二维点数组之间距离最短的一对索引。

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr1 (np.ndarray): 一个形状为(N, 2)的NumPy数组，表示N个二维点。
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.
        arr2 (np.ndarray): 一个形状为(M, 2)的NumPy数组，表示M个二维点。

    Returns:
        (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
        (tuple): 一个元组，包含在arr1和arr2中距离最短的点的索引。
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)  # Calculate the squared distance between each pair of points
    # 计算每对点之间的平方距离
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)  # Return the indices of the minimum distance
    # 返回最小距离的索引


def merge_multi_segment(segments):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    通过连接每个段之间最小距离的坐标，将多个段合并为一个列表。

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].
        segments (List[List]): COCO的JSON文件中的原始分割。每个元素都是坐标的列表，如[segmentation1, segmentation2,...]。

    Returns:
        s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
        s (List[np.ndarray]): 作为NumPy数组表示的连接段的列表。
    """
    s = []  # Initialize an empty list to hold the merged segments
    # 初始化一个空列表以保存合并的段
    segments = [np.array(i).reshape(-1, 2) for i in segments]  # Convert each segment to a NumPy array and reshape
    # 将每个段转换为NumPy数组并重塑
    idx_list = [[] for _ in range(len(segments))]  # Create a list to hold the indices of minimum distances
    # 创建一个列表以保存最小距离的索引

    # Record the indexes with min distance between each segment
    # 记录每个段之间最小距离的索引
    for i in range(1, len(segments)):  # Iterate through segments starting from the second one
        # 从第二个段开始遍历段
        idx1, idx2 = min_index(segments[i - 1], segments[i])  # Find the closest points between the current and previous segment
        # 找到当前段和前一个段之间最近的点
        idx_list[i - 1].append(idx1)  # Append the index of the previous segment
        idx_list[i].append(idx2)  # Append the index of the current segment

    # Use two rounds to connect all the segments
    # 使用两轮连接所有段
    for k in range(2):  # Iterate twice for forward and backward connections
        # 为正向和反向连接迭代两次
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):  # Iterate through the index list
                # 遍历索引列表
                # Middle segments have two indexes, reverse the index of middle segments
                # 中间段有两个索引，反转中间段的索引
                if len(idx) == 2 and idx[0] > idx[1]:  # Check if the current segment has two indices and reverse if necessary
                    # 检查当前段是否有两个索引，如果需要则反转
                    idx = idx[::-1]  # Reverse the index
                    segments[i] = segments[i][::-1, :]  # Reverse the segment order

                segments[i] = np.roll(segments[i], -idx[0], axis=0)  # Roll the segment to align with the minimum distance
                # 滚动段以与最小距离对齐
                segments[i] = np.concatenate([segments[i], segments[i][:1]])  # Concatenate the first point to close the segment
                # 连接第一个点以闭合段
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:  # If it's the first or last segment, append it directly
                    # 如果是第一个或最后一个段，直接附加
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]  # Calculate the range for middle segments
                    s.append(segments[i][idx[0]: idx[1] + 1])  # Append the relevant part of the segment

        else:  # Backward connection
            for i in range(len(idx_list) - 1, -1, -1):  # Iterate backward through the index list
                # 反向遍历索引列表
                if i not in {0, len(idx_list) - 1}:  # Skip the first and last segments
                    # 跳过第一个和最后一个段
                    idx = idx_list[i]  # Get the index for the current segment
                    nidx = abs(idx[1] - idx[0])  # Calculate the absolute difference of the indices
                    s.append(segments[i][nidx:])  # Append the segment from the calculated index
    return s  # Return the merged segments
    # 返回合并的段


def yolo_bbox2segment(im_dir, save_dir=None, sam_model="sam_b.pt", device=None):
    """
    Converts existing object detection dataset (bounding boxes) to segmentation dataset or oriented bounding box (OBB)
    in YOLO format. Generates segmentation data using SAM auto-annotator as needed.
    将现有的目标检测数据集（边界框）转换为YOLO格式的分割数据集或定向边界框（OBB）。根据需要使用SAM自动标注器生成分割数据。

    Args:
        im_dir (str | Path): Path to image directory to convert.
        im_dir (str | Path): 要转换的图像目录的路径。
        save_dir (str | Path): Path to save the generated labels, labels will be saved
            into `labels-segment` in the same directory level of `im_dir` if save_dir is None. Default: None.
        save_dir (str | Path): 保存生成标签的路径，如果save_dir为None，则标签将保存在与im_dir同一目录级别的`labels-segment`中。默认值：None。
        sam_model (str): Segmentation model to use for intermediate segmentation data; optional.
        sam_model (str): 用于中间分割数据的分割模型；可选。
        device (int | str): The specific device to run SAM models. Default: None.
        device (int | str): 运行SAM模型的特定设备。默认值：None。

    Notes:
        The input directory structure assumed for dataset:
        假设数据集的输入目录结构：

            - im_dir
                ├─ 001.jpg
                ├─ ...
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ...
                └─ NNN.txt
    """
    from ultralytics import SAM  # Import the SAM model for segmentation
    from ultralytics.data import YOLODataset  # Import the YOLO dataset class
    from ultralytics.utils import LOGGER  # Import the logger for logging information
    from ultralytics.utils.ops import xywh2xyxy  # Import the function to convert bounding box formats

    # NOTE: add placeholder to pass class index check
    # 注意：添加占位符以通过类索引检查
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000))))  # Create a YOLO dataset object
    # 创建一个YOLO数据集对象
    if len(dataset.labels[0]["segments"]) > 0:  # if it's segment data
        # 如果是分割数据
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")  # Log that segmentation labels are already present
        # 记录分割标签已存在的信息
        return  # Exit the function

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")  # Log that detection labels are found
    # 记录检测标签已找到的信息
    sam_model = SAM(sam_model)  # Load the SAM model for segmentation
    # 加载用于分割的SAM模型
    for label in TQDM(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):  # Iterate through labels with a progress bar
        # 遍历标签并显示进度条
        h, w = label["shape"]  # Get the shape of the image
        # 获取图像的形状
        boxes = label["bboxes"]  # Get the bounding boxes from the label
        # 从标签中获取边界框
        if len(boxes) == 0:  # skip empty labels
            # 跳过空标签
            continue
        boxes[:, [0, 2]] *= w  # Scale the x-coordinates of the bounding boxes
        # 缩放边界框的x坐标
        boxes[:, [1, 3]] *= h  # Scale the y-coordinates of the bounding boxes
        # 缩放边界框的y坐标
        im = cv2.imread(label["im_file"])  # Read the image file
        # 读取图像文件
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)  # Generate segmentation using SAM
        # 使用SAM生成分割
        label["segments"] = sam_results[0].masks.xyn  # Store the segmentation results in the label
        # 将分割结果存储在标签中

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"  # Define the save directory
    # 定义保存目录
    save_dir.mkdir(parents=True, exist_ok=True)  # Create the save directory if it doesn't exist
    # 如果保存目录不存在，则创建
    for label in dataset.labels:  # Iterate through each label in the dataset
        # 遍历数据集中的每个标签
        texts = []  # Initialize a list to hold the text lines for the label
        # 初始化一个列表以保存标签的文本行
        lb_name = Path(label["im_file"]).with_suffix(".txt").name  # Get the label filename corresponding to the image
        # 获取与图像对应的标签文件名
        txt_file = save_dir / lb_name  # Define the path for the label text file
        # 定义标签文本文件的路径
        cls = label["cls"]  # Get the class index from the label
        # 从标签中获取类索引
        for i, s in enumerate(label["segments"]):  # Iterate through each segment in the label
            # 遍历标签中的每个段
            if len(s) == 0:  # Skip empty segments
                # 跳过空段
                continue
            line = (int(cls[i]), *s.reshape(-1))  # Create a line with the class index and segment coordinates
            # 创建一行，包含类索引和段坐标
            texts.append(("%g " * len(line)).rstrip() % line)  # Format the line and append to the texts list
            # 格式化行并附加到文本列表中
        with open(txt_file, "a") as f:  # Open the text file for appending
            # 打开文本文件以进行追加
            f.writelines(text + "\n" for text in texts)  # Write the formatted lines to the file
            # 将格式化的行写入文件
    LOGGER.info(f"Generated segment labels saved in {save_dir}")  # Log the completion of label generation
    # 记录标签生成完成的信息


def create_synthetic_coco_dataset():
    """
    Creates a synthetic COCO dataset with random images based on filenames from label lists.
    创建一个基于标签列表中的文件名的随机图像合成COCO数据集。

    This function downloads COCO labels, reads image filenames from label list files,
    creates synthetic images for train2017 and val2017 subsets, and organizes
    them in the COCO dataset structure. It uses multithreading to generate images efficiently.
    此函数下载COCO标签，从标签列表文件中读取图像文件名，为train2017和val2017子集创建合成图像，并将它们组织在COCO数据集结构中。它使用多线程高效生成图像。

    Examples:
        >>> from ultralytics.data.converter import create_synthetic_coco_dataset
        >>> create_synthetic_coco_dataset()
    示例：
        >>> from ultralytics.data.converter import create_synthetic_coco_dataset
        >>> create_synthetic_coco_dataset()

    Notes:
        - Requires internet connection to download label files.
        - 需要互联网连接以下载标签文件。
        - Generates random RGB images of varying sizes (480x480 to 640x640 pixels).
        - 生成不同大小（480x480到640x640像素）的随机RGB图像。
        - Existing test2017 directory is removed as it's not needed.
        - 删除现有的test2017目录，因为不需要。
        - Reads image filenames from train2017.txt and val2017.txt files.
        - 从train2017.txt和val2017.txt文件中读取图像文件名。
    """

    def create_synthetic_image(image_file):
        """Generates synthetic images with random sizes and colors for dataset augmentation or testing purposes.
        生成具有随机大小和颜色的合成图像，用于数据集增强或测试目的。
        """
        if not image_file.exists():  # Check if the image file already exists
            # 检查图像文件是否已存在
            size = (random.randint(480, 640), random.randint(480, 640))  # Generate random size for the image
            # 生成图像的随机大小
            Image.new(
                "RGB",
                size=size,
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            ).save(image_file)  # Create and save a new image with random color
            # 创建并保存具有随机颜色的新图像

    # Download labels
    dir = DATASETS_DIR / "coco"  # Define the directory for COCO dataset
    # 定义COCO数据集的目录
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"  # Base URL for downloading labels
    # 下载标签的基本URL
    label_zip = "coco2017labels-segments.zip"  # Define the label zip file name
    # 定义标签zip文件名
    download([url + label_zip], dir=dir.parent)  # Download the label zip file
    # 下载标签zip文件

    # Create synthetic images
    shutil.rmtree(dir / "labels" / "test2017", ignore_errors=True)  # Remove test2017 directory as not needed
    # 删除test2017目录，因为不需要
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:  # Create a thread pool for concurrent image generation
        # 创建线程池以并发生成图像
        for subset in ["train2017", "val2017"]:  # Iterate through the train and validation subsets
            # 遍历训练和验证子集
            subset_dir = dir / "images" / subset  # Define the directory for the current subset
            # 定义当前子集的目录
            subset_dir.mkdir(parents=True, exist_ok=True)  # Create the subset directory if it doesn't exist
            # 如果子集目录不存在，则创建

            # Read image filenames from label list file
            label_list_file = dir / f"{subset}.txt"  # Define the path for the label list file
            # 定义标签列表文件的路径
            if label_list_file.exists():  # Check if the label list file exists
                # 检查标签列表文件是否存在
                with open(label_list_file) as f:  # Open the label list file
                    # 打开标签列表文件
                    image_files = [dir / line.strip() for line in f]  # Read image filenames from the file
                    # 从文件中读取图像文件名

                # Submit all tasks
                futures = [executor.submit(create_synthetic_image, image_file) for image_file in image_files]  # Submit tasks to create images
                # 提交任务以创建图像
                for _ in TQDM(as_completed(futures), total=len(futures), desc=f"Generating images for {subset}"):
                    pass  # The actual work is done in the background
                    # 实际工作在后台完成
            else:
                print(f"Warning: Labels file {label_list_file} does not exist. Skipping image creation for {subset}.")
                # 警告：标签文件{label_list_file}不存在。跳过{subset}的图像创建。

    print("Synthetic COCO dataset created successfully.")  # Print success message
    # 打印成功消息