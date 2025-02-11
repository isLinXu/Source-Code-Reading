# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 segment model on a segment dataset.

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate COCO-segments

Usage - formats:
    $ python segment/val.py --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg_openvino_label     # OpenVINO
                                      yolov5s-seg.engine             # TensorRT
                                      yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                      yolov5s-seg_saved_model        # TensorFlow SavedModel
                                      yolov5s-seg.pb                 # TensorFlow GraphDef
                                      yolov5s-seg.tflite             # TensorFlow Lite
                                      yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                      yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse  # 导入argparse库，用于处理命令行参数
import json  # 导入json库，用于处理JSON数据
import os  # 导入os库，用于与操作系统交互
import subprocess  # 导入subprocess库，用于执行子进程
import sys  # 导入sys库，用于访问与Python解释器相关的变量和函数
from multiprocessing.pool import ThreadPool  # 从multiprocessing.pool导入ThreadPool类，用于创建线程池
from pathlib import Path  # 从pathlib导入Path类，用于处理文件路径

import numpy as np  # 导入numpy库，常用于数值计算
import torch  # 导入PyTorch库，用于深度学习
from tqdm import tqdm  # 从tqdm导入tqdm类，用于显示进度条

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[1]  # 获取YOLOv5根目录（当前文件的父目录的父目录）
if str(ROOT) not in sys.path:  # 如果根目录不在系统路径中
    sys.path.append(str(ROOT))  # 将根目录添加到系统路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 获取相对于当前工作目录的根目录路径

import torch.nn.functional as F  # 导入PyTorch的功能性模块，通常用于神经网络的操作

from models.common import DetectMultiBackend  # 从models.common导入DetectMultiBackend类
from models.yolo import SegmentationModel  # 从models.yolo导入SegmentationModel类
from utils.callbacks import Callbacks  # 从utils.callbacks导入Callbacks类
from utils.general import (  # 从utils.general导入多个实用函数
    LOGGER,  # 日志记录器
    NUM_THREADS,  # 线程数量
    TQDM_BAR_FORMAT,  # tqdm进度条格式
    Profile,  # 性能分析类
    check_dataset,  # 检查数据集的函数
    check_img_size,  # 检查图像大小的函数
    check_requirements,  # 检查依赖项的函数
    check_yaml,  # 检查YAML文件的函数
    coco80_to_coco91_class,  # COCO 80类到91类的转换函数
    colorstr,  # 颜色字符串格式化函数
    increment_path,  # 增加路径的函数
    non_max_suppression,  # 非极大值抑制函数
    print_args,  # 打印参数的函数
    scale_boxes,  # 缩放框的函数
    xywh2xyxy,  # 从xywh格式转换到xyxy格式的函数
    xyxy2xywh,  # 从xyxy格式转换到xywh格式的函数
)
from utils.metrics import ConfusionMatrix, box_iou  # 从utils.metrics导入混淆矩阵和框的IOU计算函数
from utils.plots import output_to_target, plot_val_study  # 从utils.plots导入输出到目标和绘制验证研究的函数
from utils.segment.dataloaders import create_dataloader  # 从utils.segment.dataloaders导入创建数据加载器的函数
from utils.segment.general import (  # 从utils.segment.general导入多个实用函数
    mask_iou,  # 计算掩码的IOU函数
    process_mask,  # 处理掩码的函数
    process_mask_native,  # 原生处理掩码的函数
    scale_image,  # 缩放图像的函数
)
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask  # 从utils.segment.metrics导入Metrics类和按类别计算AP的函数
from utils.segment.plots import plot_images_and_masks  # 从utils.segment.plots导入绘制图像和掩码的函数
from utils.torch_utils import de_parallel, select_device, smart_inference_mode  # 从utils.torch_utils导入多个实用函数

def save_one_txt(predn, save_conf, shape, file):  # 定义一个函数，保存检测结果到txt文件
    """Saves detection results in txt format; includes class, xywh (normalized), optionally confidence if `save_conf` is
    True.  # 将检测结果以txt格式保存；包括类别、xywh（归一化），如果`save_conf`为True则可选保存置信度。
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh  # 归一化增益whwh
    for *xyxy, conf, cls in predn.tolist():  # 遍历预测结果，将xyxy、置信度和类别解包
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh  # 归一化的xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format  # 标签格式
        with open(file, "a") as f:  # 以追加模式打开文件
            f.write(("%g " * len(line)).rstrip() % line + "\n")  # 将结果写入文件


def save_one_json(predn, jdict, path, class_map, pred_masks):
    """
    Saves a JSON file with detection results including bounding boxes, category IDs, scores, and segmentation masks.
    保存一个包含检测结果的 JSON 文件，包括边界框、类别 ID、得分和分割掩码。

    Example JSON result: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}.
    示例 JSON 结果：{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}。
    """
    from pycocotools.mask import encode
    # 从 pycocotools.mask 导入 encode 函数，用于编码分割掩码。

    def single_encode(x):
        # 定义一个内部函数 single_encode，用于对单个掩码进行编码。
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        # 将输入数组 x 转换为 RLE（游程长度编码）格式。
        rle["counts"] = rle["counts"].decode("utf-8")
        # 将 RLE 的 counts 字段从字节解码为字符串。
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    # 从路径中提取图像 ID，如果路径的 stem 是数字，则转换为整数，否则保持为字符串。
    box = xyxy2xywh(predn[:, :4])  # xywh
    # 将预测的边界框从 (x1, y1, x2, y2) 格式转换为 (x_center, y_center, width, height) 格式。
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    # 将中心坐标转换为左上角坐标，通过减去宽度和高度的一半。
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    # 转置预测掩码的维度，以便于后续处理，调整维度顺序为 (num_masks, height, width)。
    with ThreadPool(NUM_THREADS) as pool:
        # 使用线程池并行处理掩码编码。
        rles = pool.map(single_encode, pred_masks)
        # 对每个预测掩码调用 single_encode 函数进行编码，返回 RLE 列表。

    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        # 遍历预测结果和边界框，使用 enumerate 获取索引 i。
        jdict.append(
            {
                "image_id": image_id,
                # 添加图像 ID 到 JSON 字典。
                "category_id": class_map[int(p[5])],
                # 根据预测结果中的类别索引获取类别 ID。
                "bbox": [round(x, 3) for x in b],
                # 将边界框坐标四舍五入到小数点后三位。
                "score": round(p[4], 5),
                # 将得分四舍五入到小数点后五位。
                "segmentation": rles[i],
                # 将编码后的分割掩码添加到 JSON 字典中。
            }
        )


def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Return correct prediction matrix
    返回正确的预测矩阵
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        detections（数组[N, 6]），x1, y1, x2, y2, 置信度, 类别
        labels (array[M, 5]), class, x1, y1, x2, y2
        labels（数组[M, 5]），类别, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
        correct（数组[N, 10]），用于 10 个 IoU 水平
    """
    if masks:
        # 如果使用掩码
        if overlap:
            # 如果允许重叠
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            # 创建一个索引数组，形状为(nl, 1, 1)，用于后续处理
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            # 将 gt_masks 扩展为(nl, 640, 640)的形状
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            # 将 gt_masks 中等于索引的值设为 1.0，其余设为 0.0
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            # 如果 gt_masks 和 pred_masks 的形状不一致
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            # 使用双线性插值调整 gt_masks 的大小以匹配 pred_masks
            gt_masks = gt_masks.gt_(0.5)
            # 将 gt_masks 中大于 0.5 的值设为 True
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        # 计算 gt_masks 和 pred_masks 之间的 IoU
    else:  # boxes
        # 如果不使用掩码，则计算边界框的 IoU
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    # 创建一个正确预测矩阵，初始化为 False
    correct_class = labels[:, 0:1] == detections[:, 5]
    # 检查预测类别与真实类别是否匹配
    for i in range(len(iouv)):
        # 遍历每个 IoU 阈值
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        # 找到 IoU 大于当前阈值且类别匹配的预测
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            # 将匹配的标签、检测和 IoU 组合成一个数组
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 按照 IoU 从大到小排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # 去除重复的检测
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                # 去除重复的标签
            correct[matches[:, 1].astype(int), i] = True
            # 更新正确预测矩阵
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
    # 返回正确预测矩阵作为布尔张量


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    # 模型路径
    batch_size=32,  # batch size
    # 批处理大小
    imgsz=640,  # inference size (pixels)
    # 推理图像大小（像素）
    conf_thres=0.001,  # confidence threshold
    # 置信度阈值
    iou_thres=0.6,  # NMS IoU threshold
    # NMS IoU 阈值
    max_det=300,  # maximum detections per image
    # 每张图像的最大检测数
    task="val",  # train, val, test, speed or study
    # 任务类型：训练、验证、测试、速度或学习
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    # CUDA 设备，例如 0 或 0,1,2,3 或 cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    # 最大数据加载器工作线程（在 DDP 模式下每个 RANK）
    single_cls=False,  # treat as single-class dataset
    # 将数据集视为单类数据集
    augment=False,  # augmented inference
    # 增强推理
    verbose=False,  # verbose output
    # 详细输出
    save_txt=False,  # save results to *.txt
    # 将结果保存到 *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    # 将标签+预测混合结果保存到 *.txt
    save_conf=False,  # save confidences in --save-txt labels
    # 在 --save-txt 标签中保存置信度
    save_json=False,  # save a COCO-JSON results file
    # 保存 COCO-JSON 结果文件
    project=ROOT / "runs/val-seg",  # save to project/name
    # 保存到项目/名称
    name="exp",  # save to project/name
    # 保存到项目/名称
    exist_ok=False,  # existing project/name ok, do not increment
    # 允许存在的项目/名称，不递增
    half=True,  # use FP16 half-precision inference
    # 使用 FP16 半精度推理
    dnn=False,  # use OpenCV DNN for ONNX inference
    # 使用 OpenCV DNN 进行 ONNX 推理
    model=None,
    # 模型
    dataloader=None,
    # 数据加载器
    save_dir=Path(""),
    # 保存目录
    plots=True,
    # 是否生成图表
    overlap=False,
    # 是否允许重叠
    mask_downsample_ratio=1,
    # 掩码下采样比例
    compute_loss=None,
    # 计算损失函数
    callbacks=Callbacks(),
):
    if save_json:
        # 如果需要保存 JSON
        check_requirements("pycocotools>=2.0.6")
        # 检查 pycocotools 的版本要求
        process = process_mask_native  # more accurate
        # 使用更精确的掩码处理
    else:
        process = process_mask  # faster
        # 使用更快的掩码处理

    # Initialize/load model and set device
    # 初始化/加载模型并设置设备
    training = model is not None
    # 检查模型是否存在以确定是否在训练
    if training:  # called by train.py
        # 如果是由 train.py 调用
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        # 获取模型设备，PyTorch 模型
        half &= device.type != "cpu"  # half precision only supported on CUDA
        # 仅在 CUDA 上支持半精度
        model.half() if half else model.float()
        # 将模型转换为半精度或单精度
        nm = de_parallel(model).model[-1].nm  # number of masks
        # 获取掩码数量
    else:  # called directly
        # 如果是直接调用
        device = select_device(device, batch_size=batch_size)
        # 选择设备

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # 增加运行目录
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # 如果需要保存文本，则创建标签目录

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        # 加载多后端检测模型
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # 获取模型的步幅、PyTorch 状态、JIT 状态和引擎状态
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # 检查图像大小
        half = model.fp16  # FP16 supported on limited backends with CUDA
        # FP16 仅在有限的后端支持
        nm = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks
        # 获取掩码数量，如果模型是分割模型
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                # export.py 模型默认为批处理大小 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")
                # 日志记录，强制非 PyTorch 模型使用批处理大小 1

        # Data
        data = check_dataset(data)  # check
        # 检查数据集
    
    # Configure
    model.eval()
    # 设置模型为评估模式
    cuda = device.type != "cpu"
    # 检查设备是否为 CUDA
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    # 检查数据集是否为 COCO 数据集
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    # 确定类别数量
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    # 创建 IoU 向量用于计算 mAP
    niou = iouv.numel()
    # 获取 IoU 向量的元素数量

    # Dataloader
    if not training:
        # 如果不是在训练模式
        if pt and not single_cls:  # check --weights are trained on --data
            # 如果使用 PyTorch 模型并且不是单类数据集
            ncm = model.model.nc
            # 获取模型的类别数量
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
            # 确保模型训练的类别数量与传入的数据集类别数量一致
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        # 对模型进行预热，设置图像大小
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        # 根据任务类型设置填充和矩形推理标志
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        # 确保任务类型有效，默认为验证模式
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
            overlap_mask=overlap,
            mask_downsample_ratio=mask_downsample_ratio,
        )[0]
        # 创建数据加载器，返回第一个元素

    seen = 0
    # 记录已处理的样本数量
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 初始化混淆矩阵
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    # 获取类别名称
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
        # 如果名称是列表或元组，则转换为字典格式
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 如果数据集是 COCO 格式，则获取 COCO 80 到 COCO 91 的类别映射，否则使用 0 到 999 的范围
    s = ("%22s" + "%11s" * 10) % (
        "Class",
        "Images",
        "Instances",
        "Box(P",
        "R",
        "mAP50",
        "mAP50-95)",
        "Mask(P",
        "R",
        "mAP50",
        "mAP50-95)",
    )
    # 定义输出格式字符串
    dt = Profile(device=device), Profile(device=device), Profile(device=device)
    # 创建三个性能分析器
    metrics = Metrics()
    # 初始化度量对象
    loss = torch.zeros(4, device=device)
    # 初始化损失为零
    jdict, stats = [], []
    # 初始化 JSON 字典和统计信息列表
    # callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    # 创建进度条

    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        # 遍历数据加载器中的每个批次
        # callbacks.run('on_val_batch_start')
        with dt[0]:
            # 使用第一个性能分析器
            if cuda:
                im = im.to(device, non_blocking=True)
                # 将图像数据移动到设备上
                targets = targets.to(device)
                # 将目标数据移动到设备上
                masks = masks.to(device)
                # 将掩码数据移动到设备上
            masks = masks.float()
            # 将掩码转换为浮点型
            im = im.half() if half else im.float()  # uint8 to fp16/32
            # 将图像数据转换为半精度或单精度
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 将图像数据归一化到 [0.0, 1.0] 范围
            nb, _, height, width = im.shape  # batch size, channels, height, width
            # 获取批次大小、通道数、高度和宽度

        # Inference
        with dt[1]:
            # 使用第二个性能分析器
            preds, protos, train_out = model(im) if compute_loss else (*model(im, augment=augment)[:2], None)
            # 进行推理，获取预测结果和原型输出

        # Loss
        if compute_loss:
            # 如果需要计算损失
            loss += compute_loss((train_out, protos), targets, masks)[1]  # box, obj, cls
            # 更新损失值

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # 将目标框的坐标转换为像素值
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        # 为每个样本准备标签
        with dt[2]:
            # 使用第三个性能分析器
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det, nm=nm
            )
            # 进行非极大值抑制

        # Metrics
        plot_masks = []  # masks for plotting
        # 初始化绘图掩码列表
        for si, (pred, proto) in enumerate(zip(preds, protos)):
            # 遍历每个预测和原型
            labels = targets[targets[:, 0] == si, 1:]
            # 获取当前样本的标签
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            # 获取标签和预测的数量
            path, shape = Path(paths[si]), shapes[si][0]
            # 获取当前样本的路径和形状
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            # 初始化正确掩码
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            # 初始化正确边界框
            seen += 1
            # 更新已处理样本计数

            if npr == 0:
                # 如果没有预测
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    # 记录统计信息
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                        # 更新混淆矩阵
                continue

            # Masks
            midx = [si] if overlap else targets[:, 0] == si
            # 根据重叠情况选择掩码索引
            gt_masks = masks[midx]
            # 获取真实掩码
            pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=im[si].shape[1:])
            # 处理预测掩码

            # Predictions
            if single_cls:
                pred[:, 5] = 0
                # 如果是单类数据集，将类别索引设为 0
            predn = pred.clone()
            # 克隆预测结果
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            # 将预测框的坐标转换为原始空间

            # Evaluate
            if nl:
                # 如果有标签
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                # 将目标框从 (x_center, y_center, width, height) 转换为 (x1, y1, x2, y2) 格式
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # 将目标框的坐标转换为原始空间
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                # 合并标签和目标框
                correct_bboxes = process_batch(predn, labelsn, iouv)
                # 处理边界框
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)
                # 处理掩码
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
                    # 更新混淆矩阵
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls)
            # 记录统计信息

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            # 将预测掩码转换为 uint8 类型
            if plots and batch_i < 3:
                plot_masks.append(pred_masks[:15])  # filter top 15 to plot
                # 过滤前 15 个掩码用于绘图

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
                # 保存预测结果到文本文件
            if save_json:
                pred_masks = scale_image(
                    im[si].shape[1:], pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1]
                )
                # 对预测掩码进行缩放
                save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary
                # 将预测结果添加到 COCO-JSON 字典中
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
            
            # Plot images
            # 绘制图像
            if plots and batch_i < 3:
                # 如果需要绘制图像并且当前批次小于3
                if len(plot_masks):
                    # 如果有掩码
                    plot_masks = torch.cat(plot_masks, dim=0)
                    # 将所有掩码在第0维度上连接成一个张量
                plot_images_and_masks(im, targets, masks, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)
                # 绘制真实图像和掩码，并保存为“val_batch{batch_i}_labels.jpg”
                plot_images_and_masks(
                    im,
                    output_to_target(preds, max_det=15),
                    plot_masks,
                    paths,
                    save_dir / f"val_batch{batch_i}_pred.jpg",
                    names,
                )  # pred
                # 绘制预测图像和掩码，并保存为“val_batch{batch_i}_pred.jpg”

            # callbacks.run('on_val_batch_end')
            # 调用回调函数，执行验证批次结束时的操作

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # 将统计信息从张量转换为 NumPy 数组，使用 torch.cat 将每个统计信息按维度 0 连接起来，并将其移动到 CPU 上。

    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        # 如果 stats 不为空且第一个统计信息中有数据，则调用 ap_per_class_box_and_mask 函数计算每个类别的 AP（平均精度）及其他指标。

        metrics.update(results)
        # 更新 metrics 对象，加入新计算的结果。

    nt = np.bincount(stats[4].astype(int), minlength=nc)  # number of targets per class
    # 计算每个类别的目标数量，使用 np.bincount 统计 stats[4] 中每个类别的出现次数，确保长度至少为 nc。

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 8  # print format
    # 定义打印格式，包含类别名称、已见目标数量、每个类别的目标数量和其他指标的格式。

    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
    # 打印所有类别的结果，包括总目标数量和平均结果。

    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")
        # 如果没有找到任何标签，发出警告，无法计算指标。

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))
            # 如果需要详细输出或类别数小于 50 且不是训练状态，则打印每个类别的结果。

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    # 计算每张图片的处理速度，将时间转换为毫秒。

    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)
        # 如果不是训练状态，打印每张图片的预处理、推理和 NMS 的速度。

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # 如果需要绘制图表，绘制混淆矩阵并保存到指定目录。

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()
    # 从 metrics 中获取平均结果，包括多种指标。

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        # 如果需要保存 JSON 且 jdict 不为空，获取权重的文件名。

        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        # 定义 COCO 数据集的注释文件路径。

        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        # 定义预测结果的 JSON 文件路径。

        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        # 打印信息，表示正在评估 pycocotools 的 mAP，并保存预测结果。

        with open(pred_json, "w") as f:
            json.dump(jdict, f)
            # 将 jdict 保存为 JSON 文件。

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            # 导入 COCO API 的相关模块。

            anno = COCO(anno_json)  # init annotations api
            # 初始化 COCO 注释 API。

            pred = anno.loadRes(pred_json)  # init predictions api
            # 加载预测结果，初始化预测 API。

            results = []
            for eval in COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm"):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # img ID to evaluate
                    # 如果是 COCO 数据集，设置要评估的图像 ID。

                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])  # update results (mAP@0.5:0.95, mAP@0.5)
                # 评估、累积并总结结果，将 mAP 相关的统计信息添加到结果列表中。

            map_bbox, map50_bbox, map_mask, map50_mask = results
            # 从结果中获取不同的 mAP 指标。

        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")
            # 捕捉异常，如果 pycocotools 无法运行，记录错误信息。

    # Return results
    model.float()  # for training
    # 将模型转换为浮点模式，以备训练使用。

    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # 如果不是训练状态，记录保存的标签数量和保存目录。

    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask
    # 最终的指标结果，包括多种指标。

    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t
    # 返回最终指标、每个类别的 mAP 和处理速度。


def parse_opt():
    """Parses command line arguments for configuring YOLOv5 options like dataset path, weights, batch size, and
    inference settings.
    """
    # 解析命令行参数以配置 YOLOv5 选项，例如数据集路径、权重、批大小和推理设置。
    
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象。

    parser.add_argument("--data", type=str, default=ROOT / "data/coco128-seg.yaml", help="dataset.yaml path")
    # 添加数据集 YAML 文件路径参数，默认为 coco128-seg.yaml。

    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-seg.pt", help="model path(s)")
    # 添加权重文件路径参数，支持多个权重文件，默认为 yolov5s-seg.pt。

    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    # 添加批大小参数，默认为 32。

    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    # 添加图像大小参数，默认为 640 像素。

    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    # 添加置信度阈值参数，默认为 0.001。

    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    # 添加 NMS 的 IoU 阈值参数，默认为 0.6。

    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    # 添加每张图像最大检测数参数，默认为 300。

    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    # 添加任务类型参数，默认为验证（val）。

    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 添加设备参数，指定使用的 CUDA 设备或 CPU。

    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    # 添加数据加载器的最大工作线程数参数，默认为 8。

    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    # 添加单类数据集参数，若设置则将数据集视为单类。

    parser.add_argument("--augment", action="store_true", help="augmented inference")
    # 添加增强推理参数，若设置则使用增强推理。

    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    # 添加详细输出参数，若设置则按类别报告 mAP。

    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    # 添加保存结果到文本文件的参数，若设置则保存结果。

    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    # 添加保存混合结果到文本文件的参数，若设置则保存标签和预测的混合结果。

    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    # 添加保存置信度到文本文件的参数，若设置则在保存的标签中包含置信度。

    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    # 添加保存 COCO-JSON 结果文件的参数，若设置则保存为 JSON 格式。

    parser.add_argument("--project", default=ROOT / "runs/val-seg", help="save results to project/name")
    # 添加项目保存路径参数，默认为 runs/val-seg。

    parser.add_argument("--name", default="exp", help="save to project/name")
    # 添加项目名称参数，默认为 "exp"。

    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 添加允许存在的项目名称参数，若设置则不递增项目名称。

    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # 添加使用 FP16 半精度推理的参数，若设置则启用半精度。

    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    # 添加使用 OpenCV DNN 进行 ONNX 推理的参数，若设置则启用 DNN。

    opt = parser.parse_args()
    # 解析命令行参数并将结果存储在 opt 中。

    opt.data = check_yaml(opt.data)  # check YAML
    # 检查数据集 YAML 文件的有效性。

    # opt.save_json |= opt.data.endswith('coco.yaml')
    # 如果数据集 YAML 文件以 coco.yaml 结尾，则设置保存 JSON 的参数。

    opt.save_txt |= opt.save_hybrid
    # 如果设置了保存混合结果，则也设置保存文本文件的参数。

    print_args(vars(opt))
    # 打印解析后的参数。

    return opt
    # 返回解析后的参数对象。


def main(opt):
    """Executes YOLOv5 tasks including training, validation, testing, speed, and study with configurable options."""
    # 执行 YOLOv5 任务，包括训练、验证、测试、速度评估和研究，使用可配置选项。

    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    # 检查所需的 Python 包是否已安装，排除 tensorboard 和 thop。

    if opt.task in ("train", "val", "test"):  # run normally
        # 如果任务是训练、验证或测试，则正常运行。
        
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.warning(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
            # 如果置信度阈值大于 0.001，则发出警告，可能导致无效结果。

        if opt.save_hybrid:
            LOGGER.warning("WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone")
            # 如果设置了保存混合结果，则发出警告，说明高 mAP 是由于混合标签，而非仅仅是预测结果。

        run(**vars(opt))
        # 调用 run 函数执行任务，传入所有参数。

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        # 如果权重是列表则直接使用，否则将其转换为列表。

        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        # 如果可用 CUDA 且设备不是 CPU，则设置使用 FP16 以获得最快的结果。

        if opt.task == "speed":  # speed benchmarks
            # 如果任务是速度评估，则设置相关参数。
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            # 设置置信度阈值、IoU 阈值和保存 JSON 的参数。

            for opt.weights in weights:
                run(**vars(opt), plots=False)
                # 对每个权重调用 run 函数，执行速度评估，不绘制图表。

        elif opt.task == "study":  # speed vs mAP benchmarks
            # 如果任务是研究速度与 mAP 的关系，则执行相关操作。
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                # 生成保存结果的文件名。

                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                # 定义 x 轴（图像尺寸）和 y 轴（结果）。

                for opt.imgsz in x:  # img-size
                    # 遍历不同的图像尺寸。
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    # 打印当前运行的信息。

                    r, _, t = run(**vars(opt), plots=False)
                    # 调用 run 函数运行，并获取结果和时间。

                    y.append(r + t)  # results and times
                    # 将结果和时间添加到 y 轴数据中。

                np.savetxt(f, y, fmt="%10.4g")  # save
                # 将结果保存到文本文件中。

            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            # 将所有研究结果文件压缩为 study.zip。

            plot_val_study(x=x)  # plot
            # 绘制研究结果的图表。

        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')
            # 如果任务不在已实现的范围内，则抛出未实现的异常。


if __name__ == "__main__":
    opt = parse_opt()
    # 如果该脚本是主程序，则解析命令行参数。

    main(opt)
    # 调用 main 函数执行任务。