# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:  # 检查根目录是否在系统路径中
    sys.path.append(str(ROOT))  # 将根目录添加到系统路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from models.common import DetectMultiBackend  # 从 models.common 导入 DetectMultiBackend 类
from utils.callbacks import Callbacks  # 从 utils.callbacks 导入 Callbacks 类
from utils.dataloaders import create_dataloader  # 从 utils.dataloaders 导入 create_dataloader 函数
from utils.general import (  # 从 utils.general 导入多个工具函数
    LOGGER,  # 日志记录器
    TQDM_BAR_FORMAT,  # TQDM 进度条格式
    Profile,  # 性能分析类
    check_dataset,  # 检查数据集的函数
    check_img_size,  # 检查图像大小的函数
    check_requirements,  # 检查依赖项的函数
    check_yaml,  # 检查 YAML 文件的函数
    coco80_to_coco91_class,  # COCO 80 类映射到 COCO 91 类的函数
    colorstr,  # 颜色字符串格式化函数
    increment_path,  # 增加路径的函数
    non_max_suppression,  # 非极大值抑制函数
    print_args,  # 打印参数的函数
    scale_boxes,  # 缩放边界框的函数
    xywh2xyxy,  # 将 xywh 格式转换为 xyxy 格式的函数
    xyxy2xywh,  # 将 xyxy 格式转换为 xywh 格式的函数
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou  # 从 utils.metrics 导入混淆矩阵、每类平均精度和框的 IoU 计算函数
from utils.plots import output_to_target, plot_images, plot_val_study  # 从 utils.plots 导入绘图相关函数
from utils.torch_utils import select_device, smart_inference_mode  # 从 utils.torch_utils 导入设备选择和智能推理模式函数


def save_one_txt(predn, save_conf, shape, file):
    """Saves one detection result to a txt file in normalized xywh format, optionally including confidence."""
    # 将一个检测结果保存到 txt 文件中，采用归一化的 xywh 格式，选项上可以包含置信度
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh  # 归一化增益
    for *xyxy, conf, cls in predn.tolist():  # 遍历预测结果
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh  # 归一化的 xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format  # 标签格式
        with open(file, "a") as f:  # 以追加模式打开文件
            f.write(("%g " * len(line)).rstrip() % line + "\n")  # 写入结果


def save_one_json(predn, jdict, path, class_map):
    """
    Saves one JSON detection result with image ID, category ID, bounding box, and score.

    Example: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    """
    # 保存一个 JSON 格式的检测结果，包括图像 ID、类别 ID、边界框和分数
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem  # 获取图像 ID
    box = xyxy2xywh(predn[:, :4])  # xywh  # 将预测结果转换为 xywh 格式
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner  # 从中心点转换为左上角坐标
    for p, b in zip(predn.tolist(), box.tolist()):  # 遍历预测结果和边界框
        jdict.append(  # 将结果添加到 JSON 字典中
            {
                "image_id": image_id,  # 图像 ID
                "category_id": class_map[int(p[5])],  # 类别 ID
                "bbox": [round(x, 3) for x in b],  # 边界框，保留三位小数
                "score": round(p[4], 5),  # 分数，保留五位小数
            }
        )


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class  # 检测结果数组
        labels (array[M, 5]), class, x1, y1, x2, y2  # 标签数组
    Returns:
        correct (array[N, 10]), for 10 IoU levels  # 返回正确预测矩阵
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)  # 初始化正确预测矩阵
    iou = box_iou(labels[:, 1:], detections[:, :4])  # 计算 IoU
    correct_class = labels[:, 0:1] == detections[:, 5]  # 类别匹配
    for i in range(len(iouv)):  # 遍历 IoU 阈值
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match  # IoU 大于阈值且类别匹配
        if x[0].shape[0]:  # 如果有匹配
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:  # 如果有多个匹配
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按照 IoU 降序排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 去重
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 去重
            correct[matches[:, 1].astype(int), i] = True  # 更新正确预测矩阵
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)  # 返回正确预测矩阵

@smart_inference_mode()  # 使用智能推理模式装饰器
def run(
    data,  # 输入数据
    weights=None,  # model.pt path(s)  # 模型权重路径
    batch_size=32,  # batch size  # 批处理大小
    imgsz=640,  # inference size (pixels)  # 推理图像大小（像素）
    conf_thres=0.001,  # confidence threshold  # 置信度阈值
    iou_thres=0.6,  # NMS IoU threshold  # 非极大值抑制的 IoU 阈值
    max_det=300,  # maximum detections per image  # 每张图像的最大检测数量
    task="val",  # train, val, test, speed or study  # 任务类型：训练、验证、测试、速度测试或研究
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu  # 设备选择，例如 CUDA 设备或 CPU
    workers=8,  # max dataloader workers (per RANK in DDP mode)  # 最大数据加载工作线程数（在 DDP 模式下每个 RANK）
    single_cls=False,  # treat as single-class dataset  # 将数据集视为单类数据集
    augment=False,  # augmented inference  # 是否使用增强推理
    verbose=False,  # verbose output  # 是否输出详细信息
    save_txt=False,  # save results to *.txt  # 是否将结果保存到 *.txt 文件
    save_hybrid=False,  # save label+prediction hybrid results to *.txt  # 是否保存标签和预测的混合结果到 *.txt 文件
    save_conf=False,  # save confidences in --save-txt labels  # 是否在 --save-txt 标签中保存置信度
    save_json=False,  # save a COCO-JSON results file  # 是否保存 COCO-JSON 格式的结果文件
    project=ROOT / "runs/val",  # save to project/name  # 保存到项目/名称
    name="exp",  # save to project/name  # 保存到项目/名称
    exist_ok=False,  # existing project/name ok, do not increment  # 如果项目/名称已存在，则不递增
    half=True,  # use FP16 half-precision inference  # 是否使用 FP16 半精度推理
    dnn=False,  # use OpenCV DNN for ONNX inference  # 是否使用 OpenCV DNN 进行 ONNX 推理
    model=None,  # 模型
    dataloader=None,  # 数据加载器
    save_dir=Path(""),  # 保存目录
    plots=True,  # 是否绘制图像
    callbacks=Callbacks(),  # 回调函数
    compute_loss=None,  # 计算损失的函数
):
    # Initialize/load model and set device  # 初始化/加载模型并设置设备
    training = model is not None  # 检查模型是否存在
    if training:  # called by train.py  # 如果是由 train.py 调用
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model  # 获取模型设备，PyTorch 模型
        half &= device.type != "cpu"  # half precision only supported on CUDA  # 半精度仅在 CUDA 上支持
        model.half() if half else model.float()  # 设置模型为半精度或单精度
    else:  # called directly  # 如果是直接调用
        device = select_device(device, batch_size=batch_size)  # 选择设备

        # Directories  # 目录设置
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run  # 增加运行次数
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir  # 创建目录

        # Load model  # 加载模型
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 初始化多后端检测模型
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine  # 获取模型的步幅和其他参数
        imgsz = check_img_size(imgsz, s=stride)  # check image size  # 检查图像大小
        half = model.fp16  # FP16 supported on limited backends with CUDA  # FP16 仅在有限的 CUDA 后端支持
        if engine:  # 如果模型是引擎类型
            batch_size = model.batch_size  # 获取模型的批处理大小
        else:
            device = model.device  # 获取模型设备
            if not (pt or jit):  # 如果不是 PyTorch 或 JIT 模型
                batch_size = 1  # export.py models default to batch-size 1  # export.py 模型默认批处理大小为 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")  # 强制设置批处理大小为 1

        # Data  # 数据处理
        data = check_dataset(data)  # check  # 检查数据集

    # Configure  # 配置模型
    model.eval()  # 设置模型为评估模式
    cuda = device.type != "cpu"  # 检查是否使用 CUDA
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset  # 检查是否为 COCO 数据集
    nc = 1 if single_cls else int(data["nc"])  # number of classes  # 类别数量
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95  # 创建 IoU 向量
    niou = iouv.numel()  # IoU 数量

    # Dataloader  # 数据加载器
    if not training:  # 如果不是训练模式
        if pt and not single_cls:  # check --weights are trained on --data  # 检查权重是否在数据集上训练
            ncm = model.model.nc  # 获取模型类别数量
            assert ncm == nc, (  # 断言类别数量一致
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup  # 预热模型
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks  # 根据任务设置填充和矩形推理
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images  # 设置任务类型
        dataloader = create_dataloader(  # 创建数据加载器
            data[task],  # 任务对应的数据
            imgsz,  # 图像大小
            batch_size,  # 批处理大小
            stride,  # 步幅
            single_cls,  # 是否为单类
            pad=pad,  # 填充
            rect=rect,  # 矩形推理
            workers=workers,  # 工作线程数
            prefix=colorstr(f"{task}: "),  # 前缀
        )[0]  # 返回数据加载器
    
    seen = 0  # 已处理的图像数量
    confusion_matrix = ConfusionMatrix(nc=nc)  # 初始化混淆矩阵，类别数量为 nc
    names = model.names if hasattr(model, "names") else model.module.names  # get class names  # 获取类别名称
    if isinstance(names, (list, tuple)):  # old format  # 如果是旧格式
        names = dict(enumerate(names))  # 将类别名称转换为字典
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))  # 根据是否为 COCO 数据集设置类别映射
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")  # 打印格式
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 初始化指标
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times  # 初始化性能分析
    loss = torch.zeros(3, device=device)  # 初始化损失
    jdict, stats, ap, ap_class = [], [], [], []  # 初始化结果字典和统计信息
    callbacks.run("on_val_start")  # 运行验证开始的回调
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar  # 进度条
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):  # 遍历数据加载器
        callbacks.run("on_val_batch_start")  # 运行每个批次开始的回调
        with dt[0]:  # 记录时间
            if cuda:  # 如果使用 CUDA
                im = im.to(device, non_blocking=True)  # 将图像移动到设备
                targets = targets.to(device)  # 将目标移动到设备
            im = im.half() if half else im.float()  # uint8 to fp16/32  # 将图像转换为半精度或单精度
            im /= 255  # 0 - 255 to 0.0 - 1.0  # 将图像归一化到 [0, 1]
            nb, _, height, width = im.shape  # batch size, channels, height, width  # 获取图像的形状

        # Inference  # 推理
        with dt[1]:  # 记录时间
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)  # 进行推理

        # Loss  # 计算损失
        if compute_loss:  # 如果需要计算损失
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls  # 累加损失

        # NMS  # 非极大值抑制
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels  # 将目标转换为像素坐标
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling  # 用于自动标注
        with dt[2]:  # 记录时间
            preds = non_max_suppression(  # 进行非极大值抑制
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        # Metrics  # 计算指标
        for si, pred in enumerate(preds):  # 遍历每个预测结果
            labels = targets[targets[:, 0] == si, 1:]  # 获取当前图像的标签
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions  # 标签和预测的数量
            path, shape = Path(paths[si]), shapes[si][0]  # 获取图像路径和形状
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init  # 初始化正确预测矩阵
            seen += 1  # 已处理图像数量增加

            if npr == 0:  # 如果没有预测结果
                if nl:  # 如果有标签
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))  # 记录统计信息
                    if plots:  # 如果需要绘图
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])  # 处理混淆矩阵
                continue  # 继续下一个批次

            # Predictions  # 处理预测结果
            if single_cls:  # 如果是单类
                pred[:, 5] = 0  # 将类别设置为 0
            predn = pred.clone()  # 克隆预测结果
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred  # 将预测结果缩放到原始空间

            # Evaluate  # 评估
            if nl:  # 如果有标签
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes  # 获取目标框
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels  # 将目标框缩放到原始空间
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels  # 合并标签
                correct = process_batch(predn, labelsn, iouv)  # 处理批次
                if plots:  # 如果需要绘图
                    confusion_matrix.process_batch(predn, labelsn)  # 处理混淆矩阵
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # 记录统计信息

            # Save/log  # 保存/记录结果
            if save_txt:  # 如果需要保存为 txt 文件
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)  # 创建标签目录
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")  # 保存预测结果到 txt 文件
            if save_json:  # 如果需要保存为 JSON 文件
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary  # 添加到 COCO-JSON 字典
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])  # 运行每个图像结束的回调

        # Plot images  # 绘制图像
        if plots and batch_i < 3:  # 如果需要绘图且批次小于 3
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels  # 绘制标签图像
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred  # 绘制预测图像

        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)  # 运行每个批次结束的回调

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # 将统计数据中的每个元素沿着第0维连接，并将结果从GPU转移到CPU，最后转换为NumPy数组

    if len(stats) and stats[0].any():
        # 检查统计数据是否存在且第一个统计数据中有任意值
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        # 调用ap_per_class函数计算每个类别的真阳性、假阳性、精确率、召回率、F1分数、平均精确度和类别

        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # 获取AP@0.5的值和AP在不同阈值下的平均值

        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # 计算平均精确率、平均召回率、AP@0.5的平均值和总体AP的平均值

    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    # 统计每个类别的目标数量，使用np.bincount对统计数据进行计数

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    # 定义打印格式

    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    # 记录所有类别的结果，包括已见样本数、目标总数、平均精确率、平均召回率、AP@0.5的平均值和总体AP

    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")
        # 如果没有找到任何标签，发出警告，无法计算指标

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        # 如果需要详细输出或类别少于50且不是训练模式，并且统计数据存在
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            # 记录每个类别的结果，包括类别名称、已见样本数、目标数、精确率、召回率、AP@0.5和AP

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    # 计算每张图像的处理速度，单位为毫秒

    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)
        # 如果不是训练模式，记录每张图像的预处理、推理和NMS的速度

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # 如果需要绘制图表，绘制混淆矩阵并保存到指定目录

        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)
        # 运行回调函数，传递验证结束时的统计数据

    # Save JSON
    if save_json and len(jdict):
        # 如果需要保存JSON并且jdict不为空
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        # 获取权重文件的基本名称

        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        # 设置COCO格式的注释文件路径

        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
            # 如果注释文件不存在，使用数据路径下的注释文件

        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        # 设置预测结果的保存路径

        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        # 记录正在评估pycocotools的mAP，并保存预测结果

        with open(pred_json, "w") as f:
            json.dump(jdict, f)
            # 将预测结果以JSON格式保存

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            # 检查pycocotools的版本要求

            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            # 导入pycocotools库

            anno = COCO(anno_json)  # init annotations api
            # 初始化注释API

            pred = anno.loadRes(pred_json)  # init predictions api
            # 加载预测结果

            eval = COCOeval(anno, pred, "bbox")
            # 初始化COCO评估对象，评估目标框

            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
                # 如果是COCO数据集，设置要评估的图像ID

            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            # 进行评估、累积结果并总结

            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            # 更新结果，获取mAP@0.5:0.95和mAP@0.5的值
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")
            # 如果运行pycocotools时发生异常，记录错误信息

    # Return results
    model.float()  # for training
    # 将模型转换为浮点模式，以便进行训练

    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        # 如果不是训练模式，记录保存的标签文件数量

        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # 记录结果保存的目录

    maps = np.zeros(nc) + map
    # 创建一个与类别数量相同的数组，并将map的值赋给它

    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # 将每个类别的AP值存入maps数组

    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    # 返回平均精确率、平均召回率、AP@0.5的平均值、总体AP、损失值列表、maps数组和处理速度


def parse_opt():
    """Parses command-line options for YOLOv5 model inference configuration."""
    # 解析YOLOv5模型推理配置的命令行选项
    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器

    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    # 添加数据集路径参数，默认为coco128.yaml

    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    # 添加权重文件路径参数，支持多个权重文件，默认为yolov5s.pt

    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    # 添加批处理大小参数，默认为32

    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    # 添加图像大小参数，默认为640像素

    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    # 添加置信度阈值参数，默认为0.001

    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    # 添加NMS的IoU阈值参数，默认为0.6

    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    # 添加每张图像的最大检测数量参数，默认为300

    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    # 添加任务类型参数，默认为验证（val）

    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 添加设备参数，指定使用的CUDA设备或CPU

    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    # 添加最大数据加载工作线程数参数，默认为8

    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    # 添加单类数据集标志，如果设置则将数据集视为单类

    parser.add_argument("--augment", action="store_true", help="augmented inference")
    # 添加增强推理标志，如果设置则启用增强推理

    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    # 添加详细输出标志，如果设置则按类别报告mAP

    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    # 添加保存结果到文本文件的标志

    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    # 添加保存标签和预测混合结果到文本文件的标志

    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    # 添加保存置信度到文本文件的标志

    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    # 添加保存COCO格式JSON结果文件的标志

    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    # 添加项目保存路径参数，默认为runs/val

    parser.add_argument("--name", default="exp", help="save to project/name")
    # 添加实验名称参数，默认为"exp"

    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 添加允许存在的项目名称标志，如果设置则不递增项目名称

    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # 添加使用FP16半精度推理的标志

    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    # 添加使用OpenCV DNN进行ONNX推理的标志

    opt = parser.parse_args()
    # 解析命令行参数

    opt.data = check_yaml(opt.data)  # check YAML
    # 检查YAML文件的有效性

    opt.save_json |= opt.data.endswith("coco.yaml")
    # 如果数据集路径以coco.yaml结尾，则设置保存JSON的标志

    opt.save_txt |= opt.save_hybrid
    # 如果设置了保存混合结果的标志，则也设置保存文本的标志

    print_args(vars(opt))
    # 打印解析后的参数

    return opt
    # 返回解析后的选项


def main(opt):
    """Executes YOLOv5 tasks like training, validation, testing, speed, and study benchmarks based on provided
    options.
    """
    # 根据提供的选项执行YOLOv5任务，如训练、验证、测试、速度和研究基准

    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    # 检查所需的依赖项，排除tensorboard和thop

    if opt.task in ("train", "val", "test"):  # run normally
        # 如果任务为训练、验证或测试，正常运行
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
            # 如果置信度阈值大于0.001，记录警告信息

        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
            # 如果设置了保存混合结果的标志，记录警告信息

        run(**vars(opt))
        # 运行主程序，传递解析后的选项参数

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        # 如果权重是列表，则使用该列表，否则将单个权重放入列表中

        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        # 如果CUDA可用且设备不是CPU，则设置使用FP16以获得最快的结果

        if opt.task == "speed":  # speed benchmarks
            # 如果任务为速度基准测试
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            # 设置置信度阈值、IoU阈值和保存JSON的标志

            for opt.weights in weights:
                run(**vars(opt), plots=False)
                # 对每个权重运行主程序，不绘制图表

        elif opt.task == "study":  # speed vs mAP benchmarks
            # 如果任务为研究基准测试
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                # 生成用于保存结果的文件名

                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                # 设置x轴为图像大小范围，y轴为空列表

                for opt.imgsz in x:  # img-size
                    # 对每个图像大小进行循环
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    # 记录正在运行的图像大小

                    r, _, t = run(**vars(opt), plots=False)
                    # 运行主程序，获得结果和时间，不绘制图表

                    y.append(r + t)  # results and times
                    # 将结果和时间添加到y轴列表中

                np.savetxt(f, y, fmt="%10.4g")  # save
                # 将y轴数据保存到文本文件

            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            # 将所有研究结果文件压缩为study.zip

            plot_val_study(x=x)  # plot
            # 绘制研究结果图

        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')
            # 如果任务不在已定义的范围内，则引发未实现错误


if __name__ == "__main__":
    opt = parse_opt()
    # 解析命令行选项

    main(opt)
    # 执行主程序