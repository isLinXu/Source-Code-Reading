# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset.

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
                                       yolov5s-cls.torchscript        # TorchScript
                                       yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                       yolov5s-cls_openvino_model     # OpenVINO
                                       yolov5s-cls.engine             # TensorRT
                                       yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                       yolov5s-cls_saved_model        # TensorFlow SavedModel
                                       yolov5s-cls.pb                 # TensorFlow GraphDef
                                       yolov5s-cls.tflite             # TensorFlow Lite
                                       yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                       yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse  # 导入argparse模块，用于解析命令行参数
import os  # 导入os模块，用于与操作系统交互
import sys  # 导入sys模块，提供对Python解释器的访问
from pathlib import Path  # 从pathlib模块导入Path类，用于处理文件路径

import torch  # 导入PyTorch库
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[1]  # YOLOv5根目录
if str(ROOT) not in sys.path:  # 如果根目录不在系统路径中
    sys.path.append(str(ROOT))  # 将根目录添加到系统路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 计算相对路径

from models.common import DetectMultiBackend  # 从models.common模块导入DetectMultiBackend类
from utils.dataloaders import create_classification_dataloader  # 从utils.dataloaders模块导入create_classification_dataloader函数
from utils.general import (  # 从utils.general模块导入多个实用函数
    LOGGER,  # 日志记录器
    TQDM_BAR_FORMAT,  # tqdm进度条格式
    Profile,  # 性能分析类
    check_img_size,  # 检查图像大小的函数
    check_requirements,  # 检查依赖项的函数
    colorstr,  # 颜色字符串处理函数
    increment_path,  # 增加路径的函数
    print_args,  # 打印参数的函数
)
from utils.torch_utils import select_device, smart_inference_mode  # 从utils.torch_utils模块导入设备选择和智能推理模式的函数


@smart_inference_mode()  # 使用智能推理模式装饰器
def run(  # 定义run函数
    data=ROOT / "../datasets/mnist",  # 数据集目录
    weights=ROOT / "yolov5s-cls.pt",  # 模型权重文件路径
    batch_size=128,  # 批处理大小
    imgsz=224,  # 推理图像大小（像素）
    device="",  # CUDA设备，例如0或0,1,2,3或cpu
    workers=8,  # 最大数据加载器工作线程数（在DDP模式下每个RANK）
    verbose=False,  # 是否输出详细信息
    project=ROOT / "runs/val-cls",  # 保存到项目/名称
    name="exp",  # 保存到项目/名称
    exist_ok=False,  # 允许存在的项目/名称，不递增
    half=False,  # 使用FP16半精度推理
    dnn=False,  # 使用OpenCV DNN进行ONNX推理
    model=None,  # 模型
    dataloader=None,  # 数据加载器
    criterion=None,  # 损失函数
    pbar=None,  # 进度条
):
    # Initialize/load model and set device  # 初始化/加载模型并设置设备
    training = model is not None  # 判断模型是否存在，若存在则为训练模式
    if training:  # 如果是训练模式
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # 获取模型设备、PyTorch模型标志
        half &= device.type != "cpu"  # 仅在CUDA上支持半精度
        model.half() if half else model.float()  # 根据半精度设置模型为half或float
    else:  # 如果直接调用
        device = select_device(device, batch_size=batch_size)  # 选择设备

        # Directories  # 目录设置
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增加运行目录
        save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录

        # Load model  # 加载模型
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)  # 初始化DetectMultiBackend模型
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine  # 获取模型的步幅和其他信息
        imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小
        half = model.fp16  # FP16在支持CUDA的有限后端上可用
        if engine:  # 如果有引擎
            batch_size = model.batch_size  # 使用模型的批处理大小
        else:  # 否则
            device = model.device  # 获取模型设备
            if not (pt or jit):  # 如果不是PyTorch或JIT模型
                batch_size = 1  # export.py模型默认使用批处理大小1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")  # 记录信息

        # Dataloader  # 数据加载器设置
        data = Path(data)  # 将数据路径转换为Path对象
        test_dir = data / "test" if (data / "test").exists() else data / "val"  # 数据/test或数据/val
        dataloader = create_classification_dataloader(  # 创建分类数据加载器
            path=test_dir, imgsz=imgsz, batch_size=batch_size, augment=False, rank=-1, workers=workers
        )

    model.eval()  # 设置模型为评估模式
    pred, targets, loss, dt = [], [], 0, (Profile(device=device), Profile(device=device), Profile(device=device))  # 初始化预测、目标、损失和性能分析器
    n = len(dataloader)  # 获取批次数量
    action = "validating" if dataloader.dataset.root.stem == "val" else "testing"  # 判断当前是验证还是测试
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"  # 设置进度条描述
    bar = tqdm(dataloader, desc, n, not training, bar_format=TQDM_BAR_FORMAT, position=0)  # 初始化进度条
    with torch.cuda.amp.autocast(enabled=device.type != "cpu"):  # 在CUDA设备上启用自动混合精度
        for images, labels in bar:  # 遍历数据加载器中的图像和标签
            with dt[0]:  # 记录预处理时间
                images, labels = images.to(device, non_blocking=True), labels.to(device)  # 将图像和标签移动到设备上

            with dt[1]:  # 记录推理时间
                y = model(images)  # 执行模型推理

            with dt[2]:  # 记录后处理时间
                pred.append(y.argsort(1, descending=True)[:, :5])  # 记录前5个预测
                targets.append(labels)  # 记录标签
                if criterion:  # 如果存在损失函数
                    loss += criterion(y, labels)  # 计算损失并累加

    loss /= n  # 计算平均损失
    pred, targets = torch.cat(pred), torch.cat(targets)  # 合并所有预测和目标
    correct = (targets[:, None] == pred).float()  # 计算正确预测
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # 计算top1和top5准确率
    top1, top5 = acc.mean(0).tolist()  # 获取top1和top5准确率的平均值

    if pbar:  # 如果存在进度条
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"  # 更新进度条描述
    if verbose:  # 如果需要详细输出
        LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")  # 打印表头
        LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")  # 打印所有类的准确率
        for i, c in model.names.items():  # 遍历每个类
            acc_i = acc[targets == i]  # 获取当前类的准确率
            top1i, top5i = acc_i.mean(0).tolist()  # 计算当前类的top1和top5准确率
            LOGGER.info(f"{c:>24}{acc_i.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")  # 打印当前类的结果

        # Print results  # 打印结果
        t = tuple(x.t / len(dataloader.dataset.samples) * 1e3 for x in dt)  # 计算每张图像的处理速度
        shape = (1, 3, imgsz, imgsz)  # 定义图像形状
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}" % t)  # 打印速度信息
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")  # 打印结果保存路径

    return top1, top5, loss  # 返回top1、top5准确率和损失

def parse_opt():  # 定义parse_opt函数
    """Parses and returns command line arguments for YOLOv5 model evaluation and inference settings."""  # 解析并返回YOLOv5模型评估和推理设置的命令行参数
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象，用于解析命令行参数
    parser.add_argument("--data", type=str, default=ROOT / "../datasets/mnist", help="dataset path")  # 添加数据集路径参数
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-cls.pt", help="model.pt path(s)")  # 添加模型权重参数，支持多个路径
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")  # 添加批处理大小参数
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="inference size (pixels)")  # 添加图像大小参数，支持多个名称
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  # 添加设备选择参数
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")  # 添加最大数据加载器工作线程数参数
    parser.add_argument("--verbose", nargs="?", const=True, default=True, help="verbose output")  # 添加详细输出参数
    parser.add_argument("--project", default=ROOT / "runs/val-cls", help="save to project/name")  # 添加项目保存路径参数
    parser.add_argument("--name", default="exp", help="save to project/name")  # 添加实验名称参数
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")  # 添加允许存在的项目名称参数
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")  # 添加使用FP16半精度推理的参数
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")  # 添加使用OpenCV DNN进行ONNX推理的参数
    opt = parser.parse_args()  # 解析命令行参数并将结果存储在opt中
    print_args(vars(opt))  # 打印解析后的参数
    return opt  # 返回解析后的参数


def main(opt):  # 定义main函数，接收解析后的参数
    """Executes the YOLOv5 model prediction workflow, handling argument parsing and requirement checks."""  # 执行YOLOv5模型预测工作流，处理参数解析和依赖检查
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))  # 检查依赖项，排除tensorboard和thop
    run(**vars(opt))  # 调用run函数，传入解析后的参数


if __name__ == "__main__":  # 如果当前脚本是主程序
    opt = parse_opt()  # 解析命令行参数
    main(opt)  # 调用main函数
