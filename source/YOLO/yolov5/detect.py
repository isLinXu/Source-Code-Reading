# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
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

import argparse  # 导入 argparse 模块，用于解析命令行参数
import csv  # 导入 csv 模块，用于处理 CSV 文件
import os  # 导入 os 模块，用于与操作系统交互
import platform  # 导入 platform 模块，用于获取操作系统信息
import sys  # 导入 sys 模块，用于访问与 Python 解释器相关的变量和函数
from pathlib import Path  # 从 pathlib 模块导入 Path 类，用于处理文件路径

import torch  # 导入 PyTorch 库

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:  # 如果根目录不在系统路径中
    sys.path.append(str(ROOT))  # 将根目录添加到系统路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 计算相对路径

from ultralytics.utils.plotting import Annotator, colors, save_one_box  # 导入绘图工具

from models.common import DetectMultiBackend  # 导入多后端检测模型
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams  # 导入数据加载工具
from utils.general import (  # 导入常用工具函数
    LOGGER,  # 日志记录器
    Profile,  # 性能分析器
    check_file,  # 检查文件存在性
    check_img_size,  # 检查图像大小
    check_imshow,  # 检查图像显示
    check_requirements,  # 检查依赖项
    colorstr,  # 颜色字符串
    cv2,  # OpenCV 库
    increment_path,  # 增加路径
    non_max_suppression,  # 非极大值抑制
    print_args,  # 打印参数
    scale_boxes,  # 缩放边界框
    strip_optimizer,  # 清理优化器
    xyxy2xywh,  # 转换边界框格式
)
from utils.torch_utils import select_device, smart_inference_mode  # 导入设备选择和智能推理模式工具

@smart_inference_mode()  # 使用智能推理模式装饰器
def run(  # 定义运行函数，接收多个参数
    weights=ROOT / "yolov5s.pt",  # model path or triton URL  # 模型路径或 Triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)  # 输入源（文件、目录、URL、屏幕、摄像头）
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path  # 数据集配置文件路径
    imgsz=(640, 640),  # inference size (height, width)  # 推理图像大小（高度，宽度）
    conf_thres=0.25,  # confidence threshold  # 置信度阈值
    iou_thres=0.45,  # NMS IOU threshold  # NMS IOU 阈值
    max_det=1000,  # maximum detections per image  # 每张图像的最大检测数量
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu  # CUDA 设备，例如 0 或 0,1,2,3 或 cpu
    view_img=False,  # show results  # 是否显示结果
    save_txt=False,  # save results to *.txt  # 是否将结果保存为 *.txt
    save_csv=False,  # save results in CSV format  # 是否将结果保存为 CSV 格式
    save_conf=False,  # save confidences in --save-txt labels  # 是否在 --save-txt 标签中保存置信度
    save_crop=False,  # save cropped prediction boxes  # 是否保存裁剪的预测框
    nosave=False,  # do not save images/videos  # 是否不保存图像/视频
    classes=None,  # filter by class: --class 0, or --class 0 2 3  # 按类别过滤
    agnostic_nms=False,  # class-agnostic NMS  # 类别无关的 NMS
    augment=False,  # augmented inference  # 是否使用增强推理
    visualize=False,  # visualize features  # 是否可视化特征
    update=False,  # update all models  # 是否更新所有模型
    project=ROOT / "runs/detect",  # save results to project/name  # 将结果保存到项目/名称
    name="exp",  # save results to project/name  # 将结果保存到项目/名称
    exist_ok=False,  # existing project/name ok, do not increment  # 允许存在的项目/名称，不递增
    line_thickness=3,  # bounding box thickness (pixels)  # 边界框厚度（像素）
    hide_labels=False,  # hide labels  # 是否隐藏标签
    hide_conf=False,  # hide confidences  # 是否隐藏置信度
    half=False,  # use FP16 half-precision inference  # 是否使用 FP16 半精度推理
    dnn=False,  # use OpenCV DNN for ONNX inference  # 是否使用 OpenCV DNN 进行 ONNX 推理
    vid_stride=1,  # video frame-rate stride  # 视频帧率步长
):
    source = str(source)  # 将输入源转换为字符串
    save_img = not nosave and not source.endswith(".txt")  # save inference images  # 是否保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 检查输入源是否为文件格式
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))  # 检查输入源是否为 URL
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)  # 检查输入源是否为摄像头
    screenshot = source.lower().startswith("screen")  # 检查输入源是否为屏幕截图
    if is_url and is_file:  # 如果输入源是 URL 且是文件
        source = check_file(source)  # download  # 检查文件

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run  # 增加运行目录
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir  # 创建目录，如果需要保存文本文件则创建 labels 子目录

    # Load model
    device = select_device(device)  # 选择设备（CPU 或 GPU）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 加载多后端检测模型
    stride, names, pt = model.stride, model.names, model.pt  # 获取模型的步幅、类别名称和是否为 PyTorch 模型
    imgsz = check_img_size(imgsz, s=stride)  # check image size  # 检查图像大小是否符合要求

    # Dataloader
    bs = 1  # batch_size  # 设置批量大小为 1
    if webcam:  # 如果输入源为摄像头
        view_img = check_imshow(warn=True)  # 检查是否可以显示图像，并发出警告
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载摄像头流
        bs = len(dataset)  # 更新批量大小为数据集的长度
    elif screenshot:  # 如果输入源为屏幕截图
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)  # 加载屏幕截图
    else:  # 如果输入源为文件或其他类型
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载图像
    vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化视频路径和视频写入器

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup  # 进行模型预热
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))  # 初始化计数器、窗口和性能分析器
    for path, im, im0s, vid_cap, s in dataset:  # 遍历数据集中的每个项目
        with dt[0]:  # 记录第一个阶段的时间
            im = torch.from_numpy(im).to(model.device)  # 将 NumPy 图像转换为 PyTorch 张量并移动到模型设备
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32  # 根据模型设置转换数据类型
            im /= 255  # 0 - 255 to 0.0 - 1.0  # 将图像归一化到 [0.0, 1.0]
            if len(im.shape) == 3:  # 如果图像维度为 3
                im = im[None]  # expand for batch dim  # 扩展维度以适应批量
            if model.xml and im.shape[0] > 1:  # 如果模型为 XML 格式且图像数量大于 1
                ims = torch.chunk(im, im.shape[0], 0)  # 将图像分块

        # Inference
        with dt[1]:  # 记录第二个阶段的时间
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # 如果需要可视化，则创建可视化目录
            if model.xml and im.shape[0] > 1:  # 如果模型为 XML 格式且图像数量大于 1
                pred = None  # 初始化预测结果
                for image in ims:  # 遍历每个图像块
                    if pred is None:  # 如果预测结果为空
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)  # 进行推理并添加维度
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)  # 合并预测结果
                pred = [pred, None]  # 将预测结果封装为列表
            else:
                pred = model(im, augment=augment, visualize=visualize)  # 进行推理

        # NMS
        with dt[2]:  # 记录第三个阶段的时间
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 进行非极大值抑制

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)  # 应用第二阶段分类器（可选）

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"  # 定义 CSV 文件的保存路径


        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):  # 定义写入 CSV 文件的函数
            """Writes prediction data for an image to a CSV file, appending if the file exists."""  # 将图像的预测数据写入 CSV 文件，如果文件存在则追加
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}  # 创建包含图像名称、预测和置信度的数据字典
            with open(csv_path, mode="a", newline="") as f:  # 以追加模式打开 CSV 文件
                writer = csv.DictWriter(f, fieldnames=data.keys())  # 创建字典写入器
                if not csv_path.is_file():  # 如果 CSV 文件不存在
                    writer.writeheader()  # 写入表头
                writer.writerow(data)  # 写入数据行

        # Process predictions
        for i, det in enumerate(pred):  # per image  # 遍历每个图像的检测结果
            seen += 1  # 统计已处理的图像数量
            if webcam:  # batch_size >= 1  # 如果输入源为摄像头
                p, im0, frame = path[i], im0s[i].copy(), dataset.count  # 获取当前路径、原始图像和帧数
                s += f"{i}: "  # 更新状态字符串
            else:  # 如果输入源不是摄像头
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)  # 获取路径、原始图像和帧数

            p = Path(p)  # 将路径转换为 Path 对象
            save_path = str(save_dir / p.name)  # 定义保存图像的路径
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # 定义保存标签的路径
            s += "%gx%g " % im.shape[2:]  # 将图像尺寸添加到状态字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh  # 计算归一化增益
            imc = im0.copy() if save_crop else im0  # 如果需要裁剪，则复制原始图像
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 初始化标注器
            if len(det):  # 如果检测结果不为空
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # 将检测框从图像大小缩放到原始图像大小

                # Print results
                for c in det[:, 5].unique():  # 遍历每个唯一的类别
                    n = (det[:, 5] == c).sum()  # 计算每个类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 将检测数量和类别名称添加到状态字符串

                # Write results
                for *xyxy, conf, cls in reversed(det):  # 遍历检测结果，提取边界框、置信度和类别
                    c = int(cls)  # integer class  # 将类别转换为整数
                    label = names[c] if hide_conf else f"{names[c]}"  # 如果不隐藏置信度，则生成标签
                    confidence = float(conf)  # 转换置信度为浮点数
                    confidence_str = f"{confidence:.2f}"  # 格式化置信度字符串

                    if save_csv:  # 如果需要保存 CSV
                        write_to_csv(p.name, label, confidence_str)  # 写入 CSV 文件

                    if save_txt:  # Write to file  # 如果需要保存 TXT 文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 转换为归一化的 xywh 格式
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 定义标签格式
                        with open(f"{txt_path}.txt", "a") as f:  # 以追加模式打开 TXT 文件
                            f.write(("%g " * len(line)).rstrip() % line + "\n")  # 写入检测结果

                    if save_img or save_crop or view_img:  # Add bbox to image  # 如果需要保存图像或裁剪图像或查看图像
                        c = int(cls)  # integer class  # 将类别转换为整数
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")  # 生成标签
                        annotator.box_label(xyxy, label, color=colors(c, True))  # 在图像上绘制边界框
                    if save_crop:  # 如果需要裁剪图像
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)  # 保存裁剪的边界框图像

            # Stream results
            im0 = annotator.result()  # 获取标注后的结果图像
            if view_img:  # 如果需要查看图像
                if platform.system() == "Linux" and p not in windows:  # 如果是 Linux 系统且窗口不在列表中
                    windows.append(p)  # 将路径添加到窗口列表
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小（Linux）
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])  # 调整窗口大小
                cv2.imshow(str(p), im0)  # 显示结果图像
                cv2.waitKey(1)  # 1 毫秒等待

            # Save results (image with detections)
            if save_img:  # 如果需要保存图像
                if dataset.mode == "image":  # 如果数据集模式为图像
                    cv2.imwrite(save_path, im0)  # 保存图像
                else:  # 'video' or 'stream'  # 如果数据集模式为视频或流
                    if vid_path[i] != save_path:  # 如果保存路径不同
                        vid_path[i] = save_path  # 更新视频路径
                        if isinstance(vid_writer[i], cv2.VideoWriter):  # 如果视频写入器存在
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 如果是视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
                        else:  # 如果是流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # 设置默认帧率和图像大小
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # 强制将结果视频后缀设置为 *.mp4
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # 创建视频写入器
                    vid_writer[i].write(im0)  # 写入当前帧

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")  # 打印推理时间

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image  # 计算每张图像的速度
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)  # 打印速度信息
        if save_txt or save_img:  # 如果需要保存 TXT 或图像
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""  # 统计保存的标签数量
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")  # 打印保存结果的信息
        if update:  # 如果需要更新模型
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)  # 更新模型以修复 SourceChangeWarning


def parse_opt():  # 定义解析命令行参数的函数
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""  # 解析 YOLOv5 检测的命令行参数，设置推理选项和模型配置
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")  # 添加权重参数，支持多个权重路径
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")  # 添加输入源参数
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")  # 添加数据集配置文件路径参数
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")  # 添加图像大小参数
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")  # 添加置信度阈值参数
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")  # 添加 NMS IOU 阈值参数
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")  # 添加每张图像的最大检测数量参数
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  # 添加设备参数
    parser.add_argument("--view-img", action="store_true", help="show results")  # 添加显示结果的参数
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")  # 添加保存结果为 TXT 文件的参数
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")  # 添加保存结果为 CSV 文件的参数
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")  # 添加保存置信度到 TXT 标签的参数
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")  # 添加保存裁剪预测框的参数
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")  # 添加不保存图像/视频的参数
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")  # 添加按类别过滤的参数
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")  # 添加类别无关的 NMS 参数
    parser.add_argument("--augment", action="store_true", help="augmented inference")  # 添加增强推理的参数
    parser.add_argument("--visualize", action="store_true", help="visualize features")  # 添加可视化特征的参数
    parser.add_argument("--update", action="store_true", help="update all models")  # 添加更新所有模型的参数
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")  # 添加保存结果到项目的参数
    parser.add_argument("--name", default="exp", help="save results to project/name")  # 添加保存结果的名称参数
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")  # 添加允许存在的项目/名称参数
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")  # 添加边界框厚度参数
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")  # 添加隐藏标签的参数
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")  # 添加隐藏置信度的参数
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")  # 添加使用 FP16 半精度推理的参数
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")  # 添加使用 OpenCV DNN 进行 ONNX 推理的参数
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")  # 添加视频帧率步长的参数
    opt = parser.parse_args()  # 解析命令行参数
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand  # 如果图像大小参数只有一个值，则将其扩展为两个值
    print_args(vars(opt))  # 打印参数
    return opt  # 返回解析后的参数对象


def main(opt):  # 定义主函数
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""  # 执行 YOLOv5 模型推理，检查要求后运行模型
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))  # 检查依赖项
    run(**vars(opt))  # 运行模型，传入解析后的参数


if __name__ == "__main__":  # 如果是主程序
    opt = parse_opt()  # 解析命令行参数
    main(opt)  # 调用主函数
