# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 classification inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python classify/predict.py --weights yolov5s-cls.pt --source 0                               # webcam
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
    $ python classify/predict.py --weights yolov5s-cls.pt                 # PyTorch
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
import platform  # 导入platform模块，用于获取操作系统信息
import sys  # 导入sys模块，提供对Python解释器的访问
from pathlib import Path  # 从pathlib模块导入Path类，用于处理文件路径

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[1]  # YOLOv5根目录
if str(ROOT) not in sys.path:  # 如果根目录不在系统路径中
    sys.path.append(str(ROOT))  # 将根目录添加到系统路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 计算相对路径

from ultralytics.utils.plotting import Annotator  # 从ultralytics.utils.plotting模块导入Annotator类

from models.common import DetectMultiBackend  # 从models.common模块导入DetectMultiBackend类
from utils.augmentations import classify_transforms  # 从utils.augmentations模块导入分类变换函数
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams  # 从utils.dataloaders模块导入图像和视频格式以及加载函数
from utils.general import (  # 从utils.general模块导入多个实用函数
    LOGGER,  # 日志记录器
    Profile,  # 性能分析类
    check_file,  # 检查文件的函数
    check_img_size,  # 检查图像大小的函数
    check_imshow,  # 检查图像显示的函数
    check_requirements,  # 检查依赖项的函数
    colorstr,  # 颜色字符串处理函数
    cv2,  # OpenCV库
    increment_path,  # 增加路径的函数
    print_args,  # 打印参数的函数
    strip_optimizer,  # 去除优化器的函数
)
from utils.torch_utils import select_device, smart_inference_mode  # 从utils.torch_utils模块导入设备选择和智能推理模式的函数


@smart_inference_mode()  # 使用智能推理模式装饰器
def run(  # 定义run函数
    weights=ROOT / "yolov5s-cls.pt",  # model.pt路径
    source=ROOT / "data/images",  # 文件/目录/URL/屏幕/0（网络摄像头）
    data=ROOT / "data/coco128.yaml",  # 数据集yaml路径
    imgsz=(224, 224),  # 推理图像大小（高度，宽度）
    device="",  # CUDA设备，例如0或0,1,2,3或cpu
    view_img=False,  # 是否显示结果
    save_txt=False,  # 是否将结果保存到*.txt
    nosave=False,  # 是否不保存图像/视频
    augment=False,  # 是否进行增强推理
    visualize=False,  # 是否可视化特征
    update=False,  # 是否更新所有模型
    project=ROOT / "runs/predict-cls",  # 保存结果到项目/名称
    name="exp",  # 保存结果到项目/名称
    exist_ok=False,  # 允许存在的项目/名称，不递增
    half=False,  # 使用FP16半精度推理
    dnn=False,  # 使用OpenCV DNN进行ONNX推理
    vid_stride=1,  # 视频帧率步幅
):
    source = str(source)  # 将源转换为字符串
    save_img = not nosave and not source.endswith(".txt")  # 是否保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 判断源是否为文件
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))  # 判断源是否为URL
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)  # 判断源是否为网络摄像头
    screenshot = source.lower().startswith("screen")  # 判断源是否为屏幕截图
    if is_url and is_file:  # 如果源是URL且是文件
        source = check_file(source)  # 下载文件

    # Directories  # 目录设置
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增加运行目录
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # Load model  # 加载模型
    device = select_device(device)  # 选择设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 初始化DetectMultiBackend模型
    stride, names, pt = model.stride, model.names, model.pt  # 获取模型的步幅、名称和PyTorch标志
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小

    # Dataloader  # 数据加载器设置
    bs = 1  # 批处理大小
    if webcam:  # 如果是网络摄像头
        view_img = check_imshow(warn=True)  # 检查并显示图像
        dataset = LoadStreams(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)  # 加载流数据集
        bs = len(dataset)  # 更新批处理大小
    elif screenshot:  # 如果是屏幕截图
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)  # 加载屏幕截图数据集
    else:  # 否则
        dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)  # 加载图像数据集
    vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化视频路径和写入器

    # Run inference  # 运行推理
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # 预热模型
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))  # 初始化已见数据、窗口和性能分析器

    
    for path, im, im0s, vid_cap, s in dataset:  # 遍历数据集中的每个样本，获取路径、图像、原始图像、视频捕获对象和字符串
        with dt[0]:  # 记录预处理时间
            im = torch.Tensor(im).to(model.device)  # 将图像转换为张量并移动到模型的设备上
            im = im.half() if model.fp16 else im.float()  # uint8 转换为 fp16/32
            if len(im.shape) == 3:  # 如果图像只有三个维度（高度、宽度、通道）
                im = im[None]  # 扩展为批次维度

        # Inference
        with dt[1]:  # 记录推理时间
            results = model(im)  # 使用模型进行推理

        # Post-process
        with dt[2]:  # 记录后处理时间
            pred = F.softmax(results, dim=1)  # 计算概率分布

        # Process predictions
        for i, prob in enumerate(pred):  # 遍历每张图像的预测结果
            seen += 1  # 已处理的图像数量增加
            if webcam:  # 如果使用网络摄像头（batch_size >= 1）
                p, im0, frame = path[i], im0s[i].copy(), dataset.count  # 获取当前图像的路径、原始图像和帧数
                s += f"{i}: "  # 更新字符串以包含当前索引
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)  # 否则获取路径、所有原始图像和帧数

            p = Path(p)  # 将路径转换为 Path 对象
            save_path = str(save_dir / p.name)  # 构建保存图像的路径（im.jpg）
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # 构建保存标签的路径（im.txt）

            s += "%gx%g " % im.shape[2:]  # 更新字符串以包含图像的宽度和高度
            annotator = Annotator(im0, example=str(names), pil=True)  # 创建 Annotator 对象，用于在图像上添加注释

            # Print results
            top5i = prob.argsort(0, descending=True)[:5].tolist()  # 获取前 5 个索引（概率最高的 5 个类别）
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "  # 更新字符串以包含前 5 个类别及其概率

            # Write results
            text = "\n".join(f"{prob[j]:.2f} {names[j]}" for j in top5i)  # 构建输出文本，包含类别及其概率
            if save_img or view_img:  # 如果需要保存图像或查看图像
                annotator.text([32, 32], text, txt_color=(255, 255, 255))  # 在图像上添加文本注释
            if save_txt:  # 如果需要保存文本文件
                with open(f"{txt_path}.txt", "a") as f:  # 以追加模式打开文本文件
                    f.write(text + "\n")  # 写入文本

            # Stream results
            im0 = annotator.result()  # 获取添加注释后的图像
            if view_img:  # 如果需要查看图像
                if platform.system() == "Linux" and p not in windows:  # 如果是 Linux 系统且路径不在窗口列表中
                    windows.append(p)  # 将路径添加到窗口列表
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 创建可调整大小的窗口
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])  # 调整窗口大小
                cv2.imshow(str(p), im0)  # 显示图像
                cv2.waitKey(1)  # 等待 1 毫秒

            # Save results (image with detections)
            if save_img:  # 如果需要保存图像
                if dataset.mode == "image":  # 如果数据集模式是图像
                    cv2.imwrite(save_path, im0)  # 保存图像
                else:  # 如果是 'video' 或 'stream'
                    if vid_path[i] != save_path:  # 如果保存路径与视频路径不同（新视频）
                        vid_path[i] = save_path  # 更新视频路径
                        if isinstance(vid_writer[i], cv2.VideoWriter):  # 如果视频写入器已存在
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 如果是视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
                        else:  # 如果是流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # 设置默认帧率和图像尺寸
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # 强制结果视频使用 *.mp4 后缀
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # 创建视频写入器
                    vid_writer[i].write(im0)  # 写入当前图像到视频

        # Print time (inference-only)
        LOGGER.info(f"{s}{dt[1].dt * 1E3:.1f}ms")  # 记录推理时间

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # 计算每张图像的处理速度
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)  # 记录处理速度信息
    if save_txt or save_img:  # 如果需要保存文本或图像
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""  # 更新字符串以包含保存的标签数量
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")  # 记录结果保存路径
    if update:  # 如果需要更新模型
        strip_optimizer(weights[0])  # 更新模型以修复 SourceChangeWarning


def parse_opt():
    """Parses command line arguments for YOLOv5 inference settings including model, source, device, and image size."""
    # 解析命令行参数，用于YOLOv5推理设置，包括模型、源、设备和图像大小

    parser = argparse.ArgumentParser()  # 创建一个解析器对象
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-cls.pt", help="model path(s)")
    # 添加参数 --weights，接收一个或多个字符串，默认值为 yolov5s-cls.pt 模型路径

    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    # 添加参数 --source，接收一个字符串，默认值为数据图像路径

    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    # 添加参数 --data，接收一个字符串，默认值为 coco128.yaml 数据集路径

    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[224], help="inference size h,w")
    # 添加参数 --imgsz，接收一个或多个整数，默认值为 [224]，表示推理图像的高度和宽度

    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 添加参数 --device，接收一个字符串，表示使用的CUDA设备，默认值为空

    parser.add_argument("--view-img", action="store_true", help="show results")
    # 添加参数 --view-img，若存在则为True，表示显示结果

    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    # 添加参数 --save-txt，若存在则为True，表示将结果保存到 *.txt 文件

    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    # 添加参数 --nosave，若存在则为True，表示不保存图像/视频

    parser.add_argument("--augment", action="store_true", help="augmented inference")
    # 添加参数 --augment，若存在则为True，表示进行增强推理

    parser.add_argument("--visualize", action="store_true", help="visualize features")
    # 添加参数 --visualize，若存在则为True，表示可视化特征

    parser.add_argument("--update", action="store_true", help="update all models")
    # 添加参数 --update，若存在则为True，表示更新所有模型

    parser.add_argument("--project", default=ROOT / "runs/predict-cls", help="save results to project/name")
    # 添加参数 --project，接收一个字符串，默认值为 runs/predict-cls，表示保存结果的项目名称

    parser.add_argument("--name", default="exp", help="save results to project/name")
    # 添加参数 --name，接收一个字符串，默认值为 exp，表示保存结果的名称

    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 添加参数 --exist-ok，若存在则为True，表示允许使用已存在的项目/名称，不进行递增

    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # 添加参数 --half，若存在则为True，表示使用FP16半精度推理

    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    # 添加参数 --dnn，若存在则为True，表示使用OpenCV DNN进行ONNX推理

    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    # 添加参数 --vid-stride，接收一个整数，默认值为1，表示视频帧率步幅

    opt = parser.parse_args()  # 解析命令行参数并返回结果

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 如果图像大小参数只有一个值，则将其扩展为两个相同的值

    print_args(vars(opt))  # 打印解析后的参数
    return opt  # 返回解析后的参数对象


def main(opt):
    """Executes YOLOv5 model inference with options for ONNX DNN and video frame-rate stride adjustments."""
    # 执行YOLOv5模型推理，支持ONNX DNN和视频帧率步幅调整

    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))  # 检查依赖项
    run(**vars(opt))  # 运行推理，传入解析后的参数


if __name__ == "__main__":
    opt = parse_opt()  # 解析命令行参数
    main(opt)  # 调用主函数执行推理