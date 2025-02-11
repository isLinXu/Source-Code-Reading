# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
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
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
# 导入 argparse 模块，用于解析命令行参数。

import os
# 导入 os 模块，用于与操作系统进行交互。

import platform
# 导入 platform 模块，用于获取操作系统信息。

import sys
# 导入 sys 模块，用于访问与 Python 解释器相关的变量和函数。

from pathlib import Path
# 从 pathlib 模块导入 Path 类，用于处理文件路径。

import torch
# 导入 PyTorch 库，用于深度学习模型的构建和训练。

FILE = Path(__file__).resolve()
# 获取当前文件的绝对路径。

ROOT = FILE.parents[1]  # YOLOv5 root directory
# 设置 YOLOv5 的根目录为当前文件的父目录的父目录。

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    # 如果根目录不在系统路径中，则将其添加到系统路径中。

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# 将根目录转换为相对于当前工作目录的路径。

from ultralytics.utils.plotting import Annotator, colors, save_one_box
# 从 ultralytics.utils.plotting 模块导入 Annotator、colors 和 save_one_box，用于绘图和保存框。

from models.common import DetectMultiBackend
# 从 models.common 模块导入 DetectMultiBackend 类，用于加载多种后端模型。

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# 从 utils.dataloaders 模块导入图像和视频格式常量以及加载图像、截图和流的类。

from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    scale_segments,
    strip_optimizer,
)
# 从 utils.general 模块导入各种实用函数和类，包括日志记录器、文件检查、图像大小检查、NMS、路径增量等。

from utils.segment.general import masks2segments, process_mask, process_mask_native
# 从 utils.segment.general 模块导入处理掩码的函数。

from utils.torch_utils import select_device, smart_inference_mode
# 从 utils.torch_utils 模块导入设备选择和智能推理模式的函数。

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
    # 设置模型权重的默认路径。
    
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    # 设置输入源的默认路径，可以是文件、目录、URL、屏幕或摄像头。

    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    # 设置数据集 YAML 文件的默认路径。

    imgsz=(640, 640),  # inference size (height, width)
    # 设置推理图像的默认大小（高度，宽度）。

    conf_thres=0.25,  # confidence threshold
    # 设置置信度阈值的默认值。

    iou_thres=0.45,  # NMS IOU threshold
    # 设置 NMS 的 IoU 阈值的默认值。

    max_det=1000,  # maximum detections per image
    # 设置每张图像的最大检测数量的默认值。

    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    # 设置使用的设备，默认为空。

    view_img=False,  # show results
    # 设置是否显示推理结果的默认值。

    save_txt=False,  # save results to *.txt
    # 设置是否将结果保存到文本文件的默认值。

    save_conf=False,  # save confidences in --save-txt labels
    # 设置是否在保存的标签中保存置信度的默认值。

    save_crop=False,  # save cropped prediction boxes
    # 设置是否保存裁剪的预测框的默认值。

    nosave=False,  # do not save images/videos
    # 设置是否不保存图像/视频的默认值。

    classes=None,  # filter by class: --class 0, or --class 0 2 3
    # 设置按类别过滤的默认值。

    agnostic_nms=False,  # class-agnostic NMS
    # 设置是否使用类无关的 NMS 的默认值。

    augment=False,  # augmented inference
    # 设置是否使用增强推理的默认值。

    visualize=False,  # visualize features
    # 设置是否可视化特征的默认值。

    update=False,  # update all models
    # 设置是否更新所有模型的默认值。

    project=ROOT / "runs/predict-seg",  # save results to project/name
    # 设置结果保存目录的默认路径。

    name="exp",  # save results to project/name
    # 设置结果保存名称的默认值。

    exist_ok=False,  # existing project/name ok, do not increment
    # 设置是否允许存在的项目名称的默认值。

    line_thickness=3,  # bounding box thickness (pixels)
    # 设置边界框的厚度的默认值。

    hide_labels=False,  # hide labels
    # 设置是否隐藏标签的默认值。

    hide_conf=False,  # hide confidences
    # 设置是否隐藏置信度的默认值。

    half=False,  # use FP16 half-precision inference
    # 设置是否使用 FP16 半精度推理的默认值。

    dnn=False,  # use OpenCV DNN for ONNX inference
    # 设置是否使用 OpenCV DNN 进行 ONNX 推理的默认值。

    vid_stride=1,  # video frame-rate stride
    # 设置视频帧率步长的默认值。

    retina_masks=False,
):
    # 定义 run 函数，接收多个参数以进行推理。

    source = str(source)
    # 将输入源转换为字符串格式。

    save_img = not nosave and not source.endswith(".txt")  # save inference images
    # 确定是否保存推理图像，条件为不设置 nosave 且输入源不是文本文件。

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 检查输入源是否为文件，且后缀在支持的图像和视频格式中。

    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    # 检查输入源是否为 URL。

    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    # 检查输入源是否为摄像头。

    screenshot = source.lower().startswith("screen")
    # 检查输入源是否为屏幕截图。

    if is_url and is_file:
        source = check_file(source)  # download
        # 如果输入源是 URL 且是文件，则下载文件。

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 创建保存结果的目录，若存在则递增名称。

    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # 创建保存标签的目录，如果设置了保存文本，则创建 labels 子目录。

    # Load model
    device = select_device(device)
    # 选择计算设备。

    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 加载多后端模型。

    stride, names, pt = model.stride, model.names, model.pt
    # 获取模型的步幅、类别名称和模型类型。

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # 检查图像大小是否符合模型要求。

    # Dataloader
    bs = 1  # batch_size
    # 设置批大小为 1。

    if webcam:
        view_img = check_imshow(warn=True)
        # 如果输入源是摄像头，检查是否可以显示图像，并发出警告。

        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 加载摄像头流，创建数据集。

        bs = len(dataset)
        # 更新批大小为数据集的长度。
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        # 如果输入源是屏幕截图，加载屏幕截图数据集。
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 否则，加载图像数据集。

    vid_path, vid_writer = [None] * bs, [None] * bs
    # 初始化视频路径和视频写入器，长度为批大小。

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    # 对模型进行预热，以便在推理时提高性能。

    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    # 初始化已见图像计数器、窗口列表和性能分析器。

    for path, im, im0s, vid_cap, s in dataset:
        # 遍历数据集，获取每个图像的路径、处理后的图像、原始图像、视频捕获对象和字符串信息。

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            # 将 NumPy 数组转换为 PyTorch 张量，并移动到模型所在的设备。

            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 根据模型是否使用 FP16，将图像转换为半精度或单精度浮点数。

            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 将图像像素值从 0-255 归一化到 0.0-1.0。

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                # 如果图像只有三个维度（高度、宽度、通道），则在第一个维度扩展为批次维度。

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 如果设置了可视化，则创建保存路径。

            pred, proto = model(im, augment=augment, visualize=visualize)[:2]
            # 对图像进行推理，获取预测结果和原始特征图。

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            # 对预测结果应用非极大值抑制（NMS），以去除重叠框。

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # 可选的第二阶段分类器，注释掉了。

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # 遍历每张图像的检测结果。

            seen += 1
            # 更新已见图像计数器。

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                # 如果输入源是摄像头，获取当前图像的路径、原始图像和帧数。

                s += f"{i}: "
                # 更新字符串信息。

            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                # 否则，获取当前图像的路径、原始图像和帧数。

            p = Path(p)  # to Path
            # 将路径转换为 Path 对象。

            save_path = str(save_dir / p.name)  # im.jpg
            # 设置保存图像的路径。

            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            # 设置保存标签的路径。

            s += "%gx%g " % im.shape[2:]  # print string
            # 更新字符串信息，包含图像的宽度和高度。

            imc = im0.copy() if save_crop else im0  # for save_crop
            # 如果设置了保存裁剪，则复制原始图像。

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 创建 Annotator 对象，用于绘制边界框和标签。

            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    # 如果使用视网膜掩码，首先缩放边界框。

                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    # 处理原生掩码。

                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    # 处理掩码。

                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    # 缩放边界框到原始图像大小。

                # Segments
                if save_txt:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))
                    ]
                    # 如果设置了保存文本，则处理并缩放分段结果。

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # 计算每个类别的检测数量。

                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # 更新字符串信息，包含当前类别的检测数量。

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous()
                    / 255
                    if retina_masks
                    else im[i],
                )
                # 绘制掩码。

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # 遍历每个检测结果。

                    if save_txt:  # Write to file
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        # 将分段结果重塑为一维数组。

                        line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                        # 根据是否保存置信度设置标签格式。

                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                            # 将结果写入文本文件。

                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 如果设置了保存图像、裁剪或查看图像，则在图像上添加边界框。

                        c = int(cls)  # integer class
                        # 将类别转换为整数。

                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        # 根据设置决定标签内容。

                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # 在图像上绘制边界框和标签。

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
                            # 如果设置了保存裁剪，则保存裁剪的图像。



            # Stream results
            # im0 = annotator.result()
            # if view_img:
            #     if platform.system() == "Linux" and p not in windows:
            #         windows.append(p)
            #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            #     cv2.imshow(str(p), im0)
            #     if cv2.waitKey(1) == ord("q"):  # 1 millisecond
            #         exit()

            # Stream results
            im0 = annotator.result()
            # 获取绘制了边界框和标签的图像结果。

            if view_img:
                # 如果设置了查看图像，则执行以下操作。
                
                if platform.system() == "Linux" and p not in windows:
                    # 如果操作系统是 Linux 且当前窗口不在窗口列表中，则执行以下操作。
                    
                    windows.append(p)
                    # 将当前窗口路径添加到窗口列表中。
                    
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    # 创建一个可调整大小的窗口。
                    
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    # 根据图像的宽度和高度调整窗口大小。
                
                cv2.imshow(str(p), im0)
                # 显示图像结果。
                
                if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                    exit()
                    # 如果按下 'q' 键，则退出程序。

            # Save results (image with detections)
            if save_img:
                # 如果设置了保存图像，则执行以下操作。
                
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    # 如果数据集模式为图像，则将结果保存为图像文件。
                
                else:  # 'video' or 'stream'
                    # 否则，处理视频或流的情况。
                    
                    if vid_path[i] != save_path:  # new video
                        # 如果当前视频路径与保存路径不同，则执行以下操作。
                        
                        vid_path[i] = save_path
                        # 更新视频路径为保存路径。
                        
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                            # 如果视频写入器存在，则释放上一个视频写入器。
                        
                        if vid_cap:  # video
                            # 如果存在视频捕获对象，则获取视频的帧率和尺寸。
                            
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        else:  # stream
                            # 否则，处理流的情况。
                            
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            # 设置帧率为 30，宽度和高度为图像的宽度和高度。
                        
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        # 将保存路径的后缀强制设置为 .mp4。
                        
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        # 创建视频写入器。
                    
                    vid_writer[i].write(im0)
                    # 将当前图像写入视频文件。


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        # 打印推理时间信息，包含检测结果的数量。

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    # 计算每张图像的处理速度。

    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    # 打印每个阶段的处理速度信息。

    if save_txt or save_img:
        # 如果设置了保存文本或图像，则执行以下操作。
        
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        # 如果保存文本，则获取保存的标签数量。
        
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # 打印结果保存的路径信息。

    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        # 如果需要更新模型，则调用 strip_optimizer 函数更新模型。


def parse_opt():
    """Parses command-line options for YOLOv5 inference including model paths, data sources, inference settings, and
    output preferences.
    """
    # 解析 YOLOv5 推理的命令行选项，包括模型路径、数据源、推理设置和输出偏好。
    
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象。

    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-seg.pt", help="model path(s)")
    # 添加权重文件路径参数，支持多个权重文件，默认为 yolov5s-seg.pt。

    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    # 添加输入源参数，默认为图像数据目录。

    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    # 添加数据集 YAML 文件路径参数，默认为 coco128.yaml。

    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    # 添加图像大小参数，默认为 640。

    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    # 添加置信度阈值参数，默认为 0.25。

    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    # 添加 NMS 的 IoU 阈值参数，默认为 0.45。

    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    # 添加每张图像最大检测数参数，默认为 1000。

    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 添加设备参数，指定使用的 CUDA 设备或 CPU。

    parser.add_argument("--view-img", action="store_true", help="show results")
    # 添加查看图像参数，若设置则显示推理结果。

    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    # 添加保存结果到文本文件的参数，若设置则保存结果。

    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    # 添加保存置信度到文本文件的参数，若设置则在保存的标签中包含置信度。

    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    # 添加保存裁剪预测框的参数，若设置则保存裁剪的图像。

    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    # 添加不保存图像/视频的参数，若设置则不保存。

    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    # 添加按类别过滤的参数，支持多个类别。

    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    # 添加类无关的 NMS 参数，若设置则使用类无关的 NMS。

    parser.add_argument("--augment", action="store_true", help="augmented inference")
    # 添加增强推理参数，若设置则使用增强推理。

    parser.add_argument("--visualize", action="store_true", help="visualize features")
    # 添加可视化特征参数，若设置则可视化特征。

    parser.add_argument("--update", action="store_true", help="update all models")
    # 添加更新所有模型的参数，若设置则更新模型。

    parser.add_argument("--project", default=ROOT / "runs/predict-seg", help="save results to project/name")
    # 添加项目保存路径参数，默认为 runs/predict-seg。

    parser.add_argument("--name", default="exp", help="save results to project/name")
    # 添加项目名称参数，默认为 "exp"。

    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 添加允许存在的项目名称参数，若设置则不递增项目名称。

    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    # 添加边界框厚度参数，默认为 3 像素。

    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    # 添加隐藏标签参数，若设置则隐藏标签。

    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    # 添加隐藏置信度参数，若设置则隐藏置信度。

    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # 添加使用 FP16 半精度推理的参数，若设置则启用半精度。

    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    # 添加使用 OpenCV DNN 进行 ONNX 推理的参数，若设置则启用 DNN。

    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    # 添加视频帧率步长参数，默认为 1。

    parser.add_argument("--retina-masks", action="store_true", help="whether to plot masks in native resolution")
    # 添加是否以原生分辨率绘制掩码的参数，若设置则启用。

    opt = parser.parse_args()
    # 解析命令行参数并将结果存储在 opt 中。

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 如果图像大小参数的长度为 1，则将其扩展为两倍。

    print_args(vars(opt))
    # 打印解析后的参数。

    return opt
    # 返回解析后的参数对象。

def main(opt):
    """Executes YOLOv5 model inference with given options, checking for requirements before launching."""
    # 执行 YOLOv5 模型推理，使用给定选项，并在启动前检查所需的依赖项。
    
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    # 检查所需的 Python 包是否已安装，排除 tensorboard 和 thop。

    run(**vars(opt))
    # 调用 run 函数执行推理，传入所有参数。

if __name__ == "__main__":
    opt = parse_opt()
    # 如果该脚本是主程序，则解析命令行参数。

    main(opt)
    # 调用 main 函数执行任务。