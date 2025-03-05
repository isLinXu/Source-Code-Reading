# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolo11n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo mode=predict model=yolo11n.pt                 # PyTorch
                              yolo11n.torchscript        # TorchScript
                              yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolo11n_openvino_model     # OpenVINO
                              yolo11n.engine             # TensorRT
                              yolo11n.mlpackage          # CoreML (macOS-only)
                              yolo11n_saved_model        # TensorFlow SavedModel
                              yolo11n.pb                 # TensorFlow GraphDef
                              yolo11n.tflite             # TensorFlow Lite
                              yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolo11n_paddle_model       # PaddlePaddle
                              yolo11n.mnn                # MNN
                              yolo11n_ncnn_model         # NCNN
                              yolo11n_imx_model          # Sony IMX
                              yolo11n_rknn_model         # Rockchip RKNN
"""

import platform  # 导入平台模块
import re  # 导入正则表达式模块
import threading  # 导入线程模块
from pathlib import Path  # 从路径模块导入Path类

import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库

from ultralytics.cfg import get_cfg, get_save_dir  # 从ultralytics.cfg导入get_cfg和get_save_dir函数
from ultralytics.data import load_inference_source  # 从ultralytics.data导入load_inference_source函数
from ultralytics.data.augment import LetterBox, classify_transforms  # 从ultralytics.data.augment导入LetterBox和classify_transforms
from ultralytics.nn.autobackend import AutoBackend  # 从ultralytics.nn.autobackend导入AutoBackend类
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops  # 从ultralytics.utils导入多个工具
from ultralytics.utils.checks import check_imgsz, check_imshow  # 从ultralytics.utils.checks导入check_imgsz和check_imshow函数
from ultralytics.utils.files import increment_path  # 从ultralytics.utils.files导入increment_path函数
from ultralytics.utils.torch_utils import select_device, smart_inference_mode  # 从ultralytics.utils.torch_utils导入select_device和smart_inference_mode函数

STREAM_WARNING = """
WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""  # 定义一个警告信息，提醒用户在不使用stream=True时，推理结果会累积在内存中

class BasePredictor:
    """
    BasePredictor.

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.  # args属性，存储预测器的配置
        save_dir (Path): Directory to save results.  # save_dir属性，存储结果的目录
        done_warmup (bool): Whether the predictor has finished setup.  # done_warmup属性，指示预测器是否完成初始化
        model (nn.Module): Model used for prediction.  # model属性，存储用于预测的模型
        data (dict): Data configuration.  # data属性，存储数据配置
        device (torch.device): Device used for prediction.  # device属性，存储用于预测的设备
        dataset (Dataset): Dataset used for prediction.  # dataset属性，存储用于预测的数据集
        vid_writer (dict): Dictionary of {save_path: video_writer, ...} writer for saving video output.  # vid_writer属性，存储视频输出的写入器
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.  # cfg参数，配置文件的路径，默认为DEFAULT_CFG
            overrides (dict, optional): Configuration overrides. Defaults to None.  # overrides参数，配置覆盖，默认为None
        """
        self.args = get_cfg(cfg, overrides)  # 获取配置并赋值给args
        self.save_dir = get_save_dir(self.args)  # 获取保存目录并赋值给save_dir
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25  # 如果conf为None，设置默认置信度为0.25
        self.done_warmup = False  # 初始化done_warmup为False
        if self.args.show:
            self.args.show = check_imshow(warn=True)  # 如果show为True，检查是否可以显示图像

        # Usable if setup is done
        self.model = None  # 初始化模型为None
        self.data = self.args.data  # data_dict  # 将数据配置赋值给data
        self.imgsz = None  # 初始化图像大小为None
        self.device = None  # 初始化设备为None
        self.dataset = None  # 初始化数据集为None
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}  # 初始化视频写入器字典
        self.plotted_img = None  # 初始化绘制的图像为None
        self.source_type = None  # 初始化源类型为None
        self.seen = 0  # 初始化已处理的图像数量为0
        self.windows = []  # 初始化窗口列表
        self.batch = None  # 初始化批次为None
        self.results = None  # 初始化结果为None
        self.transforms = None  # 初始化变换为None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 获取回调函数
        self.txt_path = None  # 初始化文本路径为None
        self._lock = threading.Lock()  # for automatic thread-safe inference  # 创建线程锁以实现线程安全的推理
        callbacks.add_integration_callbacks(self)  # 添加集成回调函数

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.  # 准备输入图像，im可以是张量或图像列表
        """
        not_tensor = not isinstance(im, torch.Tensor)  # 检查im是否为张量
        if not_tensor:
            im = np.stack(self.pre_transform(im))  # 如果不是张量，进行预处理并堆叠图像
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)  # 转换图像格式
            im = np.ascontiguousarray(im)  # contiguous  # 确保数组是连续的
            im = torch.from_numpy(im)  # 将NumPy数组转换为张量

        im = im.to(self.device)  # 将图像移动到指定设备
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32  # 根据模型设置转换图像类型
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0  # 将像素值归一化到[0, 1]范围
        return im  # 返回预处理后的图像

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""  # 使用指定模型和参数对给定图像进行推理
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)  # 生成可视化路径
            if self.args.visualize and (not self.source_type.tensor)  # 如果需要可视化且源类型不是张量
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)  # 执行推理并返回结果

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.  # 在推理前对输入图像进行预变换
        """
        same_shapes = len({x.shape for x in im}) == 1  # 检查所有图像是否具有相同的形状
        letterbox = LetterBox(  # 创建LetterBox实例
            self.imgsz,
            auto=same_shapes and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),  # 自动调整
            stride=self.model.stride,  # 设置步幅
        )
        return [letterbox(image=x) for x in im]  # 对每个图像应用LetterBox变换并返回

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""  # 对图像的预测结果进行后处理并返回
        return preds  # 返回预测结果

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""  # 对图像或流执行推理
        self.stream = stream  # 设置流模式
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)  # 如果是流模式，调用流推理
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one  # 否则将结果合并为一个列表

    def predict_cli(self, source=None, model=None):
        """
        Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes
        the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Note:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """  # 用于命令行接口预测的方法，确保不会在内存中累积输出
        gen = self.stream_inference(source, model)  # 获取流推理生成器
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass  # 消耗生成器以避免内存问题

    def setup_source(self, source):
        """Sets up source and inference mode."""  # 设置源和推理模式
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size  # 检查图像大小
        self.transforms = (
            getattr(  # 获取模型的变换
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),  # 使用classify_transforms作为默认变换
            )
            if self.args.task == "classify"  # 如果任务是分类
            else None
        )
        self.dataset = load_inference_source(  # 加载推理源
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type  # 获取源类型
        if not getattr(self, "stream", True) and (  # 如果不是流模式且源类型是流或截图
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))  # videos
        ):  # videos
            LOGGER.warning(STREAM_WARNING)  # 记录警告信息
        self.vid_writer = {}  # 初始化视频写入器字典

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""  # 在摄像头视频流上进行实时推理并将结果保存到文件
        if self.args.verbose:
            LOGGER.info("")  # 如果verbose为True，记录信息

        # Setup model
        if not self.model:  # 如果模型未设置
            self.setup_model(model)  # 设置模型

        with self._lock:  # for thread-safe inference  # 使用线程锁以实现线程安全的推理
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)  # 每次调用预测时设置源

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:  # 如果需要保存结果或文本
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # 创建保存目录

            # Warmup model
            if not self.done_warmup:  # 如果模型尚未预热
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))  # 预热模型
                self.done_warmup = True  # 设置为已预热

            self.seen, self.windows, self.batch = 0, [], None  # 初始化已处理的图像数量、窗口和批次
            profilers = (  # 创建性能分析器
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")  # 运行开始预测的回调
            for self.batch in self.dataset:  # 遍历数据集中的每个批次
                self.run_callbacks("on_predict_batch_start")  # 运行开始批次预测的回调
                paths, im0s, s = self.batch  # 获取路径、图像和状态

                # Preprocess
                with profilers[0]:  # 性能分析预处理
                    im = self.preprocess(im0s)  # 预处理图像

                # Inference
                with profilers[1]:  # 性能分析推理
                    preds = self.inference(im, *args, **kwargs)  # 执行推理
                    if self.args.embed:  # 如果需要嵌入
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue  # 继续下一个批次

                # Postprocess
                with profilers[2]:  # 性能分析后处理
                    self.results = self.postprocess(preds, im, im0s)  # 后处理结果
                self.run_callbacks("on_predict_postprocess_end")  # 运行后处理结束的回调

                # Visualize, save, write results
                n = len(im0s)  # 获取当前批次的图像数量
                for i in range(n):  # 遍历每个图像
                    self.seen += 1  # 增加已处理的图像数量
                    self.results[i].speed = {  # 记录处理速度
                        "preprocess": profilers[0].dt * 1e3 / n,  # 预处理速度
                        "inference": profilers[1].dt * 1e3 / n,  # 推理速度
                        "postprocess": profilers[2].dt * 1e3 / n,  # 后处理速度
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:  # 如果需要详细输出或保存结果
                        s[i] += self.write_results(i, Path(paths[i]), im, s)  # 写入结果

                # Print batch results
                if self.args.verbose:  # 如果verbose为True
                    LOGGER.info("\n".join(s))  # 记录当前批次的结果

                self.run_callbacks("on_predict_batch_end")  # 运行批次预测结束的回调
                yield from self.results  # 返回结果

        # Release assets
        for v in self.vid_writer.values():  # 释放视频写入器
            if isinstance(v, cv2.VideoWriter):  # 如果是视频写入器
                v.release()  # 释放资源

        # Print final results
        if self.args.verbose and self.seen:  # 如果verbose为True且已处理的图像数量大于0
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image  # 计算每张图像的处理速度
            LOGGER.info(  # 记录处理速度信息
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:  # 如果需要保存结果或文本
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels  # 获取标签数量
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""  # 保存文本信息
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")  # 记录保存结果的信息
        self.run_callbacks("on_predict_end")  # 运行预测结束的回调

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""  # 初始化YOLO模型并设置为评估模式
        self.model = AutoBackend(  # 创建AutoBackend实例
            weights=model or self.args.model,  # 权重
            device=select_device(self.args.device, verbose=verbose),  # 选择设备
            dnn=self.args.dnn,  # DNN设置
            data=self.args.data,  # 数据设置
            fp16=self.args.half,  # 半精度设置
            batch=self.args.batch,  # 批次大小
            fuse=True,  # 融合设置
            verbose=verbose,  # 详细输出设置
        )

        self.device = self.model.device  # update device  # 更新设备
        self.args.half = self.model.fp16  # update half  # 更新半精度设置
        self.model.eval()  # 设置模型为评估模式

    def write_results(self, i, p, im, s):
        """Write inference results to a file or directory."""  # 将推理结果写入文件或目录
        string = ""  # print string  # 初始化输出字符串
        if len(im.shape) == 3:  # 如果图像是3维
            im = im[None]  # expand for batch dim  # 扩展为批次维度
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1  # 如果源类型是流、图像或张量
            string += f"{i}: "  # 添加索引
            frame = self.dataset.count  # 获取当前帧数
        else:
            match = re.search(r"frame (\d+)/", s[i])  # 从状态中提取帧信息
            frame = int(match[1]) if match else None  # 0 if frame undetermined  # 如果未确定帧，则为0

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))  # 设置文本路径
        string += "{:g}x{:g} ".format(*im.shape[2:])  # 添加图像尺寸信息
        result = self.results[i]  # 获取当前结果
        result.save_dir = self.save_dir.__str__()  # used in other locations  # 设置结果保存目录
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"  # 添加详细信息和推理速度

        # Add predictions to image
        if self.args.save or self.args.show:  # 如果需要保存结果或显示图像
            self.plotted_img = result.plot(  # 绘制结果
                line_width=self.args.line_width,  # 线宽
                boxes=self.args.show_boxes,  # 是否显示边框
                conf=self.args.show_conf,  # 是否显示置信度
                labels=self.args.show_labels,  # 是否显示标签
                im_gpu=None if self.args.retina_masks else im[i],  # 如果不使用视网膜掩码，则使用当前图像
            )

        # Save results
        if self.args.save_txt:  # 如果需要保存文本
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)  # 保存文本结果
        if self.args.save_crop:  # 如果需要保存裁剪结果
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)  # 保存裁剪结果
        if self.args.show:  # 如果需要显示图像
            self.show(str(p))  # 显示图像
        if self.args.save:  # 如果需要保存图像
            self.save_predicted_images(str(self.save_dir / p.name), frame)  # 保存预测图像

        return string  # 返回输出字符串

    def save_predicted_images(self, save_path="", frame=0):
        """Save video predictions as mp4 at specified path."""  # 将视频预测结果保存为MP4格式
        im = self.plotted_img  # 获取绘制的图像

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:  # 如果数据集模式是流或视频
            fps = self.dataset.fps if self.dataset.mode == "video" else 30  # 获取帧率
            frames_path = f"{save_path.split('.', 1)[0]}_frames/"  # 设置帧保存路径
            if save_path not in self.vid_writer:  # new video  # 如果是新视频
                if self.args.save_frames:  # 如果需要保存帧
                    Path(frames_path).mkdir(parents=True, exist_ok=True)  # 创建帧保存目录
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")  # 设置文件后缀和编码格式
                self.vid_writer[save_path] = cv2.VideoWriter(  # 创建视频写入器
                    filename=str(Path(save_path).with_suffix(suffix)),  # 设置文件名
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),  # 设置编码格式
                    fps=fps,  # integer required, floats produce error in MP4 codec  # 设置帧率
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)  # 设置帧大小
                )

            # Save video
            self.vid_writer[save_path].write(im)  # 写入视频帧
            if self.args.save_frames:  # 如果需要保存帧
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)  # 保存当前帧为JPEG格式

        # Save images
        else:  # 如果不是视频模式
            cv2.imwrite(str(Path(save_path).with_suffix(".jpg")), im)  # save to JPG for best support  # 保存为JPEG格式以获得最佳支持

    def show(self, p=""):
        """Display an image in a window using the OpenCV imshow function."""  # 使用OpenCV的imshow函数在窗口中显示图像
        im = self.plotted_img  # 获取绘制的图像
        if platform.system() == "Linux" and p not in self.windows:  # 如果是Linux系统且窗口未打开
            self.windows.append(p)  # 添加窗口到列表
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)  # 创建可调整大小的窗口
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (width, height)  # 调整窗口大小
        cv2.imshow(p, im)  # 显示图像
        cv2.waitKey(300 if self.dataset.mode == "image" else 1)  # 1 millisecond  # 等待指定时间以显示图像

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""  # 运行特定事件的所有注册回调
        for callback in self.callbacks.get(event, []):  # 遍历回调函数
            callback(self)  # 执行回调

    def add_callback(self, event: str, func):
        """Add callback."""  # 添加回调
        self.callbacks[event].append(func)  # 将回调函数添加到指定事件的回调列表中