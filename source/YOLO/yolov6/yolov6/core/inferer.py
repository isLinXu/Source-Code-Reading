# 导入必要的库
import os
import cv2  # OpenCV库，用于图像处理
import time
import math
import torch  # PyTorch深度学习框架
import numpy as np
import os.path as osp

from tqdm import tqdm  # 进度条显示
from pathlib import Path  # 路径处理
from PIL import ImageFont  # 图像字体处理
from collections import deque  # 双端队列

# 导入YOLOv6自定义模块
from yolov6.utils.events import LOGGER, load_yaml  # 日志和配置文件加载
from yolov6.layers.common import DetectBackend  # 检测后端
from yolov6.data.data_augment import letterbox  # 图像预处理
from yolov6.data.datasets import LoadData  # 数据加载
from yolov6.utils.nms import non_max_suppression  # 非极大值抑制
from yolov6.utils.torch_utils import get_model_info  # 模型信息获取

class Inferer:
    '''YOLOv6推理器类，用于模型推理和预测'''
    def __init__(self, source, webcam, webcam_addr, weights, device, yaml, img_size, half):
        '''
        初始化推理器
        Args:
            source: 输入源（图像/视频路径或摄像头）
            webcam: 是否使用摄像头
            webcam_addr: 摄像头地址
            weights: 模型权重文件路径
            device: 运行设备（CPU/GPU）
            yaml: 配置文件路径
            img_size: 输入图像尺寸
            half: 是否使用半精度推理
        '''
        # 更新实例属性
        self.__dict__.update(locals())

        # 初始化模型 | Init model
        self.device = device
        self.img_size = img_size
        # 检查CUDA是否可用
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        # 设置设备
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        # 加载检测模型
        self.model = DetectBackend(weights, device=self.device)
        # 获取模型步长
        self.stride = self.model.stride
        # 加载类别名称
        self.class_names = load_yaml(yaml)['names']
        # 检查并调整图像尺寸
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size | 检查图像尺寸
        self.half = half

        # Switch model to deploy status | 切换模型到部署状态
        self.model_switch(self.model.model, self.img_size)

        # Half precision | 半精度设置
        if self.half & (self.device.type != 'cpu'):
            # GPU下使用半精度
            self.model.model.half()
        else:
            # CPU下使用全精度
            self.model.model.float()
            self.half = False

        # 模型预热 | Model warmup
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        # 加载数据 | Load data
        self.webcam = webcam  # 是否使用摄像头
        self.webcam_addr = webcam_addr  # 摄像头地址
        self.files = LoadData(source, webcam, webcam_addr)  # 加载数据源
        self.source = source  # 输入源路径

    def model_switch(self, model, img_size):
        ''' Model switch to deploy status | 将模型切换到部署状态 '''
        from yolov6.layers.common import RepVGGBlock
        # 遍历模型的所有层
        for layer in model.modules():
            # 对RepVGG块进行部署模式转换
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            # 处理上采样层的兼容性问题
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility | 兼容torch 1.11.0版本

        LOGGER.info("Switch model to deploy modality.")

    def infer(self, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf, view_img=True):
        ''' Model Inference and results visualization | 模型推理和结果可视化
        Args:
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            classes: 需要检测的类别
            agnostic_nms: 是否使用类别无关的NMS
            max_det: 最大检测目标数
            save_dir: 保存目录
            save_txt: 是否保存txt结果
            save_img: 是否保存图像结果
            hide_labels: 是否隐藏标签
            hide_conf: 是否隐藏置信度
            view_img: 是否显示图像
        '''
        # 初始化视频路径、写入器和窗口列表
        vid_path, vid_writer, windows = None, None, []
        # 初始化FPS计算器
        fps_calculator = CalcFPS()
        
        # 遍历所有输入文件（图像/视频）
        for img_src, img_path, vid_cap in tqdm(self.files):
            # 处理输入图像：调整大小和预处理
            img, img_src = self.process_image(img_src, self.img_size, self.stride, self.half)
            # 将图像转移到指定设备（CPU/GPU）
            img = img.to(self.device)
            # 扩展维度以适应批处理
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim | 扩展批处理维度
                
            # 记录推理开始时间
            t1 = time.time()
            # 模型推理
            pred_results = self.model(img)
            # 执行非极大值抑制
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            # 记录推理结束时间
            t2 = time.time()

            # 设置保存路径
            if self.webcam:
                # 对于网络摄像头输入，直接使用摄像头地址作为保存路径
                save_path = osp.join(save_dir, self.webcam_addr)
                txt_path = osp.join(save_dir, self.webcam_addr)
            else:
                # Create output files in nested dirs that mirrors the structure of the images' dirs
                # 创建与输入图像目录结构相匹配的嵌套输出目录
                rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(self.source))
                save_path = osp.join(save_dir, rel_path, osp.basename(img_path))  # im.jpg
                txt_path = osp.join(save_dir, rel_path, 'labels', osp.splitext(osp.basename(img_path))[0])
                os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)

            # 计算归一化增益
            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh | 归一化增益 [宽,高,宽,高]
            # 复制原始图像用于绘制
            img_ori = img_src.copy()

            # check image and font | 检查图像和字体
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            self.font_check()

            # 如果有检测到的目标
            if len(det):
                # 将检测框坐标缩放到原始图像尺寸
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                # 遍历每个检测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file | 将结果写入文件
                        # 转换边界框格式并归一化
                        xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)
                        # 写入txt文件
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # 在图像上绘制检测结果
                        class_num = int(cls)  # integer class | 类别索引
                        # 生成标签文本
                        label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                        # 绘制边界框和标签
                        self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=self.generate_colors(class_num, True))

                # 将处理后的图像转换回numpy数组
                img_src = np.asarray(img_ori)

            # FPS counter | FPS计数器
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()

            # 在视频上绘制FPS信息
            if self.files.type == 'video':
                self.draw_text(
                    img_src,
                    f"FPS: {avg_fps:0.1f}",
                    pos=(20, 20),
                    font_scale=1.0,
                    text_color=(204, 85, 17),
                    text_color_bg=(255, 255, 255),
                    font_thickness=2,
                )

            # 实时显示检测结果
            if view_img:
                if img_path not in windows:
                    windows.append(img_path)
                    # 创建可调整大小的窗口
                    cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
                # 显示图像
                cv2.imshow(str(img_path), img_src)
                cv2.waitKey(1)  # 1 millisecond | 等待1毫秒

            # Save results (image with detections) | 保存检测结果
            if save_img:
                if self.files.type == 'image':
                    # 保存图像结果
                    cv2.imwrite(save_path, img_src)
                else:  # 'video' or 'stream' | 视频或流媒体
                    if vid_path != save_path:  # new video | 新视频
                        vid_path = save_path
                        # 释放之前的视频写入器
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer | 释放之前的视频写入器
                        if vid_cap:  # video | 视频
                            # 获取视频属性
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream | 流媒体
                            # 设置默认的视频属性
                            fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                        # 强制使用mp4格式保存
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        # 创建视频写入器
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 写入当前帧
                    vid_writer.write(img_src)

    @staticmethod
    def process_image(img_src, img_size, stride, half):
        '''Process image before image inference. | 在图像推理之前对图像进行预处理
        Args:
            img_src: 输入源图像
            img_size: 目标图像尺寸
            stride: 模型步长
            half: 是否使用半精度
        '''
        # 使用letterbox进行图像缩放，保持原始图像比例
        image = letterbox(img_src, img_size, stride=stride)[0]
        
        # Convert | 转换图像格式和颜色空间
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB | 维度转换：HWC到CHW，颜色空间：BGR到RGB
        # 确保数组内存连续，提高运行效率
        image = torch.from_numpy(np.ascontiguousarray(image))
        # 根据half参数决定是否使用半精度
        image = image.half() if half else image.float()  # uint8 to fp16/32 | uint8转换为fp16或fp32
        # 归一化像素值到0-1范围
        image /= 255  # 0 - 255 to 0.0 - 1.0 | 将像素值从0-255归一化到0.0-1.0

        return image, img_src

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image. | 
        确保图像尺寸在每个维度上都是步长s的倍数，并返回新的图像形状列表
        Args:
            img_size: 输入图像尺寸，可以是整数或列表
            s: 步长，默认32
            floor: 最小尺寸限制，默认0
        """
        # 处理整数类型的输入尺寸
        if isinstance(img_size, int):  # integer i.e. img_size=640 | 整数类型，例如img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        # 处理列表类型的输入尺寸
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480] | 列表类型，例如img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        # 如果尺寸发生变化，输出警告信息
        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        # 返回新的尺寸，如果输入是整数则返回相同的宽高
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor. | 向上修正值x，使其能被除数整除
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):
        """在图像上绘制文本
        Args:
            img: 输入图像
            text: 要绘制的文本
            font: 字体类型，默认使用FONT_HERSHEY_SIMPLEX
            pos: 文本位置，默认(0,0)
            font_scale: 字体缩放比例，默认1
            font_thickness: 字体粗细，默认2
            text_color: 文本颜色，默认绿色(0,255,0)
            text_color_bg: 背景颜色，默认黑色(0,0,0)
        """
        # 设置文本边界偏移量
        offset = (5, 5)
        x, y = pos
        # 获取文本大小
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)[0]  # text width, height | 文本宽度和高度
        text_w, text_h = text_size
        # 计算背景矩形的起始和结束坐标
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        # 绘制背景矩形
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        # 绘制文本
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),  # 文本位置
            font,  # 字体类型
            font_scale,  # 字体缩放
            text_color,  # 文本颜色
            font_thickness,  # 字体粗细
            cv2.LINE_AA,  # 抗锯齿
        )

        return text_size

    def font_check(self, font='./yolov6/utils/Arial.ttf', size=10):
        '''检查并加载字体
        Args:
            font: 字体文件路径，默认为Arial.ttf
            size: 字体大小，默认为10
        Returns:
            PIL TrueType Font对象
        '''
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary | 返回PIL TrueType字体对象，如果需要则下载到配置目录
        # 确保字体文件存在
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            # 尝试加载字体文件
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing | 如果字体缺失则尝试下载
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
        '''在图像上绘制边界框和标签
        Args:
            image: 输入图像
            lw: 线条宽度
            box: 边界框坐标 [x1,y1,x2,y2]
            label: 标签文本，默认为空
            color: 边界框颜色，默认为灰色
            txt_color: 文本颜色，默认为白色
            font: 字体类型，默认为FONT_HERSHEY_COMPLEX
        '''
        # Add one xyxy box to image with label | 在图像上添加一个带标签的边界框
        # 转换边界框坐标为整数类型
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # 绘制边界框
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        
        # 如果有标签文本，则绘制标签
        if label:
            tf = max(lw - 1, 1)  # font thickness | 字体粗细
            # 获取文本尺寸
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height | 文本宽度和高度
            # 判断标签是否适合放在边界框外部
            outside = p1[1] - h - 3 >= 0  # label fits outside box | 判断标签是否能放在框的上方
            # 计算文本背景框的位置
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # 绘制文本背景框
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled | 填充背景色
            # 绘制文本
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)  # 使用抗锯齿线条绘制文本

    @staticmethod
    def box_convert(x):
        '''将边界框坐标从[x1, y1, x2, y2]格式转换为[x, y, w, h]格式
        Args:
            x: 输入边界框坐标，形状为[n, 4]
               其中x1y1为左上角坐标，x2y2为右下角坐标
        Returns:
            转换后的边界框坐标[x中心点, y中心点, 宽度, 高度]
        '''
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        # 根据输入类型选择适当的复制方法
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center | 计算中心点x坐标
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center | 计算中心点y坐标
        y[:, 2] = x[:, 2] - x[:, 0]  # width | 计算宽度
        y[:, 3] = x[:, 3] - x[:, 1]  # height | 计算高度
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        '''生成用于可视化的颜色列表
        Args:
            i: 颜色索引
            bgr: 是否返回BGR格式的颜色，默认为RGB格式
        Returns:
            tuple: 颜色元组 (R,G,B) 或 (B,G,R)
        '''
        # 预定义的16进制颜色值
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        # 存储转换后的RGB颜色值
        palette = []
        # 将16进制颜色转换为RGB元组
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        # 计算颜色数量
        num = len(palette)
        # 使用取模运算循环选择颜色
        color = palette[int(i) % num]
        # 根据需要返回BGR或RGB格式
        return (color[2], color[1], color[0]) if bgr else color

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape | 将输出框的坐标缩放回原始图像尺寸
        Args:
            ori_shape: 原始图像形状
            boxes: 检测框坐标
            target_shape: 目标图像形状
        '''
        # 计算缩放比例，取宽高中的最小值确保完整包含图像
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        # 计算填充像素数，确保图像居中
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        # 减去填充，还原到原始坐标系
        boxes[:, [0, 2]] -= padding[0]  # 减去x方向的填充
        boxes[:, [1, 3]] -= padding[1]  # 减去y方向的填充
        # 根据比例还原坐标值
        boxes[:, :4] /= ratio

        # 将坐标值限制在有效范围内
        boxes[:, 0].clamp_(0, target_shape[1])  # x1 | 限制x1的范围
        boxes[:, 1].clamp_(0, target_shape[0])  # y1 | 限制y1的范围
        boxes[:, 2].clamp_(0, target_shape[1])  # x2 | 限制x2的范围
        boxes[:, 3].clamp_(0, target_shape[0])  # y2 | 限制y2的范围

        return boxes

class CalcFPS:
    '''FPS计算器类，用于计算和平均帧率'''
    def __init__(self, nsamples: int = 50):
        '''初始化FPS计算器
        Args:
            nsamples: 用于计算平均值的样本数量，默认50帧
        '''
        # 使用双端队列存储帧率数据，限制最大长度为nsamples
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        '''更新帧率数据
        Args:
            duration: 每帧处理时间（秒）
        '''
        # 将新的帧率添加到队列中
        self.framerate.append(duration)

    def accumulate(self):
        '''计算平均帧率
        Returns:
            float: 平均帧率，如果样本数小于2则返回0
        '''
        # 当有足够的样本时计算平均值
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0