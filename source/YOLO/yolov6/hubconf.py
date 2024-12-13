# 导入必要的库
import os          # 操作系统接口，用于文件和路径操作
import cv2         # OpenCV库，用于图像处理
import math        # 数学运算库
import pathlib     # 路径处理库，提供面向对象的文件系统路径操作
import torch       # PyTorch深度学习框架
import numpy as np # 数值计算库
from PIL import Image          # Python图像处理库
import matplotlib.pyplot as plt # 数据可视化库

# 导入YOLOv6自定义模块
from yolov6.layers.common import DetectBackend     # 检测后端模块
from yolov6.utils.nms import non_max_suppression  # 非极大值抑制
from yolov6.data.data_augment import letterbox   # 图像预处理：信箱填充
from yolov6.core.inferer import Inferer          # 推理器
from yolov6.utils.events import LOGGER           # 日志工具
from yolov6.utils.events import load_yaml        # YAML配置文件加载器

# 全局常量定义
PATH_YOLOv6 = pathlib.Path(__file__).parent     # 获取当前文件所在目录作为YOLOv6根目录
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择计算设备（GPU优先）
CLASS_NAMES = load_yaml(str(PATH_YOLOv6/"data/coco.yaml"))['names']   # 加载COCO数据集的类别名称

def visualize_detections(image,
                        boxes,      # 检测框坐标 [x1, y1, x2, y2]
                        classes,    # 检测到的类别名称
                        scores,     # 检测置信度分数
                        min_score=0.4,    # 置信度阈值
                        figsize=(16, 16), # 显示图像大小
                        linewidth=2,      # 边框线宽
                        color='lawngreen' # 边框颜色
                        ):
    """
    可视化目标检测结果的函数
    """
    # 将输入图像转换为numpy数组，并确保类型为uint8
    image = np.array(image, dtype=np.uint8)
    
    # 创建一个新的图形窗口，设置大小
    fig = plt.figure(figsize=figsize)
    
    # 关闭坐标轴显示
    plt.axis("off")
    
    # 显示原始图像
    plt.imshow(image)
    
    # 获取当前坐标轴对象
    ax = plt.gca()
    
    # 遍历所有检测结果
    for box, name, score in zip(boxes, classes, scores):
        # 只显示置信度大于阈值的检测结果
        if score >= min_score:
            # 格式化显示文本：类别名称和置信度分数
            text = "{}: {:.2f}".format(name, score)
            
            # 获取边界框坐标
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1  # 计算框的宽度和高度
            
            # 创建一个矩形patch（检测框）
            patch = plt.Rectangle(
                [x1, y1],        # 左上角坐标
                w, h,            # 宽度和高度
                fill=False,      # 不填充
                edgecolor=color, # 边框颜色
                linewidth=linewidth  # 线宽
            )
            
            # 将矩形框添加到图像上
            ax.add_patch(patch)
            
            # 添加类别和置信度文本标签
            ax.text(
                x1, y1,          # 文本位置
                text,            # 显示的文本
                bbox={"facecolor": color, "alpha": 0.8},  # 文本框样式
                clip_box=ax.clipbox,  # 裁剪区域
                clip_on=True,         # 启用裁剪
            )
    
    # 显示图像
    plt.show()


def check_img_size(img_size, s=32, floor=0):
    """
    检查和调整输入图像尺寸，确保其是步长(stride)的整数倍
    参数:
        img_size: 输入图像尺寸（可以是整数或列表）
        s: 步长，默认32
        floor: 最小尺寸限制，默认0
    """
    # 内部辅助函数：将数值调整为除数的整数倍
    def make_divisible(x, divisor):
        return math.ceil(x / divisor) * divisor

    # 处理单个整数尺寸的情况（如img_size=640）
    if isinstance(img_size, int):
        new_size = max(make_divisible(img_size, int(s)), floor)
    # 处理列表尺寸的情况（如img_size=[640, 480]）
    elif isinstance(img_size, list):
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        # 不支持其他类型的输入
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    # 如果尺寸发生变化，输出警告信息
    if new_size != img_size:
        LOGGER.info(
            f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    # 返回调整后的尺寸，如果输入是单个整数，返回[new_size, new_size]
    return new_size if isinstance(img_size, list) else [new_size] * 2


def process_image(path, img_size, stride):
    '''
    图像预处理函数
    参数:
        path: 图像路径
        img_size: 目标图像尺寸
        stride: 模型步长
    '''
    try:
        # 首先尝试使用OpenCV读取图像
        img_src = cv2.imread(path)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)  # 转换颜色空间从BGR到RGB
        assert img_src is not None, f"opencv cannot read image correctly or {path} not exists"
    except:
        # 如果OpenCV失败，尝试使用PIL读取
        img_src = np.asarray(Image.open(path))
        assert img_src is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

    # 使用letterbox进行图像预处理（保持宽高比的resize）
    image = letterbox(img_src, img_size, stride=stride)[0]
    
    # 转换图像格式：HWC（高度，宽度，通道）到CHW（通道，高度，宽度）
    image = image.transpose((2, 0, 1))
    
    # 转换为连续的numpy数组，然后转换为PyTorch张量
    image = torch.from_numpy(np.ascontiguousarray(image))
    
    # 转换为浮点类型
    image = image.float()
    
    # 归一化：将像素值缩放到0-1范围
    image /= 255
    
    return image, img_src


class Detector(DetectBackend):
    """
    YOLOv6检测器类，继承自DetectBackend
    实现了模型加载、推理和可视化功能
    """
    def __init__(self,
                 ckpt_path,      # 模型权重文件路径
                 class_names,    # 类别名称列表
                 device,         # 计算设备（CPU/GPU）
                 img_size=640,   # 输入图像尺寸
                 conf_thres=0.25,# 置信度阈值
                 iou_thres=0.45, # IoU阈值（用于NMS）
                 max_det=1000):  # 最大检测数量
        # 初始化父类
        super().__init__(ckpt_path, device)
        
        # 设置类属性
        self.class_names = class_names
        self.model.float()  # 将模型转换为浮点类型
        self.device = device
        self.img_size = check_img_size(img_size)  # 检查并调整图像尺寸
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def forward(self, x, src_shape):
        """
        模型前向推理
        参数:
            x: 输入张量
            src_shape: 原始图像形状
        """
        # 执行模型推理
        pred_results = super().forward(x)
        
        # 进行非极大值抑制(NMS)处理
        classes = None  # 需要保留的类别，None表示保留所有类别
        det = non_max_suppression(pred_results, 
                                self.conf_thres, 
                                self.iou_thres,
                                classes, 
                                agnostic=False, 
                                max_det=self.max_det)[0]

        # 将检测框坐标转换回原始图像尺寸
        det[:, :4] = Inferer.rescale(x.shape[2:], det[:, :4], src_shape).round()
        
        # 提取检测结果
        boxes = det[:, :4]      # 边界框坐标
        scores = det[:, 4]      # 置信度分数
        labels = det[:, 5].long()  # 类别标签
        
        # 整理预测结果
        prediction = {'boxes': boxes, 'scores': scores, 'labels': labels}
        return predictiondef predict(self, img_path):
            """
            对单张图片进行预测
            参数:
                img_path: 图像路径
            返回:
                out: 包含检测结果的字典
            """
            # 预处理图像
            img, img_src = process_image(img_path, self.img_size, 32)
            
            # 将图像转移到指定设备（CPU/GPU）
            img = img.to(self.device)
            
            # 如果图像是3维的，添加batch维度使其变为4维
            if len(img.shape) == 3:
                img = img[None]  # 扩展为 [1, channels, height, width]
        
            # 执行前向推理
            prediction = self.forward(img, img_src.shape)
            
            # 将结果转换为numpy数组并移到CPU
            out = {k: v.cpu().numpy() for k, v in prediction.items()}
            
            # 将数字标签转换为对应的类别名称
            out['classes'] = [self.class_names[i] for i in out['labels']]
            return out
        
        def show_predict(self,
                         img_path,          # 图像路径
                         min_score=0.5,     # 最小置信度阈值
                         figsize=(16, 16),  # 显示图像大小
                         color='lawngreen', # 边框颜色
                         linewidth=2):      # 边框线宽
            """
            预测并可视化检测结果
            """
            # 执行预测
            prediction = self.predict(img_path)
            
            # 提取预测结果
            boxes, scores, classes = prediction['boxes'], prediction['scores'], prediction['classes']
            
            # 调用可视化函数显示结果
            visualize_detections(Image.open(img_path),
                                boxes, classes, scores,
                                min_score=min_score, 
                                figsize=figsize,  
                                color=color, 
                                linewidth=linewidth
                                )
        
        def create_model(model_name, class_names=CLASS_NAMES, device=DEVICE,
                         img_size=640, conf_thres=0.25, iou_thres=0.45, max_det=1000):
            """
            创建并加载模型
            参数:
                model_name: 模型名称
                class_names: 类别名称列表
                device: 计算设备
                img_size: 输入图像尺寸
                conf_thres: 置信度阈值
                iou_thres: IoU阈值
                max_det: 最大检测数量
            """
            # 确保权重目录存在
            if not os.path.exists(str(PATH_YOLOv6/'weights')):
                os.mkdir(str(PATH_YOLOv6/'weights'))
            
            # 如果本地没有模型权重文件，从GitHub下载
            if not os.path.exists(str(PATH_YOLOv6/'weights') + f'/{model_name}.pt'):
                torch.hub.load_state_dict_from_url(
                    f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{model_name}.pt",
                    str(PATH_YOLOv6/'weights'))
            
            # 创建并返回检测器实例
            return Detector(str(PATH_YOLOv6/'weights') + f'/{model_name}.pt',
                            class_names, device, img_size=img_size, conf_thres=conf_thres,
                            iou_thres=iou_thres, max_det=max_det)
        
        # 以下是不同版本YOLOv6模型的快捷创建函数
        def yolov6n(class_names=CLASS_NAMES, device=DEVICE, img_size=640, conf_thres=0.25, iou_thres=0.45, max_det=1000):
            """
            创建YOLOv6-nano版本模型
            nano版本是最轻量级的版本，适合在资源受限的设备上运行
            """
            return create_model('yolov6n', class_names, device, img_size=img_size, conf_thres=conf_thres,
                                iou_thres=iou_thres, max_det=max_det)
        
        def yolov6s(class_names=CLASS_NAMES, device=DEVICE, img_size=640, conf_thres=0.25, iou_thres=0.45, max_det=1000):
            """
            创建YOLOv6-small版本模型
            small版本在速度和精度上有较好的平衡
            """
            return create_model('yolov6s', class_names, device, img_size=img_size, conf_thres=conf_thres,
                                iou_thres=iou_thres, max_det=max_det)
        
        def yolov6m(class_names=CLASS_NAMES, device=DEVICE, img_size=640, conf_thres=0.25, iou_thres=0.45, max_det=1000):
            """
            创建YOLOv6-medium版本模型
            medium版本提供更好的检测精度，但计算量相应增加
            """
            return create_model('yolov6m', class_names, device, img_size=img_size, conf_thres=conf_thres,
                                iou_thres=iou_thres, max_det=max_det)
        
        def yolov6l(class_names=CLASS_NAMES, device=DEVICE, img_size=640, conf_thres=0.25, iou_thres=0.45, max_det=1000):
            """
            创建YOLOv6-large版本模型
            large版本提供最高的检测精度，但需要更多计算资源
            """
            return create_model('yolov6l', class_names, device, img_size=img_size, conf_thres=conf_thres,
                                iou_thres=iou_thres, max_det=max_det)
        
        def custom(ckpt_path, class_names, device=DEVICE, img_size=640, conf_thres=0.25, iou_thres=0.45, max_det=1000):
            """
            加载自定义的YOLOv6模型
            参数:
                ckpt_path: 自定义模型权重文件路径
                class_names: 自定义类别名称列表
            """
            return Detector(ckpt_path, class_names, device, img_size=img_size, conf_thres=conf_thres,
                            iou_thres=iou_thres, max_det=max_det)