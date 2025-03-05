# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLO PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo11n.pt
TorchScript             | `torchscript`             | yolo11n.torchscript
ONNX                    | `onnx`                    | yolo11n.onnx
OpenVINO                | `openvino`                | yolo11n_openvino_model/
TensorRT                | `engine`                  | yolo11n.engine
CoreML                  | `coreml`                  | yolo11n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo11n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo11n.pb
TensorFlow Lite         | `tflite`                  | yolo11n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolo11n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolo11n_web_model/
PaddlePaddle            | `paddle`                  | yolo11n_paddle_model/
MNN                     | `mnn`                     | yolo11n.mnn
NCNN                    | `ncnn`                    | yolo11n_ncnn_model/
IMX                     | `imx`                     | yolo11n_imx_model/
RKNN                    | `rknn`                    | yolo11n_rknn_model/

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolo11n.pt format=onnx

Inference:
    $ yolo predict model=yolo11n.pt                 # PyTorch
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
                         yolo11n_imx_model          # IMX

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolo11n_web_model public/yolo11n_web_model
    $ npm start
"""

import gc  # 导入gc模块，用于垃圾回收
import json  # 导入json模块，用于处理JSON数据
import os  # 导入os模块，用于操作系统相关功能
import shutil  # 导入shutil模块，用于文件和目录的高级操作
import subprocess  # 导入subprocess模块，用于执行外部命令
import time  # 导入time模块，用于时间相关操作
import warnings  # 导入warnings模块，用于发出警告
from copy import deepcopy  # 从copy模块导入deepcopy函数，用于深拷贝
from datetime import datetime  # 从datetime模块导入datetime类，用于处理日期和时间
from pathlib import Path  # 从pathlib模块导入Path类，用于处理路径

import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习

from ultralytics.cfg import TASK2DATA, get_cfg  # 从ultralytics.cfg模块导入TASK2DATA和get_cfg函数
from ultralytics.data import build_dataloader  # 从ultralytics.data模块导入build_dataloader函数
from ultralytics.data.dataset import YOLODataset  # 从ultralytics.data.dataset模块导入YOLODataset类
from ultralytics.data.utils import check_cls_dataset, check_det_dataset  # 从ultralytics.data.utils模块导入数据集检查函数
from ultralytics.nn.autobackend import check_class_names, default_class_names  # 从ultralytics.nn.autobackend模块导入检查类名和默认类名的函数
from ultralytics.nn.modules import C2f, Classify, Detect, RTDETRDecoder  # 从ultralytics.nn.modules模块导入模型模块
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, WorldModel  # 从ultralytics.nn.tasks模块导入不同任务的模型类
from ultralytics.utils import (  # 从ultralytics.utils模块导入工具函数和常量
    ARM64,
    DEFAULT_CFG,
    IS_COLAB,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    PYTHON_VERSION,
    RKNN_CHIPS,
    ROOT,
    WINDOWS,
    __version__,
    callbacks,
    colorstr,
    get_default_args,
    yaml_save,
)
from ultralytics.utils.checks import (  # 从ultralytics.utils.checks模块导入检查函数
    check_imgsz,
    check_is_path_safe,
    check_requirements,
    check_version,
    is_sudo_available,
)
from ultralytics.utils.downloads import attempt_download_asset, get_github_assets, safe_download  # 从ultralytics.utils.downloads模块导入下载相关函数
from ultralytics.utils.files import file_size, spaces_in_path  # 从ultralytics.utils.files模块导入文件大小和路径空格检查函数
from ultralytics.utils.ops import Profile, nms_rotated, xywh2xyxy  # 从ultralytics.utils.ops模块导入操作相关函数
from ultralytics.utils.torch_utils import TORCH_1_13, get_latest_opset, select_device  # 从ultralytics.utils.torch_utils模块导入PyTorch相关工具函数


def export_formats():
    """Ultralytics YOLO export formats.
    Ultralytics YOLO导出格式。"""
    x = [  # 定义导出格式的列表
        ["PyTorch", "-", ".pt", True, True, []],  # PyTorch格式
        ["TorchScript", "torchscript", ".torchscript", True, True, ["batch", "optimize", "nms"]],  # TorchScript格式
        ["ONNX", "onnx", ".onnx", True, True, ["batch", "dynamic", "half", "opset", "simplify", "nms"]],  # ONNX格式
        ["OpenVINO", "openvino", "_openvino_model", True, False, ["batch", "dynamic", "half", "int8", "nms"]],  # OpenVINO格式
        ["TensorRT", "engine", ".engine", False, True, ["batch", "dynamic", "half", "int8", "simplify", "nms"]],  # TensorRT格式
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "half", "int8", "nms"]],  # CoreML格式
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True, ["batch", "int8", "keras", "nms"]],  # TensorFlow SavedModel格式
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"]],  # TensorFlow GraphDef格式
        ["TensorFlow Lite", "tflite", ".tflite", True, False, ["batch", "half", "int8", "nms"]],  # TensorFlow Lite格式
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False, []],  # TensorFlow Edge TPU格式
        ["TensorFlow.js", "tfjs", "_web_model", True, False, ["batch", "half", "int8", "nms"]],  # TensorFlow.js格式
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"]],  # PaddlePaddle格式
        ["MNN", "mnn", ".mnn", True, True, ["batch", "half", "int8"]],  # MNN格式
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "half"]],  # NCNN格式
        ["IMX", "imx", "_imx_model", True, True, ["int8"]],  # IMX格式
        ["RKNN", "rknn", "_rknn_model", False, False, ["batch", "name"]],  # RKNN格式
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))  # 返回格式字典


def validate_args(format, passed_args, valid_args):
    """
    Validates arguments based on format.
    根据格式验证参数。

    Args:
        format (str): The export format.
        passed_args (Namespace): The arguments used during export.
        valid_args (dict): List of valid arguments for the format.

    Raises:
        AssertionError: If an argument that's not supported by the export format is used, or if format doesn't have the supported arguments listed.
    """
    # Only check valid usage of these args
    export_args = ["half", "int8", "dynamic", "keras", "nms", "batch"]  # 定义有效参数列表

    assert valid_args is not None, f"ERROR ❌️ valid arguments for '{format}' not listed."  # 检查有效参数是否存在
    custom = {"batch": 1, "data": None, "device": None}  # exporter默认参数
    default_args = get_cfg(DEFAULT_CFG, custom)  # 获取默认配置
    for arg in export_args:  # 遍历有效参数
        not_default = getattr(passed_args, arg, None) != getattr(default_args, arg, None)  # 检查参数是否为默认值
        if not_default:  # 如果参数不是默认值
            assert arg in valid_args, f"ERROR ❌️ argument '{arg}' is not supported for format='{format}'"  # 抛出不支持参数的错误


def gd_outputs(gd):
    """TensorFlow GraphDef model output node names.
    TensorFlow GraphDef模型输出节点名称。"""
    name_list, input_list = [], []  # 初始化名称和输入列表
    for node in gd.node:  # 遍历节点
        name_list.append(node.name)  # 添加节点名称
        input_list.extend(node.input)  # 添加节点输入
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))  # 返回输出节点名称


def try_export(inner_func):
    """YOLO export decorator, i.e. @try_export.
    YOLO导出装饰器，即@try_export。"""
    inner_args = get_default_args(inner_func)  # 获取内部函数的默认参数

    def outer_func(*args, **kwargs):
        """Export a model.
        导出模型。"""
        prefix = inner_args["prefix"]  # 获取前缀
        try:
            with Profile() as dt:  # 记录时间
                f, model = inner_func(*args, **kwargs)  # 调用内部函数
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{f}' ({file_size(f):.1f} MB)")  # 日志记录导出成功
            return f, model  # 返回文件和模型
        except Exception as e:
            LOGGER.error(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")  # 日志记录导出失败
            raise e  # 抛出异常

    return outer_func  # 返回外部函数

class Exporter:
    """
    A class for exporting a model.
    用于导出模型的类。

    Attributes:
        args (SimpleNamespace): Configuration for the exporter.
        callbacks (list, optional): List of callback functions. Defaults to None.
        # 属性：
        #     args (SimpleNamespace): 导出的配置。
        #     callbacks (list, optional): 回调函数列表。默认为None。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the Exporter class.
        初始化Exporter类。

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            # 参数：
            #     cfg (str, optional): 配置文件的路径。默认为DEFAULT_CFG。
            overrides (dict, optional): Configuration overrides. Defaults to None.
            #     overrides (dict, optional): 配置覆盖。默认为None。
            _callbacks (dict, optional): Dictionary of callback functions. Defaults to None.
            #     _callbacks (dict, optional): 回调函数字典。默认为None。
        """
        self.args = get_cfg(cfg, overrides)  # 获取配置

        if self.args.format.lower() in {"coreml", "mlmodel"}:  # fix attempt for protobuf<3.20.x errors
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # 必须在TensorBoard回调之前运行

        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 设置回调函数
        callbacks.add_integration_callbacks(self)  # 添加集成回调

    def __call__(self, model=None) -> str:
        """Returns list of exported files/dirs after running callbacks.
        在运行回调后返回导出文件/目录的列表。"""
        self.run_callbacks("on_export_start")  # 运行导出开始的回调
        t = time.time()  # 记录开始时间
        fmt = self.args.format.lower()  # 将格式转换为小写
        if fmt in {"tensorrt", "trt"}:  # 'engine' 别名
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' 别名
            fmt = "coreml"
        fmts_dict = export_formats()  # 获取导出格式字典
        fmts = tuple(fmts_dict["Argument"][1:])  # 可用的导出格式
        if fmt not in fmts:  # 如果格式无效
            import difflib  # 导入difflib模块用于查找相似项

            # Get the closest match if format is invalid
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 60%相似度才能匹配
            if not matches:  # 如果没有匹配项
                raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")  # 抛出无效格式错误
            LOGGER.warning(f"WARNING ⚠️ Invalid export format='{fmt}', updating to format='{matches[0]}'")  # 日志记录无效格式警告
            fmt = matches[0]  # 更新格式为匹配的格式
        flags = [x == fmt for x in fmts]  # 创建格式标志
        if sum(flags) != 1:  # 如果格式标志不唯一
            raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")  # 抛出无效格式错误
        (jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, mnn, ncnn, imx, rknn) = (
            flags  # 导出布尔值
        )

        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))  # 检查是否为TensorFlow格式

        # Device
        dla = None  # 初始化DLA为None
        if fmt == "engine" and self.args.device is None:  # 如果格式为engine且未指定设备
            LOGGER.warning("WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0")  # 日志记录警告
            self.args.device = "0"  # 自动分配设备为0
        if fmt == "engine" and "dla" in str(self.args.device):  # 如果设备为DLA
            dla = self.args.device.split(":")[-1]  # 获取DLA设备编号
            self.args.device = "0"  # 更新设备为"0"
            assert dla in {"0", "1"}, f"Expected self.args.device='dla:0' or 'dla:1, but got {self.args.device}."  # 检查DLA设备编号
        self.device = select_device("cpu" if self.args.device is None else self.args.device)  # 选择设备

        # Argument compatibility checks
        fmt_keys = fmts_dict["Arguments"][flags.index(True) + 1]  # 获取格式对应的参数
        validate_args(fmt, self.args, fmt_keys)  # 验证参数兼容性
        if imx and not self.args.int8:  # 如果格式为IMX且未设置int8
            LOGGER.warning("WARNING ⚠️ IMX only supports int8 export, setting int8=True.")  # 日志记录警告
            self.args.int8 = True  # 设置int8为True
        if not hasattr(model, "names"):  # 如果模型没有names属性
            model.names = default_class_names()  # 设置默认类名
        model.names = check_class_names(model.names)  # 检查类名
        if self.args.half and self.args.int8:  # 如果同时设置了half和int8
            LOGGER.warning("WARNING ⚠️ half=True and int8=True are mutually exclusive, setting half=False.")  # 日志记录警告
            self.args.half = False  # 设置half为False
        if self.args.half and onnx and self.device.type == "cpu":  # 如果在CPU上使用half和onnx
            LOGGER.warning("WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0")  # 日志记录警告
            self.args.half = False  # 设置half为False
            assert not self.args.dynamic, "half=True not compatible with dynamic=True, i.e. use only one."  # 检查动态参数
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # 检查图像大小
        if self.args.int8 and engine:  # 如果格式为engine且设置了int8
            self.args.dynamic = True  # 强制动态导出TensorRT INT8
        if self.args.optimize:  # 如果设置了优化
            assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"  # 检查ncnn兼容性
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"  # 检查CUDA设备兼容性
        if rknn:  # 如果格式为RKNN
            if not self.args.name:  # 如果未设置名称
                LOGGER.warning(
                    "WARNING ⚠️ Rockchip RKNN export requires a missing 'name' arg for processor type. Using default name='rk3588'."  # 日志记录警告
                )
                self.args.name = "rk3588"  # 设置默认名称
            self.args.name = self.args.name.lower()  # 将名称转换为小写
            assert self.args.name in RKNN_CHIPS, (
                f"Invalid processor name '{self.args.name}' for Rockchip RKNN export. Valid names are {RKNN_CHIPS}."  # 检查RKNN处理器名称
            )
        if self.args.int8 and tflite:  # 如果设置了int8且格式为tflite
            assert not getattr(model, "end2end", False), "TFLite INT8 export not supported for end2end models."  # 检查end2end模型兼容性
        if self.args.nms:  # 如果设置了非极大值抑制
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."  # 检查分类模型兼容性
            if getattr(model, "end2end", False):  # 如果是end2end模型
                LOGGER.warning("WARNING ⚠️ 'nms=True' is not available for end2end models. Forcing 'nms=False'.")  # 日志记录警告
                self.args.nms = False  # 强制nms为False
            self.args.conf = self.args.conf or 0.25  # 设置nms导出的默认置信度
        if edgetpu:  # 如果格式为Edge TPU
            if not LINUX:  # 检查操作系统
                raise SystemError("Edge TPU export only supported on Linux. See https://coral.ai/docs/edgetpu/compiler")  # 抛出系统错误
            elif self.args.batch != 1:  # 检查批次大小
                LOGGER.warning("WARNING ⚠️ Edge TPU export requires batch size 1, setting batch=1.")  # 日志记录警告
                self.args.batch = 1  # 设置批次大小为1
        if isinstance(model, WorldModel):  # 如果模型是WorldModel
            LOGGER.warning(
                "WARNING ⚠️ YOLOWorld (original version) export is not supported to any format.\n"
                "WARNING ⚠️ YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."  # 日志记录警告
            )
            model.clip_model = None  # openvino int8导出错误
        if self.args.int8 and not self.args.data:  # 如果设置了int8且未指定数据
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # 设置默认数据
            LOGGER.warning(
                "WARNING ⚠️ INT8 export requires a missing 'data' arg for calibration. "
                f"Using default 'data={self.args.data}'."  # 日志记录警告
            )

        # Input
        im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)  # 创建输入张量
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )  # 获取模型文件路径
        if file.suffix in {".yaml", ".yml"}:  # 如果文件是yaml格式
            file = Path(file.name)  # 获取文件名

        # Update model
        model = deepcopy(model).to(self.device)  # 深拷贝模型并移动到指定设备
        for p in model.parameters():  # 遍历模型参数
            p.requires_grad = False  # 禁用梯度计算
        model.eval()  # 设置模型为评估模式
        model.float()  # 转换模型为浮点模式
        model = model.fuse()  # 融合模型层

        if imx:  # 如果格式为IMX
            from ultralytics.utils.torch_utils import FXModel  # 导入FXModel

            model = FXModel(model)  # 将模型转换为FXModel
        for m in model.modules():  # 遍历模型模块
            if isinstance(m, Classify):  # 如果模块是Classify
                m.export = True  # 设置导出标志
            if isinstance(m, (Detect, RTDETRDecoder)):  # 如果模块是Detect或RTDETRDecoder
                m.dynamic = self.args.dynamic  # 设置动态标志
                m.export = True  # 设置导出标志
                m.format = self.args.format  # 设置格式
                m.max_det = self.args.max_det  # 设置最大检测数
            elif isinstance(m, C2f) and not is_tf_format:  # 如果模块是C2f且不是TensorFlow格式
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split  # 使用分割的前向传播
            if isinstance(m, Detect) and imx:  # 如果模块是Detect且格式为IMX
                from ultralytics.utils.tal import make_anchors  # 导入make_anchors函数

                m.anchors, m.strides = (
                    x.transpose(0, 1)  # 转置锚点和步幅
                    for x in make_anchors(
                        torch.cat([s / m.stride.unsqueeze(-1) for s in self.imgsz], dim=1), m.stride, 0.5
                    )
                )

        y = None  # 初始化输出为None
        for _ in range(2):  # 干跑
            y = NMSModel(model, self.args)(im) if self.args.nms and not coreml else model(im)  # 运行模型
        if self.args.half and onnx and self.device.type != "cpu":  # 如果设置了half且格式为onnx且设备不是CPU
            im, model = im.half(), model.half()  # 转换为FP16

        # Filter warnings
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # 抑制TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)  # 抑制形状缺失的ONNX警告
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # 抑制CoreML np.bool弃用警告

        # Assign
        self.im = im  # 设置输入
        self.model = model  # 设置模型
        self.file = file  # 设置文件路径
        self.output_shape = (
            tuple(y.shape)  # 设置输出形状
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")  # 设置美化名称
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""  # 获取数据
        description = f"Ultralytics {self.pretty_name} model {f'trained on {data}' if data else ''}"  # 设置描述
        self.metadata = {  # 设置模型元数据
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": model.names,
            "args": {k: v for k, v in self.args if k in fmt_keys},
        }  # 模型元数据
        if dla is not None:  # 如果DLA不为None
            self.metadata["dla"] = dla  # 确保AutoBackend使用正确的DLA设备
        if model.task == "pose":  # 如果任务是姿态估计
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape  # 设置关键点形状

        LOGGER.info(  # 日志记录导出信息
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"
        )

        # Exports
        f = [""] * len(fmts)  # 导出文件名
        if jit or ncnn:  # TorchScript
            f[0], _ = self.export_torchscript()  # 导出TorchScript模型
        if engine:  # TensorRT需要在ONNX之前导出
            f[1], _ = self.export_engine(dla=dla)  # 导出TensorRT模型
        if onnx:  # ONNX
            f[2], _ = self.export_onnx()  # 导出ONNX模型
        if xml:  # OpenVINO
            f[3], _ = self.export_openvino()  # 导出OpenVINO模型
        if coreml:  # CoreML
            f[4], _ = self.export_coreml()  # 导出CoreML模型
        if is_tf_format:  # TensorFlow格式
            self.args.int8 |= edgetpu  # 设置int8为True
            f[5], keras_model = self.export_saved_model()  # 导出TensorFlow SavedModel
            if pb or tfjs:  # pb是tfjs的前提
                f[6], _ = self.export_pb(keras_model=keras_model)  # 导出TensorFlow GraphDef
            if tflite:  # 导出TensorFlow Lite
                f[7], _ = self.export_tflite(keras_model=keras_model, nms=False, agnostic_nms=self.args.agnostic_nms)
            if edgetpu:  # 导出Edge TPU
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")
            if tfjs:  # 导出TensorFlow.js
                f[9], _ = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10], _ = self.export_paddle()  # 导出PaddlePaddle模型
        if mnn:  # MNN
            f[11], _ = self.export_mnn()  # 导出MNN模型
        if ncnn:  # NCNN
            f[12], _ = self.export_ncnn()  # 导出NCNN模型
        if imx:  # IMX
            f[13], _ = self.export_imx()  # 导出IMX模型
        if rknn:  # RKNN
            f[14], _ = self.export_rknn()  # 导出RKNN模型

        # Finish
        f = [str(x) for x in f if x]  # 过滤掉空字符串和None
        if any(f):  # 如果有导出文件
            f = str(Path(f[-1]))  # 获取最后一个文件的路径
            square = self.imgsz[0] == self.imgsz[1]  # 检查是否为正方形图像
            s = (
                ""
                if square
                else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
                f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."  # 日志记录警告
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")  # 设置图像大小
            predict_data = f"data={data}" if model.task == "segment" and fmt == "pb" else ""  # 设置预测数据
            q = "int8" if self.args.int8 else "half" if self.args.half else ""  # 量化
            LOGGER.info(
                f"\nExport complete ({time.time() - t:.1f}s)"
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f"\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}"
                f"\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}"
                f"\nVisualize:       https://netron.app"  # 日志记录导出完成信息
            )

        self.run_callbacks("on_export_end")  # 运行导出结束的回调
        return f  # 返回导出文件/目录的列表

    def get_int8_calibration_dataloader(self, prefix=""):
        """Build and return a dataloader suitable for calibration of INT8 models.
        构建并返回适合INT8模型校准的数据加载器。"""
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")  # 日志记录校准图像收集信息
        data = (check_cls_dataset if self.model.task == "classify" else check_det_dataset)(self.args.data)  # 检查数据集
        # TensorRT INT8 calibration should use 2x batch size
        batch = self.args.batch * (2 if self.args.format == "engine" else 1)  # 设置批次大小
        dataset = YOLODataset(
            data[self.args.split or "val"],  # 获取数据集
            data=data,
            task=self.model.task,
            imgsz=self.imgsz[0],  # 设置图像大小
            augment=False,  # 不进行数据增强
            batch_size=batch,  # 设置批次大小
        )
        n = len(dataset)  # 获取数据集大小
        if n < self.args.batch:  # 如果数据集大小小于批次大小
            raise ValueError(
                f"The calibration dataset ({n} images) must have at least as many images as the batch size ('batch={self.args.batch}')."
            )  # 抛出错误
        elif n < 300:  # 如果数据集大小小于300
            LOGGER.warning(f"{prefix} WARNING ⚠️ >300 images recommended for INT8 calibration, found {n} images.")  # 日志记录警告
        return build_dataloader(dataset, batch=batch, workers=0)  # 返回数据加载器

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """YOLO TorchScript model export.
        YOLO TorchScript模型导出。"""
        LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")  # 日志记录导出开始信息
        f = self.file.with_suffix(".torchscript")  # 设置导出文件名

        ts = torch.jit.trace(NMSModel(self.model, self.args) if self.args.nms else self.model, self.im, strict=False)  # 跟踪模型
        extra_files = {"config.txt": json.dumps(self.metadata)}  # 附加文件
        if self.args.optimize:  # 如果设置了优化
            LOGGER.info(f"{prefix} optimizing for mobile...")  # 日志记录优化信息
            from torch.utils.mobile_optimizer import optimize_for_mobile  # 导入优化函数

            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)  # 保存优化后的模型
        else:
            ts.save(str(f), _extra_files=extra_files)  # 保存模型
        return f, None  # 返回文件名和None

        
    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """YOLO ONNX export."""  # YOLO ONNX 导出
        requirements = ["onnx>=1.12.0"]  # 需要的依赖库
        if self.args.simplify:
            requirements += ["onnxslim", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]  # 如果需要简化，添加依赖
        check_requirements(requirements)  # 检查依赖库是否满足

        import onnx  # noqa  # 导入 ONNX 库

        opset_version = self.args.opset or get_latest_opset()  # 获取 opset 版本
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")  # 日志记录导出信息
        f = str(self.file.with_suffix(".onnx"))  # 设置输出文件名
        output_names = ["output0", "output1"] if isinstance(self.model, SegmentationModel) else ["output0"]  # 根据模型类型设置输出名称
        dynamic = self.args.dynamic  # 获取动态参数
        if dynamic:
            self.model.cpu()  # dynamic=True 仅支持 CPU
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)
            if self.args.nms:  # 只有在 NMS 时 batch size 是动态的
                dynamic["output0"].pop(2)  # 移除动态输出的第三维

        if self.args.nms and self.model.task == "obb":
            self.args.opset = opset_version  # 对于 NMSModel
            # OBB error https://github.com/pytorch/pytorch/issues/110859#issuecomment-1757841865
            try:
                torch.onnx.register_custom_op_symbolic("aten::lift_fresh", lambda g, x: x, opset_version)  # 注册自定义操作
            except RuntimeError:  # 如果已经注册则会失败
                pass
            check_requirements("onnxslim>=0.1.46")  # 检查 onnxslim 版本

        torch.onnx.export(
            NMSModel(self.model, self.args) if self.args.nms else self.model,  # 导出模型
            self.im.cpu() if dynamic else self.im,  # 输入图像
            f,  # 输出文件
            verbose=False,  # 是否详细输出
            opset_version=opset_version,  # opset 版本
            do_constant_folding=True,  # 警告：torch>=1.12 可能需要 do_constant_folding=False
            input_names=["images"],  # 输入名称
            output_names=output_names,  # 输出名称
            dynamic_axes=dynamic or None,  # 动态轴
        )

        # Checks
        model_onnx = onnx.load(f)  # 加载 ONNX 模型

        # Simplify
        if self.args.simplify:
            try:
                import onnxslim  # 导入 onnxslim 库

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")  # 日志记录简化信息
                model_onnx = onnxslim.slim(model_onnx)  # 简化模型

            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")  # 记录简化失败的信息

        # Metadata
        for k, v in self.metadata.items():  # 遍历元数据
            meta = model_onnx.metadata_props.add()  # 添加元数据属性
            meta.key, meta.value = k, str(v)  # 设置键值对

        onnx.save(model_onnx, f)  # 保存 ONNX 模型
        return f, model_onnx  # 返回文件路径和模型

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        """YOLO OpenVINO export."""  # YOLO OpenVINO 导出
        check_requirements("openvino>=2024.5.0")  # 检查 OpenVINO 版本
        import openvino as ov  # 导入 OpenVINO 库

        LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")  # 日志记录导出信息
        assert TORCH_1_13, f"OpenVINO export requires torch>=1.13.0 but torch=={torch.__version__} is installed"  # 检查 PyTorch 版本
        ov_model = ov.convert_model(
            NMSModel(self.model, self.args) if self.args.nms else self.model,  # 转换模型
            input=None if self.args.dynamic else [self.im.shape],  # 输入形状
            example_input=self.im,  # 示例输入
        )

        def serialize(ov_model, file):
            """Set RT info, serialize and save metadata YAML."""  # 设置运行时信息，序列化并保存元数据 YAML
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])  # 设置模型类型
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])  # 设置反转输入通道
            ov_model.set_rt_info(114, ["model_info", "pad_value"])  # 设置填充值
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])  # 设置缩放值
            ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])  # 设置 IOU 阈值
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])  # 设置标签
            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])  # 设置调整类型

            ov.runtime.save_model(ov_model, file, compress_to_fp16=self.args.half)  # 保存模型
            yaml_save(Path(file).parent / "metadata.yaml", self.metadata)  # 添加元数据 YAML

        if self.args.int8:
            fq = str(self.file).replace(self.file.suffix, f"_int8_openvino_model{os.sep}")  # 设置输出文件名
            fq_ov = str(Path(fq) / self.file.with_suffix(".xml").name)  # 设置 OpenVINO 输出文件名
            check_requirements("nncf>=2.14.0")  # 检查 NNCF 版本
            import nncf  # 导入 NNCF 库

            def transform_fn(data_item) -> np.ndarray:
                """Quantization transform function."""  # 量化转换函数
                data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item  # 获取图像数据
                assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"  # 检查数据类型
                im = data_item.numpy().astype(np.float32) / 255.0  # uint8 转换为 fp16/32，并将范围从 0-255 转换为 0.0 - 1.0
                return np.expand_dims(im, 0) if im.ndim == 3 else im  # 返回处理后的图像

            # Generate calibration data for integer quantization
            ignored_scope = None  # 初始化忽略范围
            if isinstance(self.model.model[-1], Detect):  # 如果模型最后一层是 Detect
                # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])  # 获取头模块名称
                ignored_scope = nncf.IgnoredScope(  # 忽略操作
                    patterns=[
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                        f".*{head_module_name}\\.dfl.*",
                    ],
                    types=["Sigmoid"],
                )

            quantized_ov_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=nncf.Dataset(self.get_int8_calibration_dataloader(prefix), transform_fn),  # 量化数据集
                preset=nncf.QuantizationPreset.MIXED,  # 量化预设
                ignored_scope=ignored_scope,  # 忽略范围
            )
            serialize(quantized_ov_model, fq_ov)  # 序列化量化模型
            return fq, None  # 返回文件路径

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")  # 设置输出文件名
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)  # 设置 OpenVINO 输出文件名

        serialize(ov_model, f_ov)  # 序列化模型
        return f, None  # 返回文件路径

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """YOLO Paddle export."""  # YOLO Paddle 导出
        check_requirements(("paddlepaddle-gpu" if torch.cuda.is_available() else "paddlepaddle", "x2paddle"))  # 检查 PaddlePaddle 依赖
        import x2paddle  # noqa  # 导入 x2paddle 库
        from x2paddle.convert import pytorch2paddle  # noqa  # 从 x2paddle 导入转换函数

        LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")  # 日志记录导出信息
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")  # 设置输出文件名

        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.im])  # 导出模型
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # 添加元数据 YAML
        return f, None  # 返回文件路径

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):
        """YOLOv8 MNN export using MNN https://github.com/alibaba/MNN."""  # 使用 MNN 导出 YOLOv8
        f_onnx, _ = self.export_onnx()  # 首先获取 ONNX 模型

        check_requirements("MNN>=2.9.6")  # 检查 MNN 版本
        import MNN  # noqa  # 导入 MNN 库
        from MNN.tools import mnnconvert  # 导入 MNN 转换工具

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with MNN {MNN.version()}...")  # 日志记录导出信息
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"  # 检查 ONNX 文件是否存在
        f = str(self.file.with_suffix(".mnn"))  # 设置 MNN 模型文件
        args = ["", "-f", "ONNX", "--modelFile", f_onnx, "--MNNModel", f, "--bizCode", json.dumps(self.metadata)]  # 设置转换参数
        if self.args.int8:
            args.extend(("--weightQuantBits", "8"))  # 添加权重量化位数参数
        if self.args.half:
            args.append("--fp16")  # 添加 FP16 参数
        mnnconvert.convert(args)  # 执行转换
        # remove scratch file for model convert optimize
        convert_scratch = Path(self.file.parent / ".__convert_external_data.bin")  # 设置临时文件路径
        if convert_scratch.exists():
            convert_scratch.unlink()  # 删除临时文件
        return f, None  # 返回文件路径

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """YOLO NCNN export using PNNX https://github.com/pnnx/pnnx."""  # 使用 PNNX 导出 YOLO NCNN
        check_requirements("ncnn")  # 检查 NCNN 依赖
        import ncnn  # noqa  # 导入 NCNN 库

        LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__}...")  # 日志记录导出信息
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))  # 设置输出文件名
        f_ts = self.file.with_suffix(".torchscript")  # 设置 TorchScript 文件名

        name = Path("pnnx.exe" if WINDOWS else "pnnx")  # PNNX 文件名
        pnnx = name if name.is_file() else (ROOT / name)  # 获取 PNNX 路径
        if not pnnx.is_file():
            LOGGER.warning(
                f"{prefix} WARNING ⚠️ PNNX not found. Attempting to download binary file from "
                "https://github.com/pnnx/pnnx/.\nNote PNNX Binary file must be placed in current working directory "
                f"or in {ROOT}. See PNNX repo for full installation instructions."
            )  # 日志记录 PNNX 未找到的警告
            system = "macos" if MACOS else "windows" if WINDOWS else "linux-aarch64" if ARM64 else "linux"  # 获取系统类型
            try:
                release, assets = get_github_assets(repo="pnnx/pnnx")  # 获取 GitHub 资产
                asset = [x for x in assets if f"{system}.zip" in x][0]  # 获取对应系统的资产
                assert isinstance(asset, str), "Unable to retrieve PNNX repo assets"  # 检查资产类型
                LOGGER.info(f"{prefix} successfully found latest PNNX asset file {asset}")  # 日志记录成功找到资产
            except Exception as e:
                release = "20240410"  # 默认版本
                asset = f"pnnx-{release}-{system}.zip"  # 默认资产
                LOGGER.warning(f"{prefix} WARNING ⚠️ PNNX GitHub assets not found: {e}, using default {asset}")  # 日志记录未找到资产的警告
            unzip_dir = safe_download(f"https://github.com/pnnx/pnnx/releases/download/{release}/{asset}", delete=True)  # 下载资产
            if check_is_path_safe(Path.cwd(), unzip_dir):  # 避免路径遍历安全漏洞
                shutil.move(src=unzip_dir / name, dst=pnnx)  # 移动二进制文件到 ROOT
                pnnx.chmod(0o777)  # 设置权限
                shutil.rmtree(unzip_dir)  # 删除解压目录

        ncnn_args = [
            f"ncnnparam={f / 'model.ncnn.param'}",  # NCNN 参数文件
            f"ncnnbin={f / 'model.ncnn.bin'}",  # NCNN 二进制文件
            f"ncnnpy={f / 'model_ncnn.py'}",  # NCNN Python 文件
        ]

        pnnx_args = [
            f"pnnxparam={f / 'model.pnnx.param'}",  # PNNX 参数文件
            f"pnnxbin={f / 'model.pnnx.bin'}",  # PNNX 二进制文件
            f"pnnxpy={f / 'model_pnnx.py'}",  # PNNX Python 文件
            f"pnnxonnx={f / 'model.pnnx.onnx'}",  # PNNX ONNX 文件
        ]

        cmd = [
            str(pnnx),  # PNNX 命令
            str(f_ts),  # TorchScript 文件
            *ncnn_args,  # NCNN 参数
            *pnnx_args,  # PNNX 参数
            f"fp16={int(self.args.half)}",  # FP16 参数
            f"device={self.device.type}",  # 设备类型
            f'inputshape="{[self.args.batch, 3, *self.imgsz]}"',  # 输入形状
        ]
        f.mkdir(exist_ok=True)  # 创建 ncnn_model 目录
        LOGGER.info(f"{prefix} running '{' '.join(cmd)}'")  # 日志记录运行命令
        subprocess.run(cmd, check=True)  # 执行命令

        # Remove debug files
        pnnx_files = [x.split("=")[-1] for x in pnnx_args]  # 获取 PNNX 文件列表
        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_files):  # 遍历调试文件
            Path(f_debug).unlink(missing_ok=True)  # 删除调试文件

        yaml_save(f / "metadata.yaml", self.metadata)  # 添加元数据 YAML
        return str(f), None  # 返回文件路径

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """YOLO CoreML export."""  # YOLO CoreML 导出
        mlmodel = self.args.format.lower() == "mlmodel"  # 检查是否请求 legacy *.mlmodel 导出格式
        check_requirements("coremltools>=6.0,<=6.2" if mlmodel else "coremltools>=7.0")  # 检查 coremltools 版本
        import coremltools as ct  # noqa  # 导入 coremltools 库

        LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")  # 日志记录导出信息
        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."  # 检查操作系统
        assert self.args.batch == 1, "CoreML batch sizes > 1 are not supported. Please retry at 'batch=1'."  # 检查批大小
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")  # 设置输出文件名
        if f.is_dir():
            shutil.rmtree(f)  # 删除已存在的目录

        bias = [0.0, 0.0, 0.0]  # 偏置
        scale = 1 / 255  # 缩放因子
        classifier_config = None  # 初始化分类器配置
        if self.model.task == "classify":
            classifier_config = ct.ClassifierConfig(list(self.model.names.values())) if self.args.nms else None  # 设置分类器配置
            model = self.model  # 获取模型
        elif self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im) if self.args.nms else self.model  # 检查模型类型
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} WARNING ⚠️ 'nms=True' is only available for Detect models like 'yolo11n.pt'.")  # 日志记录警告
                # TODO CoreML Segment and Pose model pipelining
            model = self.model  # 获取模型

        ts = torch.jit.trace(model.eval(), self.im, strict=False)  # TorchScript 模型
        ct_model = ct.convert(
            ts,
            inputs=[ct.ImageType("image", shape=self.im.shape, scale=scale, bias=bias)],  # 设置输入类型
            classifier_config=classifier_config,  # 设置分类器配置
            convert_to="neuralnetwork" if mlmodel else "mlprogram",  # 转换类型
        )
        bits, mode = (8, "kmeans") if self.args.int8 else (16, "linear") if self.args.half else (32, None)  # 设置量化位数和模式
        if bits < 32:
            if "kmeans" in mode:
                check_requirements("scikit-learn")  # 检查 scikit-learn 依赖
            if mlmodel:
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)  # 量化权重
            elif bits == 8:  # mlprogram 已经量化为 FP16
                import coremltools.optimize.coreml as cto  # 导入优化模块

                op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)  # 设置优化配置
                config = cto.OptimizationConfig(global_config=op_config)  # 设置全局配置
                ct_model = cto.palettize_weights(ct_model, config=config)  # 进行权重调色

        if self.args.nms and self.model.task == "detect":
            if mlmodel:
                # coremltools<=6.2 NMS export requires Python<3.11
                check_version(PYTHON_VERSION, "<3.11", name="Python ", hard=True)  # 检查 Python 版本
                weights_dir = None  # 初始化权重目录
            else:
                ct_model.save(str(f))  # 保存模型，否则权重目录不存在
                weights_dir = str(f / "Data/com.apple.CoreML/weights")  # 获取权重目录

            ct_model = self._pipeline_coreml(ct_model, weights_dir=weights_dir)  # 进行 CoreML 管道处理

        m = self.metadata  # 获取元数据字典
        ct_model.short_description = m.pop("description")  # 设置简短描述
        ct_model.author = m.pop("author")  # 设置作者
        ct_model.license = m.pop("license")  # 设置许可证
        ct_model.version = m.pop("version")  # 设置版本
        ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})  # 更新用户定义的元数据
        try:
            ct_model.save(str(f))  # 保存 *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} WARNING ⚠️ CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )  # 日志记录保存失败的警告
            f = f.with_suffix(".mlmodel")  # 回退到 *.mlmodel
            ct_model.save(str(f))  # 保存模型
        return f, ct_model  # 返回文件路径和模型

    @try_export
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        """YOLO TensorRT export https://developer.nvidia.com/tensorrt."""  # YOLO TensorRT 导出
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"  # 确保在 GPU 上导出
        f_onnx, _ = self.export_onnx()  # 运行 ONNX 导出，确保在 TRT 导入之前 https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa  # 导入 TensorRT 库
        except ImportError:
            if LINUX:
                check_requirements("tensorrt>7.0.0,!=10.1.0")  # 检查 TensorRT 版本
            import tensorrt as trt  # noqa  # 再次导入 TensorRT 库
        check_version(trt.__version__, ">=7.0.0", hard=True)  # 检查 TensorRT 版本是否大于等于 7.0.0
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")  # 检查 TensorRT 版本是否不等于 10.1.0

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")  # 日志记录导出信息
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # 检查 TensorRT 版本是否大于等于 10
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"  # 检查 ONNX 文件是否存在
        f = self.file.with_suffix(".engine")  # 设置 TensorRT 引擎文件名
        logger = trt.Logger(trt.Logger.INFO)  # 创建 TensorRT 日志记录器
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE  # 如果需要详细日志，设置日志级别

        # Engine builder
        builder = trt.Builder(logger)  # 创建 TensorRT 构建器
        config = builder.create_builder_config()  # 创建构建配置
        workspace = int(self.args.workspace * (1 << 30)) if self.args.workspace is not None else 0  # 设置工作区大小
        if is_trt10 and workspace > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)  # 设置工作区内存限制
        elif workspace > 0:  # TensorRT 版本 7 和 8
            config.max_workspace_size = workspace  # 设置最大工作区大小
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # 设置网络定义标志
        network = builder.create_network(flag)  # 创建网络
        half = builder.platform_has_fast_fp16 and self.args.half  # 检查是否支持 FP16
        int8 = builder.platform_has_fast_int8 and self.args.int8  # 检查是否支持 INT8

        # Optionally switch to DLA if enabled
        if dla is not None:
            if not IS_JETSON:
                raise ValueError("DLA is only available on NVIDIA Jetson devices")  # DLA 仅在 NVIDIA Jetson 设备上可用
            LOGGER.info(f"{prefix} enabling DLA on core {dla}...")  # 日志记录启用 DLA 信息
            if not self.args.half and not self.args.int8:
                raise ValueError(
                    "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
                )  # DLA 需要启用 FP16 或 INT8
            config.default_device_type = trt.DeviceType.DLA  # 设置默认设备类型为 DLA
            config.DLA_core = int(dla)  # 设置 DLA 核心
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # 设置 GPU 回退标志

        # Read ONNX file
        parser = trt.OnnxParser(network, logger)  # 创建 ONNX 解析器
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")  # 解析 ONNX 文件失败

        # Network inputs
        inputs = [network.get_input(i) for i in range(network.num_inputs)]  # 获取网络输入
        outputs = [network.get_output(i) for i in range(network.num_outputs)]  # 获取网络输出
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')  # 日志记录输入信息
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')  # 日志记录输出信息

        if self.args.dynamic:
            shape = self.im.shape  # 获取输入图像形状
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'")  # 日志记录动态模型的警告
            profile = builder.create_optimization_profile()  # 创建优化配置文件
            min_shape = (1, shape[1], 32, 32)  # 最小输入形状
            max_shape = (*shape[:2], *(int(max(1, workspace) * d) for d in shape[2:]))  # 最大输入形状
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)  # 设置输入形状
            config.add_optimization_profile(profile)  # 添加优化配置文件

        LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}")  # 日志记录构建引擎信息
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)  # 设置 INT8 标志
            config.set_calibration_profile(profile)  # 设置校准配置文件
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # 设置详细的性能分析

            class EngineCalibrator(trt.IInt8Calibrator):
                def __init__(
                    self,
                    dataset,  # ultralytics.data.build.InfiniteDataLoader
                    batch: int,
                    cache: str = "",
                ) -> None:
                    trt.IInt8Calibrator.__init__(self)  # 初始化基类
                    self.dataset = dataset  # 保存数据集
                    self.data_iter = iter(dataset)  # 创建数据迭代器
                    self.algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2  # 设置校准算法
                    self.batch = batch  # 设置批大小
                    self.cache = Path(cache)  # 设置缓存路径

                def get_algorithm(self) -> trt.CalibrationAlgoType:
                    """Get the calibration algorithm to use."""  # 获取校准算法
                    return self.algo

                def get_batch_size(self) -> int:
                    """Get the batch size to use for calibration."""  # 获取用于校准的批大小
                    return self.batch or 1  # 如果未设置，则返回 1

                def get_batch(self, names) -> list:
                    """Get the next batch to use for calibration, as a list of device memory pointers."""  # 获取下一个用于校准的批次
                    try:
                        im0s = next(self.data_iter)["img"] / 255.0  # 获取图像并归一化
                        im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s  # 移动到 GPU
                        return [int(im0s.data_ptr())]  # 返回设备内存指针
                    except StopIteration:
                        # Return [] or None, signal to TensorRT there is no calibration data remaining
                        return None  # 返回 None，表示没有剩余的校准数据

                def read_calibration_cache(self) -> bytes:
                    """Use existing cache instead of calibrating again, otherwise, implicitly return None."""  # 使用现有缓存而不是重新校准
                    if self.cache.exists() and self.cache.suffix == ".cache":
                        return self.cache.read_bytes()  # 读取缓存数据

                def write_calibration_cache(self, cache) -> None:
                    """Write calibration cache to disk."""  # 将校准缓存写入磁盘
                    _ = self.cache.write_bytes(cache)  # 写入缓存

            # Load dataset w/ builder (for batching) and calibrate
            config.int8_calibrator = EngineCalibrator(
                dataset=self.get_int8_calibration_dataloader(prefix),  # 获取 INT8 校准数据加载器
                batch=2 * self.args.batch,  # TensorRT INT8 校准应使用 2 倍的批大小
                cache=str(self.file.with_suffix(".cache")),  # 设置缓存文件路径
            )

        elif half:
            config.set_flag(trt.BuilderFlag.FP16)  # 设置 FP16 标志

        # Free CUDA memory
        del self.model  # 删除模型以释放 CUDA 内存
        gc.collect()  # 垃圾回收
        torch.cuda.empty_cache()  # 清空 CUDA 缓存

        # Write file
        build = builder.build_serialized_network if is_trt10 else builder.build_engine  # 根据 TensorRT 版本选择构建方法
        with build(network, config) as engine, open(f, "wb") as t:  # 构建引擎并打开文件进行写入
            # Metadata
            meta = json.dumps(self.metadata)  # 将元数据转换为 JSON 字符串
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))  # 写入元数据长度
            t.write(meta.encode())  # 写入元数据
            # Model
            t.write(engine if is_trt10 else engine.serialize())  # 写入模型数据

        return f, None  # 返回文件路径

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """YOLO TensorFlow SavedModel export."""  # YOLO TensorFlow SavedModel 导出
        cuda = torch.cuda.is_available()  # 检查是否可用 CUDA
        try:
            import tensorflow as tf  # noqa  # 导入 TensorFlow 库
        except ImportError:
            suffix = "-macos" if MACOS else "-aarch64" if ARM64 else "" if cuda else "-cpu"  # 根据平台设置后缀
            version = ">=2.0.0"  # 设置版本要求
            check_requirements(f"tensorflow{suffix}{version}")  # 检查 TensorFlow 依赖
            import tensorflow as tf  # noqa  # 再次导入 TensorFlow 库
        check_requirements(
            (
                "keras",  # required by 'onnx2tf' package
                "tf_keras",  # required by 'onnx2tf' package
                "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
                "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
                "onnx>=1.12.0",  # ONNX 版本要求
                "onnx2tf>1.17.5,<=1.26.3",  # ONNX2TF 版本要求
                "onnxslim>=0.1.31",  # ONNX Slim 版本要求
                "tflite_support<=0.4.3" if IS_JETSON else "tflite_support",  # 修复 Jetson 的导入错误
                "flatbuffers>=23.5.26,<100",  # 更新 TensorFlow 包内的旧 flatbuffers
                "onnxruntime-gpu" if cuda else "onnxruntime",  # 根据是否可用 CUDA 设置 ONNX Runtime
            ),
            cmds="--extra-index-url https://pypi.ngc.nvidia.com",  # ONNX GraphSurgeon 仅在 NVIDIA 上可用
        )

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")  # 日志记录导出信息
        check_version(
            tf.__version__,
            ">=2.0.0",
            name="tensorflow",
            verbose=True,
            msg="https://github.com/ultralytics/ultralytics/issues/5161",
        )  # 检查 TensorFlow 版本

        import onnx2tf  # 导入 ONNX2TF 库

        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))  # 设置输出文件路径
        if f.is_dir():
            shutil.rmtree(f)  # 删除输出文件夹

        # Pre-download calibration file to fix https://github.com/PINTO0309/onnx2tf/issues/545
        onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")  # 校准文件路径
        if not onnx2tf_file.exists():
            attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)  # 下载校准文件

        # Export to ONNX
        self.args.simplify = True  # 设置简化参数
        f_onnx, _ = self.export_onnx()  # 导出 ONNX 模型

        # Export to TF
        np_data = None  # 初始化 NumPy 数据
        if self.args.int8:
            tmp_file = f / "tmp_tflite_int8_calibration_images.npy"  # INT8 校准图像文件
            if self.args.data:
                f.mkdir()  # 创建输出目录
                images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]  # 获取校准图像
                images = torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz).permute(
                    0, 2, 3, 1
                )  # 调整图像大小并调整维度
                np.save(str(tmp_file), images.numpy().astype(np.float32))  # 保存图像为 NumPy 文件
                np_data = ["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]  # 设置输入数据

        LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")  # 日志记录 TFLite 导出信息
        keras_model = onnx2tf.convert(
            input_onnx_file_path=f_onnx,  # 输入 ONNX 文件路径
            output_folder_path=str(f),  # 输出文件夹路径
            not_use_onnxsim=True,  # 不使用 ONNX 简化
            verbosity="error",  # 设置日志详细程度
            output_integer_quantized_tflite=self.args.int8,  # 是否输出 INT8 TFLite 模型
            quant_type="per-tensor",  # 量化类型
            custom_input_op_name_np_data_path=np_data,  # 自定义输入操作名称的 NumPy 数据路径
            disable_group_convolution=True,  # 禁用分组卷积以兼容模型
            enable_batchmatmul_unfold=True,  # 启用批量矩阵展开以兼容模型
        )
        yaml_save(f / "metadata.yaml", self.metadata)  # 添加元数据 YAML

        # Remove/rename TFLite models
        if self.args.int8:
            tmp_file.unlink(missing_ok=True)  # 删除临时文件
            for file in f.rglob("*_dynamic_range_quant.tflite"):  # 遍历动态范围量化文件
                file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))  # 重命名文件
            for file in f.rglob("*_integer_quant_with_int16_act.tflite"):  # 遍历额外的 FP16 激活 TFLite 文件
                file.unlink()  # 删除额外的文件

        # Add TFLite metadata
        for file in f.rglob("*.tflite"):  # 遍历所有 TFLite 文件
            f.unlink() if "quant_with_int16_act.tflite" in str(f) else self._add_tflite_metadata(file)  # 添加元数据

        return str(f), None  # 返回文件路径

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """YOLO TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow."""  # YOLO TensorFlow GraphDef 导出
        import tensorflow as tf  # noqa  # 导入 TensorFlow 库
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa  # 导入转换函数

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")  # 日志记录导出信息
        f = self.file.with_suffix(".pb")  # 设置输出文件路径

        m = tf.function(lambda x: keras_model(x))  # 完整模型
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))  # 获取具体函数
        frozen_func = convert_variables_to_constants_v2(m)  # 转换为常量
        frozen_func.graph.as_graph_def()  # 获取图形定义
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)  # 写入图形
        return f, None  # 返回文件路径

    @try_export
    def export_tflite(self, keras_model, nms, agnostic_nms, prefix=colorstr("TensorFlow Lite:")):
        """YOLO TensorFlow Lite export."""  # YOLO TensorFlow Lite 导出
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        import tensorflow as tf  # noqa  # 导入 TensorFlow 库

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")  # 日志记录导出信息
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))  # 设置保存模型路径
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # 设置 INT8 输出文件路径
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # 设置 FP16 输出文件路径
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"  # 设置 FP32 输出文件路径
        return str(f), None  # 返回文件路径

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """YOLO Edge TPU export https://coral.ai/docs/edgetpu/models-intro/."""  # YOLO Edge TPU 导出
        LOGGER.warning(f"{prefix} WARNING ⚠️ Edge TPU known bug https://github.com/ultralytics/ultralytics/issues/1185")  # 日志记录警告

        cmd = "edgetpu_compiler --version"  # 检查 Edge TPU 编译器版本
        help_url = "https://coral.ai/docs/edgetpu/compiler/"  # Edge TPU 编译器帮助链接
        assert LINUX, f"export only supported on Linux. See {help_url}"  # 确保在 Linux 上导出
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")  # 日志记录安装信息
            for c in (
                "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",  # 添加 GPG 密钥
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | '
                "sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",  # 添加 Edge TPU 源
                "sudo apt-get update",  # 更新包列表
                "sudo apt-get install edgetpu-compiler",  # 安装 Edge TPU 编译器
            ):
                subprocess.run(c if is_sudo_available() else c.replace("sudo ", ""), shell=True, check=True)  # 执行安装命令
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]  # 获取 Edge TPU 编译器版本

        LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")  # 日志记录导出信息
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # 设置 Edge TPU 模型文件名

        cmd = (
            "edgetpu_compiler "
            f'--out_dir "{Path(f).parent}" '  # 设置输出目录
            "--show_operations "  # 显示操作
            "--search_delegate "  # 搜索委托
            "--delegate_search_step 30 "  # 委托搜索步长
            "--timeout_sec 180 "  # 设置超时时间
            f'"{tflite_model}"'  # 输入 TFLite 模型
        )
        LOGGER.info(f"{prefix} running '{cmd}'")  # 日志记录运行命令
        subprocess.run(cmd, shell=True)  # 执行命令
        self._add_tflite_metadata(f)  # 添加 TFLite 元数据
        return f, None  # 返回文件路径

    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """YOLO TensorFlow.js export."""  # YOLO TensorFlow.js 导出
        check_requirements("tensorflowjs")  # 检查 TensorFlow.js 依赖
        if ARM64:
            # Fix error: `np.object` was a deprecated alias for the builtin `object` when exporting to TF.js on ARM64
            check_requirements("numpy==1.23.5")  # 检查 NumPy 版本
        import tensorflow as tf  # 导入 TensorFlow 库
        import tensorflowjs as tfjs  # noqa  # 导入 TensorFlow.js 库

        LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")  # 日志记录导出信息
        f = str(self.file).replace(self.file.suffix, "_web_model")  # 设置 JS 目录
        f_pb = str(self.file.with_suffix(".pb"))  # 设置 *.pb 路径

        gd = tf.Graph().as_graph_def()  # TF 图形定义
        with open(f_pb, "rb") as file:
            gd.ParseFromString(file.read())  # 从文件读取图形定义
        outputs = ",".join(gd_outputs(gd))  # 获取输出节点名称
        LOGGER.info(f"\n{prefix} output node names: {outputs}")  # 日志记录输出节点名称

        quantization = "--quantize_float16" if self.args.half else "--quantize_uint8" if self.args.int8 else ""  # 设置量化参数
        with spaces_in_path(f_pb) as fpb_, spaces_in_path(f) as f_:  # 导出器无法处理路径中的空格
            cmd = (
                "tensorflowjs_converter "
                f'--input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'  # 设置转换命令
            )
            LOGGER.info(f"{prefix} running '{cmd}'")  # 日志记录运行命令
            subprocess.run(cmd, shell=True)  # 执行命令

        if " " in f:
            LOGGER.warning(f"{prefix} WARNING ⚠️ your model may not work correctly with spaces in path '{f}'.")  # 日志记录路径中的空格警告

        # Add metadata
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # 添加元数据 YAML
        return f, None  # 返回文件路径

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):
        """YOLO RKNN model export."""  # YOLO RKNN 模型导出
        LOGGER.info(f"\n{prefix} starting export with rknn-toolkit2...")  # 日志记录导出信息

        check_requirements("rknn-toolkit2")  # 检查 RKNN 工具包依赖
        if IS_COLAB:
            # Prevent 'exit' from closing the notebook https://github.com/airockchip/rknn-toolkit2/issues/259
            import builtins

            builtins.exit = lambda: None  # 防止退出关闭笔记本

        from rknn.api import RKNN  # 导入 RKNN API

        f, _ = self.export_onnx()  # 导出 ONNX 模型
        export_path = Path(f"{Path(f).stem}_rknn_model")  # 设置导出路径
        export_path.mkdir(exist_ok=True)  # 创建导出目录

        rknn = RKNN(verbose=False)  # 创建 RKNN 实例
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=self.args.name)  # 设置均值和标准差
        rknn.load_onnx(model=f)  # 加载 ONNX 模型
        rknn.build(do_quantization=False)  # 构建 RKNN 模型，不进行量化
        f = f.replace(".onnx", f"-{self.args.name}.rknn")  # 设置 RKNN 文件名
        rknn.export_rknn(f"{export_path / f}")  # 导出 RKNN 模型
        yaml_save(export_path / "metadata.yaml", self.metadata)  # 添加元数据 YAML
        return export_path, None  # 返回导出路径

    @try_export
    def export_imx(self, prefix=colorstr("IMX:")):
        """YOLO IMX export."""  # YOLO IMX 导出
        gptq = False  # 设置 gptq 为 False，表示不使用梯度后训练量化
        assert LINUX, (  # 确保在 Linux 系统上运行
            "export only supported on Linux. See https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter"
        )
        if getattr(self.model, "end2end", False):  # 检查模型是否为端到端模型
            raise ValueError("IMX export is not supported for end2end models.")  # 不支持端到端模型的导出
        if "C2f" not in self.model.__str__():  # 检查模型字符串中是否包含 "C2f"
            raise ValueError("IMX export is only supported for YOLOv8n detection models")  # 仅支持 YOLOv8n 检测模型的导出
        check_requirements(("model-compression-toolkit==2.1.1", "sony-custom-layers==0.2.0", "tensorflow==2.12.0"))  # 检查依赖项
        check_requirements("imx500-converter[pt]==3.14.3")  # 检查 imx500-converter 的依赖项
    
        import model_compression_toolkit as mct  # 导入模型压缩工具包
        import onnx  # 导入 ONNX 库
        from sony_custom_layers.pytorch.object_detection.nms import multiclass_nms  # 从自定义层导入多类非极大值抑制
    
        LOGGER.info(f"\n{prefix} starting export with model_compression_toolkit {mct.__version__}...")  # 记录导出开始的信息
    
        try:
            out = subprocess.run(
                ["java", "--version"], check=True, capture_output=True
            )  # 检查 Java 版本，imx500-converter 需要 Java 17
            if "openjdk 17" not in str(out.stdout):  # 如果输出不包含 OpenJDK 17
                raise FileNotFoundError  # 抛出文件未找到异常
        except FileNotFoundError:
            c = ["apt", "install", "-y", "openjdk-17-jdk", "openjdk-17-jre"]  # 安装 OpenJDK 17
            if is_sudo_available():  # 检查是否有 sudo 权限
                c.insert(0, "sudo")  # 如果有，添加 sudo
            subprocess.run(c, check=True)  # 执行安装命令
    
        def representative_dataset_gen(dataloader=self.get_int8_calibration_dataloader(prefix)):  # 生成代表性数据集
            for batch in dataloader:  # 遍历数据加载器中的每个批次
                img = batch["img"]  # 获取图像
                img = img / 255.0  # 标准化图像
                yield [img]  # 生成标准化后的图像
    
        tpc = mct.get_target_platform_capabilities(  # 获取目标平台能力
            fw_name="pytorch", target_platform_name="imx500", target_platform_version="v1"
        )
    
        config = mct.core.CoreConfig(  # 配置核心设置
            mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=10),  # 设置混合精度配置
            quantization_config=mct.core.QuantizationConfig(concat_threshold_update=True),  # 设置量化配置
        )
    
        resource_utilization = mct.core.ResourceUtilization(weights_memory=3146176 * 0.76)  # 设置资源利用率
    
        quant_model = (  # 量化模型
            mct.gptq.pytorch_gradient_post_training_quantization(  # 执行基于梯度的后训练量化
                model=self.model,
                representative_data_gen=representative_dataset_gen,
                target_resource_utilization=resource_utilization,
                gptq_config=mct.gptq.get_pytorch_gptq_config(n_epochs=1000, use_hessian_based_weights=False),
                core_config=config,
                target_platform_capabilities=tpc,
            )[0]
            if gptq  # 如果 gptq 为 True
            else mct.ptq.pytorch_post_training_quantization(  # 执行后训练量化
                in_module=self.model,
                representative_data_gen=representative_dataset_gen,
                target_resource_utilization=resource_utilization,
                core_config=config,
                target_platform_capabilities=tpc,
            )[0]
        )
    
        class NMSWrapper(torch.nn.Module):  # 定义 NMS 包装类
            def __init__(self, model: torch.nn.Module, score_threshold: float = 0.001, iou_threshold: float = 0.7, max_detections: int = 300):
                """
                Wrapping PyTorch Module with multiclass_nms layer from sony_custom_layers.
    
                Args:
                    model (nn.Module): Model instance.  # 模型实例
                    score_threshold (float): Score threshold for non-maximum suppression.  # 非极大值抑制的分数阈值
                    iou_threshold (float): Intersection over union threshold for non-maximum suppression.  # 非极大值抑制的交并比阈值
                    max_detections (float): The number of detections to return.  # 返回的检测数量
                """
                super().__init__()  # 调用父类构造函数
                self.model = model  # 保存模型
                self.score_threshold = score_threshold  # 保存分数阈值
                self.iou_threshold = iou_threshold  # 保存交并比阈值
                self.max_detections = max_detections  # 保存最大检测数量
    
            def forward(self, images):  # 前向传播
                # model inference
                outputs = self.model(images)  # 模型推理
    
                boxes = outputs[0]  # 获取边界框
                scores = outputs[1]  # 获取分数
                nms = multiclass_nms(  # 执行多类非极大值抑制
                    boxes=boxes,
                    scores=scores,
                    score_threshold=self.score_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections,
                )
                return nms  # 返回 NMS 结果
    
        quant_model = NMSWrapper(  # 包装量化模型
            model=quant_model,
            score_threshold=self.args.conf or 0.001,  # 使用配置中的分数阈值
            iou_threshold=self.args.iou,  # 使用配置中的交并比阈值
            max_detections=self.args.max_det,  # 使用配置中的最大检测数量
        ).to(self.device)  # 移动到设备上
    
        f = Path(str(self.file).replace(self.file.suffix, "_imx_model"))  # 创建保存模型路径
        f.mkdir(exist_ok=True)  # 创建目录
        onnx_model = f / Path(str(self.file.name).replace(self.file.suffix, "_imx.onnx"))  # 设置 ONNX 模型路径
        mct.exporter.pytorch_export_model(  # 导出 PyTorch 模型
            model=quant_model, save_model_path=onnx_model, repr_dataset=representative_dataset_gen
        )
    
        model_onnx = onnx.load(onnx_model)  # 加载 ONNX 模型
        for k, v in self.metadata.items():  # 遍历元数据
            meta = model_onnx.metadata_props.add()  # 添加元数据属性
            meta.key, meta.value = k, str(v)  # 设置元数据键值对
    
        onnx.save(model_onnx, onnx_model)  # 保存 ONNX 模型
    
        subprocess.run(  # 执行 imxconv-pt 命令
            ["imxconv-pt", "-i", str(onnx_model), "-o", str(f), "--no-input-persistency", "--overwrite-output"],
            check=True,
        )
    
        # Needed for imx models.
        with open(f / "labels.txt", "w") as file:  # 创建标签文件
            file.writelines([f"{name}\n" for _, name in self.model.names.items()])  # 写入模型名称
    
        return f, None  # 返回模型路径和 None
    
    def _add_tflite_metadata(self, file):  # 添加 TFLite 模型的元数据
        """Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata."""  # 根据文档添加元数据
        import flatbuffers  # 导入 flatbuffers 库
    
        try:
            # TFLite Support bug https://github.com/tensorflow/tflite-support/issues/954#issuecomment-2108570845
            from tensorflow_lite_support.metadata import metadata_schema_py_generated as schema  # noqa
            from tensorflow_lite_support.metadata.python import metadata  # noqa
        except ImportError:  # ARM64 系统可能没有 'tensorflow_lite_support' 包
            from tflite_support import metadata  # noqa
            from tflite_support import metadata_schema_py_generated as schema  # noqa
    
        # Create model info
        model_meta = schema.ModelMetadataT()  # 创建模型元数据
        model_meta.name = self.metadata["description"]  # 设置模型名称
        model_meta.version = self.metadata["version"]  # 设置模型版本
        model_meta.author = self.metadata["author"]  # 设置模型作者
        model_meta.license = self.metadata["license"]  # 设置模型许可证
    
        # Label file
        tmp_file = Path(file).parent / "temp_meta.txt"  # 创建临时文件
        with open(tmp_file, "w") as f:  # 打开临时文件以写入
            f.write(str(self.metadata))  # 写入元数据
    
        label_file = schema.AssociatedFileT()  # 创建关联文件
        label_file.name = tmp_file.name  # 设置文件名
        label_file.type = schema.AssociatedFileType.TENSOR_AXIS_LABELS  # 设置文件类型
    
        # Create input info
        input_meta = schema.TensorMetadataT()  # 创建输入元数据
        input_meta.name = "image"  # 设置输入名称
        input_meta.description = "Input image to be detected."  # 设置输入描述
        input_meta.content = schema.ContentT()  # 创建内容对象
        input_meta.content.contentProperties = schema.ImagePropertiesT()  # 设置内容属性
        input_meta.content.contentProperties.colorSpace = schema.ColorSpaceType.RGB  # 设置颜色空间
        input_meta.content.contentPropertiesType = schema.ContentProperties.ImageProperties  # 设置内容属性类型
    
        # Create output info
        output1 = schema.TensorMetadataT()  # 创建输出元数据
        output1.name = "output"  # 设置输出名称
        output1.description = "Coordinates of detected objects, class labels, and confidence score"  # 设置输出描述
        output1.associatedFiles = [label_file]  # 关联标签文件
        if self.model.task == "segment":  # 如果模型任务是分割
            output2 = schema.TensorMetadataT()  # 创建第二个输出元数据
            output2.name = "output"  # 设置输出名称
            output2.description = "Mask protos"  # 设置输出描述
            output2.associatedFiles = [label_file]  # 关联标签文件
    
        # Create subgraph info
        subgraph = schema.SubGraphMetadataT()  # 创建子图元数据
        subgraph.inputTensorMetadata = [input_meta]  # 设置输入张量元数据
        subgraph.outputTensorMetadata = [output1, output2] if self.model.task == "segment" else [output1]  # 设置输出张量元数据
        model_meta.subgraphMetadata = [subgraph]  # 设置模型元数据的子图元数据
    
        b = flatbuffers.Builder(0)  # 创建 flatbuffers 构建器
        b.Finish(model_meta.Pack(b), metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)  # 完成构建
        metadata_buf = b.Output()  # 获取输出缓冲区
    
        populator = metadata.MetadataPopulator.with_model_file(str(file))  # 创建元数据填充器
        populator.load_metadata_buffer(metadata_buf)  # 加载元数据缓冲区
        populator.load_associated_files([str(tmp_file)])  # 加载关联文件
        populator.populate()  # 填充元数据
        tmp_file.unlink()  # 删除临时文件
    
    def _pipeline_coreml(self, model, weights_dir=None, prefix=colorstr("CoreML Pipeline:")):  # YOLO CoreML 管道
        """YOLO CoreML pipeline."""  # YOLO CoreML 管道
        import coremltools as ct  # noqa
    
        LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")  # 记录管道开始的信息
        _, _, h, w = list(self.im.shape)  # BCHW
    
        # Output shapes
        spec = model.get_spec()  # 获取模型规格
        out0, out1 = iter(spec.description.output)  # 获取输出
        if MACOS:
            from PIL import Image  # 导入图像处理库
    
            img = Image.new("RGB", (w, h))  # 创建新的 RGB 图像
            out = model.predict({"image": img})  # 进行预测
            out0_shape = out[out0.name].shape  # 获取第一个输出的形状
            out1_shape = out[out1.name].shape  # 获取第二个输出的形状
        else:  # linux 和 windows 无法运行 model.predict()，从 PyTorch 模型输出 y 获取大小
            out0_shape = self.output_shape[2], self.output_shape[1] - 4  # 获取第一个输出的形状
            out1_shape = self.output_shape[2], 4  # 获取第二个输出的形状
    
        # Checks
        names = self.metadata["names"]  # 获取模型名称
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height  # 获取输入图像的宽高
        _, nc = out0_shape  # 获取锚点数量和类别数量
        assert len(names) == nc, f"{len(names)} names found for nc={nc}"  # 检查名称数量与类别数量是否一致
    
        # Define output shapes (missing)
        out0.type.multiArrayType.shape[:] = out0_shape  # 设置第一个输出的形状
        out1.type.multiArrayType.shape[:] = out1_shape  # 设置第二个输出的形状
    
        # Model from spec
        model = ct.models.MLModel(spec, weights_dir=weights_dir)  # 创建 CoreML 模型
    
        # 3. Create NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()  # 创建 NMS protobuf 模型
        nms_spec.specificationVersion = 5  # 设置规格版本
        for i in range(2):
            decoder_output = model._spec.description.output[i].SerializeToString()  # 序列化输出
            nms_spec.description.input.add()  # 添加输入
            nms_spec.description.input[i].ParseFromString(decoder_output)  # 解析输入
            nms_spec.description.output.add()  # 添加输出
            nms_spec.description.output[i].ParseFromString(decoder_output)  # 解析输出
    
        nms_spec.description.output[0].name = "confidence"  # 设置第一个输出名称为 confidence
        nms_spec.description.output[1].name = "coordinates"  # 设置第二个输出名称为 coordinates
    
        output_sizes = [nc, 4]  # 定义输出大小
        for i in range(2):
            ma_type = nms_spec.description.output[i].type.multiArrayType  # 获取多维数组类型
            ma_type.shapeRange.sizeRanges.add()  # 添加大小范围
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0  # 设置下界
            ma_type.shapeRange.sizeRanges[0].upperBound = -1  # 设置上界
            ma_type.shapeRange.sizeRanges.add()  # 添加大小范围
            ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]  # 设置下界
            ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]  # 设置上界
            del ma_type.shape[:]  # 删除原有形状
    
        nms = nms_spec.nonMaximumSuppression  # 获取非极大值抑制对象
        nms.confidenceInputFeatureName = out0.name  # 设置置信度输入特征名称
        nms.coordinatesInputFeatureName = out1.name  # 设置坐标输入特征名称
        nms.confidenceOutputFeatureName = "confidence"  # 设置置信度输出特征名称
        nms.coordinatesOutputFeatureName = "coordinates"  # 设置坐标输出特征名称
        nms.iouThresholdInputFeatureName = "iouThreshold"  # 设置 IoU 阈值输入特征名称
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"  # 设置置信度阈值输入特征名称
        nms.iouThreshold = self.args.iou  # 设置 IoU 阈值
        nms.confidenceThreshold = self.args.conf  # 设置置信度阈值
        nms.pickTop.perClass = True  # 每个类别选择前 N 个
        nms.stringClassLabels.vector.extend(names.values())  # 添加类别标签
        nms_model = ct.models.MLModel(nms_spec)  # 创建 NMS 模型
    
        # 4. Pipeline models together
        pipeline = ct.models.pipeline.Pipeline(  # 创建管道模型
            input_features=[
                ("image", ct.models.datatypes.Array(3, ny, nx)),  # 输入特征
                ("iouThreshold", ct.models.datatypes.Double()),  # IoU 阈值特征
                ("confidenceThreshold", ct.models.datatypes.Double()),  # 置信度阈值特征
            ],
            output_features=["confidence", "coordinates"],  # 输出特征
        )
        pipeline.add_model(model)  # 添加模型
        pipeline.add_model(nms_model)  # 添加 NMS 模型
    
        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())  # 解析输入特征
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())  # 解析输出特征
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())  # 解析输出特征
    
        # Update metadata
        pipeline.spec.specificationVersion = 5  # 更新规格版本
        pipeline.spec.description.metadata.userDefined.update(  # 更新用户定义的元数据
            {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
        )
    
        # Save the model
        model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)  # 保存模型
        model.input_description["image"] = "Input image"  # 设置输入描述
        model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"  # 设置 IoU 阈值描述
        model.input_description["confidenceThreshold"] = (  # 设置置信度阈值描述
            f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
        )
        model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'  # 设置置信度输出描述
        model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"  # 设置坐标输出描述
        LOGGER.info(f"{prefix} pipeline success")  # 记录管道成功的信息
        return model  # 返回模型
    
    def add_callback(self, event: str, callback):  # 添加回调函数
        """Appends the given callback."""  # 附加给定的回调
        self.callbacks[event].append(callback)  # 将回调添加到事件列表中
    
    def run_callbacks(self, event: str):  # 运行指定事件的所有回调
        """Execute all callbacks for a given event."""  # 执行给定事件的所有回调
        for callback in self.callbacks.get(event, []):  # 遍历事件的所有回调
            callback(self)  # 执行回调

class IOSDetectModel(torch.nn.Module):  # 定义 IOSDetectModel 类，继承自 torch.nn.Module
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""  # 为 Apple iOS CoreML 导出包装 Ultralytics YOLO 模型

    def __init__(self, model, im):  # 初始化 IOSDetectModel 类，接受一个 YOLO 模型和示例图像
        """Initialize the IOSDetectModel class with a YOLO model and example image."""  # 使用 YOLO 模型和示例图像初始化 IOSDetectModel 类
        super().__init__()  # 调用父类构造函数
        _, _, h, w = im.shape  # 获取图像的形状，分别为 batch, channel, height, width
        self.model = model  # 保存传入的 YOLO 模型
        self.nc = len(model.names)  # 获取类别数量
        if w == h:  # 如果宽度和高度相等
            self.normalize = 1.0 / w  # 设置归一化因子为宽度的倒数（标量）
        else:  # 如果宽度和高度不相等
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # 设置归一化因子为张量（广播方式，较慢，较小）

    def forward(self, x):  # 前向传播方法
        """Normalize predictions of object detection model with input size-dependent factors."""  # 使用输入大小相关因子来归一化目标检测模型的预测
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)  # 获取模型的输出，并进行转置和拆分
        return cls, xywh * self.normalize  # 返回类别和归一化后的坐标（置信度 (3780, 80)，坐标 (3780, 4)）


class NMSModel(torch.nn.Module):  # 定义 NMSModel 类，继承自 torch.nn.Module
    """Model wrapper with embedded NMS for Detect, Segment, Pose and OBB."""  # 包含嵌入式 NMS 的模型包装器，用于检测、分割、姿态和 OBB

    def __init__(self, model, args):  # 初始化 NMSModel 类，接受一个模型和参数
        """
        Initialize the NMSModel.

        Args:
            model (torch.nn.module): The model to wrap with NMS postprocessing.  # 要包装的模型，带有 NMS 后处理
            args (Namespace): The export arguments.  # 导出参数
        """
        super().__init__()  # 调用父类构造函数
        self.model = model  # 保存传入的模型
        self.args = args  # 保存传入的参数
        self.obb = model.task == "obb"  # 检查模型任务是否为 OBB
        self.is_tf = self.args.format in frozenset({"saved_model", "tflite", "tfjs"})  # 检查格式是否为 TensorFlow 相关格式

    def forward(self, x):  # 前向传播方法
        """
        Performs inference with NMS post-processing. Supports Detect, Segment, OBB and Pose.

        Args:
            x (torch.Tensor): The preprocessed tensor with shape (N, 3, H, W).  # 预处理后的张量，形状为 (N, 3, H, W)

        Returns:
            out (torch.Tensor): The post-processed results with shape (N, max_det, 4 + 2 + extra_shape).  # 后处理结果，形状为 (N, max_det, 4 + 2 + extra_shape)
        """
        from functools import partial  # 导入 partial 函数

        from torchvision.ops import nms  # 从 torchvision 导入 nms 函数

        preds = self.model(x)  # 执行模型推理
        pred = preds[0] if isinstance(preds, tuple) else preds  # 获取预测结果
        pred = pred.transpose(-1, -2)  # 将形状从 (1, 84, 6300) 转换为 (1, 6300, 84)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))  # 计算额外的形状，来自 Segment、OBB、Pose
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)  # 拆分预测结果为边界框、分数和额外信息
        scores, classes = scores.max(dim=-1)  # 获取每个框的最大分数和对应的类别
        self.args.max_det = min(pred.shape[1], self.args.max_det)  # 确保 max_det 不超过预测数量
        # (N, max_det, 4 coords + 1 class score + 1 class label + extra_shape).
        out = torch.zeros(  # 初始化输出张量
            boxes.shape[0],  # N
            self.args.max_det,  # max_det
            boxes.shape[-1] + 2 + extra_shape,  # 4 个坐标 + 1 个类分数 + 1 个类标签 + 额外形状
            device=boxes.device,  # 使用相同的设备
            dtype=boxes.dtype,  # 使用相同的数据类型
        )
        for i, (box, cls, score, extra) in enumerate(zip(boxes, classes, scores, extras)):  # 遍历每个框、类别、分数和额外信息
            mask = score > self.args.conf  # 创建掩码，筛选出分数大于阈值的框
            if self.is_tf:  # 如果是 TensorFlow 格式
                # TFLite GatherND error if mask is empty
                score *= mask  # 将分数乘以掩码
                # Explicit length otherwise reshape error, hardcoded to `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices  # 获取前 N 个分数的索引
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]  # 根据掩码筛选框、分数、类别和额外信息
            if not self.obb:  # 如果不是 OBB
                box = xywh2xyxy(box)  # 将坐标从 xywh 转换为 xyxy
                if self.is_tf:  # 如果是 TensorFlow 格式
                    # TFlite bug returns less boxes
                    box = torch.nn.functional.pad(box, (0, 0, 0, mask.shape[0] - box.shape[0]))  # 填充框以匹配最大数量
            nmsbox = box.clone()  # 克隆框以进行 NMS
            # `8` is the minimum value experimented to get correct NMS results for obb
            multiplier = 8 if self.obb else 1  # 设置乘数，OBB 使用 8
            # Normalize boxes for NMS since large values for class offset causes issue with int8 quantization
            if self.args.format == "tflite":  # 如果格式为 TFLite
                nmsbox *= multiplier  # 乘以乘数
            else:
                nmsbox = multiplier * nmsbox / torch.tensor(x.shape[2:], device=box.device, dtype=box.dtype).max()  # 归一化框
            if not self.args.agnostic_nms:  # 如果不是类别无关的 NMS
                end = 2 if self.obb else 4  # 设置结束索引
                # fully explicit expansion otherwise reshape error
                # large max_wh causes issues when quantizing
                cls_offset = cls.reshape(-1, 1).expand(nmsbox.shape[0], end)  # 扩展类别偏移量
                offbox = nmsbox[:, :end] + cls_offset * multiplier  # 计算偏移框
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)  # 拼接偏移框和 NMS 框
            nms_fn = (  # 选择 NMS 函数
                partial(
                    nms_rotated,  # 使用旋转 NMS
                    use_triu=not (
                        self.is_tf
                        or (self.args.opset or 14) < 14
                        or (self.args.format == "openvino" and self.args.int8)  # OpenVINO int8 error with triu
                    ),
                )
                if self.obb  # 如果是 OBB
                else nms  # 否则使用普通 NMS
            )
            keep = nms_fn(  # 执行 NMS
                torch.cat([nmsbox, extra], dim=-1) if self.obb else nmsbox,  # 如果是 OBB，拼接额外信息
                score,  # 使用分数
                self.args.iou,  # 使用 IoU 阈值
            )[: self.args.max_det]  # 保留前 max_det 个结果
            dets = torch.cat(  # 拼接检测结果
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            # Zero-pad to max_det size to avoid reshape error
            pad = (0, 0, 0, self.args.max_det - dets.shape[0])  # 计算填充大小
            out[i] = torch.nn.functional.pad(dets, pad)  # 填充检测结果
        return (out, preds[1]) if self.model.task == "segment" else out  # 如果任务是分割，返回 (out, preds[1])，否则返回 out