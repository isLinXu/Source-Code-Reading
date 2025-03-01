# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import ast
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import ARM64, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, PYTHON_VERSION, ROOT, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml, is_rockchip
from ultralytics.utils.downloads import attempt_download_asset, is_url


# def check_class_names(names):
#     """
#     Check class names.

#     Map imagenet class codes to human-readable names if required. Convert lists to dicts.
#     """
#     if isinstance(names, list):  # names is a list
#         names = dict(enumerate(names))  # convert to dict
#     if isinstance(names, dict):
#         # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
#         names = {int(k): str(v) for k, v in names.items()}
#         n = len(names)
#         if max(names.keys()) >= n:
#             raise KeyError(
#                 f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
#                 f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
#             )
#         if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
#             names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
#             names = {k: names_map[v] for k, v in names.items()}
#     return names


# def default_class_names(data=None):
#     """Applies default class names to an input YAML file or returns numerical class names."""
#     if data:
#         try:
#             return yaml_load(check_yaml(data))["names"]
#         except Exception:
#             pass
#     return {i: f"class{i}" for i in range(999)}  # return default if above errors


# class AutoBackend(nn.Module):
#     """
#     Handles dynamic backend selection for running inference using Ultralytics YOLO models.

#     The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
#     range of formats, each with specific naming conventions as outlined below:

#         Supported Formats and Naming Conventions:
#             | Format                | File Suffix       |
#             | --------------------- | ----------------- |
#             | PyTorch               | *.pt              |
#             | TorchScript           | *.torchscript     |
#             | ONNX Runtime          | *.onnx            |
#             | ONNX OpenCV DNN       | *.onnx (dnn=True) |
#             | OpenVINO              | *openvino_model/  |
#             | CoreML                | *.mlpackage       |
#             | TensorRT              | *.engine          |
#             | TensorFlow SavedModel | *_saved_model/    |
#             | TensorFlow GraphDef   | *.pb              |
#             | TensorFlow Lite       | *.tflite          |
#             | TensorFlow Edge TPU   | *_edgetpu.tflite  |
#             | PaddlePaddle          | *_paddle_model/   |
#             | MNN                   | *.mnn             |
#             | NCNN                  | *_ncnn_model/     |
#             | IMX                   | *_imx_model/      |
#             | RKNN                  | *_rknn_model/     |

#     This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
#     models across various platforms.
#     """

#     @torch.no_grad()
#     def __init__(
#         self,
#         weights="yolo11n.pt",
#         device=torch.device("cpu"),
#         dnn=False,
#         data=None,
#         fp16=False,
#         batch=1,
#         fuse=True,
#         verbose=True,
#     ):
#         """
#         Initialize the AutoBackend for inference.

#         Args:
#             weights (str | torch.nn.Module): Path to the model weights file or a module instance. Defaults to 'yolo11n.pt'.
#             device (torch.device): Device to run the model on. Defaults to CPU.
#             dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
#             data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
#             fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
#             batch (int): Batch-size to assume for inference.
#             fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
#             verbose (bool): Enable verbose logging. Defaults to True.
#         """
#         super().__init__()
#         w = str(weights[0] if isinstance(weights, list) else weights)
#         nn_module = isinstance(weights, torch.nn.Module)
#         (
#             pt,
#             jit,
#             onnx,
#             xml,
#             engine,
#             coreml,
#             saved_model,
#             pb,
#             tflite,
#             edgetpu,
#             tfjs,
#             paddle,
#             mnn,
#             ncnn,
#             imx,
#             rknn,
#             triton,
#         ) = self._model_type(w)
#         fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
#         nhwc = coreml or saved_model or pb or tflite or edgetpu or rknn  # BHWC formats (vs torch BCWH)
#         stride = 32  # default stride
#         end2end = False  # default end2end
#         model, metadata, task = None, None, None

#         # Set device
#         cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
#         if cuda and not any([nn_module, pt, jit, engine, onnx, paddle]):  # GPU dataloader formats
#             device = torch.device("cpu")
#             cuda = False

#         # Download if not local
#         if not (pt or triton or nn_module):
#             w = attempt_download_asset(w)

#         # In-memory PyTorch model
#         if nn_module:
#             model = weights.to(device)
#             if fuse:
#                 model = model.fuse(verbose=verbose)
#             if hasattr(model, "kpt_shape"):
#                 kpt_shape = model.kpt_shape  # pose-only
#             stride = max(int(model.stride.max()), 32)  # model stride
#             names = model.module.names if hasattr(model, "module") else model.names  # get class names
#             model.half() if fp16 else model.float()
#             self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
#             pt = True

#         # PyTorch
#         elif pt:
#             from ultralytics.nn.tasks import attempt_load_weights

#             model = attempt_load_weights(
#                 weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
#             )
#             if hasattr(model, "kpt_shape"):
#                 kpt_shape = model.kpt_shape  # pose-only
#             stride = max(int(model.stride.max()), 32)  # model stride
#             names = model.module.names if hasattr(model, "module") else model.names  # get class names
#             model.half() if fp16 else model.float()
#             self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

#         # TorchScript
#         elif jit:
#             LOGGER.info(f"Loading {w} for TorchScript inference...")
#             extra_files = {"config.txt": ""}  # model metadata
#             model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
#             model.half() if fp16 else model.float()
#             if extra_files["config.txt"]:  # load metadata dict
#                 metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

#         # ONNX OpenCV DNN
#         elif dnn:
#             LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
#             check_requirements("opencv-python>=4.5.4")
#             net = cv2.dnn.readNetFromONNX(w)

#         # ONNX Runtime and IMX
#         elif onnx or imx:
#             LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
#             check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
#             if IS_RASPBERRYPI or IS_JETSON:
#                 # Fix 'numpy.linalg._umath_linalg' has no attribute '_ilp64' for TF SavedModel on RPi and Jetson
#                 check_requirements("numpy==1.23.5")
#             import onnxruntime

#             providers = ["CPUExecutionProvider"]
#             if cuda and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
#                 providers.insert(0, "CUDAExecutionProvider")
#             elif cuda:  # Only log warning if CUDA was requested but unavailable
#                 LOGGER.warning("WARNING ⚠️ Failed to start ONNX Runtime with CUDA. Using CPU...")
#                 device = torch.device("cpu")
#                 cuda = False
#             LOGGER.info(f"Using ONNX Runtime {providers[0]}")
#             if onnx:
#                 session = onnxruntime.InferenceSession(w, providers=providers)
#             else:
#                 check_requirements(
#                     ["model-compression-toolkit==2.1.1", "sony-custom-layers[torch]==0.2.0", "onnxruntime-extensions"]
#                 )
#                 w = next(Path(w).glob("*.onnx"))
#                 LOGGER.info(f"Loading {w} for ONNX IMX inference...")
#                 import mct_quantizers as mctq
#                 from sony_custom_layers.pytorch.object_detection import nms_ort  # noqa

#                 session = onnxruntime.InferenceSession(
#                     w, mctq.get_ort_session_options(), providers=["CPUExecutionProvider"]
#                 )
#                 task = "detect"

#             output_names = [x.name for x in session.get_outputs()]
#             metadata = session.get_modelmeta().custom_metadata_map
#             dynamic = isinstance(session.get_outputs()[0].shape[0], str)
#             fp16 = True if "float16" in session.get_inputs()[0].type else False
#             if not dynamic:
#                 io = session.io_binding()
#                 bindings = []
#                 for output in session.get_outputs():
#                     out_fp16 = "float16" in output.type
#                     y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(device)
#                     io.bind_output(
#                         name=output.name,
#                         device_type=device.type,
#                         device_id=device.index if cuda else 0,
#                         element_type=np.float16 if out_fp16 else np.float32,
#                         shape=tuple(y_tensor.shape),
#                         buffer_ptr=y_tensor.data_ptr(),
#                     )
#                     bindings.append(y_tensor)

#         # OpenVINO
#         elif xml:
#             LOGGER.info(f"Loading {w} for OpenVINO inference...")
#             check_requirements("openvino>=2024.0.0")
#             import openvino as ov

#             core = ov.Core()
#             w = Path(w)
#             if not w.is_file():  # if not *.xml
#                 w = next(w.glob("*.xml"))  # get *.xml file from *_openvino_model dir
#             ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
#             if ov_model.get_parameters()[0].get_layout().empty:
#                 ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

#             # OpenVINO inference modes are 'LATENCY', 'THROUGHPUT' (not recommended), or 'CUMULATIVE_THROUGHPUT'
#             inference_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 else "LATENCY"
#             LOGGER.info(f"Using OpenVINO {inference_mode} mode for batch={batch} inference...")
#             ov_compiled_model = core.compile_model(
#                 ov_model,
#                 device_name="AUTO",  # AUTO selects best available device, do not modify
#                 config={"PERFORMANCE_HINT": inference_mode},
#             )
#             input_name = ov_compiled_model.input().get_any_name()
#             metadata = w.parent / "metadata.yaml"

#         # TensorRT
#         elif engine:
#             LOGGER.info(f"Loading {w} for TensorRT inference...")

#             if IS_JETSON and PYTHON_VERSION <= "3.8.0":
#                 # fix error: `np.bool` was a deprecated alias for the builtin `bool` for JetPack 4 with Python <= 3.8.0
#                 check_requirements("numpy==1.23.5")

#             try:
#                 import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
#             except ImportError:
#                 if LINUX:
#                     check_requirements("tensorrt>7.0.0,!=10.1.0")
#                 import tensorrt as trt  # noqa
#             check_version(trt.__version__, ">=7.0.0", hard=True)
#             check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
#             if device.type == "cpu":
#                 device = torch.device("cuda:0")
#             Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
#             logger = trt.Logger(trt.Logger.INFO)
#             # Read file
#             with open(w, "rb") as f, trt.Runtime(logger) as runtime:
#                 try:
#                     meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
#                     metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
#                 except UnicodeDecodeError:
#                     f.seek(0)  # engine file may lack embedded Ultralytics metadata
#                 dla = metadata.get("dla", None)
#                 if dla is not None:
#                     runtime.DLA_core = int(dla)
#                 model = runtime.deserialize_cuda_engine(f.read())  # read engine

#             # Model context
#             try:
#                 context = model.create_execution_context()
#             except Exception as e:  # model is None
#                 LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
#                 raise e

#             bindings = OrderedDict()
#             output_names = []
#             fp16 = False  # default updated below
#             dynamic = False
#             is_trt10 = not hasattr(model, "num_bindings")
#             num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
#             for i in num:
#                 if is_trt10:
#                     name = model.get_tensor_name(i)
#                     dtype = trt.nptype(model.get_tensor_dtype(name))
#                     is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
#                     if is_input:
#                         if -1 in tuple(model.get_tensor_shape(name)):
#                             dynamic = True
#                             context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
#                         if dtype == np.float16:
#                             fp16 = True
#                     else:
#                         output_names.append(name)
#                     shape = tuple(context.get_tensor_shape(name))
#                 else:  # TensorRT < 10.0
#                     name = model.get_binding_name(i)
#                     dtype = trt.nptype(model.get_binding_dtype(i))
#                     is_input = model.binding_is_input(i)
#                     if model.binding_is_input(i):
#                         if -1 in tuple(model.get_binding_shape(i)):  # dynamic
#                             dynamic = True
#                             context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
#                         if dtype == np.float16:
#                             fp16 = True
#                     else:
#                         output_names.append(name)
#                     shape = tuple(context.get_binding_shape(i))
#                 im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
#                 bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
#             binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
#             batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size

#         # CoreML
#         elif coreml:
#             LOGGER.info(f"Loading {w} for CoreML inference...")
#             import coremltools as ct

#             model = ct.models.MLModel(w)
#             metadata = dict(model.user_defined_metadata)

#         # TF SavedModel
#         elif saved_model:
#             LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
#             import tensorflow as tf

#             keras = False  # assume TF1 saved_model
#             model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
#             metadata = Path(w) / "metadata.yaml"

#         # TF GraphDef
#         elif pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
#             LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
#             import tensorflow as tf

#             from ultralytics.engine.exporter import gd_outputs

#             def wrap_frozen_graph(gd, inputs, outputs):
#                 """Wrap frozen graphs for deployment."""
#                 x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
#                 ge = x.graph.as_graph_element
#                 return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

#             gd = tf.Graph().as_graph_def()  # TF GraphDef
#             with open(w, "rb") as f:
#                 gd.ParseFromString(f.read())
#             frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
#             try:  # find metadata in SavedModel alongside GraphDef
#                 metadata = next(Path(w).resolve().parent.rglob(f"{Path(w).stem}_saved_model*/metadata.yaml"))
#             except StopIteration:
#                 pass

#         # TFLite or TFLite Edge TPU
#         elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
#             try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
#                 from tflite_runtime.interpreter import Interpreter, load_delegate
#             except ImportError:
#                 import tensorflow as tf

#                 Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
#             if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
#                 device = device[3:] if str(device).startswith("tpu") else ":0"
#                 LOGGER.info(f"Loading {w} on device {device[1:]} for TensorFlow Lite Edge TPU inference...")
#                 delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
#                     platform.system()
#                 ]
#                 interpreter = Interpreter(
#                     model_path=w,
#                     experimental_delegates=[load_delegate(delegate, options={"device": device})],
#                 )
#                 device = "cpu"  # Required, otherwise PyTorch will try to use the wrong device
#             else:  # TFLite
#                 LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
#                 interpreter = Interpreter(model_path=w)  # load TFLite model
#             interpreter.allocate_tensors()  # allocate
#             input_details = interpreter.get_input_details()  # inputs
#             output_details = interpreter.get_output_details()  # outputs
#             # Load metadata
#             try:
#                 with zipfile.ZipFile(w, "r") as model:
#                     meta_file = model.namelist()[0]
#                     metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))
#             except zipfile.BadZipFile:
#                 pass

#         # TF.js
#         elif tfjs:
#             raise NotImplementedError("YOLOv8 TF.js inference is not currently supported.")

#         # PaddlePaddle
#         elif paddle:
#             LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
#             check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
#             import paddle.inference as pdi  # noqa

#             w = Path(w)
#             if not w.is_file():  # if not *.pdmodel
#                 w = next(w.rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
#             config = pdi.Config(str(w), str(w.with_suffix(".pdiparams")))
#             if cuda:
#                 config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
#             predictor = pdi.create_predictor(config)
#             input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
#             output_names = predictor.get_output_names()
#             metadata = w.parents[1] / "metadata.yaml"

#         # MNN
#         elif mnn:
#             LOGGER.info(f"Loading {w} for MNN inference...")
#             check_requirements("MNN")  # requires MNN
#             import os

#             import MNN

#             config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
#             rt = MNN.nn.create_runtime_manager((config,))
#             net = MNN.nn.load_module_from_file(w, [], [], runtime_manager=rt, rearrange=True)

#             def torch_to_mnn(x):
#                 return MNN.expr.const(x.data_ptr(), x.shape)

#             metadata = json.loads(net.get_info()["bizCode"])

#         # NCNN
#         elif ncnn:
#             LOGGER.info(f"Loading {w} for NCNN inference...")
#             check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn")  # requires NCNN
#             import ncnn as pyncnn

#             net = pyncnn.Net()
#             net.opt.use_vulkan_compute = cuda
#             w = Path(w)
#             if not w.is_file():  # if not *.param
#                 w = next(w.glob("*.param"))  # get *.param file from *_ncnn_model dir
#             net.load_param(str(w))
#             net.load_model(str(w.with_suffix(".bin")))
#             metadata = w.parent / "metadata.yaml"

#         # NVIDIA Triton Inference Server
#         elif triton:
#             check_requirements("tritonclient[all]")
#             from ultralytics.utils.triton import TritonRemoteModel

#             model = TritonRemoteModel(w)
#             metadata = model.metadata

def check_class_names(names):
    """
    Check class names.  # 检查类名

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.  # 如果需要，将imagenet类代码映射到可读的名称。将列表转换为字典。
    """
    if isinstance(names, list):  # names is a list  # 如果names是一个列表
        names = dict(enumerate(names))  # convert to dict  # 转换为字典
    if isinstance(names, dict):  # 如果names是一个字典
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'  # 将1) 字符串键转换为整数，例如'0'转换为0，非字符串值转换为字符串，例如True转换为'True'
        names = {int(k): str(v) for k, v in names.items()}  # 将键转换为整数，值转换为字符串
        n = len(names)  # 获取字典的长度
        if max(names.keys()) >= n:  # 如果最大键值大于等于字典长度
            raise KeyError(  # 引发KeyError异常
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "  # "{n}-类数据集需要类索引0-{n - 1}，但您有无效的类索引"
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."  # "{min(names.keys())}-{max(names.keys())} 在您的数据集YAML中定义。"
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'  # imagenet类代码，例如'n01440764'
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names  # 可读名称
            names = {k: names_map[v] for k, v in names.items()}  # 将类代码映射到可读名称
    return names  # 返回处理后的类名

def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names.  # 将默认类名应用于输入YAML文件或返回数字类名。"""
    if data:  # 如果提供了数据
        try:
            return yaml_load(check_yaml(data))["names"]  # 尝试加载YAML文件并返回类名
        except Exception:  # 如果发生异常
            pass  # 忽略异常
    return {i: f"class{i}" for i in range(999)}  # return default if above errors  # 如果发生上述错误，则返回默认类名

class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models.  # 处理使用Ultralytics YOLO模型进行推理的动态后端选择。

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide  # AutoBackend类旨在为各种推理引擎提供抽象层。它支持广泛的
    range of formats, each with specific naming conventions as outlined below:  # 格式，每种格式都有特定的命名约定，如下所示：

        Supported Formats and Naming Conventions:  # 支持的格式和命名约定：
            | Format                | File Suffix       |  # | 格式                | 文件后缀       |
            | --------------------- | ----------------- |  # | --------------------- | ----------------- |
            | PyTorch               | *.pt              |  # | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |  # | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |  # | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |  # | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |  # | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |  # | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |  # | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |  # | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |  # | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Lite       | *.tflite          |  # | TensorFlow Lite       | *.tflite          |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |  # | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |  # | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |  # | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |  # | NCNN                  | *_ncnn_model/     |
            | IMX                   | *_imx_model/      |  # | IMX                   | *_imx_model/      |
            | RKNN                  | *_rknn_model/     |  # | RKNN                  | *_rknn_model/     |

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy  # 此类提供基于输入模型格式的动态后端切换功能，使得在不同平台上部署
    models across various platforms.  # 模型变得更加容易。
    """

    @torch.no_grad()  # 不计算梯度
    def __init__(  # 构造函数
        self,
        weights="yolo11n.pt",  # 权重文件路径，默认为'yolo11n.pt'
        device=torch.device("cpu"),  # 运行模型的设备，默认为CPU
        dnn=False,  # 是否使用OpenCV DNN模块进行ONNX推理，默认为False
        data=None,  # 额外的数据.yaml文件路径，包含类名，可选
        fp16=False,  # 是否启用半精度推理，仅在特定后端支持，默认为False
        batch=1,  # 假设的推理批次大小
        fuse=True,  # 是否融合Conv2D + BatchNorm层以优化，默认为True
        verbose=True,  # 是否启用详细日志，默认为True
    ):
        """
        Initialize the AutoBackend for inference.  # 初始化AutoBackend进行推理。

        Args:  # 参数：
            weights (str | torch.nn.Module): Path to the model weights file or a module instance. Defaults to 'yolo11n.pt'.  # 权重（str | torch.nn.Module）：模型权重文件的路径或模块实例。默认为'yolo11n.pt'。
            device (torch.device): Device to run the model on. Defaults to CPU.  # 设备（torch.device）：运行模型的设备。默认为CPU。
            dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.  # dnn（bool）：使用OpenCV DNN模块进行ONNX推理。默认为False。
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.  # data（str | Path | 可选）：包含类名的额外data.yaml文件的路径。可选。
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.  # fp16（bool）：启用半精度推理。仅在特定后端支持。默认为False。
            batch (int): Batch-size to assume for inference.  # batch（int）：假设的推理批次大小。
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.  # fuse（bool）：融合Conv2D + BatchNorm层以优化。默认为True。
            verbose (bool): Enable verbose logging. Defaults to True.  # verbose（bool）：启用详细日志。默认为True。
        """
        super().__init__()  # 调用父类构造函数
        w = str(weights[0] if isinstance(weights, list) else weights)  # 如果weights是列表，取第一个元素，否则直接使用weights
        nn_module = isinstance(weights, torch.nn.Module)  # 检查weights是否为torch.nn.Module类型
        (
            pt,  # PyTorch模型
            jit,  # TorchScript模型
            onnx,  # ONNX模型
            xml,  # OpenVINO模型
            engine,  # TensorRT模型
            coreml,  # CoreML模型
            saved_model,  # TensorFlow SavedModel模型
            pb,  # TensorFlow GraphDef模型
            tflite,  # TensorFlow Lite模型
            edgetpu,  # TensorFlow Edge TPU模型
            tfjs,  # TensorFlow.js模型
            paddle,  # PaddlePaddle模型
            mnn,  # MNN模型
            ncnn,  # NCNN模型
            imx,  # IMX模型
            rknn,  # RKNN模型
            triton,  # NVIDIA Triton模型
        ) = self._model_type(w)  # 确定模型类型
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu or rknn  # BHWC格式（与torch BCWH相对）
        stride = 32  # default stride  # 默认步幅
        end2end = False  # default end2end  # 默认end2end
        model, metadata, task = None, None, None  # 初始化模型、元数据和任务

        # Set device  # 设置设备
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA  # 使用CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx, paddle]):  # GPU dataloader formats  # GPU数据加载器格式
            device = torch.device("cpu")  # 如果不支持CUDA，则使用CPU
            cuda = False  # 设置cuda为False

        # Download if not local  # 如果不是本地文件，则下载
        if not (pt or triton or nn_module):  # 如果不是PyTorch、Triton或nn模块
            w = attempt_download_asset(w)  # 尝试下载资产

        # In-memory PyTorch model  # 内存中的PyTorch模型
        if nn_module:  # 如果weights是nn模块
            model = weights.to(device)  # 将模型移动到指定设备
            if fuse:  # 如果需要融合
                model = model.fuse(verbose=verbose)  # 融合模型
            if hasattr(model, "kpt_shape"):  # 如果模型有关键点形状属性
                kpt_shape = model.kpt_shape  # 仅用于姿态检测
            stride = max(int(model.stride.max()), 32)  # 获取模型步幅
            names = model.module.names if hasattr(model, "module") else model.names  # 获取类名
            model.half() if fp16 else model.float()  # 根据fp16设置模型为半精度或单精度
            self.model = model  # 显式赋值给self.model，以便后续调用to()、cpu()、cuda()、half()
            pt = True  # 设置pt为True

        # PyTorch  # PyTorch模型
        elif pt:  # 如果是PyTorch模型
            from ultralytics.nn.tasks import attempt_load_weights  # 从Ultralytics加载权重

            model = attempt_load_weights(  # 尝试加载权重
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            if hasattr(model, "kpt_shape"):  # 如果模型有关键点形状属性
                kpt_shape = model.kpt_shape  # 仅用于姿态检测
            stride = max(int(model.stride.max()), 32)  # 获取模型步幅
            names = model.module.names if hasattr(model, "module") else model.names  # 获取类名
            model.half() if fp16 else model.float()  # 根据fp16设置模型为半精度或单精度
            self.model = model  # 显式赋值给self.model，以便后续调用to()、cpu()、cuda()、half()

        # TorchScript  # TorchScript模型
        elif jit:  # 如果是TorchScript模型
            LOGGER.info(f"Loading {w} for TorchScript inference...")  # 记录加载信息
            extra_files = {"config.txt": ""}  # model metadata  # 模型元数据
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)  # 加载TorchScript模型

        # ONNX OpenCV DNN  # ONNX OpenCV DNN模型
        elif dnn:  # 如果是DNN模型
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")  # 记录加载信息
            check_requirements("opencv-python>=4.5.4")  # 检查OpenCV要求
            net = cv2.dnn.readNetFromONNX(w)  # 从ONNX加载DNN网络

        # ONNX Runtime and IMX  # ONNX Runtime和IMX模型
        elif onnx or imx:  # 如果是ONNX或IMX模型
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")  # 记录加载信息
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))  # 检查ONNX要求
            if IS_RASPBERRYPI or IS_JETSON:  # 如果是树莓派或Jetson
                # Fix 'numpy.linalg._umath_linalg' has no attribute '_ilp64' for TF SavedModel on RPi and Jetson  # 修复树莓派和Jetson上TF SavedModel的错误
                check_requirements("numpy==1.23.5")  # 检查numpy版本
            import onnxruntime  # 导入ONNX Runtime

            providers = ["CPUExecutionProvider"]  # 设置执行提供者为CPU
            if cuda and "CUDAExecutionProvider" in onnxruntime.get_available_providers():  # 如果支持CUDA
                providers.insert(0, "CUDAExecutionProvider")  # 将CUDA提供者插入到列表中
            elif cuda:  # 如果请求了CUDA但不可用
                LOGGER.warning("WARNING ⚠️ Failed to start ONNX Runtime with CUDA. Using CPU...")  # 记录警告信息
                device = torch.device("cpu")  # 使用CPU
                cuda = False  # 设置cuda为False
            LOGGER.info(f"Using ONNX Runtime {providers[0]}")  # 记录使用的ONNX Runtime提供者
            if onnx:  # 如果是ONNX模型
                session = onnxruntime.InferenceSession(w, providers=providers)  # 创建ONNX推理会话
            else:  # 如果是IMX模型
                check_requirements(  # 检查IMX要求
                    ["model-compression-toolkit==2.1.1", "sony-custom-layers[torch]==0.2.0", "onnxruntime-extensions"]
                )
                w = next(Path(w).glob("*.onnx"))  # 获取IMX模型的ONNX文件
                LOGGER.info(f"Loading {w} for ONNX IMX inference...")  # 记录加载信息
                import mct_quantizers as mctq  # 导入量化工具
                from sony_custom_layers.pytorch.object_detection import nms_ort  # noqa  # 导入自定义层

                session = onnxruntime.InferenceSession(  # 创建ONNX推理会话
                    w, mctq.get_ort_session_options(), providers=["CPUExecutionProvider"]
                )
                task = "detect"  # 设置任务为检测

            output_names = [x.name for x in session.get_outputs()]  # 获取输出名称
            metadata = session.get_modelmeta().custom_metadata_map  # 获取模型元数据
            dynamic = isinstance(session.get_outputs()[0].shape[0], str)  # 检查输出形状是否动态
            fp16 = True if "float16" in session.get_inputs()[0].type else False  # 检查输入类型是否为float16
            if not dynamic:  # 如果不是动态
                io = session.io_binding()  # 获取IO绑定
                bindings = []  # 初始化绑定列表
                for output in session.get_outputs():  # 遍历输出
                    out_fp16 = "float16" in output.type  # 检查输出类型是否为float16
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(device)  # 创建输出张量
                    io.bind_output(  # 绑定输出
                        name=output.name,
                        device_type=device.type,
                        device_id=device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    bindings.append(y_tensor)  # 添加到绑定列表

        # OpenVINO  # OpenVINO模型
        elif xml:  # 如果是OpenVINO模型
            LOGGER.info(f"Loading {w} for OpenVINO inference...")  # 记录加载信息
            check_requirements("openvino>=2024.0.0")  # 检查OpenVINO要求
            import openvino as ov  # 导入OpenVINO

            core = ov.Core()  # 创建OpenVINO核心
            w = Path(w)  # 将权重路径转换为Path对象
            if not w.is_file():  # 如果不是*.xml文件
                w = next(w.glob("*.xml"))  # 获取*_openvino_model目录中的*.xml文件
            ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))  # 读取OpenVINO模型
            if ov_model.get_parameters()[0].get_layout().empty:  # 如果布局为空
                ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))  # 设置布局为NCHW

            # OpenVINO inference modes are 'LATENCY', 'THROUGHPUT' (not recommended), or 'CUMULATIVE_THROUGHPUT'  # OpenVINO推理模式为'LATENCY'、'THROUGHPUT'（不推荐）或'CUMULATIVE_THROUGHPUT'
            inference_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 else "LATENCY"  # 设置推理模式
            LOGGER.info(f"Using OpenVINO {inference_mode} mode for batch={batch} inference...")  # 记录使用的推理模式
            ov_compiled_model = core.compile_model(  # 编译OpenVINO模型
                ov_model,
                device_name="AUTO",  # AUTO选择最佳可用设备，请勿修改
                config={"PERFORMANCE_HINT": inference_mode},  # 设置性能提示
            )
            input_name = ov_compiled_model.input().get_any_name()  # 获取输入名称
            metadata = w.parent / "metadata.yaml"  # 设置元数据路径

        # TensorRT  # TensorRT模型
        elif engine:  # 如果是TensorRT模型
            LOGGER.info(f"Loading {w} for TensorRT inference...")  # 记录加载信息

            if IS_JETSON and PYTHON_VERSION <= "3.8.0":  # 如果是Jetson并且Python版本小于等于3.8.0
                # fix error: `np.bool` was a deprecated alias for the builtin `bool` for JetPack 4 with Python <= 3.8.0  # 修复错误
                check_requirements("numpy==1.23.5")  # 检查numpy版本

            try:
                import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download  # 导入TensorRT
            except ImportError:  # 如果导入失败
                if LINUX:  # 如果是Linux
                    check_requirements("tensorrt>7.0.0,!=10.1.0")  # 检查TensorRT要求
                import tensorrt as trt  # noqa  # 重新导入TensorRT
            check_version(trt.__version__, ">=7.0.0", hard=True)  # 检查TensorRT版本
            check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")  # 检查TensorRT版本
            if device.type == "cpu":  # 如果设备是CPU
                device = torch.device("cuda:0")  # 切换到CUDA设备
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))  # 定义绑定元组
            logger = trt.Logger(trt.Logger.INFO)  # 创建TensorRT日志记录器
            # Read file  # 读取文件
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:  # 打开权重文件
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")  # 读取元数据长度
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))  # 读取元数据
                except UnicodeDecodeError:  # 如果解码错误
                    f.seek(0)  # 如果引擎文件缺少嵌入的Ultralytics元数据
                dla = metadata.get("dla", None)  # 获取DLA信息
                if dla is not None:  # 如果DLA信息存在
                    runtime.DLA_core = int(dla)  # 设置DLA核心
                model = runtime.deserialize_cuda_engine(f.read())  # 反序列化CUDA引擎

            # Model context  # 模型上下文
            try:
                context = model.create_execution_context()  # 创建执行上下文
            except Exception as e:  # 如果模型为None
                LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")  # 记录错误信息
                raise e  # 引发异常

            bindings = OrderedDict()  # 初始化绑定字典
            output_names = []  # 初始化输出名称列表
            fp16 = False  # default updated below  # 默认更新为False
            dynamic = False  # 默认动态为False
            is_trt10 = not hasattr(model, "num_bindings")  # 检查TensorRT版本
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)  # 获取绑定数量
            for i in num:  # 遍历绑定
                if is_trt10:  # 如果是TensorRT 10
                    name = model.get_tensor_name(i)  # 获取张量名称
                    dtype = trt.nptype(model.get_tensor_dtype(name))  # 获取张量数据类型
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT  # 检查是否为输入
                    if is_input:  # 如果是输入
                        if -1 in tuple(model.get_tensor_shape(name)):  # 动态绑定
                            dynamic = True  # 设置动态为True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))  # 设置输入形状
                        if dtype == np.float16:  # 如果数据类型为float16
                            fp16 = True  # 设置fp16为True
                    else:
                        output_names.append(name)  # 添加到输出名称列表
                    shape = tuple(context.get_tensor_shape(name))  # 获取张量形状
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)  # 获取绑定名称
                    dtype = trt.nptype(model.get_binding_dtype(i))  # 获取绑定数据类型
                    is_input = model.binding_is_input(i)  # 检查是否为输入
                    if model.binding_is_input(i):  # 如果是输入
                        if -1 in tuple(model.get_binding_shape(i)):  # 动态绑定
                            dynamic = True  # 设置动态为True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))  # 设置绑定形状
                        if dtype == np.float16:  # 如果数据类型为float16
                            fp16 = True  # 设置fp16为True
                    else:
                        output_names.append(name)  # 添加到输出名称列表
                    shape = tuple(context.get_binding_shape(i))  # 获取绑定形状
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)  # 创建输入张量
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 添加到绑定字典
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())  # 获取绑定地址字典
            batch_size = bindings["images"].shape[0]  # 如果是动态，则这是批大小

        # RKNN
        elif rknn:  # 如果是 RKNN 格式
            if not is_rockchip():  # 如果不是 Rockchip 设备
                raise OSError("RKNN inference is only supported on Rockchip devices.")  # 抛出 OSError，提示 RKNN 推理仅支持 Rockchip 设备
            LOGGER.info(f"Loading {w} for RKNN inference...")  # 记录加载 RKNN 推理模型的信息
            check_requirements("rknn-toolkit-lite2")  # 检查 RKNN 工具包的要求
            from rknnlite.api import RKNNLite  # 从 rknnlite 导入 RKNNLite 类

            w = Path(w)  # 将权重路径转换为 Path 对象
            if not w.is_file():  # if not *.rknn  # 如果不是 *.rknn 文件
                w = next(w.rglob("*.rknn"))  # get *.rknn file from *_rknn_model dir  # 从 *_rknn_model 目录中获取 *.rknn 文件
            rknn_model = RKNNLite()  # 创建 RKNNLite 实例
            rknn_model.load_rknn(w)  # 加载 RKNN 模型
            rknn_model.init_runtime()  # 初始化运行时
            metadata = Path(w).parent / "metadata.yaml"  # 设置元数据路径

        # Any other format (unsupported)  # 其他格式（不支持）
        else:  # 如果不是已知格式
            from ultralytics.engine.exporter import export_formats  # 从 ultralytics 导入 export_formats

            raise TypeError(  # 抛出 TypeError
                f"model='{w}' is not a supported model format. Ultralytics supports: {export_formats()['Format']}\n"  # "{w} 不是支持的模型格式。Ultralytics 支持的格式为："
                f"See https://docs.ultralytics.com/modes/predict for help."  # "请参阅 https://docs.ultralytics.com/modes/predict 获取帮助。"
            )

        # Load external metadata YAML  # 加载外部元数据 YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():  # 检查 metadata 是否为字符串或 Path 且存在
            metadata = yaml_load(metadata)  # 加载元数据
        if metadata and isinstance(metadata, dict):  # 如果 metadata 存在且为字典
            for k, v in metadata.items():  # 遍历元数据项
                if k in {"stride", "batch"}:  # 如果键是 "stride" 或 "batch"
                    metadata[k] = int(v)  # 将值转换为整数
                elif k in {"imgsz", "names", "kpt_shape", "args"} and isinstance(v, str):  # 如果键是 "imgsz"、"names"、"kpt_shape" 或 "args" 且值为字符串
                    metadata[k] = eval(v)  # 评估字符串并更新值
            stride = metadata["stride"]  # 获取步幅
            task = metadata["task"]  # 获取任务
            batch = metadata["batch"]  # 获取批次大小
            imgsz = metadata["imgsz"]  # 获取图像大小
            names = metadata["names"]  # 获取类名
            kpt_shape = metadata.get("kpt_shape")  # 获取关键点形状
            end2end = metadata.get("args", {}).get("nms", False)  # 获取是否启用 NMS 的参数
        elif not (pt or triton or nn_module):  # 如果不是 PyTorch、Triton 或 nn_module
            LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")  # 记录警告信息，提示未找到模型的元数据

        # Check names  # 检查类名
        if "names" not in locals():  # names missing  # 如果类名未定义
            names = default_class_names(data)  # 使用默认类名
        names = check_class_names(names)  # 检查类名

        # Disable gradients  # 禁用梯度
        if pt:  # 如果是 PyTorch 模型
            for p in model.parameters():  # 遍历模型参数
                p.requires_grad = False  # 禁用梯度计算

        self.__dict__.update(locals())  # 将所有局部变量赋值给实例字典

    def forward(self, im, augment=False, visualize=False, embed=None):  # 定义前向推理方法
        """
        Runs inference on the YOLOv8 MultiBackend model.  # 在 YOLOv8 MultiBackend 模型上运行推理

        Args:  # 参数：
            im (torch.Tensor): The image tensor to perform inference on.  # im（torch.Tensor）：用于推理的图像张量。
            augment (bool): whether to perform data augmentation during inference, defaults to False  # augment（bool）：是否在推理期间执行数据增强，默认为 False
            visualize (bool): whether to visualize the output predictions, defaults to False  # visualize（bool）：是否可视化输出预测，默认为 False
            embed (list, optional): A list of feature vectors/embeddings to return.  # embed（列表，可选）：要返回的特征向量/嵌入列表。

        Returns:  # 返回：
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)  # （元组）：包含原始输出张量和可视化处理输出的元组（如果 visualize=True）
        """
        b, ch, h, w = im.shape  # batch, channel, height, width  # 获取输入图像的批次、通道、高度和宽度
        if self.fp16 and im.dtype != torch.float16:  # 如果启用了半精度且输入图像不是 float16 类型
            im = im.half()  # 转换为 FP16
        if self.nhwc:  # 如果使用 NHWC 格式
            im = im.permute(0, 2, 3, 1)  # torch BCHW 转换为 numpy BHWC 形状（1, 320, 192, 3）

        # PyTorch  # PyTorch 推理
        if self.pt or self.nn_module:  # 如果是 PyTorch 模型或 nn.Module
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)  # 执行推理

        # TorchScript  # TorchScript 推理
        elif self.jit:  # 如果是 TorchScript 模型
            y = self.model(im)  # 执行推理

        # ONNX OpenCV DNN  # ONNX OpenCV DNN 推理
        elif self.dnn:  # 如果是 DNN 模型
            im = im.cpu().numpy()  # torch 转换为 numpy
            self.net.setInput(im)  # 设置输入
            y = self.net.forward()  # 执行前向推理

        # ONNX Runtime  # ONNX Runtime 推理
        elif self.onnx or self.imx:  # 如果是 ONNX 或 IMX 模型
            if self.dynamic:  # 如果是动态模型
                im = im.cpu().numpy()  # torch 转换为 numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})  # 执行推理
            else:  # 如果不是动态模型
                if not self.cuda:  # 如果不使用 CUDA
                    im = im.cpu()  # 转换为 CPU
                self.io.bind_input(  # 绑定输入
                    name="images",
                    device_type=im.device.type,
                    device_id=im.device.index if im.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(im.shape),
                    buffer_ptr=im.data_ptr(),
                )
                self.session.run_with_iobinding(self.io)  # 使用 I/O 绑定执行推理
                y = self.bindings  # 获取绑定输出
            if self.imx:  # 如果是 IMX 模型
                # boxes, conf, cls  # 盒子、置信度、类别
                y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)  # 合并输出

        # OpenVINO  # OpenVINO 推理
        elif self.xml:  # 如果是 OpenVINO 模型
            im = im.cpu().numpy()  # FP32  # 转换为 FP32

            if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:  # 优化大批次推理
                n = im.shape[0]  # batch 中图像的数量
                results = [None] * n  # 预分配与图像数量相同的结果列表

                def callback(request, userdata):  # 回调函数
                    """Places result in preallocated list using userdata index."""  # 使用用户数据索引将结果放入预分配列表
                    results[userdata] = request.results  # 将结果存储在指定位置

                # Create AsyncInferQueue, set the callback and start asynchronous inference for each input image  # 创建异步推理队列，设置回调并为每个输入图像开始异步推理
                async_queue = self.ov.runtime.AsyncInferQueue(self.ov_compiled_model)  # 创建异步推理队列
                async_queue.set_callback(callback)  # 设置回调函数
                for i in range(n):  # 遍历每个图像
                    # Start async inference with userdata=i to specify the position in results list  # 使用 userdata=i 开始异步推理，以指定结果列表中的位置
                    async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)  # 保持图像为 BCHW
                async_queue.wait_all()  # 等待所有推理请求完成
                y = np.concatenate([list(r.values())[0] for r in results])  # 合并结果

            else:  # inference_mode = "LATENCY"，优化以最快速度返回结果，批次大小为 1
                y = list(self.ov_compiled_model(im).values())  # 执行推理并获取结果

        # TensorRT  # TensorRT 推理
        elif self.engine:  # 如果是 TensorRT 模型
            if self.dynamic and im.shape != self.bindings["images"].shape:  # 如果是动态模型且输入形状不匹配
                if self.is_trt10:  # 如果是 TensorRT 10 版本
                    self.context.set_input_shape("images", im.shape)  # 设置输入形状
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)  # 更新绑定形状
                    for name in self.output_names:  # 遍历输出名称
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))  # 调整输出形状
                else:  # TensorRT < 10.0
                    i = self.model.get_binding_index("images")  # 获取输入绑定索引
                    self.context.set_binding_shape(i, im.shape)  # 设置绑定形状
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)  # 更新绑定形状
                    for name in self.output_names:  # 遍历输出名称
                        i = self.model.get_binding_index(name)  # 获取输出绑定索引
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))  # 调整输出形状

            s = self.bindings["images"].shape  # 获取输入绑定的形状
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"  # 断言输入形状与模型最大形状匹配
            self.binding_addrs["images"] = int(im.data_ptr())  # 获取输入数据指针
            self.context.execute_v2(list(self.binding_addrs.values()))  # 执行推理
            y = [self.bindings[x].data for x in sorted(self.output_names)]  # 获取输出数据

        # CoreML  # CoreML 推理
        elif self.coreml:  # 如果是 CoreML 模型
            im = im[0].cpu().numpy()  # 获取输入图像并转换为 numpy
            im_pil = Image.fromarray((im * 255).astype("uint8"))  # 转换为 PIL 图像
            # im = im.resize((192, 320), Image.BILINEAR)  # 可选：调整图像大小
            y = self.model.predict({"image": im_pil})  # 执行推理，返回坐标
            if "confidence" in y:  # 如果返回结果中包含置信度
                raise TypeError(  # 抛出类型错误
                    "Ultralytics only supports inference of non-pipelined CoreML models exported with "  # "Ultralytics 仅支持未管道化的 CoreML 模型推理，导出时需使用"
                    f"'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."  # "'nms=True' 导出的模型。"
                )
                # TODO: CoreML NMS inference handling  # TODO: CoreML NMS 推理处理
                # from ultralytics.utils.ops import xywh2xyxy
                # box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                # conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float32)
                # y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            y = list(y.values())  # 将结果转换为列表
            if len(y) == 2 and len(y[1].shape) != 4:  # segmentation model  # 如果是分割模型
                y = list(reversed(y))  # reversed for segmentation models (pred, proto)  # 反转结果顺序

        # PaddlePaddle  # PaddlePaddle 推理
        elif self.paddle:  # 如果是 PaddlePaddle 模型
            im = im.cpu().numpy().astype(np.float32)  # 转换为 numpy 并设置数据类型
            self.input_handle.copy_from_cpu(im)  # 从 CPU 复制输入数据
            self.predictor.run()  # 执行推理
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]  # 获取输出数据并复制到 CPU

        # MNN  # MNN 推理
        elif self.mnn:  # 如果是 MNN 模型
            input_var = self.torch_to_mnn(im)  # 将 PyTorch 张量转换为 MNN 张量
            output_var = self.net.onForward([input_var])  # 执行前向推理
            y = [x.read() for x in output_var]  # 读取输出数据

        # NCNN  # NCNN 推理
        elif self.ncnn:  # 如果是 NCNN 模型
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())  # 将输入数据转换为 NCNN 格式
            with self.net.create_extractor() as ex:  # 创建提取器
                ex.input(self.net.input_names()[0], mat_in)  # 设置输入
                # WARNING: 'output_names' sorted as a temporary fix for https://github.com/pnnx/pnnx/issues/130  # 警告：'output_names' 排序是临时修复
                y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]  # 获取输出数据

        # NVIDIA Triton Inference Server  # NVIDIA Triton 推理服务器
        elif self.triton:  # 如果是 Triton 模型
            im = im.cpu().numpy()  # torch 转换为 numpy
            y = self.model(im)  # 执行推理

        # RKNN  # RKNN 推理
        elif self.rknn:  # 如果是 RKNN 模型
            im = (im.cpu().numpy() * 255).astype("uint8")  # 转换为 numpy 并设置数据类型
            im = im if isinstance(im, (list, tuple)) else [im]  # 如果不是列表，则转换为列表
            y = self.rknn_model.inference(inputs=im)  # 执行推理

        # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)  # TensorFlow（SavedModel、GraphDef、Lite、Edge TPU）
        else:  # 如果不是已知格式
            im = im.cpu().numpy()  # 转换为 numpy
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)  # 执行推理
                if not isinstance(y, list):  # 如果返回结果不是列表
                    y = [y]  # 转换为列表
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))  # 执行推理
            else:  # Lite or Edge TPU
                details = self.input_details[0]  # 获取输入详情
                is_int = details["dtype"] in {np.int8, np.int16}  # 检查是否为 TFLite 量化 int8 或 int16 模型
                if is_int:  # 如果是量化模型
                    scale, zero_point = details["quantization"]  # 获取量化参数
                    im = (im / scale + zero_point).astype(details["dtype"])  # 反量化
                self.interpreter.set_tensor(details["index"], im)  # 设置输入张量
                self.interpreter.invoke()  # 执行推理
                y = []  # 初始化输出
                for output in self.output_details:  # 遍历输出详情
                    x = self.interpreter.get_tensor(output["index"])  # 获取输出张量
                    if is_int:  # 如果是量化模型
                        scale, zero_point = output["quantization"]  # 获取量化参数
                        x = (x.astype(np.float32) - zero_point) * scale  # 反量化
                    if x.ndim == 3:  # 如果任务不是分类，且不包括掩码（ndim=4）
                        # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695  # 根据图像大小反归一化 xywh
                        # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models  # xywh 在 TFLite/EdgeTPU 中归一化，以减少整数模型的量化误差
                        if x.shape[-1] == 6 or self.end2end:  # end-to-end model  # 如果是端到端模型
                            x[:, :, [0, 2]] *= w  # 反归一化宽度
                            x[:, :, [1, 3]] *= h  # 反归一化高度
                            if self.task == "pose":  # 如果任务是姿态检测
                                x[:, :, 6::3] *= w  # 反归一化关键点 x 坐标
                                x[:, :, 7::3] *= h  # 反归一化关键点 y 坐标
                        else:  # 如果不是端到端模型
                            x[:, [0, 2]] *= w  # 反归一化宽度
                            x[:, [1, 3]] *= h  # 反归一化高度
                            if self.task == "pose":  # 如果任务是姿态检测
                                x[:, 5::3] *= w  # 反归一化关键点 x 坐标
                                x[:, 6::3] *= h  # 反归一化关键点 y 坐标
                    y.append(x)  # 添加输出
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed  # TensorFlow 分割修复：导出顺序与 ONNX 导出相反，原型被转置
            if len(y) == 2:  # segment with (det, proto) output order reversed  # 如果是分割模型，输出顺序为（检测，原型）
                if len(y[1].shape) != 4:  # 如果原型的形状不是 4 维
                    y = list(reversed(y))  # 反转输出顺序
                if y[1].shape[-1] == 6:  # end-to-end model  # 如果是端到端模型
                    y = [y[1]]  # 仅返回原型
                else:
                    y[1] = np.transpose(y[1], (0, 3, 1, 2))  # 转置原型
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]  # 确保输出为 numpy 数组

        # for x in y:  # 调试输出形状
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
        if isinstance(y, (list, tuple)):  # 如果输出是列表或元组
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined  # 如果类名未定义
                nc = y[0].shape[1] - y[1].shape[1] - 4  # 计算类别数量
                self.names = {i: f"class{i}" for i in range(nc)}  # 创建类名字典
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]  # 返回输出
        else:
            return self.from_numpy(y)  # 返回输出

    def from_numpy(self, x):  # 将 numpy 数组转换为张量
        """
        Convert a numpy array to a tensor.  # 将 numpy 数组转换为张量

        Args:  # 参数：
            x (np.ndarray): The array to be converted.  # x（np.ndarray）：要转换的数组。

        Returns:  # 返回：
            (torch.Tensor): The converted tensor  # （torch.Tensor）：转换后的张量
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x  # 转换为张量并移动到设备

    def warmup(self, imgsz=(1, 3, 640, 640)):  # 预热模型
        """
        Warm up the model by running one forward pass with a dummy input.  # 通过使用虚拟输入运行一次前向传递来预热模型

        Args:  # 参数：
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)  # imgsz（元组）：虚拟输入张量的形状，格式为（批次大小、通道数、高度、宽度）
        """
        import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)  # 导入 torchvision

        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module  # 预热类型
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):  # 如果需要预热且设备不是 CPU 或使用 Triton
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # 创建虚拟输入
            for _ in range(2 if self.jit else 1):  # 根据是否为 JIT 进行预热
                self.forward(im)  # 运行前向推理

    @staticmethod
    def _model_type(p="path/to/model.pt"):  # 确定模型类型
        """
        Takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml, engine, coreml,  # 接受模型文件路径并返回模型类型。可能的类型有 pt、jit、onnx、xml、engine、coreml、
        saved_model, pb, tflite, edgetpu, tfjs, ncnn or paddle.  # saved_model、pb、tflite、edgetpu、tfjs、ncnn 或 paddle。

        Args:  # 参数：
            p (str): path to the model file. Defaults to path/to/model.pt  # p（字符串）：模型文件的路径。默认为 path/to/model.pt

        Examples:  # 示例：
            >>> model = AutoBackend(weights="path/to/model.onnx")  # 创建 AutoBackend 实例
            >>> model_type = model._model_type()  # returns "onnx"  # 返回 "onnx"
        """
        from ultralytics.engine.exporter import export_formats  # 导入导出格式

        sf = export_formats()["Suffix"]  # export suffixes  # 获取导出后缀
        if not is_url(p) and not isinstance(p, str):  # 检查路径是否为 URL 或字符串
            check_suffix(p, sf)  # 检查后缀
        name = Path(p).name  # 获取文件名
        types = [s in name for s in sf]  # 检查文件名是否包含后缀
        types[5] |= name.endswith(".mlmodel")  # retain support for older Apple CoreML *.mlmodel formats  # 保留对旧版 Apple CoreML *.mlmodel 格式的支持
        types[8] &= not types[9]  # tflite &= not edgetpu  # tflite 仅当不是 edgetpu 时有效
        if any(types):  # 如果有匹配的类型
            triton = False  # 设置 Triton 为 False
        else:  # 如果没有匹配的类型
            from urllib.parse import urlsplit  # 导入 URL 解析模块

            url = urlsplit(p)  # 解析 URL
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}  # 检查是否为 Triton URL

        return types + [triton]  # 返回类型列表