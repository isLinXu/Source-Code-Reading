# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Benchmark a YOLO model formats for speed and accuracy.

Usage:
    from ultralytics.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolo11n.yaml', 'yolov8s.yaml']).profile()
    benchmark(model='yolo11n.pt', imgsz=160)

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
RKNN                    | `rknn`                    | yolo11n_rknn_model/
"""

import glob  # 导入 glob 模块
import os  # 导入 os 模块
import platform  # 导入 platform 模块
import re  # 导入 re 模块
import shutil  # 导入 shutil 模块
import time  # 导入 time 模块
from pathlib import Path  # 从 pathlib 导入 Path

import numpy as np  # 导入 numpy 作为 np
import torch.cuda  # 导入 torch.cuda 模块
import yaml  # 导入 yaml 模块

from ultralytics import YOLO, YOLOWorld  # 从 ultralytics 导入 YOLO 和 YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC  # 从 ultralytics.cfg 导入 TASK2DATA 和 TASK2METRIC
from ultralytics.engine.exporter import export_formats  # 从 ultralytics.engine.exporter 导入 export_formats
from ultralytics.utils import ARM64, ASSETS, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR  # 从 ultralytics.utils 导入相关常量
from ultralytics.utils.checks import IS_PYTHON_3_12, check_imgsz, check_requirements, check_yolo, is_rockchip  # 从 ultralytics.utils.checks 导入检查函数
from ultralytics.utils.downloads import safe_download  # 从 ultralytics.utils.downloads 导入 safe_download
from ultralytics.utils.files import file_size  # 从 ultralytics.utils.files 导入 file_size
from ultralytics.utils.torch_utils import get_cpu_info, select_device  # 从 ultralytics.utils.torch_utils 导入 get_cpu_info 和 select_device


def benchmark(
    model=WEIGHTS_DIR / "yolo11n.pt",  # 模型的路径，默认为 yolo11n.pt
    data=None,  # 数据集，默认为 None
    imgsz=160,  # 图像大小，默认为 160
    half=False,  # 是否使用半精度，默认为 False
    int8=False,  # 是否使用 int8 精度，默认为 False
    device="cpu",  # 运行基准测试的设备，默认为 'cpu'
    verbose=False,  # 是否详细输出，默认为 False
    eps=1e-3,  # 防止除以零的 epsilon 值
    format="",  # 导出格式，默认为空
):
    """
    Benchmark a YOLO model across different formats for speed and accuracy.  # 在不同格式下对 YOLO 模型进行速度和准确性基准测试。

    Args:  # 参数：
        model (str | Path): Path to the model file or directory.  # model (str | Path): 模型文件或目录的路径。
        data (str | None): Dataset to evaluate on, inherited from TASK2DATA if not passed.  # data (str | None): 要评估的数据集，如果未传递，则从 TASK2DATA 中继承。
        imgsz (int): Image size for the benchmark.  # imgsz (int): 基准测试的图像大小。
        half (bool): Use half-precision for the model if True.  # half (bool): 如果为 True，则对模型使用半精度。
        int8 (bool): Use int8-precision for the model if True.  # int8 (bool): 如果为 True，则对模型使用 int8 精度。
        device (str): Device to run the benchmark on, either 'cpu' or 'cuda'.  # device (str): 运行基准测试的设备，可以是 'cpu' 或 'cuda'。
        verbose (bool | float): If True or a float, assert benchmarks pass with given metric.  # verbose (bool | float): 如果为 True 或浮点数，则断言基准测试通过给定的指标。
        eps (float): Epsilon value for divide by zero prevention.  # eps (float): 防止除以零的 epsilon 值。
        format (str): Export format for benchmarking. If not supplied all formats are benchmarked.  # format (str): 基准测试的导出格式。如果未提供，则对所有格式进行基准测试。

    Returns:  # 返回：
        (pandas.DataFrame): A pandas DataFrame with benchmark results for each format, including file size, metric,  # (pandas.DataFrame): 包含每种格式基准测试结果的 pandas DataFrame，包括文件大小、指标，
            and inference time.  # 和推理时间。

    Examples:  # 示例：
        Benchmark a YOLO model with default settings:  # 使用默认设置对 YOLO 模型进行基准测试：
        >>> from ultralytics.utils.benchmarks import benchmark  # 从 ultralytics.utils.benchmarks 导入 benchmark
        >>> benchmark(model="yolo11n.pt", imgsz=640)  # 对 yolo11n.pt 模型进行基准测试，图像大小为 640
    """
    imgsz = check_imgsz(imgsz)  # 检查图像大小
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."  # 确保图像大小是正方形

    import pandas as pd  # scope for faster 'import ultralytics'  # 为了更快的 'import ultralytics' 而导入 pandas

    pd.options.display.max_columns = 10  # 设置 pandas 显示的最大列数
    pd.options.display.width = 120  # 设置 pandas 显示的宽度
    device = select_device(device, verbose=False)  # 选择设备
    if isinstance(model, (str, Path)):  # 如果模型是字符串或路径
        model = YOLO(model)  # 创建 YOLO 模型实例
    is_end2end = getattr(model.model.model[-1], "end2end", False)  # 检查模型是否为端到端模型
    data = data or TASK2DATA[model.task]  # task to dataset, i.e. coco8.yaml for task=detect  # 任务对应的数据集，例如任务为检测时对应 coco8.yaml
    key = TASK2METRIC[model.task]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect  # 任务对应的指标，例如任务为检测时对应 metrics/mAP50-95(B)

    y = []  # 初始化结果列表
    t0 = time.time()  # 记录开始时间

    format_arg = format.lower()  # 将格式转换为小写
    if format_arg:  # 如果提供了格式
        formats = frozenset(export_formats()["Argument"])  # 获取可用的导出格式
        assert format in formats, f"Expected format to be one of {formats}, but got '{format_arg}'."  # 确保格式有效
    for i, (name, format, suffix, cpu, gpu, _) in enumerate(zip(*export_formats().values())):  # 遍历可用的导出格式
        emoji, filename = "❌", None  # 导出默认值
        try:
            if format_arg and format_arg != format:  # 如果指定了格式且不匹配
                continue  # 跳过

            # Checks  # 检查
            if i == 7:  # TF GraphDef
                assert model.task != "obb", "TensorFlow GraphDef not supported for OBB task"  # 检查任务是否支持
            elif i == 9:  # Edge TPU
                assert LINUX and not ARM64, "Edge TPU export only supported on non-aarch64 Linux"  # 检查是否支持 Edge TPU 导出
            elif i in {5, 10}:  # CoreML and TF.js
                assert MACOS or (LINUX and not ARM64), (  # 检查是否支持 CoreML 和 TF.js 导出
                    "CoreML and TF.js export only supported on macOS and non-aarch64 Linux"
                )
            if i in {5}:  # CoreML
                assert not IS_PYTHON_3_12, "CoreML not supported on Python 3.12"  # 检查 Python 版本
            if i in {6, 7, 8}:  # TF SavedModel, TF GraphDef, and TFLite
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"  # 检查模型类型
            if i in {9, 10}:  # TF EdgeTPU and TF.js
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"  # 检查模型类型
            if i == 11:  # Paddle
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Paddle exports not supported yet"  # 检查模型类型
                assert not is_end2end, "End-to-end models not supported by PaddlePaddle yet"  # 检查模型类型
                assert LINUX or MACOS, "Windows Paddle exports not supported yet"  # 检查操作系统
            if i == 12:  # MNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 MNN exports not supported yet"  # 检查模型类型
            if i == 13:  # NCNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 NCNN exports not supported yet"  # 检查模型类型
            if i == 14:  # IMX
                assert not is_end2end  # 检查模型类型
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 IMX exports not supported"  # 检查模型类型
                assert model.task == "detect", "IMX only supported for detection task"  # 检查任务类型
                assert "C2f" in model.__str__(), "IMX only supported for YOLOv8"  # 检查模型类型
            if i == 15:  # RKNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 RKNN exports not supported yet"  # 检查模型类型
                assert not is_end2end, "End-to-end models not supported by RKNN yet"  # 检查模型类型
                assert LINUX, "RKNN only supported on Linux"  # 检查操作系统
                assert not is_rockchip(), "RKNN Inference only supported on Rockchip devices"  # 检查设备类型
            if "cpu" in device.type:  # 如果设备类型为 CPU
                assert cpu, "inference not supported on CPU"  # 检查推理是否支持
            if "cuda" in device.type:  # 如果设备类型为 CUDA
                assert gpu, "inference not supported on GPU"  # 检查推理是否支持

            # Export  # 导出
            if format == "-":  # 如果格式为 "-"
                filename = model.pt_path or model.ckpt_path or model.model_name  # 获取模型文件名
                exported_model = model  # PyTorch format  # PyTorch 格式
            else:
                filename = model.export(  # 导出模型
                    imgsz=imgsz, format=format, half=half, int8=int8, data=data, device=device, verbose=False
                )
                exported_model = YOLO(filename, task=model.task)  # 创建 YOLO 模型实例
                assert suffix in str(filename), "export failed"  # 检查导出是否成功
            emoji = "❎"  # indicates export succeeded  # 指示导出成功

            # Predict  # 预测
            assert model.task != "pose" or i != 7, "GraphDef Pose inference is not supported"  # 检查任务类型
            assert i not in {9, 10}, "inference not supported"  # Edge TPU and TF.js are unsupported  # Edge TPU 和 TF.js 不支持
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  # CoreML 仅支持 macOS
            if i in {13}:  # NCNN
                assert not is_end2end, "End-to-end torch.topk operation is not supported for NCNN prediction yet"  # 检查模型类型
            exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half, verbose=False)  # 进行预测

            # Validate  # 验证
            results = exported_model.val(  # 验证模型
                data=data, batch=1, imgsz=imgsz, plots=False, device=device, half=half, int8=int8, verbose=False
            )
            metric, speed = results.results_dict[key], results.speed["inference"]  # 获取指标和速度
            fps = round(1000 / (speed + eps), 2)  # frames per second  # 帧率
            y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])  # 添加结果
        except Exception as e:  # 捕获异常
            if verbose:  # 如果详细输出
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"  # 检查异常类型
            LOGGER.warning(f"ERROR ❌️ Benchmark failure for {name}: {e}")  # 记录错误信息
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # Print results  # 打印结果
    check_yolo(device=device)  # print system info  # 打印系统信息
    df = pd.DataFrame(y, columns=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"])  # 创建结果 DataFrame

    name = model.model_name  # 获取模型名称
    dt = time.time() - t0  # 计算耗时
    legend = "Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed"  # 基准测试图例
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({dt:.2f}s)\n{legend}\n{df.fillna('-')}\n"  # 格式化输出字符串
    LOGGER.info(s)  # 记录基准测试结果
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:  # 打开日志文件
        f.write(s)  # 写入结果

    if verbose and isinstance(verbose, float):  # 如果详细输出且为浮点数
        metrics = df[key].array  # values to compare to floor  # 获取比较的值
        floor = verbose  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n  # 最小指标阈值
        assert all(x > floor for x in metrics if pd.notna(x)), f"Benchmark failure: metric(s) < floor {floor}"  # 检查指标是否满足阈值

    return df  # 返回结果 DataFrame


class RF100Benchmark:
    """Benchmark YOLO model performance across various formats for speed and accuracy.  # 对 YOLO 模型在不同格式下的速度和准确性进行基准测试。"""

    def __init__(self):
        """Initialize the RF100Benchmark class for benchmarking YOLO model performance across various formats.  # 初始化 RF100Benchmark 类以对 YOLO 模型在不同格式下的性能进行基准测试。"""
        self.ds_names = []  # 数据集名称列表
        self.ds_cfg_list = []  # 数据集配置列表
        self.rf = None  # Roboflow 实例
        self.val_metrics = ["class", "images", "targets", "precision", "recall", "map50", "map95"]  # 验证指标列表

    def set_key(self, api_key):
        """
        Set Roboflow API key for processing.  # 设置 Roboflow API 密钥以进行处理。

        Args:  # 参数：
            api_key (str): The API key.  # api_key (str): API 密钥。

        Examples:  # 示例：
            Set the Roboflow API key for accessing datasets:  # 设置 Roboflow API 密钥以访问数据集：
            >>> benchmark = RF100Benchmark()  # 创建 RF100Benchmark 实例
            >>> benchmark.set_key("your_roboflow_api_key")  # 设置 API 密钥
        """
        check_requirements("roboflow")  # 检查是否安装了 roboflow
        from roboflow import Roboflow  # 从 roboflow 导入 Roboflow

        self.rf = Roboflow(api_key=api_key)  # 创建 Roboflow 实例

    def parse_dataset(self, ds_link_txt="datasets_links.txt"):
        """
        Parse dataset links and download datasets.  # 解析数据集链接并下载数据集。

        Args:  # 参数：
            ds_link_txt (str): Path to the file containing dataset links.  # ds_link_txt (str): 包含数据集链接的文件路径。

        Examples:  # 示例：
            >>> benchmark = RF100Benchmark()  # 创建 RF100Benchmark 实例
            >>> benchmark.set_key("api_key")  # 设置 API 密钥
            >>> benchmark.parse_dataset("datasets_links.txt")  # 解析数据集
        """
        (shutil.rmtree("rf-100"), os.mkdir("rf-100")) if os.path.exists("rf-100") else os.mkdir("rf-100")  # 如果 rf-100 目录存在则删除并重新创建
        os.chdir("rf-100")  # 切换到 rf-100 目录
        os.mkdir("ultralytics-benchmarks")  # 创建 ultralytics-benchmarks 目录
        safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/datasets_links.txt")  # 下载数据集链接文件

        with open(ds_link_txt) as file:  # 打开数据集链接文件
            for line in file:  # 遍历每一行
                try:
                    _, url, workspace, project, version = re.split("/+", line.strip())  # 解析链接
                    self.ds_names.append(project)  # 添加项目名称到列表
                    proj_version = f"{project}-{version}"  # 创建项目版本字符串
                    if not Path(proj_version).exists():  # 如果项目版本目录不存在
                        self.rf.workspace(workspace).project(project).version(version).download("yolov8")  # 下载数据集
                    else:
                        print("Dataset already downloaded.")  # 数据集已下载
                    self.ds_cfg_list.append(Path.cwd() / proj_version / "data.yaml")  # 添加数据配置文件路径到列表
                except Exception:  # 捕获异常
                    continue  # 继续下一个循环

        return self.ds_names, self.ds_cfg_list  # 返回数据集名称和配置列表

    @staticmethod
    def fix_yaml(path):
        """
        Fixes the train and validation paths in a given YAML file.  # 修复给定 YAML 文件中的训练和验证路径。

        Args:  # 参数：
            path (str): Path to the YAML file to be fixed.  # path (str): 要修复的 YAML 文件路径。

        Examples:  # 示例：
            >>> RF100Benchmark.fix_yaml("path/to/data.yaml")  # 修复数据 YAML 文件
        """
        with open(path) as file:  # 打开 YAML 文件
            yaml_data = yaml.safe_load(file)  # 加载 YAML 数据
        yaml_data["train"] = "train/images"  # 设置训练路径
        yaml_data["val"] = "valid/images"  # 设置验证路径
        with open(path, "w") as file:  # 以写入模式打开 YAML 文件
            yaml.safe_dump(yaml_data, file)  # 保存修复后的 YAML 数据

    def evaluate(self, yaml_path, val_log_file, eval_log_file, list_ind):
        """
        Evaluate model performance on validation results.  # 在验证结果上评估模型性能。

        Args:  # 参数：
            yaml_path (str): Path to the YAML configuration file.  # yaml_path (str): YAML 配置文件的路径。
            val_log_file (str): Path to the validation log file.  # val_log_file (str): 验证日志文件的路径。
            eval_log_file (str): Path to the evaluation log file.  # eval_log_file (str): 评估日志文件的路径。
            list_ind (int): Index of the current dataset in the list.  # list_ind (int): 当前数据集在列表中的索引。

        Returns:  # 返回：
            (float): The mean average precision (mAP) value for the evaluated model.  # (float): 评估模型的平均精度 (mAP) 值。

        Examples:  # 示例：
            Evaluate a model on a specific dataset  # 在特定数据集上评估模型
            >>> benchmark = RF100Benchmark()  # 创建 RF100Benchmark 实例
            >>> benchmark.evaluate("path/to/data.yaml", "path/to/val_log.txt", "path/to/eval_log.txt", 0)  # 评估模型
        """
        skip_symbols = ["🚀", "⚠️", "💡", "❌"]  # 跳过的符号列表
        with open(yaml_path) as stream:  # 打开 YAML 配置文件
            class_names = yaml.safe_load(stream)["names"]  # 加载类别名称
        with open(val_log_file, encoding="utf-8") as f:  # 打开验证日志文件
            lines = f.readlines()  # 读取所有行
            eval_lines = []  # 初始化评估行列表
            for line in lines:  # 遍历每一行
                if any(symbol in line for symbol in skip_symbols):  # 如果行中包含跳过的符号
                    continue  # 跳过该行
                entries = line.split(" ")  # 按空格分割行
                entries = list(filter(lambda val: val != "", entries))  # 过滤空值
                entries = [e.strip("\n") for e in entries]  # 去除换行符
                eval_lines.extend(  # 添加评估行
                    {
                        "class": entries[0],  # 类别
                        "images": entries[1],  # 图像数量
                        "targets": entries[2],  # 目标数量
                        "precision": entries[3],  # 精度
                        "recall": entries[4],  # 召回率
                        "map50": entries[5],  # mAP50
                        "map95": entries[6],  # mAP95
                    }
                    for e in entries  # 遍历条目
                    if e in class_names or (e == "all" and "(AP)" not in entries and "(AR)" not in entries)  # 检查类别名称
                )
        map_val = 0.0  # 初始化 mAP 值
        if len(eval_lines) > 1:  # 如果有多条评估行
            print("There's more dicts")  # 有多个字典
            for lst in eval_lines:  # 遍历评估行
                if lst["class"] == "all":  # 如果类别为 "all"
                    map_val = lst["map50"]  # 获取 mAP50 值
        else:  # 只有一条评估行
            print("There's only one dict res")  # 只有一个字典
            map_val = [res["map50"] for res in eval_lines][0]  # 获取 mAP50 值

        with open(eval_log_file, "a") as f:  # 以追加模式打开评估日志文件
            f.write(f"{self.ds_names[list_ind]}: {map_val}\n")  # 写入数据集名称和 mAP 值

class ProfileModels:
    """
    ProfileModels class for profiling different models on ONNX and TensorRT.  # ProfileModels 类用于对不同模型在 ONNX 和 TensorRT 上进行性能分析。

    This class profiles the performance of different models, returning results such as model speed and FLOPs.  # 此类分析不同模型的性能，返回模型速度和 FLOPs 等结果。

    Attributes:  # 属性：
        paths (List[str]): Paths of the models to profile.  # paths (List[str]): 要分析的模型路径列表。
        num_timed_runs (int): Number of timed runs for the profiling.  # num_timed_runs (int): 性能分析的计时运行次数。
        num_warmup_runs (int): Number of warmup runs before profiling.  # num_warmup_runs (int): 在性能分析前的预热运行次数。
        min_time (float): Minimum number of seconds to profile for.  # min_time (float): 性能分析的最小时间（秒）。
        imgsz (int): Image size used in the models.  # imgsz (int): 模型使用的图像大小。
        half (bool): Flag to indicate whether to use FP16 half-precision for TensorRT profiling.  # half (bool): 标志，指示是否在 TensorRT 性能分析中使用 FP16 半精度。
        trt (bool): Flag to indicate whether to profile using TensorRT.  # trt (bool): 标志，指示是否使用 TensorRT 进行性能分析。
        device (torch.device): Device used for profiling.  # device (torch.device): 用于性能分析的设备。

    Methods:  # 方法：
        profile: Profiles the models and prints the result.  # profile: 分析模型并打印结果。

    Examples:  # 示例：
        Profile models and print results  # 分析模型并打印结果
        >>> from ultralytics.utils.benchmarks import ProfileModels  # 从 ultralytics.utils.benchmarks 导入 ProfileModels
        >>> profiler = ProfileModels(["yolo11n.yaml", "yolov8s.yaml"], imgsz=640)  # 创建 ProfileModels 实例
        >>> profiler.profile()  # 执行性能分析
    """

    def __init__(
        self,
        paths: list,  # paths (list): 要分析的模型路径列表
        num_timed_runs=100,  # num_timed_runs (int): 性能分析的计时运行次数，默认为 100
        num_warmup_runs=10,  # num_warmup_runs (int): 在性能分析前的预热运行次数，默认为 10
        min_time=60,  # min_time (float): 性能分析的最小时间（秒），默认为 60
        imgsz=640,  # imgsz (int): 模型使用的图像大小，默认为 640
        half=True,  # half (bool): 是否使用 FP16 半精度，默认为 True
        trt=True,  # trt (bool): 是否使用 TensorRT 进行性能分析，默认为 True
        device=None,  # device (torch.device | None): 用于性能分析的设备，默认为 None
    ):
        """
        Initialize the ProfileModels class for profiling models.  # 初始化 ProfileModels 类以对模型进行性能分析。

        Args:  # 参数：
            paths (List[str]): List of paths of the models to be profiled.  # paths (List[str]): 要分析的模型路径列表。
            num_timed_runs (int): Number of timed runs for the profiling.  # num_timed_runs (int): 性能分析的计时运行次数。
            num_warmup_runs (int): Number of warmup runs before the actual profiling starts.  # num_warmup_runs (int): 在实际性能分析开始前的预热运行次数。
            min_time (float): Minimum time in seconds for profiling a model.  # min_time (float): 性能分析的最小时间（秒）。
            imgsz (int): Size of the image used during profiling.  # imgsz (int): 性能分析中使用的图像大小。
            half (bool): Flag to indicate whether to use FP16 half-precision for TensorRT profiling.  # half (bool): 标志，指示是否在 TensorRT 性能分析中使用 FP16 半精度。
            trt (bool): Flag to indicate whether to profile using TensorRT.  # trt (bool): 标志，指示是否使用 TensorRT 进行性能分析。
            device (torch.device | None): Device used for profiling. If None, it is determined automatically.  # device (torch.device | None): 用于性能分析的设备。如果为 None，则自动确定。

        Notes:  # 注意：
            FP16 'half' argument option removed for ONNX as slower on CPU than FP32.  # 对于 ONNX，已移除 FP16 'half' 参数选项，因为在 CPU 上比 FP32 更慢。

        Examples:  # 示例：
            Initialize and profile models  # 初始化并分析模型
            >>> from ultralytics.utils.benchmarks import ProfileModels  # 从 ultralytics.utils.benchmarks 导入 ProfileModels
            >>> profiler = ProfileModels(["yolo11n.yaml", "yolov8s.yaml"], imgsz=640)  # 创建 ProfileModels 实例
            >>> profiler.profile()  # 执行性能分析
        """
        self.paths = paths  # 设置模型路径
        self.num_timed_runs = num_timed_runs  # 设置计时运行次数
        self.num_warmup_runs = num_warmup_runs  # 设置预热运行次数
        self.min_time = min_time  # 设置最小分析时间
        self.imgsz = imgsz  # 设置图像大小
        self.half = half  # 设置半精度标志
        self.trt = trt  # run TensorRT profiling  # 运行 TensorRT 性能分析
        self.device = device or torch.device(0 if torch.cuda.is_available() else "cpu")  # 设置设备，优先使用 CUDA

    def profile(self):
        """Profiles YOLO models for speed and accuracy across various formats including ONNX and TensorRT.  # 对 YOLO 模型在不同格式下（包括 ONNX 和 TensorRT）进行速度和准确性分析。"""
        files = self.get_files()  # 获取模型文件

        if not files:  # 如果没有找到文件
            print("No matching *.pt or *.onnx files found.")  # 打印提示信息
            return  # 返回

        table_rows = []  # 初始化表格行列表
        output = []  # 初始化输出列表
        for file in files:  # 遍历每个文件
            engine_file = file.with_suffix(".engine")  # 创建引擎文件名
            if file.suffix in {".pt", ".yaml", ".yml"}:  # 如果文件后缀为 .pt、.yaml 或 .yml
                model = YOLO(str(file))  # 创建 YOLO 模型实例
                model.fuse()  # to report correct params and GFLOPs in model.info()  # 融合模型以报告正确的参数和 GFLOPs
                model_info = model.info()  # 获取模型信息
                if self.trt and self.device.type != "cpu" and not engine_file.is_file():  # 如果使用 TensorRT 且不是 CPU 设备且引擎文件不存在
                    engine_file = model.export(  # 导出模型为引擎格式
                        format="engine",
                        half=self.half,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                onnx_file = model.export(  # 导出模型为 ONNX 格式
                    format="onnx",
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            elif file.suffix == ".onnx":  # 如果文件后缀为 .onnx
                model_info = self.get_onnx_model_info(file)  # 获取 ONNX 模型信息
                onnx_file = file  # 设置 ONNX 文件
            else:  # 其他情况
                continue  # 跳过

            t_engine = self.profile_tensorrt_model(str(engine_file))  # 对 TensorRT 模型进行性能分析
            t_onnx = self.profile_onnx_model(str(onnx_file))  # 对 ONNX 模型进行性能分析
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))  # 生成表格行
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))  # 生成结果字典

        self.print_table(table_rows)  # 打印表格
        return output  # 返回输出结果

    def get_files(self):
        """Returns a list of paths for all relevant model files given by the user.  # 返回用户提供的所有相关模型文件的路径列表。"""
        files = []  # 初始化文件列表
        for path in self.paths:  # 遍历每个路径
            path = Path(path)  # 将路径转换为 Path 对象
            if path.is_dir():  # 如果路径是目录
                extensions = ["*.pt", "*.onnx", "*.yaml"]  # 定义文件扩展名
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])  # 查找匹配的文件并添加到列表
            elif path.suffix in {".pt", ".yaml", ".yml"}:  # 如果路径是文件且后缀为 .pt、.yaml 或 .yml
                files.append(str(path))  # 添加文件路径到列表
            else:  # 其他情况
                files.extend(glob.glob(str(path)))  # 查找匹配的文件并添加到列表

        print(f"Profiling: {sorted(files)}")  # 打印正在分析的文件
        return [Path(file) for file in sorted(files)]  # 返回排序后的文件路径列表

    @staticmethod
    def get_onnx_model_info(onnx_file: str):
        """Extracts metadata from an ONNX model file including parameters, GFLOPs, and input shape.  # 从 ONNX 模型文件中提取元数据，包括参数、GFLOPs 和输入形状。"""
        return 0.0, 0.0, 0.0, 0.0  # return (num_layers, num_params, num_gradients, num_flops)  # 返回 (num_layers, num_params, num_gradients, num_flops)

    @staticmethod
    def iterative_sigma_clipping(data, sigma=2, max_iters=3):
        """Applies iterative sigma clipping to data to remove outliers based on specified sigma and iteration count.  # 对数据应用迭代 sigma 剪切，以根据指定的 sigma 和迭代次数去除异常值。"""
        data = np.array(data)  # 将数据转换为 numpy 数组
        for _ in range(max_iters):  # 进行最大迭代次数
            mean, std = np.mean(data), np.std(data)  # 计算均值和标准差
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]  # 剪切数据
            if len(clipped_data) == len(data):  # 如果剪切后的数据长度与原数据相同
                break  # 退出循环
            data = clipped_data  # 更新数据
        return data  # 返回剪切后的数据

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        """Profiles YOLO model performance with TensorRT, measuring average run time and standard deviation.  # 使用 TensorRT 对 YOLO 模型性能进行分析，测量平均运行时间和标准差。"""
        if not self.trt or not Path(engine_file).is_file():  # 如果不使用 TensorRT 或引擎文件不存在
            return 0.0, 0.0  # 返回 0.0，0.0

        # Model and input  # 模型和输入
        model = YOLO(engine_file)  # 创建 YOLO 模型实例
        input_data = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)  # use uint8 for Classify  # 使用 uint8 类型的输入数据

        # Warmup runs  # 预热运行
        elapsed = 0.0  # 初始化已用时间
        for _ in range(3):  # 进行 3 次预热
            start_time = time.time()  # 记录开始时间
            for _ in range(self.num_warmup_runs):  # 进行预热运行
                model(input_data, imgsz=self.imgsz, verbose=False)  # 执行模型推理
            elapsed = time.time() - start_time  # 计算已用时间

        # Compute number of runs as higher of min_time or num_timed_runs  # 计算运行次数，取 min_time 和 num_timed_runs 中的较大值
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)  # 计算运行次数

        # Timed runs  # 计时运行
        run_times = []  # 初始化运行时间列表
        for _ in TQDM(range(num_runs), desc=engine_file):  # 进行计时运行
            results = model(input_data, imgsz=self.imgsz, verbose=False)  # 执行模型推理
            run_times.append(results[0].speed["inference"])  # Convert to milliseconds  # 转换为毫秒

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping  # 进行 sigma 剪切
        return np.mean(run_times), np.std(run_times)  # 返回平均运行时间和标准差

    def profile_onnx_model(self, onnx_file: str, eps: float = 1e-3):
        """Profiles an ONNX model, measuring average inference time and standard deviation across multiple runs.  # 对 ONNX 模型进行性能分析，测量多次运行的平均推理时间和标准差。"""
        check_requirements("onnxruntime")  # 检查是否安装了 onnxruntime
        import onnxruntime as ort  # 导入 onnxruntime

        # Session with either 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'  # 使用 'TensorrtExecutionProvider'、'CUDAExecutionProvider' 或 'CPUExecutionProvider' 创建会话
        sess_options = ort.SessionOptions()  # 创建会话选项
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 设置图优化级别
        sess_options.intra_op_num_threads = 8  # Limit the number of threads  # 限制线程数量
        sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])  # 创建推理会话

        input_tensor = sess.get_inputs()[0]  # 获取输入张量
        input_type = input_tensor.type  # 获取输入类型
        dynamic = not all(isinstance(dim, int) and dim >= 0 for dim in input_tensor.shape)  # dynamic input shape  # 检查输入形状是否为动态
        input_shape = (1, 3, self.imgsz, self.imgsz) if dynamic else input_tensor.shape  # 设置输入形状

        # Mapping ONNX datatype to numpy datatype  # 将 ONNX 数据类型映射到 numpy 数据类型
        if "float16" in input_type:  # 如果输入类型为 float16
            input_dtype = np.float16  # 设置输入数据类型为 float16
        elif "float" in input_type:  # 如果输入类型为 float
            input_dtype = np.float32  # 设置输入数据类型为 float32
        elif "double" in input_type:  # 如果输入类型为 double
            input_dtype = np.float64  # 设置输入数据类型为 float64
        elif "int64" in input_type:  # 如果输入类型为 int64
            input_dtype = np.int64  # 设置输入数据类型为 int64
        elif "int32" in input_type:  # 如果输入类型为 int32
            input_dtype = np.int32  # 设置输入数据类型为 int32
        else:  # 其他情况
            raise ValueError(f"Unsupported ONNX datatype {input_type}")  # 抛出不支持的数据类型异常

        input_data = np.random.rand(*input_shape).astype(input_dtype)  # 创建随机输入数据
        input_name = input_tensor.name  # 获取输入张量名称
        output_name = sess.get_outputs()[0].name  # 获取输出张量名称

        # Warmup runs  # 预热运行
        elapsed = 0.0  # 初始化已用时间
        for _ in range(3):  # 进行 3 次预热
            start_time = time.time()  # 记录开始时间
            for _ in range(self.num_warmup_runs):  # 进行预热运行
                sess.run([output_name], {input_name: input_data})  # 执行推理
            elapsed = time.time() - start_time  # 计算已用时间

        # Compute number of runs as higher of min_time or num_timed_runs  # 计算运行次数，取 min_time 和 num_timed_runs 中的较大值
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs)  # 计算运行次数

        # Timed runs  # 计时运行
        run_times = []  # 初始化运行时间列表
        for _ in TQDM(range(num_runs), desc=onnx_file):  # 进行计时运行
            start_time = time.time()  # 记录开始时间
            sess.run([output_name], {input_name: input_data})  # 执行推理
            run_times.append((time.time() - start_time) * 1000)  # Convert to milliseconds  # 转换为毫秒

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)  # sigma clipping  # 进行 sigma 剪切
        return np.mean(run_times), np.std(run_times)  # 返回平均运行时间和标准差

    def generate_table_row(self, model_name, t_onnx, t_engine, model_info):
        """Generates a table row string with model performance metrics including inference times and model details.  # 生成包含模型性能指标（包括推理时间和模型详细信息）的表格行字符串。"""
        layers, params, gradients, flops = model_info  # 解包模型信息
        return (  # 返回格式化的字符串
            f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.1f}±{t_onnx[1]:.1f} ms | {t_engine[0]:.1f}±"
            f"{t_engine[1]:.1f} ms | {params / 1e6:.1f} | {flops:.1f} |"
        )

    @staticmethod
    def generate_results_dict(model_name, t_onnx, t_engine, model_info):
        """Generates a dictionary of profiling results including model name, parameters, GFLOPs, and speed metrics.  # 生成包含模型名称、参数、GFLOPs 和速度指标的性能分析结果字典。"""
        layers, params, gradients, flops = model_info  # 解包模型信息
        return {  # 返回结果字典
            "model/name": model_name,  # 模型名称
            "model/parameters": params,  # 模型参数
            "model/GFLOPs": round(flops, 3),  # 模型 GFLOPs
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),  # ONNX 模型速度
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),  # TensorRT 模型速度
        }

    @staticmethod
    def print_table(table_rows):
        """Prints a formatted table of model profiling results, including speed and accuracy metrics.  # 打印格式化的模型性能分析结果表，包括速度和准确性指标。"""
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"  # 获取 GPU 名称
        headers = [  # 表头
            "Model",  # 模型
            "size<br><sup>(pixels)",  # 大小（像素）
            "mAP<sup>val<br>50-95",  # mAP 值（50-95）
            f"Speed<br><sup>CPU ({get_cpu_info()}) ONNX<br>(ms)",  # CPU 上的 ONNX 模型速度
            f"Speed<br><sup>{gpu} TensorRT<br>(ms)",  # GPU 上的 TensorRT 模型速度
            "params<br><sup>(M)",  # 参数数量（百万）
            "FLOPs<br><sup>(B)",  # FLOPs（十亿）
        ]
        header = "|" + "|".join(f" {h} " for h in headers) + "|"  # 格式化表头
        separator = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"  # 格式化分隔符

        print(f"\n\n{header}")  # 打印表头
        print(separator)  # 打印分隔符
        for row in table_rows:  # 遍历每一行
            print(row)  # 打印行