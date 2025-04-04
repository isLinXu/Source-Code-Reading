# Copyright (c) Facebook, Inc. and its affiliates.
import importlib  # 导入importlib模块，用于动态导入模块
import numpy as np  # 导入numpy库，常用于数值计算
import os  # 导入os模块，用于操作系统相关功能
import re  # 导入正则表达式模块
import subprocess  # 导入子进程模块，用于执行系统命令
import sys  # 导入sys模块，提供系统相关的变量和函数
from collections import defaultdict  # 从collections导入defaultdict类型
import PIL  # 导入PIL(Python Imaging Library)，用于图像处理
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，PyTorch的计算机视觉工具包
from tabulate import tabulate  # 从tabulate导入tabulate函数，用于格式化表格输出

__all__ = ["collect_env_info"]  # 指定模块公开的函数列表


def collect_torch_env():
    try:
        import torch.__config__  # 尝试导入torch.__config__模块

        return torch.__config__.show()  # 返回PyTorch配置信息
    except ImportError:
        # compatible with older versions of pytorch
        # 兼容旧版PyTorch
        from torch.utils.collect_env import get_pretty_env_info  # 导入环境信息收集函数

        return get_pretty_env_info()  # 返回格式化的环境信息


def get_env_module():
    var_name = "DETECTRON2_ENV_MODULE"  # 定义环境变量名
    return var_name, os.environ.get(var_name, "<not set>")  # 返回环境变量名和其值，若未设置则返回"<not set>"


def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")  # 构建cuobjdump工具的路径
        if os.path.isfile(cuobjdump):  # 检查cuobjdump工具是否存在
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True
            )  # 执行cuobjdump命令获取so文件的ELF信息
            output = output.decode("utf-8").strip().split("\n")  # 解码输出并按行分割
            arch = []  # 初始化架构列表
            for line in output:  # 遍历每一行输出
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]  # 提取SM架构版本号
                arch.append(".".join(line))  # 将版本号格式化并添加到架构列表
            arch = sorted(set(arch))  # 排序并去重架构列表
            return ", ".join(arch)  # 返回以逗号分隔的架构列表字符串
        else:
            return so_file + "; cannot find cuobjdump"  # 如果找不到cuobjdump工具则返回错误信息
    except Exception:
        # unhandled failure
        # 未处理的异常
        return so_file  # 发生异常时返回so文件路径


def collect_env_info():
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    # 检查是否有可用的GPU，对CUDA和ROCM都适用
    torch_version = torch.__version__  # 获取PyTorch版本

    # NOTE that CUDA_HOME/ROCM_HOME could be None even when CUDA runtime libs are functional
    # 注意，即使CUDA运行时库可用，CUDA_HOME/ROCM_HOME也可能为None
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME  # 导入CUDA和ROCM的路径变量

    has_rocm = False  # 初始化has_rocm为False
    if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
        has_rocm = True  # 如果PyTorch支持HIP且ROCM_HOME存在，则设置has_rocm为True
    has_cuda = has_gpu and (not has_rocm)  # 如果有GPU且不是ROCM，则为CUDA

    data = []  # 初始化数据列表
    data.append(("sys.platform", sys.platform))  # check-template.yml depends on it
    # 添加系统平台信息，check-template.yml依赖此项
    data.append(("Python", sys.version.replace("\n", "")))  # 添加Python版本信息，移除换行符
    data.append(("numpy", np.__version__))  # 添加NumPy版本信息

    try:
        import detectron2  # noqa  # 尝试导入detectron2

        data.append(
            ("detectron2", detectron2.__version__ + " @" + os.path.dirname(detectron2.__file__))
        )  # 添加detectron2版本和路径信息
    except ImportError:
        data.append(("detectron2", "failed to import"))  # 导入失败时添加错误信息
    except AttributeError:
        data.append(("detectron2", "imported a wrong installation"))  # 导入错误安装时添加错误信息

    try:
        import detectron2._C as _C  # 尝试导入detectron2的C++扩展
    except ImportError as e:
        data.append(("detectron2._C", f"not built correctly: {e}"))  # 导入失败时添加错误信息

        # print system compilers when extension fails to build
        # 当扩展构建失败时打印系统编译器信息
        if sys.platform != "win32":  # don't know what to do for windows
            # 对于非Windows平台
            try:
                # this is how torch/utils/cpp_extensions.py choose compiler
                # 这是torch/utils/cpp_extensions.py选择编译器的方式
                cxx = os.environ.get("CXX", "c++")  # 获取C++编译器，默认为"c++"
                cxx = subprocess.check_output("'{}' --version".format(cxx), shell=True)  # 获取编译器版本
                cxx = cxx.decode("utf-8").strip().split("\n")[0]  # 解码并获取第一行信息
            except subprocess.SubprocessError:
                cxx = "Not found"  # 如果发生子进程错误，设置为"Not found"
            data.append(("Compiler ($CXX)", cxx))  # 添加编译器信息

            if has_cuda and CUDA_HOME is not None:  # 如果有CUDA且CUDA_HOME存在
                try:
                    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")  # 构建nvcc编译器路径
                    nvcc = subprocess.check_output("'{}' -V".format(nvcc), shell=True)  # 获取nvcc版本
                    nvcc = nvcc.decode("utf-8").strip().split("\n")[-1]  # 解码并获取最后一行信息
                except subprocess.SubprocessError:
                    nvcc = "Not found"  # 如果发生子进程错误，设置为"Not found"
                data.append(("CUDA compiler", nvcc))  # 添加CUDA编译器信息
        if has_cuda and sys.platform != "win32":  # 如果有CUDA且不是Windows平台
            try:
                so_file = importlib.util.find_spec("detectron2._C").origin  # 获取_C扩展的文件路径
            except (ImportError, AttributeError):
                pass  # 忽略导入错误
            else:
                data.append(
                    ("detectron2 arch flags", detect_compute_compatibility(CUDA_HOME, so_file))
                )  # 添加detectron2架构标志信息
    else:
        # print compilers that are used to build extension
        # 打印用于构建扩展的编译器
        data.append(("Compiler", _C.get_compiler_version()))  # 添加编译器版本信息
        data.append(("CUDA compiler", _C.get_cuda_version()))  # cuda or hip
        # 添加CUDA编译器版本信息，可能是CUDA或HIP
        if has_cuda and getattr(_C, "has_cuda", lambda: True)():  # 如果有CUDA且_C.has_cuda()返回True
            data.append(
                ("detectron2 arch flags", detect_compute_compatibility(CUDA_HOME, _C.__file__))
            )  # 添加detectron2架构标志信息

    data.append(get_env_module())  # 添加环境模块信息
    data.append(("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__)))  # 添加PyTorch版本和路径信息
    data.append(("PyTorch debug build", torch.version.debug))  # 添加PyTorch调试构建信息

    if not has_gpu:  # 如果没有可用的GPU
        has_gpu_text = "No: torch.cuda.is_available() == False"  # 设置GPU可用性文本为否
    else:
        has_gpu_text = "Yes"  # 设置GPU可用性文本为是
    data.append(("GPU available", has_gpu_text))  # 添加GPU可用性信息
    if has_gpu:  # 如果有可用的GPU
        devices = defaultdict(list)  # 初始化设备字典
        for k in range(torch.cuda.device_count()):  # 遍历所有CUDA设备
            cap = ".".join((str(x) for x in torch.cuda.get_device_capability(k)))  # 获取设备计算能力
            name = torch.cuda.get_device_name(k) + f" (arch={cap})"  # 构建设备名称字符串
            devices[name].append(str(k))  # 将设备ID添加到对应设备名称下
        for name, devids in devices.items():  # 遍历所有设备名称和ID
            data.append(("GPU " + ",".join(devids), name))  # 添加GPU信息

        if has_rocm:  # 如果使用ROCM
            msg = " - invalid!" if not (ROCM_HOME and os.path.isdir(ROCM_HOME)) else ""  # 检查ROCM_HOME是否有效
            data.append(("ROCM_HOME", str(ROCM_HOME) + msg))  # 添加ROCM_HOME信息
        else:  # 如果使用CUDA
            try:
                from torch.utils.collect_env import get_nvidia_driver_version, run as _run  # 导入NVIDIA驱动程序版本获取函数

                data.append(("Driver version", get_nvidia_driver_version(_run)))  # 添加驱动程序版本信息
            except Exception:
                pass  # 忽略异常
            msg = " - invalid!" if not (CUDA_HOME and os.path.isdir(CUDA_HOME)) else ""  # 检查CUDA_HOME是否有效
            data.append(("CUDA_HOME", str(CUDA_HOME) + msg))  # 添加CUDA_HOME信息

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)  # 获取CUDA架构列表环境变量
            if cuda_arch_list:  # 如果环境变量存在
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))  # 添加CUDA架构列表信息
    data.append(("Pillow", PIL.__version__))  # 添加Pillow版本信息

    try:
        data.append(
            (
                "torchvision",
                str(torchvision.__version__) + " @" + os.path.dirname(torchvision.__file__),
            )
        )  # 添加torchvision版本和路径信息
        if has_cuda:  # 如果有CUDA
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin  # 获取torchvision._C扩展的文件路径
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)  # 检测计算兼容性
                data.append(("torchvision arch flags", msg))  # 添加torchvision架构标志信息
            except (ImportError, AttributeError):
                data.append(("torchvision._C", "Not found"))  # 找不到扩展时添加错误信息
    except AttributeError:
        data.append(("torchvision", "unknown"))  # 发生属性错误时添加未知信息

    try:
        import fvcore  # 尝试导入fvcore

        data.append(("fvcore", fvcore.__version__))  # 添加fvcore版本信息
    except (ImportError, AttributeError):
        pass  # 忽略导入错误

    try:
        import iopath  # 尝试导入iopath

        data.append(("iopath", iopath.__version__))  # 添加iopath版本信息
    except (ImportError, AttributeError):
        pass  # 忽略导入错误

    try:
        import cv2  # 尝试导入cv2

        data.append(("cv2", cv2.__version__))  # 添加cv2版本信息
    except (ImportError, AttributeError):
        data.append(("cv2", "Not found"))  # 找不到cv2时添加错误信息
    env_str = tabulate(data) + "\n"  # 使用tabulate格式化数据为表格并添加换行符
    env_str += collect_torch_env()  # 添加PyTorch环境信息
    return env_str  # 返回完整的环境信息字符串


def test_nccl_ops():
    num_gpu = torch.cuda.device_count()  # 获取GPU数量
    if os.access("/tmp", os.W_OK):  # 检查是否有/tmp目录的写权限
        import torch.multiprocessing as mp  # 导入PyTorch多进程模块

        dist_url = "file:///tmp/nccl_tmp_file"  # 设置分布式URL
        print("Testing NCCL connectivity ... this should not hang.")  # 打印测试信息
        mp.spawn(_test_nccl_worker, nprocs=num_gpu, args=(num_gpu, dist_url), daemon=False)  # 生成NCCL测试工作进程
        print("NCCL succeeded.")  # 打印成功信息


def _test_nccl_worker(rank, num_gpu, dist_url):
    import torch.distributed as dist  # 导入PyTorch分布式模块

    dist.init_process_group(backend="NCCL", init_method=dist_url, rank=rank, world_size=num_gpu)  # 初始化进程组
    dist.barrier(device_ids=[rank])  # 设置进程同步点


if __name__ == "__main__":
    try:
        from detectron2.utils.collect_env import collect_env_info as f  # 尝试从detectron2导入collect_env_info函数

        print(f())  # 打印环境信息
    except ImportError:
        print(collect_env_info())  # 导入失败时使用本地函数打印环境信息

    if torch.cuda.is_available():  # 如果有可用的GPU
        num_gpu = torch.cuda.device_count()  # 获取GPU数量
        for k in range(num_gpu):  # 遍历所有GPU
            device = f"cuda:{k}"  # 构建设备标识符
            try:
                x = torch.tensor([1, 2.0], dtype=torch.float32)  # 创建测试张量
                x = x.to(device)  # 尝试将张量移动到GPU
            except Exception as e:
                print(
                    f"Unable to copy tensor to device={device}: {e}. "
                    "Your CUDA environment is broken."
                )  # 打印错误信息
        if num_gpu > 1:  # 如果有多个GPU
            test_nccl_ops()  # 测试NCCL操作
