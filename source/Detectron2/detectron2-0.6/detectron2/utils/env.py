# Copyright (c) Facebook, Inc. and its affiliates.
import importlib  # 导入动态导入模块的库
import importlib.util  # 导入importlib的工具函数
import logging  # 导入日志模块
import numpy as np  # 导入numpy库
import os  # 导入操作系统接口模块
import random  # 导入随机数生成模块
import sys  # 导入系统相关模块
from datetime import datetime  # 从datetime模块导入datetime类
import torch  # 导入PyTorch库

__all__ = ["seed_all_rng"]  # 指定模块公开的函数列表


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])  # 将PyTorch版本转换为元组
"""
PyTorch version as a tuple of 2 ints. Useful for comparison.

PyTorch版本作为一个包含2个整数的元组。方便进行版本比较。
"""


DOC_BUILDING = os.getenv("_DOC_BUILDING", False)  # set in docs/conf.py  # 获取环境变量，检查是否在构建文档
"""
Whether we're building documentation.

是否正在构建文档。
"""


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
        
    设置torch、numpy和python中随机数生成器(RNG)的种子。

    参数:
        seed (int): 如果为None，将使用一个强随机种子。
    """
    if seed is None:  # 如果没有提供种子
        seed = (
            os.getpid()  # 获取当前进程ID
            + int(datetime.now().strftime("%S%f"))  # 加上当前时间的秒和微秒
            + int.from_bytes(os.urandom(2), "big")  # 加上2字节的系统随机数
        )
        logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
        logger.info("Using a generated random seed {}".format(seed))  # 记录生成的随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    random.seed(seed)  # 设置Python内置random模块的随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python哈希种子环境变量


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
# 根据完整路径导入模块的函数，来源于stackoverflow
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)  # 从文件路径创建模块规范
    module = importlib.util.module_from_spec(spec)  # 从规范创建模块
    spec.loader.exec_module(module)  # 执行模块
    if make_importable:  # 如果需要使模块可导入
        sys.modules[module_name] = module  # 将模块添加到sys.modules字典中
    return module  # 返回导入的模块


def _configure_libraries():
    """
    Configurations for some libraries.
    
    对一些库进行配置。
    """
    # An environment option to disable `import cv2` globally,
    # in case it leads to negative performance impact
    # 一个环境选项，用于全局禁用`import cv2`，
    # 以防它导致负面性能影响
    disable_cv2 = int(os.environ.get("DETECTRON2_DISABLE_CV2", False))  # 获取环境变量决定是否禁用cv2
    if disable_cv2:  # 如果需要禁用cv2
        sys.modules["cv2"] = None  # 将cv2模块设为None，禁止导入
    else:
        # Disable opencl in opencv since its interaction with cuda often has negative effects
        # This envvar is supported after OpenCV 3.4.0
        # 禁用opencv中的opencl，因为它与cuda的交互通常会产生负面影响
        # 此环境变量在OpenCV 3.4.0之后支持
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"  # 设置环境变量禁用OpenCL
        try:
            import cv2  # 尝试导入cv2

            if int(cv2.__version__.split(".")[0]) >= 3:  # 如果OpenCV版本大于等于3
                cv2.ocl.setUseOpenCL(False)  # 禁用OpenCL
        except ModuleNotFoundError:
            # Other types of ImportError, if happened, should not be ignored.
            # Because a failed opencv import could mess up address space
            # https://github.com/skvark/opencv-python/issues/381
            # 其他类型的ImportError（如果发生）不应被忽略。
            # 因为失败的opencv导入可能会搞乱地址空间
            # https://github.com/skvark/opencv-python/issues/381
            pass  # 忽略ModuleNotFoundError错误

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))  # 获取模块版本作为元组

    # fmt: off
    assert get_version(torch) >= (1, 4), "Requires torch>=1.4"  # 检查PyTorch版本是否大于等于1.4
    import fvcore  # 导入fvcore
    assert get_version(fvcore, 3) >= (0, 1, 2), "Requires fvcore>=0.1.2"  # 检查fvcore版本是否大于等于0.1.2
    import yaml  # 导入yaml
    assert get_version(yaml) >= (5, 1), "Requires pyyaml>=5.1"  # 检查pyyaml版本是否大于等于5.1
    # fmt: on


_ENV_SETUP_DONE = False  # 标记环境设置是否已完成的全局变量


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $DETECTRON2_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    
    执行环境设置工作。默认设置不执行任何操作，但此函数允许用户在
    $DETECTRON2_ENV_MODULE环境变量中指定Python源文件或模块，
    以执行可能对其计算环境必要的自定义设置工作。
    """
    global _ENV_SETUP_DONE  # 使用全局变量
    if _ENV_SETUP_DONE:  # 如果环境设置已完成
        return  # 直接返回
    _ENV_SETUP_DONE = True  # 标记环境设置已完成

    _configure_libraries()  # 配置库

    custom_module_path = os.environ.get("DETECTRON2_ENV_MODULE")  # 获取自定义环境模块路径

    if custom_module_path:  # 如果有自定义模块路径
        setup_custom_environment(custom_module_path)  # 设置自定义环境
    else:
        # The default setup is a no-op
        # 默认设置不执行任何操作
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    
    通过导入Python源文件或模块加载自定义环境设置，
    并运行设置函数。
    """
    if custom_module.endswith(".py"):  # 如果是Python文件
        module = _import_file("detectron2.utils.env.custom_module", custom_module)  # 导入文件
    else:  # 如果是模块名
        module = importlib.import_module(custom_module)  # 导入模块
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)  # 确保模块有setup_environment可调用属性
    module.setup_environment()  # 调用设置函数


def fixup_module_metadata(module_name, namespace, keys=None):
    """
    Fix the __qualname__ of module members to be their exported api name, so
    when they are referenced in docs, sphinx can find them. Reference:
    https://github.com/python-trio/trio/blob/6754c74eacfad9cc5c92d5c24727a2f3b620624e/trio/_util.py#L216-L241
    
    修复模块成员的__qualname__为它们的导出API名称，这样
    当它们在文档中被引用时，sphinx可以找到它们。参考：
    https://github.com/python-trio/trio/blob/6754c74eacfad9cc5c92d5c24727a2f3b620624e/trio/_util.py#L216-L241
    """
    if not DOC_BUILDING:  # 如果不是在构建文档
        return  # 直接返回
    seen_ids = set()  # 创建已处理对象ID集合，避免重复处理

    def fix_one(qualname, name, obj):
        # avoid infinite recursion (relevant when using
        # typing.Generic, for example)
        # 避免无限递归（例如使用typing.Generic时相关）
        if id(obj) in seen_ids:  # 如果对象已处理过
            return  # 直接返回
        seen_ids.add(id(obj))  # 将对象ID添加到已处理集合

        mod = getattr(obj, "__module__", None)  # 获取对象的模块
        if mod is not None and (mod.startswith(module_name) or mod.startswith("fvcore.")):  # 如果模块名以module_name或fvcore.开头
            obj.__module__ = module_name  # 设置对象的模块为module_name
            # Modules, unlike everything else in Python, put fully-qualitied
            # names into their __name__ attribute. We check for "." to avoid
            # rewriting these.
            # 模块与Python中的其他对象不同，它们将完全限定名称
            # 放入其__name__属性。我们检查"."以避免重写这些。
            if hasattr(obj, "__name__") and "." not in obj.__name__:  # 如果对象有__name__属性且不包含点
                obj.__name__ = name  # 设置对象的名称
                obj.__qualname__ = qualname  # 设置对象的限定名称
            if isinstance(obj, type):  # 如果对象是类
                for attr_name, attr_value in obj.__dict__.items():  # 遍历类的所有属性
                    fix_one(objname + "." + attr_name, attr_name, attr_value)  # 递归修复属性

    if keys is None:  # 如果未提供键列表
        keys = namespace.keys()  # 使用命名空间中的所有键
    for objname in keys:  # 遍历所有键
        if not objname.startswith("_"):  # 如果不是私有属性
            obj = namespace[objname]  # 获取对象
            fix_one(objname, objname, obj)  # 修复对象的元数据
