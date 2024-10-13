import importlib
import os

def import_modules(models_dir, namespace):
    """
    导入指定目录下的所有模块。

    参数:
    models_dir (str): 包含模块文件的目录路径。
    namespace (str): 导入模块时要使用的命名空间。
    """
    # 遍历目录中的所有文件
    for file in os.listdir(models_dir):
        # 获取文件的完整路径
        path = os.path.join(models_dir, file)
        # 检查文件是否为Python模块文件（不是以_或.开头，且以.py结尾）
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and file.endswith(".py")
        ):
            # 提取模块名称（去掉.py后缀）
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            # 动态导入模块
            importlib.import_module(namespace + "." + model_name)