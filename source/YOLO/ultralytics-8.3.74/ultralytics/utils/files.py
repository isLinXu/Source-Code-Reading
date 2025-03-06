# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib  # 导入contextlib模块，用于上下文管理
import glob  # 导入glob模块，用于文件路径匹配
import os  # 导入os模块，用于与操作系统交互
import shutil  # 导入shutil模块，用于文件和目录的操作
import tempfile  # 导入tempfile模块，用于创建临时文件
from contextlib import contextmanager  # 从contextlib导入contextmanager装饰器
from datetime import datetime  # 从datetime导入datetime类
from pathlib import Path  # 从pathlib导入Path类，用于路径操作


class WorkingDirectory(contextlib.ContextDecorator):
    """
    A context manager and decorator for temporarily changing the working directory.
    上下文管理器和装饰器，用于临时更改工作目录。

    This class allows for the temporary change of the working directory using a context manager or decorator.
    此类允许使用上下文管理器或装饰器临时更改工作目录。
    It ensures that the original working directory is restored after the context or decorated function completes.
    它确保在上下文或装饰的函数完成后，原始工作目录被恢复。

    Attributes:
        dir (Path): The new directory to switch to.
        dir (Path): 要切换到的新目录。
        cwd (Path): The original current working directory before the switch.
        cwd (Path): 切换前的原始当前工作目录。

    Methods:
        __enter__: Changes the current directory to the specified directory.
        __enter__: 将当前目录更改为指定目录。
        __exit__: Restores the original working directory on context exit.
        __exit__: 在上下文退出时恢复原始工作目录。

    Examples:
        Using as a context manager:
        >>> with WorkingDirectory('/path/to/new/dir'):
        >>> # Perform operations in the new directory
        >>>     pass

        Using as a decorator:
        >>> @WorkingDirectory('/path/to/new/dir')
        >>> def some_function():
        >>> # Perform operations in the new directory
        >>>     pass
    """

    def __init__(self, new_dir):
        """Sets the working directory to 'new_dir' upon instantiation for use with context managers or decorators.
        在实例化时将工作目录设置为'new_dir'，以便与上下文管理器或装饰器一起使用。"""
        self.dir = new_dir  # new dir 新目录
        self.cwd = Path.cwd().resolve()  # current dir 当前目录

    def __enter__(self):
        """Changes the current working directory to the specified directory upon entering the context.
        进入上下文时将当前工作目录更改为指定目录。"""
        os.chdir(self.dir)  # 更改工作目录

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        """Restores the original working directory when exiting the context.
        退出上下文时恢复原始工作目录。"""
        os.chdir(self.cwd)  # 恢复原始工作目录


@contextmanager
def spaces_in_path(path):
    """
    Context manager to handle paths with spaces in their names. If a path contains spaces, it replaces them with
    underscores, copies the file/directory to the new path, executes the context code block, then copies the
    file/directory back to its original location.
    上下文管理器，用于处理名称中包含空格的路径。如果路径包含空格，则将其替换为下划线，将文件/目录复制到新路径，执行上下文代码块，然后将文件/目录复制回其原始位置。

    Args:
        path (str | Path): The original path that may contain spaces.
        path (str | Path): 可能包含空格的原始路径。

    Yields:
        (Path): Temporary path with spaces replaced by underscores if spaces were present, otherwise the original path.
        (Path): 如果路径中存在空格，则返回临时路径（空格被替换为下划线），否则返回原始路径。

    Examples:
        Use the context manager to handle paths with spaces:
        >>> from ultralytics.utils.files import spaces_in_path
        >>> with spaces_in_path('/path/with spaces') as new_path:
        >>> # Your code here
    """
    # If path has spaces, replace them with underscores
    if " " in str(path):  # 如果路径中有空格
        string = isinstance(path, str)  # input type 输入类型
        path = Path(path)  # 转换为Path对象

        # Create a temporary directory and construct the new path
        with tempfile.TemporaryDirectory() as tmp_dir:  # 创建临时目录
            tmp_path = Path(tmp_dir) / path.name.replace(" ", "_")  # 构建新路径，空格替换为下划线

            # Copy file/directory
            if path.is_dir():  # 如果是目录
                # tmp_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, tmp_path)  # 复制目录
            elif path.is_file():  # 如果是文件
                tmp_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
                shutil.copy2(path, tmp_path)  # 复制文件

            try:
                # Yield the temporary path
                yield str(tmp_path) if string else tmp_path  # 返回临时路径

            finally:
                # Copy file/directory back
                if tmp_path.is_dir():  # 如果是目录
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)  # 将目录复制回原位置
                elif tmp_path.is_file():  # 如果是文件
                    shutil.copy2(tmp_path, path)  # 将文件复制回原位置

    else:
        # If there are no spaces, just yield the original path
        yield path  # 如果没有空格，返回原始路径


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    增加文件或目录路径，例如：runs/exp --> runs/exp{sep}2, runs/exp{sep}3，等等。

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If `mkdir` is set to True, the path will be created as a
    directory if it does not already exist.
    如果路径存在且`exist_ok`为False，则路径将通过在路径末尾附加数字和`sep`进行递增。如果路径是文件，则文件扩展名将被保留。如果路径是目录，则数字将直接附加到路径末尾。如果`mkdir`设置为True，则如果路径不存在，将创建该路径作为目录。

    Args:
        path (str | pathlib.Path): Path to increment.
        path (str | pathlib.Path): 要递增的路径。
        exist_ok (bool): If True, the path will not be incremented and returned as-is.
        exist_ok (bool): 如果为True，则路径不会递增，按原样返回。
        sep (str): Separator to use between the path and the incrementation number.
        sep (str): 在路径和递增数字之间使用的分隔符。
        mkdir (bool): Create a directory if it does not exist.
        mkdir (bool): 如果目录不存在，则创建目录。

    Returns:
        (pathlib.Path): Incremented path.
        (pathlib.Path): 递增后的路径。

    Examples:
        Increment a directory path:
        >>> from pathlib import Path
        >>> path = Path("runs/exp")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp2

        Increment a file path:
        >>> path = Path("runs/exp/results.txt")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp/results2.txt
    """
    path = Path(path)  # os-agnostic 兼容操作系统
    if path.exists() and not exist_ok:  # 如果路径存在且exist_ok为False
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")  # 获取路径和扩展名

        # Method 1
        for n in range(2, 9999):  # 循环递增数字
            p = f"{path}{sep}{n}{suffix}"  # 递增路径
            if not os.path.exists(p):  # 如果路径不存在
                break  # 退出循环
        path = Path(p)  # 更新路径

    if mkdir:  # 如果需要创建目录
        path.mkdir(parents=True, exist_ok=True)  # 创建目录

    return path  # 返回递增后的路径


def file_age(path=__file__):
    """Return days since the last modification of the specified file.
    返回指定文件自上次修改以来的天数。"""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # 计算时间差
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    """Returns the file modification date in 'YYYY-M-D' format.
    返回文件的修改日期，格式为'YYYY-M-D'。"""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)  # 获取文件的修改时间
    return f"{t.year}-{t.month}-{t.day}"  # 返回格式化的日期


def file_size(path):
    """Returns the size of a file or directory in megabytes (MB).
    返回文件或目录的大小（以兆字节为单位）。"""
    if isinstance(path, (str, Path)):  # 如果路径是字符串或Path对象
        mb = 1 << 20  # bytes to MiB (1024 ** 2) 字节转换为MiB（1024的平方）
        path = Path(path)  # 转换为Path对象
        if path.is_file():  # 如果是文件
            return path.stat().st_size / mb  # 返回文件大小（以MB为单位）
        elif path.is_dir():  # 如果是目录
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb  # 返回目录中所有文件的总大小（以MB为单位）
    return 0.0  # 如果路径无效，返回0.0


def get_latest_run(search_dir="."):
    """Returns the path to the most recent 'last.pt' file in the specified directory for resuming training.
    返回指定目录中最近的'last.pt'文件的路径，以便恢复训练。"""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)  # 查找匹配的文件
    return max(last_list, key=os.path.getctime) if last_list else ""  # 返回最新文件的路径，如果没有找到则返回空字符串


def update_models(model_names=("yolo11n.pt",), source_dir=Path("."), update_names=False):
    """
    Updates and re-saves specified YOLO models in an 'updated_models' subdirectory.
    更新并重新保存指定的YOLO模型到'updated_models'子目录。

    Args:
        model_names (Tuple[str, ...]): Model filenames to update.
        model_names (Tuple[str, ...]): 要更新的模型文件名。
        source_dir (Path): Directory containing models and target subdirectory.
        source_dir (Path): 包含模型和目标子目录的目录。
        update_names (bool): Update model names from a data YAML.
        update_names (bool): 从数据YAML更新模型名称。

    Examples:
        Update specified YOLO models and save them in 'updated_models' subdirectory:
        >>> from ultralytics.utils.files import update_models
        >>> model_names = ("yolo11n.pt", "yolov8s.pt")
        >>> update_models(model_names, source_dir=Path("/models"), update_names=True)
    """
    from ultralytics import YOLO  # 从ultralytics导入YOLO类
    from ultralytics.nn.autobackend import default_class_names  # 从ultralytics.nn.autobackend导入默认类名

    target_dir = source_dir / "updated_models"  # 目标目录
    target_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在

    for model_name in model_names:  # 遍历模型名称
        model_path = source_dir / model_name  # 获取模型路径
        print(f"Loading model from {model_path}")  # 打印加载模型信息

        # Load model
        model = YOLO(model_path)  # 加载模型
        model.half()  # 转换为半精度
        if update_names:  # 如果需要更新模型名称
            model.model.names = default_class_names("coco8.yaml")  # 从数据集YAML更新模型名称

        # Define new save path
        save_path = target_dir / model_name  # 定义新的保存路径

        # Save model using model.save()
        print(f"Re-saving {model_name} model to {save_path}")  # 打印重新保存模型信息
        model.save(save_path)  # 保存模型