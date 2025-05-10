# 导入必要的库
import os
from glob import glob


def create_new_run_directory(base_output_dir: str) -> str:
    """Creates a new sequentially numbered run directory.
    创建一个新的按顺序编号的运行目录。

    This function ensures that a new directory is created inside `base_output_dir`,
    following a numeric naming convention (e.g., `1`, `2`, `3`, etc.). It scans
    existing directories in `base_output_dir`, determines the highest existing
    number, and creates a new directory with an incremented number.
    该函数确保在`base_output_dir`内创建一个新目录，遵循数字命名约定（例如`1`、`2`、`3`等）。
    它会扫描`base_output_dir`中的现有目录，确定最大的现有编号，并创建一个递增编号的新目录。

    Args:
        base_output_dir (str):
            The base directory where the new run directory will be created.
            If it does not exist, it will be created.
            新运行目录将创建的基础目录。如果不存在，则会创建该目录。

    Returns:
        str: The absolute path to the newly created run directory.
        新创建的运行目录的绝对路径。
    """
    # 获取基础目录的绝对路径
    base_output_dir = os.path.abspath(base_output_dir)
    # 创建基础目录（如果不存在）
    os.makedirs(base_output_dir, exist_ok=True)

    # 获取基础目录中所有子目录的路径
    existing_run_dirs = [d for d in glob(os.path.join(base_output_dir, "*")) if os.path.isdir(d)]
    # 存储所有子目录的编号
    existing_numbers = []
    # 遍历所有子目录，提取编号
    for dir_path in existing_run_dirs:
        try:
            # 获取目录名并转换为整数
            dir_name = os.path.basename(dir_path)
            existing_numbers.append(int(dir_name))
        except ValueError:
            # 如果目录名不是数字，跳过
            continue

    # 计算新目录的编号（最大编号加1，如果没有编号则从1开始）
    new_run_number = max(existing_numbers, default=0) + 1
    # 创建新目录的路径
    new_run_dir = os.path.join(base_output_dir, str(new_run_number))
    # 创建新目录
    os.makedirs(new_run_dir, exist_ok=True)
    # 返回新目录的绝对路径
    return new_run_dir