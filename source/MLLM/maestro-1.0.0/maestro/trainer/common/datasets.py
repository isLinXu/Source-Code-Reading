import json
import os
from typing import Any, Callable, Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROBOFLOW_JSONL_FILENAME = "annotations.jsonl"


class RoboflowJSONLDataset(Dataset):
    """
    Dataset for loading images and annotations from a Roboflow JSONL dataset.
    用于从Roboflow JSONL数据集中加载图像和注释的数据集。

    Args:
        jsonl_file_path (str): Path to the JSONL file containing dataset entries.
            包含数据集条目的JSONL文件路径。
        image_directory_path (str): Path to the directory containing images.
            包含图像的目录路径。
    """

    def __init__(self, jsonl_file_path: str, image_directory_path: str) -> None:
        # 检查JSONL文件是否存在
        if not os.path.exists(jsonl_file_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")
        # 检查图像目录是否存在
        if not os.path.isdir(image_directory_path):
            raise NotADirectoryError(f"Image directory not found: {image_directory_path}")

        # 初始化图像目录路径
        self.image_directory_path = image_directory_path
        # 加载JSONL文件中的条目
        self.entries = self._load_entries(jsonl_file_path)

    @staticmethod
    def _load_entries(jsonl_file_path: str) -> list[dict]:
        # 打开JSONL文件并逐行加载
        with open(jsonl_file_path) as file:
            try:
                return [json.loads(line) for line in file]
            except json.JSONDecodeError:
                print("Error parsing JSONL file.")
                raise

    def __len__(self) -> int:
        # 返回数据集中的条目数量
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict]:
        # 检查索引是否超出范围
        if idx >= len(self.entries):
            raise IndexError("Index out of range")

        # 获取指定索引的条目
        entry = self.entries[idx]
        # 构建图像路径
        image_path = os.path.join(self.image_directory_path, entry["image"])
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # 打开图像并转换为RGB格式
            image = Image.open(image_path).convert("RGB")
        except OSError as e:
            raise OSError(f"Error opening image file {image_path}: {e}")

        # 返回图像和对应的条目
        return image, entry


def load_split_dataset(dataset_location: str, split_name: str) -> Optional[Dataset]:
    """
    Load a dataset split from the specified location.
    从指定位置加载数据集的分割部分。

    Args:
        dataset_location (str): Path to the dataset directory.
            数据集目录的路径。
        split_name (str): Name of the dataset split (e.g., "train", "valid", "test").
            数据集分割的名称（例如"train"、"valid"、"test"）。

    Returns:
        Optional[Dataset]: A dataset object for the split, or `None` if the split does not exist.
            分割部分的数据集对象，如果分割部分不存在则返回`None`。
    """
    # 构建JSONL文件路径和图像目录路径
    jsonl_file_path = os.path.join(dataset_location, split_name, ROBOFLOW_JSONL_FILENAME)
    image_directory_path = os.path.join(dataset_location, split_name)

    # 检查JSONL文件和图像目录是否存在
    if not os.path.exists(jsonl_file_path) or not os.path.exists(image_directory_path):
        print(f"Dataset split {split_name} not found at {dataset_location}")
        return None

    # 返回RoboflowJSONLDataset对象
    return RoboflowJSONLDataset(jsonl_file_path, image_directory_path)


def create_data_loaders(
    dataset_location: str,
    train_batch_size: int,
    train_collect_fn: Callable[[list[Any]], Any],
    train_num_workers: int = 0,
    test_batch_size: Optional[int] = None,
    test_collect_fn: Optional[Callable[[list[Any]], Any]] = None,
    test_num_workers: Optional[int] = None,
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoader instances for training, validation, and testing datasets.
    为训练、验证和测试数据集创建DataLoader实例。

    Args:
        dataset_location (str): Path to the dataset directory.
            数据集目录的路径。
        train_batch_size (int): Batch size for the training dataset. Must be a positive integer.
            训练数据集的批量大小，必须为正整数。
        train_collect_fn (Callable[[List[Any]], Any]): Function to collate training samples into a batch.
            将训练样本整理成批次的函数。
        train_num_workers (int): Number of worker threads for the training DataLoader. Defaults to 0.
            训练DataLoader的工作线程数，默认为0。
        test_batch_size (Optional[int]): Batch size for validation and test datasets. Defaults to the value of
            `train_batch_size` if not provided.
            验证和测试数据集的批量大小，如果未提供则默认为`train_batch_size`的值。
        test_collect_fn (Optional[Callable[[List[Any]], Any]]): Function to collate validation and test samples into a
            batch. Defaults to the value of `train_collect_fn` if not provided.
            将验证和测试样本整理成批次的函数，如果未提供则默认为`train_collect_fn`的值。
        test_num_workers (Optional[int]): Number of worker threads for validation and test DataLoaders. Defaults to the
            value of `train_num_workers` if not provided.
            验证和测试DataLoader的工作线程数，如果未提供则默认为`train_num_workers`的值。

    Returns:
        Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]: A tuple containing the DataLoader for the
            training dataset, and optionally for the validation and testing datasets. If a dataset split is not found,
            the corresponding DataLoader is `None`.
            包含训练数据集的DataLoader，以及可选的验证和测试数据集的DataLoader的元组。如果未找到数据集分割部分，
            则对应的DataLoader为`None`。

    Raises:
        ValueError: If batch sizes are not positive integers or no dataset splits are found.
            如果批量大小不是正整数或未找到数据集分割部分，则抛出ValueError。
    """
    # 检查训练批量大小是否为正整数
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be a positive integer.")

    # 如果未提供测试批量大小，则使用训练批量大小
    test_batch_size = test_batch_size or train_batch_size
    # 检查测试批量大小是否为正整数
    if test_batch_size <= 0:
        raise ValueError("test_batch_size must be a positive integer.")

    # 如果未提供测试工作线程数，则使用训练工作线程数
    test_num_workers = test_num_workers or train_num_workers
    # 如果未提供测试整理函数，则使用训练整理函数
    test_collect_fn = test_collect_fn or train_collect_fn

    # 加载训练、验证和测试数据集
    train_dataset = load_split_dataset(dataset_location, "train")
    valid_dataset = load_split_dataset(dataset_location, "valid")
    test_dataset = load_split_dataset(dataset_location, "test")

    # 检查是否至少加载了一个数据集分割部分
    if not any([train_dataset, valid_dataset, test_dataset]):
        raise ValueError(f"No dataset splits found at {dataset_location}. Ensure the dataset is correctly structured.")
    else:
        # 打印找到的数据集分割部分及其样本数量
        print(f"Found dataset splits at {dataset_location}:")
        if train_dataset:
            print(f"  - train: {len(train_dataset)} samples")
        if valid_dataset:
            print(f"  - valid: {len(valid_dataset)} samples")
        if test_dataset:
            print(f"  - test: {len(test_dataset)} samples")

    # 创建训练DataLoader
    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            collate_fn=train_collect_fn,
        )
        if train_dataset
        else None
    )

    # 创建验证DataLoader
    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=test_collect_fn,
        )
        if valid_dataset
        else None
    )

    # 创建测试DataLoader
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=test_collect_fn,
        )
        if test_dataset
        else None
    )

    return train_loader, valid_loader, test_loader
