from os.path import exists, join, basename  # 导入用于路径操作的函数
from os import makedirs, remove  # 导入创建目录和删除文件的函数
from six.moves import urllib  # 导入 urllib 库以处理 URL
import tarfile  # 导入 tarfile 库以处理 tar 文件
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize  # 导入图像转换工具

from dataset import DatasetFromFolder  # 从自定义数据集模块导入 DatasetFromFolder 类


def download_bsd300(dest="dataset"):
    """Download the BSDS300 dataset.
    下载 BSDS300 数据集。

    Args:
        dest: 下载数据集的目标目录
    Returns:
        output_image_dir: 输出图像目录
    """
    output_image_dir = join(dest, "BSDS300/images")  # 设置输出图像目录

    if not exists(output_image_dir):  # 如果输出目录不存在
        makedirs(dest)  # 创建目标目录
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"  # 数据集 URL
        print("downloading url ", url)  # 打印下载的 URL

        data = urllib.request.urlopen(url)  # 打开 URL 并读取数据

        file_path = join(dest, basename(url))  # 设置文件保存路径
        with open(file_path, 'wb') as f:  # 以写入二进制模式打开文件
            f.write(data.read())  # 写入下载的数据

        print("Extracting data")  # 打印提取数据的提示
        with tarfile.open(file_path) as tar:  # 打开 tar 文件
            for item in tar:  # 遍历 tar 文件中的每个项目
                tar.extract(item, dest)  # 提取项目到目标目录

        remove(file_path)  # 删除下载的 tar 文件

    return output_image_dir  # 返回输出图像目录


def calculate_valid_crop_size(crop_size, upscale_factor):
    """Calculate the valid crop size.
    计算有效的裁剪大小。

    Args:
        crop_size: 原始裁剪大小
        upscale_factor: 放大因子
    Returns:
        valid_crop_size: 有效裁剪大小
    """
    return crop_size - (crop_size % upscale_factor)  # 返回裁剪大小减去余数


def input_transform(crop_size, upscale_factor):
    """Create input transformation pipeline.
    创建输入转换管道。

    Args:
        crop_size: 裁剪大小
        upscale_factor: 放大因子
    Returns:
        transform: 转换管道
    """
    return Compose([  # 使用 Compose 组合多个转换
        CenterCrop(crop_size),  # 中心裁剪
        Resize(crop_size // upscale_factor),  # 调整大小
        ToTensor(),  # 转换为张量
    ])


def target_transform(crop_size):
    """Create target transformation pipeline.
    创建目标转换管道。

    Args:
        crop_size: 裁剪大小
    Returns:
        transform: 转换管道
    """
    return Compose([  # 使用 Compose 组合多个转换
        CenterCrop(crop_size),  # 中心裁剪
        ToTensor(),  # 转换为张量
    ])


def get_training_set(upscale_factor):
    """Get the training dataset.
    获取训练数据集。

    Args:
        upscale_factor: 放大因子
    Returns:
        dataset: 训练数据集
    """
    root_dir = download_bsd300()  # 下载数据集并获取根目录
    train_dir = join(root_dir, "train")  # 设置训练数据目录
    crop_size = calculate_valid_crop_size(256, upscale_factor)  # 计算有效裁剪大小

    return DatasetFromFolder(train_dir,  # 返回训练数据集
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    """Get the test dataset.
    获取测试数据集。

    Args:
        upscale_factor: 放大因子
    Returns:
        dataset: 测试数据集
    """
    root_dir = download_bsd300()  # 下载数据集并获取根目录
    test_dir = join(root_dir, "test")  # 设置测试数据目录
    crop_size = calculate_valid_crop_size(256, upscale_factor)  # 计算有效裁剪大小

    return DatasetFromFolder(test_dir,  # 返回测试数据集
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))