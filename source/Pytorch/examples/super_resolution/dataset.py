import torch.utils.data as data  # 导入 PyTorch 数据处理工具

from os import listdir  # 导入列出目录内容的函数
from os.path import join  # 导入路径连接函数
from PIL import Image  # 导入图像处理库 PIL


def is_image_file(filename):
    """Check if a file is an image file.
    检查文件是否为图像文件。

    Args:
        filename: 文件名
    Returns:
        bool: 如果是图像文件返回 True，否则返回 False
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])  # 检查文件扩展名


def load_img(filepath):
    """Load an image and convert it to YCbCr format.
    加载图像并转换为 YCbCr 格式。

    Args:
        filepath: 图像文件路径
    Returns:
        y: Y 通道（亮度）
    """
    img = Image.open(filepath).convert('YCbCr')  # 打开图像并转换为 YCbCr 格式
    y, _, _ = img.split()  # 分离 Y、Cb 和 Cr 通道，返回 Y 通道
    return y  # 返回 Y 通道


class DatasetFromFolder(data.Dataset):
    """Custom dataset class to load images from a folder.
    从文件夹加载图像的自定义数据集类。"""

    def __init__(self, image_dir, input_transform=None, target_transform=None):
        """Initialize the dataset.
        初始化数据集。

        Args:
            image_dir: 图像文件夹路径
            input_transform: 输入图像的转换函数
            target_transform: 目标图像的转换函数
        """
        super(DatasetFromFolder, self).__init__()  # 调用父类构造函数
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]  # 列出所有图像文件

        self.input_transform = input_transform  # 保存输入转换函数
        self.target_transform = target_transform  # 保存目标转换函数

    def __getitem__(self, index):
        """Get an item from the dataset.
        从数据集中获取一个项目。

        Args:
            index: 项目的索引
        Returns:
            input: 输入图像
            target: 目标图像
        """
        input = load_img(self.image_filenames[index])  # 加载输入图像
        target = input.copy()  # 复制输入图像作为目标图像
        if self.input_transform:  # 如果有输入转换函数
            input = self.input_transform(input)  # 应用输入转换
        if self.target_transform:  # 如果有目标转换函数
            target = self.target_transform(target)  # 应用目标转换

        return input, target  # 返回输入和目标图像

    def __len__(self):
        """Return the total number of images in the dataset.
        返回数据集中图像的总数。

        Returns:
            int: 图像数量
        """
        return len(self.image_filenames)  # 返回图像文件数量