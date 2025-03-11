import torch  # 导入 PyTorch 库
import torchvision.transforms as transforms  # 导入 torchvision 的数据转换模块
import torch.utils.data as data  # 导入 PyTorch 的数据处理模块
import os  # 导入操作系统库
import pickle  # 导入 pickle 库，用于序列化对象
import numpy as np  # 导入 NumPy 库
import nltk  # 导入 nltk 库
from PIL import Image  # 从 PIL 导入图像处理模块
from build_vocab import Vocabulary  # 从 build_vocab 导入 Vocabulary 类
from pycocotools.coco import COCO  # 从 pycocotools 导入 COCO 类，用于处理 COCO 数据集


class CocoDataset(data.Dataset):  # 定义 COCO 数据集类
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""  # 与 torch.utils.data.DataLoader 兼容的 COCO 自定义数据集
    def __init__(self, root, json, vocab, transform=None):  # 初始化方法
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.  # 图像目录
            json: coco annotation file path.  # COCO 注释文件路径
            vocab: vocabulary wrapper.  # 词汇封装器
            transform: image transformer.  # 图像转换器
        """
        self.root = root  # 保存图像目录
        self.coco = COCO(json)  # 加载 COCO 数据集
        self.ids = list(self.coco.anns.keys())  # 获取所有注释的 ID
        self.vocab = vocab  # 保存词汇封装器
        self.transform = transform  # 保存图像转换器

    def __getitem__(self, index):  # 获取数据项的方法
        """Returns one data pair (image and caption)."""  # 返回一对数据（图像和注释）
        coco = self.coco  # 获取 COCO 对象
        vocab = self.vocab  # 获取词汇封装器
        ann_id = self.ids[index]  # 获取注释 ID
        caption = coco.anns[ann_id]['caption']  # 获取注释文本
        img_id = coco.anns[ann_id]['image_id']  # 获取图像 ID
        path = coco.loadImgs(img_id)[0]['file_name']  # 加载图像文件名

        image = Image.open(os.path.join(self.root, path)).convert('RGB')  # 打开图像并转换为 RGB 格式
        if self.transform is not None:  # 如果有图像转换器
            image = self.transform(image)  # 应用图像转换

        # Convert caption (string) to word ids.  # 将注释（字符串）转换为单词 ID
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())  # 将注释文本分词并转换为小写
        caption = []  # 初始化注释列表
        caption.append(vocab('<start>'))  # 添加开始符的单词 ID
        caption.extend([vocab(token) for token in tokens])  # 将每个单词转换为 ID 并添加到注释列表
        caption.append(vocab('<end>'))  # 添加结束符的单词 ID
        target = torch.Tensor(caption)  # 将注释列表转换为张量
        return image, target  # 返回图像和目标张量

    def __len__(self):  # 获取数据集长度的方法
        return len(self.ids)  # 返回注释 ID 的数量


def collate_fn(data):  # 自定义合并函数
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).  # 数据列表，包含（图像，注释）元组
            - image: torch tensor of shape (3, 256, 256).  # 图像张量，形状为 (3, 256, 256)
            - caption: torch tensor of shape (?); variable length.  # 注释张量，形状可变

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).  # 图像张量，形状为 (batch_size, 3, 256, 256)
        targets: torch tensor of shape (batch_size, padded_length).  # 注释张量，形状为 (batch_size, padded_length)
        lengths: list; valid length for each padded caption.  # 每个填充注释的有效长度
    """
    # Sort a data list by caption length (descending order).  # 按照注释长度（降序）对数据列表进行排序
    data.sort(key=lambda x: len(x[1]), reverse=True)  # 根据注释长度排序
    images, captions = zip(*data)  # 解压图像和注释

    # Merge images (from tuple of 3D tensor to 4D tensor).  # 合并图像（从 3D 张量元组到 4D 张量）
    images = torch.stack(images, 0)  # 将图像张量堆叠成 4D 张量

    # Merge captions (from tuple of 1D tensor to 2D tensor).  # 合并注释（从 1D 张量元组到 2D 张量）
    lengths = [len(cap) for cap in captions]  # 获取每个注释的长度
    targets = torch.zeros(len(captions), max(lengths)).long()  # 创建目标张量，形状为 (batch_size, max_length)
    for i, cap in enumerate(captions):  # 遍历每个注释
        end = lengths[i]  # 获取当前注释的有效长度
        targets[i, :end] = cap[:end]  # 将有效长度的注释填入目标张量        
    return images, targets, lengths  # 返回图像、目标和长度

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):  # 获取数据加载器的方法
    """Returns torch.utils.data.DataLoader for custom coco dataset."""  # 返回 COCO 数据集的 torch.utils.data.DataLoader
    # COCO caption dataset
    coco = CocoDataset(root=root,  # 创建 COCO 数据集实例
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).  # 图像张量，形状为 (batch_size, 3, 224, 224)
    # captions: a tensor of shape (batch_size, padded_length).  # 注释张量，形状为 (batch_size, padded_length)
    # lengths: a list indicating valid length for each caption. length is (batch_size).  # 每个注释的有效长度
    data_loader = torch.utils.data.DataLoader(dataset=coco,  # 数据加载器
                                              batch_size=batch_size,  # 每个批次的样本数量
                                              shuffle=shuffle,  # 是否打乱数据
                                              num_workers=num_workers,  # 工作线程数量
                                              collate_fn=collate_fn)  # 使用自定义合并函数
    return data_loader  # 返回数据加载器