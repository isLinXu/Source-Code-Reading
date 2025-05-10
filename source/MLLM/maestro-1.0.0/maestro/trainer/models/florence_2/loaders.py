# 导入必要的模块
from typing import Any
from PIL import Image

# 定义train_collate_fn函数，用于训练数据的批处理
def train_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor):
    # 解包批次数据，将图像和数据分开
    images, data = zip(*batch)
    # 从数据中提取前缀
    prefixes = [entry["prefix"] for entry in data]
    # 从数据中提取后缀
    suffixes = [entry["suffix"] for entry in data]
    # 使用处理器处理文本和图像，返回张量格式的输入
    inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    # 获取输入token ID
    input_ids = inputs["input_ids"]
    # 获取图像张量
    pixel_values = inputs["pixel_values"]

    # 使用处理器的tokenizer对后缀进行分词，返回token ID
    labels = processor.tokenizer(
        text=suffixes, return_tensors="pt", padding=True, return_token_type_ids=False
    ).input_ids

    # 返回输入token ID、图像张量和标签
    return input_ids, pixel_values, labels


# 定义evaluation_collate_fn函数，用于评估数据的批处理
def evaluation_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor):
    # 解包批次数据，将图像和数据分开
    images, data = zip(*batch)
    # 从数据中提取前缀
    prefixes = [entry["prefix"] for entry in data]
    # 从数据中提取后缀
    suffixes = [entry["suffix"] for entry in data]
    # 使用处理器处理文本和图像，返回张量格式的输入
    inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    # 获取输入token ID
    input_ids = inputs["input_ids"]
    # 获取图像张量
    pixel_values = inputs["pixel_values"]
    # 返回输入token ID、图像张量、前缀和后缀
    return input_ids, pixel_values, prefixes, suffixes