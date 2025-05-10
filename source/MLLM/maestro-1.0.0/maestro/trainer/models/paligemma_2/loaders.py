from typing import Any  # 导入Any类型，用于类型注解
from PIL import Image  # 导入PIL的Image模块，用于处理图像
from transformers import PaliGemmaProcessor  # 导入PaliGemma处理器


def train_collate_fn(  # 定义训练数据整理函数
    batch: list[tuple[Image.Image, dict[str, Any]]],  # 输入批次数据：图像和元数据的元组列表
    processor: PaliGemmaProcessor,  # PaliGemma处理器实例
    max_length: int = 512  # 最大序列长度，默认512
):
    # 解包批次数据：将图像和元数据分开
    images, data = zip(*batch)  # 将批次数据解压为图像列表和元数据列表
    
    # 构造输入前缀（添加<image>标记）和后缀
    prefixes = ["<image>" + entry["prefix"] for entry in data]  # 为每个样本添加图像标记前缀
    suffixes = [entry["suffix"] for entry in data]  # 提取所有样本的后缀文本

    # 使用处理器处理输入数据
    inputs = processor(
        text=prefixes,  # 文本前缀列表
        images=images,  # 图像列表
        return_tensors="pt",  # 返回PyTorch张量
        suffix=suffixes,  # 后缀文本列表
        padding=True,  # 启用填充
        truncation="only_second",  # 只截断第二个序列（后缀部分）
        max_length=max_length,  # 最大序列长度限制
    )

    # 解包处理后的输入数据
    input_ids = inputs["input_ids"]  # 输入token ID张量
    attention_mask = inputs["attention_mask"]  # 注意力掩码张量
    token_type_ids = inputs["token_type_ids"]  # Token类型ID张量（区分不同文本部分）
    pixel_values = inputs["pixel_values"]  # 图像像素值张量
    labels = inputs["labels"]  # 标签张量（用于计算损失）

    return input_ids, attention_mask, token_type_ids, pixel_values, labels  # 返回整理后的训练数据


def evaluation_collate_fn(  # 定义评估数据整理函数
    batch: list[tuple[Image.Image, dict[str, Any]]],  # 输入批次数据：图像和元数据的元组列表 
    processor: PaliGemmaProcessor  # PaliGemma处理器实例
):
    # 解包批次数据：将图像和元数据分开
    images, data = zip(*batch)  # 将批次数据解压为图像列表和元数据列表
    
    # 构造输入前缀（添加<image>标记）和后缀
    prefixes = ["<image>" + entry["prefix"] for entry in data]  # 为每个样本添加图像标记前缀
    suffixes = [entry["suffix"] for entry in data]  # 提取所有样本的后缀文本

    # 使用处理器处理输入数据（评估阶段不需要标签）
    inputs = processor(
        text=prefixes,  # 文本前缀列表
        images=images,  # 图像列表
        return_tensors="pt",  # 返回PyTorch张量
        padding=True  # 启用填充
    )

    # 解包处理后的输入数据
    input_ids = inputs["input_ids"]  # 输入token ID张量
    attention_mask = inputs["attention_mask"]  # 注意力掩码张量
    pixel_values = inputs["pixel_values"]  # 图像像素值张量

    return input_ids, attention_mask, pixel_values, prefixes, suffixes  # 返回整理后的评估数据
