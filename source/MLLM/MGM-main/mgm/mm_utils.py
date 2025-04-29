from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from mgm.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    """从Base64字符串加载图像
    Args:
        image: str - Base64编码的图像字符串
    Returns:
        PIL.Image - 图像对象
    """
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    """将图像扩展为正方形
    Args:
        pil_img: PIL.Image - 输入图像
        background_color: tuple - 填充背景色 (R, G, B)
    Returns:
        PIL.Image - 正方形图像
    """
    width, height = pil_img.size
    if width == height:  # 已经是正方形
        return pil_img
    elif width > height:  # 横向扩展
        result = Image.new(pil_img.mode, (width, width), background_color)  # 创建正方形画布
        result.paste(pil_img, (0, (width - height) // 2))  # 垂直居中粘贴
        return result
    else:  # 纵向扩展
        result = Image.new(pil_img.mode, (height, height), background_color)  # 创建正方形画布
        result.paste(pil_img, ((height - width) // 2, 0))  # 水平居中粘贴
        return result


def process_images(images, image_processor, model_cfg):
    """处理图像以适应模型输入
    Args:
        images: List[PIL.Image] - 图像列表
        image_processor: transformers.ImageProcessor - 图像处理器
        model_cfg: ModelConfig - 模型配置
    Returns:
        torch.Tensor - 处理后的图像张量
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)  # 获取图像宽高比设置
    new_images = []
    if image_aspect_ratio == 'pad':  # 如果需要填充
        for image in images:
            image = expand2square(image.convert('RGB'), tuple(int(x*255) for x in image_processor.image_mean))  # 转换为RGB并填充
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]  # 预处理图像
            new_images.append(image)
    else:  # 其他情况
        return image_processor(images, return_tensors='pt')['pixel_values']  # 直接处理图像
    if all(x.shape == new_images[0].shape for x in new_images):  # 如果所有图像尺寸一致
        new_images = torch.stack(new_images, dim=0)  # 堆叠为张量
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """将包含图像标记的提示文本转换为token
    Args:
        prompt: str - 包含<image>标记的提示文本
        tokenizer: transformers.PreTrainedTokenizer - 分词器
        image_token_index: int - 图像标记的索引
        return_tensors: str - 返回张量类型（如 'pt'）
    Returns:
        List[int]/torch.Tensor - token ID列表或张量
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]  # 按<image>分割并分词

    def insert_separator(X, sep):
        """在列表元素间插入分隔符"""
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:  # 如果包含BOS标记
        offset = 1
        input_ids.append(prompt_chunks[0][0])  # 添加BOS标记

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):  # 在分割部分间插入图像标记
        input_ids.extend(x[offset:])

    if return_tensors is not None:  # 如果需要返回张量
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)  # 返回PyTorch张量
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    """从模型路径中提取模型名称
    Args:
        model_path: str - 模型路径
    Returns:
        str - 模型名称
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):  # 如果是checkpoint路径
        return model_paths[-2] + "_" + model_paths[-1]  # 返回父目录名+checkpoint名
    else:
        return model_paths[-1]  # 返回最后一级目录名


class KeywordsStoppingCriteria(StoppingCriteria):
    """基于关键词的停止条件
    Args:
        keywords: List[str] - 关键词列表
        tokenizer: transformers.PreTrainedTokenizer - 分词器
        input_ids: torch.Tensor - 输入token ID
    """
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:  # 将关键词转换为token ID
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:  # 去除BOS标记
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:  # 更新最大关键词长度
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))  # 添加关键词token ID
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]  # 记录输入长度
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """判断单个样本是否应停止生成
        Args:
            output_ids: torch.LongTensor - 输出token ID
            scores: torch.FloatTensor - 模型输出分数
        Returns:
            bool - 是否应停止生成
        """
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)  # 计算偏移量
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]  # 将关键词token ID移动到相同设备
        for keyword_id in self.keyword_ids:  # 检查是否匹配关键词
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]  # 解码输出
        for keyword in self.keywords:  # 检查是否包含关键词
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """判断整个批次是否应停止生成
        Args:
            output_ids: torch.LongTensor - 输出token ID
            scores: torch.FloatTensor - 模型输出分数
        Returns:
            bool - 是否应停止生成
        """
        outputs = []
        for i in range(output_ids.shape[0]):  # 遍历批次中的每个样本
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)  # 当所有样本都应停止时返回True