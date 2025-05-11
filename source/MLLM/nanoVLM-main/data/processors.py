from transformers import AutoTokenizer  # 从transformers库导入AutoTokenizer，用于自动加载预训练模型的tokenizer
import torchvision.transforms as transforms  # 导入torchvision.transforms模块，用于图像预处理

TOKENIZERS_CACHE = {}  # 创建一个字典作为tokenizer的缓存，避免重复加载

def get_tokenizer(name):
    if name not in TOKENIZERS_CACHE:  # 检查tokenizer是否已在缓存中
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)  # 如果不在，从预训练模型加载tokenizer，使用fast tokenizer
        tokenizer.pad_token = tokenizer.eos_token  # 将填充token设置为EOS token，这对于某些模型（如GPT-2）是常见的做法
        TOKENIZERS_CACHE[name] = tokenizer  # 将加载的tokenizer存入缓存
    return TOKENIZERS_CACHE[name]  # 返回缓存中的tokenizer

def get_image_processor(img_size):
    return transforms.Compose([  # 创建一个transforms.Compose对象，将多个图像转换操作组合起来
        transforms.Resize((img_size, img_size)),  # 添加Resize操作，将图像缩放到指定的尺寸(img_size, img_size)
        transforms.ToTensor()  # 添加ToTensor操作，将PIL Image或numpy.ndarray转换为PyTorch张量，并自动将像素值缩放到[0, 1]
    ])
