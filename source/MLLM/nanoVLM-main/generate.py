import torch  # 导入PyTorch库
from PIL import Image  # 从PIL库导入Image模块，用于图像处理
from huggingface_hub import hf_hub_download  # 从huggingface_hub导入hf_hub_download函数，用于下载模型文件

from models.vision_language_model import VisionLanguageModel  # 从本地模块导入VisionLanguageModel
from models.config import VLMConfig  # 从本地模块导入VLMConfig配置类
from data.processors import get_tokenizer, get_image_processor  # 从本地模块导入获取tokenizer和image processor的函数

torch.manual_seed(0)  # 设置PyTorch的随机种子，以确保结果的可复现性

cfg = VLMConfig()  # 创建VLMConfig配置对象
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有可用的GPU，并设置设备
print(f"Using device: {device}")  # 打印当前使用的设备

# Change to your own model path after training
# 训练后请更改为自己的模型路径
path_to_hf_file = hf_hub_download(repo_id="lusxvr/nanoVLM-222M", filename="nanoVLM-222M.pth")  # 从HuggingFace Hub下载预训练模型文件
model = VisionLanguageModel(cfg).to(device)  # 使用配置创建VisionLanguageModel实例，并将其移动到指定设备
model.load_checkpoint(path_to_hf_file)  # 加载下载的模型检查点权重
model.eval()  # 将模型设置为评估模式（关闭dropout等）

tokenizer = get_tokenizer(cfg.lm_tokenizer)  # 根据配置获取语言模型的tokenizer
image_processor = get_image_processor(cfg.vit_img_size)  # 根据配置获取视觉模型的图像处理器

text = "What is this?"  # 定义输入的文本问题
template = f"Question: {text} Answer:"  # 构建输入模板，包含问题和"Answer:"前缀
encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")  # 使用tokenizer对模板进行编码，返回PyTorch张量
tokens = encoded_batch['input_ids'].to(device)  # 获取编码后的token ID，并将其移动到指定设备

image_path = 'assets/image.png'  # 定义图像文件路径
image = Image.open(image_path)  # 使用PIL打开图像文件
image = image_processor(image)  # 使用图像处理器处理图像（如缩放、归一化等）
image = image.unsqueeze(0).to(device)  # 在批次维度上增加一个维度，并将图像张量移动到指定设备

print("Input: ")  # 打印输入提示
print(f'{text}')  # 打印输入的文本问题
print("Output:")  # 打印输出提示
num_generations = 5  # 设置生成答案的数量
for i in range(num_generations):  # 循环生成指定数量的答案
    gen = model.generate(tokens, image, max_new_tokens=20)  # 调用模型的generate方法生成新的token序列，最多生成20个新token
    print(f"Generation {i+1}: {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")  # 使用tokenizer将生成的token ID解码回文本，跳过特殊token，并打印结果