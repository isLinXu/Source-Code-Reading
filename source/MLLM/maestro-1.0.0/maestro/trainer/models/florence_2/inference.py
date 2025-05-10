# 导入必要的模块
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# 导入设备解析工具
from maestro.trainer.common.utils.device import parse_device_spec

# 定义predict_with_inputs函数，用于从预处理输入生成文本预测
def predict_with_inputs(
    model: AutoModelForCausalLM,  # Florence-2模型，用于条件文本生成
    processor: AutoProcessor,  # 处理器，用于模型输入输出的处理，包括分词和解码
    input_ids: torch.Tensor,  # 输入token ID的张量，表示文本提示
    pixel_values: torch.Tensor,  # 处理后的图像张量
    device: torch.device,  # 运行推理的设备
    max_new_tokens: int = 1024,  # 生成的最大token数量，默认为1024
) -> list[str]:  # 返回生成的文本预测列表
    """Generate text predictions from preprocessed model inputs.

    Args:
        model (AutoModelForCausalLM): The Florence-2 model for conditional text generation.
        model (AutoModelForCausalLM): 用于条件文本生成的Florence-2模型。
        processor (AutoProcessor): Processor for model inputs and outputs, handling tokenization and decoding.
        processor (AutoProcessor): 用于模型输入输出的处理器，处理分词和解码。
        input_ids (torch.Tensor): Tensor of input token IDs representing the text prompt.
        input_ids (torch.Tensor): 表示文本提示的输入token ID的张量。
        pixel_values (torch.Tensor): Processed image tensor.
        pixel_values (torch.Tensor): 处理后的图像张量。
        device (torch.device): Device on which to run inference.
        device (torch.device): 运行推理的设备。
        max_new_tokens (int): Maximum number of tokens to generate.
        max_new_tokens (int): 生成的最大token数量。

    Returns:
        list[str]: A list of generated text predictions.
        list[str]: 生成的文本预测列表。
    """
    # 使用模型生成文本
    generated_ids = model.generate(
        input_ids=input_ids.to(device),  # 将input_ids移动到指定设备
        pixel_values=pixel_values.to(device),  # 将pixel_values移动到指定设备
        max_new_tokens=max_new_tokens,  # 设置最大生成token数量
        do_sample=False,  # 不使用采样
        num_beams=3,  # 使用beam search，beam数量为3
    )
    # 使用处理器解码生成的token ID，返回文本预测
    return processor.batch_decode(generated_ids, skip_special_tokens=False)

# 定义predict函数，用于生成单个图像和文本前缀的文本预测
def predict(
    model: AutoModelForCausalLM,  # Florence-2模型，用于条件文本生成
    processor: AutoProcessor,  # 处理器，用于模型输入输出的处理，包括分词和解码
    image: Image.Image,  # 输入图像，可以是文件路径、原始字节或PIL图像
    prefix: str,  # 文本前缀，用于条件生成输出
    device: str | torch.device = "auto",  # 运行推理的设备，默认为"auto"
    max_new_tokens: int = 1024,  # 生成的最大token数量，默认为1024
) -> str:  # 返回生成的文本预测
    """Generate a text prediction for a single image and text prefix.

    Args:
        model (AutoModelForCausalLM): The Florence-2 model for conditional text generation.
        model (AutoModelForCausalLM): 用于条件文本生成的Florence-2模型。
        processor (AutoProcessor): Processor for model inputs and outputs, handling tokenization and decoding.
        processor (AutoProcessor): 用于模型输入输出的处理器，处理分词和解码。
        image (str | bytes | Image.Image): Input image as a file path, raw bytes, or a PIL Image.
        image (str | bytes | Image.Image): 输入图像，可以是文件路径、原始字节或PIL图像。
        prefix (str): Text prefix to condition the generated output.
        prefix (str): 用于条件生成输出的文本前缀。
        device (str | torch.device): Device on which to run inference (e.g., "auto", "cpu", "cuda").
        device (str | torch.device): 运行推理的设备（例如"auto", "cpu", "cuda"）。
        max_new_tokens (int): Maximum number of tokens to generate.
        max_new_tokens (int): 生成的最大token数量。

    Returns:
        str: The generated text prediction.
        str: 生成的文本预测。
    """
    # 解析设备规格
    device = parse_device_spec(device)
    # 使用处理器处理输入文本和图像
    inputs = processor(text=prefix, images=image, return_tensors="pt", padding=True)
    # 调用predict_with_inputs函数生成文本预测，并返回第一个结果
    return predict_with_inputs(
        input_ids=inputs["input_ids"],  # 输入token ID
        pixel_values=inputs["pixel_values"],  # 图像张量
        model=model,  # 模型
        processor=processor,  # 处理器
        device=device,  # 设备
        max_new_tokens=max_new_tokens,  # 最大生成token数量
    )[0]