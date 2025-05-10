import torch  # 导入PyTorch库
from PIL import Image  # 从PIL库导入Image模块，用于处理图像
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor  # 导入Hugging Face的模型和处理器

from maestro.trainer.common.utils.device import parse_device_spec  # 导入设备解析工具函数


def predict_with_inputs(  # 定义基于预处理输入的预测函数
    model: PaliGemmaForConditionalGeneration,  # PaliGemma生成模型
    processor: PaliGemmaProcessor,  # PaliGemma处理器
    input_ids: torch.Tensor,  # 输入token ID张量
    attention_mask: torch.Tensor,  # 注意力掩码张量
    pixel_values: torch.Tensor,  # 图像像素值张量
    device: torch.device,  # 运行设备
    max_new_tokens: int = 1024,  # 最大生成token数，默认1024
) -> list[str]:  # 返回字符串列表
    """Generate text predictions from preprocessed model inputs.
    根据预处理后的模型输入生成文本预测。

    Args:
        model (PaliGemmaForConditionalGeneration): The PaliGemma model for generation.
                                                  用于生成的PaliGemma模型。
        processor (PaliGemmaProcessor): Tokenizer and processor for model inputs/outputs.
                                       模型输入/输出的处理器和分词器。
        input_ids (torch.Tensor): Input token IDs.
                                 输入token ID张量。
        attention_mask (torch.Tensor): Attention mask for input tokens.
                                       输入token的注意力掩码。
        pixel_values (torch.Tensor): Processed image tensor.
                                     处理后的图像张量。
        device (torch.device): Device to run inference on.
                               运行推理的设备。
        max_new_tokens (int): Maximum number of new tokens to generate.
                              最大生成新token数。

    Returns:
        list[str]: List of generated text predictions.
                  生成的文本预测列表。
    """
    with torch.no_grad():  # 禁用梯度计算以节省内存
        generated_ids = model.generate(  # 调用模型生成方法
            pixel_values=pixel_values.to(device),  # 将图像张量移动到指定设备
            input_ids=input_ids.to(device),  # 将输入ID移动到指定设备
            attention_mask=attention_mask.to(device),  # 将注意力掩码移动到指定设备
            max_new_tokens=max_new_tokens,  # 设置最大生成token数
        )
        prefix_length = input_ids.shape[-1]  # 获取前缀长度（输入ID的最后一个维度）
        generated_ids = generated_ids[:, prefix_length:]  # 截取生成部分（去除前缀）
        return processor.batch_decode(generated_ids, skip_special_tokens=True)  # 批量解码生成结果，跳过特殊token


def predict(  # 定义单样本预测函数
    model: PaliGemmaForConditionalGeneration,  # PaliGemma生成模型
    processor: PaliGemmaProcessor,  # PaliGemma处理器
    image: str | bytes | Image.Image,  # 输入图像（路径/字节/PIL图像）
    prefix: str,  # 文本前缀
    device: str | torch.device = "auto",  # 运行设备，默认"auto"
    max_new_tokens: int = 1024,  # 最大生成token数，默认1024
) -> str:  # 返回字符串
    """Generate a text prediction for a single image and text prefix.
    为单个图像和文本前缀生成文本预测。

    Args:
        model (PaliGemmaForConditionalGeneration): The PaliGemma model for generation.
                                                  用于生成的PaliGemma模型。
        processor (PaliGemmaProcessor): Tokenizer and processor for model inputs/outputs.
                                       模型输入/输出的处理器和分词器。
        image (str | bytes | Image.Image): Input image as a file path, bytes, or PIL Image.
                                          输入图像（文件路径/字节/PIL图像）。
        prefix (str): Text prefix to condition the generation.
                     用于条件生成的文本前缀。
        device (str | torch.device): Device to run inference on.
                                    运行推理的设备。
        max_new_tokens (int): Maximum number of new tokens to generate.
                             最大生成新token数。

    Returns:
        str: Generated text prediction.
            生成的文本预测。
    """
    device = parse_device_spec(device)  # 解析设备规范字符串
    text = "<image>" + prefix  # 构造输入文本（添加图像标记）
    inputs = processor(  # 使用处理器处理输入
        text=text,  # 输入文本
        images=image,  # 输入图像
        return_tensors="pt",  # 返回PyTorch张量
        padding=True,  # 启用填充
    )
    return predict_with_inputs(  # 调用预处理输入的预测函数
        **inputs,  # 解包处理后的输入
        model=model,  # 传入模型
        processor=processor,  # 传入处理器
        device=device,  # 传入设备
        max_new_tokens=max_new_tokens  # 传入最大生成token数
    )[0]  # 返回第一个预测结果
