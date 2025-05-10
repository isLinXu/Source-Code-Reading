import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from maestro.trainer.common.utils.device import parse_device_spec
from maestro.trainer.models.qwen_2_5_vl.loaders import format_conversation


def predict_with_inputs(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 1024,
) -> list[str]:
    """
    Generates predictions from the Qwen2.5-VL model using both textual and visual inputs.
    使用文本和视觉输入从Qwen2.5-VL模型生成预测。

    Args:
        model (Qwen2_5_VLForConditionalGeneration):
            A Qwen2.5-VL model capable of conditional text generation with visual context.
            能够结合视觉上下文进行条件文本生成的Qwen2.5-VL模型。
        processor (Qwen2_5_VLProcessor):
            Preprocessing and postprocessing utility for the Qwen2.5-VL model.
            Qwen2.5-VL模型的预处理和后处理工具。
        input_ids (torch.Tensor):
            Tokenized input text IDs.
            分词后的输入文本ID。
        attention_mask (torch.Tensor):
            Attention mask corresponding to the tokenized input.
            与分词输入对应的注意力掩码。
        pixel_values (torch.Tensor):
            Preprocessed image data (pixel values) for visual inputs.
            视觉输入的预处理图像数据（像素值）。
        image_grid_thw (torch.Tensor):
            Tensor specifying the layout or shape of the provided images.
            指定提供图像的布局或形状的张量。
        device (torch.device):
            Device on which to run inference (e.g., ``torch.device("cuda")`` or ``torch.device("cpu")``).
            运行推理的设备（例如``torch.device("cuda")``或``torch.device("cpu")``）。
        max_new_tokens (int):
            Maximum number of tokens to generate.
            生成的最大token数量。

    Returns:
        list[str]: A list of decoded strings corresponding to the generated sequences.
        与生成序列对应的解码字符串列表。
    """
    # 在不计算梯度的情况下进行推理
    with torch.no_grad():
        # 使用模型生成文本
        generated_ids = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            pixel_values=pixel_values.to(device),
            image_grid_thw=image_grid_thw.to(device),
            max_new_tokens=max_new_tokens,
        )
        # 提取生成的文本部分（去掉输入部分）
        generated_ids = [
            generated_sequence[len(input_sequence) :]
            for input_sequence, generated_sequence in zip(input_ids, generated_ids)
        ]
        # 解码生成的token并返回
        return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def predict(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    image: str | bytes | Image.Image,
    prefix: str,
    system_message: str | None = None,
    device: str | torch.device = "auto",
    max_new_tokens: int = 1024,
) -> str:
    """
    Generates a single prediction from the Qwen2.5-VL model given an image and prefix text.
    给定图像和前缀文本，从Qwen2.5-VL模型生成单个预测。

    Args:
        model (Qwen2_5_VLForConditionalGeneration):
            A Qwen2.5-VL model capable of conditional text generation with visual context.
            能够结合视觉上下文进行条件文本生成的Qwen2.5-VL模型。
        processor (Qwen2_5_VLProcessor):
            Preprocessing and postprocessing utility for the Qwen2.5-VL model.
            Qwen2.5-VL模型的预处理和后处理工具。
        image (str | bytes | PIL.Image.Image):
            Image input for the model, which can be a file path, raw bytes, or a PIL Image object.
            模型的图像输入，可以是文件路径、原始字节或PIL图像对象。
        prefix (str):
            Text prompt or initial text to prepend to the conversation.
            要添加到对话中的文本提示或初始文本。
        system_message (str | None):
            A system-level instruction or context text.
            系统级别的指令或上下文文本。
        device (str | torch.device):
            Device on which to run inference. Can be ``torch.device`` or a string such
            as "auto", "cpu", "cuda", or "mps".
            运行推理的设备。可以是``torch.device``或字符串，如"auto"、"cpu"、"cuda"或"mps"。
        max_new_tokens (int):
            Maximum number of tokens to generate.
            生成的最大token数量。

    Returns:
        str: The decoded string representing the model's generated response.
        表示模型生成响应的解码字符串。
    """
    # 解析设备规格
    device = parse_device_spec(device)
    # 格式化对话内容
    conversation = format_conversation(image=image, prefix=prefix, system_message=system_message)
    # 应用聊天模板生成文本
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    # 处理视觉信息
    image_inputs, _ = process_vision_info(conversation)

    # 使用处理器准备输入
    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt",
    )
    # 调用predict_with_inputs函数进行预测并返回结果
    return predict_with_inputs(
        **inputs, model=model, processor=processor, device=device, max_new_tokens=max_new_tokens
    )[0]
