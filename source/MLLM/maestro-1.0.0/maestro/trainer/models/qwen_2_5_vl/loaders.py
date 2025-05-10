from typing import Any

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor


def format_conversation(
    image: str | bytes | Image.Image, prefix: str, suffix: str | None = None, system_message: str | None = None
) -> list[dict]:
    """
    Formats a conversation with image, text, and optional system message.
    格式化包含图像、文本和可选系统消息的对话。

    Args:
        image (str | bytes | Image.Image):
            Image input for the conversation.
            对话的图像输入。
        prefix (str):
            Text prompt or initial text to prepend to the conversation.
            要添加到对话中的文本提示或初始文本。
        suffix (str | None):
            Optional text to append to the conversation.
            可选的要附加到对话中的文本。
        system_message (str | None):
            Optional system-level instruction or context text.
            可选的系统级别指令或上下文文本。

    Returns:
        list[dict]: A list of message dictionaries representing the conversation.
        表示对话的消息字典列表。
    """
    messages = []

    if system_message is not None:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prefix,
                },
            ],
        }
    )

    if suffix is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": suffix}],
            }
        )

    return messages


def train_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]], processor: Qwen2_5_VLProcessor, system_message: str | None = None
):
    """
    Collates a batch of training data for the Qwen2.5-VL model.
    为Qwen2.5-VL模型整理一批训练数据。

    Args:
        batch (list[tuple[Image.Image, dict[str, Any]]]):
            A batch of training data consisting of images and associated metadata.
            由图像和相关元数据组成的一批训练数据。
        processor (Qwen2_5_VLProcessor):
            Preprocessing and postprocessing utility for the Qwen2.5-VL model.
            Qwen2.5-VL模型的预处理和后处理工具。
        system_message (str | None):
            Optional system-level instruction or context text.
            可选的系统级别指令或上下文文本。

    Returns:
        tuple: A tuple containing model inputs and labels.
        包含模型输入和标签的元组。
    """
    images, data = zip(*batch)
    conversations = [
        format_conversation(image, entry["prefix"], entry["suffix"], system_message)
        for image, entry in zip(images, data)
    ]

    texts = [processor.apply_chat_template(conversation=conversation, tokenize=False) for conversation in conversations]
    image_inputs = [process_vision_info(conversation)[0] for conversation in conversations]
    model_inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = model_inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_tokens = [151652, 151653, 151655]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs["image_grid_thw"]

    return input_ids, attention_mask, pixel_values, image_grid_thw, labels


def evaluation_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]], processor: Qwen2_5_VLProcessor, system_message: str | None = None
):
    """
    Collates a batch of evaluation data for the Qwen2.5-VL model.
    为Qwen2.5-VL模型整理一批评估数据。

    Args:
        batch (list[tuple[Image.Image, dict[str, Any]]]):
            A batch of evaluation data consisting of images and associated metadata.
            由图像和相关元数据组成的一批评估数据。
        processor (Qwen2_5_VLProcessor):
            Preprocessing and postprocessing utility for the Qwen2.5-VL model.
            Qwen2.5-VL模型的预处理和后处理工具。
        system_message (str | None):
            Optional system-level instruction or context text.
            可选的系统级别指令或上下文文本。

    Returns:
        tuple: A tuple containing model inputs, prefixes, and suffixes.
        包含模型输入、前缀和后缀的元组。
    """
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    conversations = [
        format_conversation(image, entry["prefix"], system_message=system_message) for image, entry in zip(images, data)
    ]

    texts = [processor.apply_chat_template(conversation=conversation, tokenize=False) for conversation in conversations]
    image_inputs = [process_vision_info(conversation)[0] for conversation in conversations]
    model_inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs["image_grid_thw"]

    return input_ids, attention_mask, pixel_values, image_grid_thw, prefixes, suffixes
