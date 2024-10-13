import os
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from .modeling_tinyllava import TinyLlavaForConditionalGeneration
from .configuration_tinyllava import TinyLlavaConfig


# 加载基础模型的checkpoint，并将key中的.base_layer去掉
def load_base_ckp_for_lora(ckp_path):
    """
    Load the checkpoint of the base model and remove '.base_layer' from the key names.

    Args:
        ckp_path (str): The path to the checkpoint file.

    Returns:
        OrderedDict: The reordered checkpoint dictionary.
    """
    # 加载模型检查点到CPU上
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))

    # 创建一个新的有序字典来存储修改后的键值对
    new_ckp = OrderedDict()
    # 遍历原始检查点的所有键值对
    for k, v in ckp.items():
        # 移除键名中的 '.base_layer'
        new_k = k.replace('.base_layer', '')
        # 将修改后的键值对添加到新的有序字典中
        new_ckp[new_k] = v

    # 返回修改后的模型检查点
    return new_ckp
    
# 加载预训练模型
def load_pretrained_model(model_name_or_path, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", **kwargs):
    """
    Load a pretrained model with optional quantization and device mapping.

    Args:
        model_name_or_path (str): The name or path of the model to load.
        load_type (str): The type of loading, default is 'hf'.
        load_8bit (bool): Whether to load the model in 8-bit mode.
        load_4bit (bool): Whether to load the model in 4-bit mode.
        device_map (str): The device map for loading the model.
        device (str): The device to load the model onto.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the model, tokenizer, image processor, and context length.
    """
    # 设置默认参数，如果device不是cuda，则将device_map设置为空字符串对应的设备
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    # 根据load_8bit和load_4bit的值设置量化配置
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        # 设置4位量化的配置
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    # 如果model_name_or_path不为None且不包含'lora'，则从预训练模型加载TinyLlavaForConditionalGeneration模型
    if model_name_or_path is not None and 'lora' not in model_name_or_path:
        model = TinyLlavaForConditionalGeneration.from_pretrained(model_name_or_path,low_cpu_mem_usage=True,torch_dtype=torch.float16)

    # 如果model_name_or_path不为None且包含'lora'，则加载LoRA模型
    elif model_name_or_path is not None and 'lora' in model_name_or_path:
        # 如果存在adapter_config.json文件，则从该路径加载TinyLlavaConfig配置
        if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
            model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
            model = TinyLlavaForConditionalGeneration(model_config)
            # 加载LoRA模型的各个部分
            language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
            language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
            model.language_model.load_state_dict(language_model_ckp)
            vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
            vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
            model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
            connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
            connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
            model.connector.load_state_dict(connector_ckp)
            # 将模型转换为float16精度
            model.to(torch.float16)
            # 加载LoRA权重并合并
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_name_or_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
    # 获取图像处理器和上下文长度
    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    # 获取tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_name_or_path, use_fast=False, padding_side="right")
    tokenizer = model.tokenizer
    #tokenizer.pad_token = tokenizer.eos_token
    # 返回模型、tokenizer、图像处理器和上下文长度
    return model, tokenizer, image_processor, context_len
