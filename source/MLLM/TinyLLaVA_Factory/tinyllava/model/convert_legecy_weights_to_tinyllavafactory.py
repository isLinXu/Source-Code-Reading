import os
import json

from huggingface_hub import hf_hub_download
import torch

from safetensors import safe_open
from .modeling_tinyllava import TinyLlavaForConditionalGeneration
from .configuration_tinyllava import TinyLlavaConfig

# 定义需要修改的权重名称映射
KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.vision_tower": "vision_tower._vision_tower",
    "model.mm_projector": "connector._connector",
    "model.embed_tokens": "language_model.model.embed_tokens",
    "model.layers": "language_model.model.layers",
    "model.norm": "language_model.model.norm",
    "lm_head": "language_model.lm_head",
    "model.final_layernorm": "language_model.model.final_layernorm"
}
# 定义模型名称映射
KEYS_TO_MODELNAME_MAPPING = {
    "TinyLlavaLlamaForCausalLM": 'TinyLlama/TinyLlama-1.1B-chat-v1.0',
    "TinyLlavaStablelmForCausalLM": 'stabilityai/stablelm-2-zephyr-1_6b',
    "TinyLlavaPhiForCausalLM": 'microsoft/phi-2',
    "bczhou/TinyLLaVA-3.1B-SigLIP": 'google/siglip-so400m-patch14-384',
    "bczhou/TinyLLaVA-2.0B-SigLIP": 'google/siglip-so400m-patch14-384',
    "bczhou/TinyLLaVA-1.5B-SigLIP": 'google/siglip-so400m-patch14-384',
}

def convert_legecy_config_to_tinyllavaconfig(old_config_path):
    """
    将旧配置转换为TinyLlava配置

    :param old_config_path: 旧配置文件路径或模型名称
    :return: TinyLlava配置对象
    """
    # 检查旧配置文件是否存在，如果存在则读取，否则从Hugging Face Hub下载

    if os.path.exists(old_config_path):
        config_path = os.path.join(old_config_path, 'config.json')
    else:
        config_path = hf_hub_download(old_config_path, "config.json")
        
    with open(config_path, 'r') as f:
        old_config = json.load(f)
    # 获取LLM和Vision模型的名称或路径
    llm_model_name_or_path = KEYS_TO_MODELNAME_MAPPING[old_config['architectures'][0]]
    vision_model_name_or_path = KEYS_TO_MODELNAME_MAPPING[old_config['mm_vision_tower']]
    # 创建TinyLlava配置对象
    model_config = TinyLlavaConfig(
        llm_model_name_or_path = llm_model_name_or_path,
        vision_model_name_or_path = vision_model_name_or_path,
        connector_type = old_config['mm_projector_type'],
        hidden_size = old_config['hidden_size'],
        vocab_size = old_config['vocab_size'],
        pad_token = old_config['pad_token'],
        tokenizer_padding_side = old_config['tokenizer_padding_side'],
        tokenizer_model_max_length = old_config['tokenizer_model_max_length'],
        vision_feature_layer = old_config['mm_vision_select_layer'],
        vision_feature_select_strategy = old_config['mm_vision_select_feature'],
        image_aspect_ratio = old_config['image_aspect_ratio'],
        use_cache = old_config['use_cache']
    )
    return model_config
        

def convert_state_dict_to_tinyllavafactory(old_state_dict_path):
    """
    将旧的状态字典转换为TinyLlavaFactory可用的状态字典

    :param old_state_dict_path: 旧状态字典文件路径或模型名称
    :return: 转换后的新状态字典
    """
    old_state_dict = []
    # 检查旧状态字典文件是否存在，如果存在则读取，否则从Hugging Face Hub下载
    if os.path.exists(old_state_dict_path):
        meta_file_name = os.path.join(old_state_dict_path, 'model.safetensors.index.json')
        if os.path.exists(meta_file_name):
            with open(meta_file_name, 'r') as f:
                meta_file = json.load(f)
            meta_file = list(set(meta_file['weight_map'].values()))
            for name in meta_file:
                old_state_dict.append(os.path.join(old_state_dict_path, name))
        else:
            old_state_dict.append(os.path.join(old_state_dict_path, 'model.safetensors'))
    else:
        try:
            meta_file_name = hf_hub_download(old_state_dict_path, 'model.safetensors.index.json')
            with open(meta_file_name, 'r') as f:
                meta_file = json.load(f)
            meta_file = list(set(meta_file['weight_map'].values()))
            for name in meta_file:
                old_state_dict.append(hf_hub_download(old_state_dict_path, name))
        except:
            old_state_dict.append(hf_hub_download(old_state_dict_path, 'model.safetensors'))
    state_dict = {}
    # 读取旧状态字典中的权重
    for osd in old_state_dict:
        with safe_open(osd, framework="pt",device=0) as f:
            for k in f.keys():
                state_dict[k]= f.get_tensor(k)

    new_state_dict={}
    # 根据映射修改权重名称
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict

def convert_legecy_weights_to_tinyllavafactory(old_state_dict_path, new_state_dict_path=None):
    """
    将旧权重转换为TinyLlavaFactory可用的权重，并保存新模型

    :param old_state_dict_path: 旧权重文件路径或模型名称
    :param new_state_dict_path: 新权重保存路径，如果为None则不保存
    :return: 转换后的TinyLlava模型
    """
    # 转换旧配置为TinyLlava配置
    model_config = convert_legecy_config_to_tinyllavaconfig(old_state_dict_path)
    # 创建TinyLlava模型
    model = TinyLlavaForConditionalGeneration(model_config)
    # For the checkpoints saved as '*.safetensors.
    # 转换旧状态字典为新状态字典
    state_dict = convert_state_dict_to_tinyllavafactory(old_state_dict_path)
    # 加载新状态字典到模型
    model.load_state_dict(state_dict, False)
    # 如果指定了新权重保存路径，则保存新模型
    if new_state_dict_path is not None:
        model.config.save_pretained(new_state_dict_path)
        model.tokenizer.save_pretrained(new_state_dict_path)
        model.save_pretrained(new_state_dict_path)
    return model
