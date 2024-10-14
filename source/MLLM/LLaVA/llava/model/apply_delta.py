"""
Usage:
python3 -m fastchat.model.apply_delta --base ~/model_weights/llama-7b --target ~/model_weights/vicuna-7b --delta lmsys/vicuna-7b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava import LlavaLlamaForCausalLM


def apply_delta(base_model_path, target_model_path, delta_path):
    """
    将delta模型加到基础模型上，并保存到目标模型路径。

    :param base_model_path: 基础模型的路径。
    :param target_model_path: 目标模型保存的路径。
    :param delta_path: delta模型的路径。
    """
    # 加载基础模型
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    # 加载delta模型及其tokenizer
    print("Loading delta")
    delta = LlavaLlamaForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)

    # 应用delta到基础模型上
    print("Applying delta")
    for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
        # 检查delta中的参数是否在基础模型中，如果不是，则跳过
        if name not in base.state_dict():
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
            continue
        # 如果参数形状相同，直接相加
        if param.data.shape == base.state_dict()[name].shape:
            param.data += base.state_dict()[name]
        # 如果参数形状不同，只对部分重叠的区域相加
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
                f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam
    # 保存目标模型及其tokenizer
    print("Saving target model")
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
