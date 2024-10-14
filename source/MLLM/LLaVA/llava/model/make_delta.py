"""
Usage:
python3 -m llava.model.make_delta --base ~/model_weights/llama-7b --target ~/model_weights/llava-7b --delta ~/model_weights/llava-7b-delta --hub-repo-id liuhaotian/llava-7b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.model.utils import auto_upgrade


def make_delta(base_model_path, target_model_path, delta_path, hub_repo_id):
    """
    计算并保存两个模型之间的差异。

    本函数旨在比较两个模型（基础模型和目标模型）之间的参数差异，并将这些差异保存下来。
    这对于在更新或微调模型时识别和跟踪具体变化非常有用。

    :param base_model_path: 基础模型的路径。
    :param target_model_path: 目标模型的路径。
    :param delta_path: 保存差异的路径。
    :param hub_repo_id: (可选) Hugging Face模型库中的仓库ID，用于推送结果模型。
    """
    # 加载基础模型
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # 加载目标模型
    print("Loading target model")
    auto_upgrade(target_model_path)
    target = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # 计算两个模型之间的差异
    print("Calculating delta")
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        # 检查参数是否存在于基础模型中，但忽略特定的例外
        if name not in base.state_dict():
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
            continue
        # 对于形状相同的参数，直接计算差异
        if param.data.shape == base.state_dict()[name].shape:
            param.data -= base.state_dict()[name]
        # 对于形状不同的参数（如嵌入层权重），处理形状不匹配的情况
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] -= bparam
    # 保存计算出的差异
    print("Saving delta")
    if hub_repo_id:
        kwargs = {"push_to_hub": True, "repo_id": hub_repo_id}
    else:
        kwargs = {}
    target.save_pretrained(delta_path, **kwargs)
    # 保存目标模型的分词器
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    target_tokenizer.save_pretrained(delta_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str, default=None)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path, args.hub_repo_id)
