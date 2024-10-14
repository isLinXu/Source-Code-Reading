"""
Usage:
python3 -m llava.model.consolidate --src ~/model_weights/llava-7b --dst ~/model_weights/llava-7b_consolidate
"""
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.model import *
from llava.model.utils import auto_upgrade


def consolidate_ckpt(src_path, dst_path):
    """
    整合模型检查点文件。

    将模型和tokenizer从源路径加载，并保存到目标路径。

    参数:
        src_path (str): 源文件夹路径，从这里加载模型和tokenizer。
        dst_path (str): 目标文件夹路径，将模型和tokenizer保存到这里。
    """
    print("Loading model")
    # 自动升级源路径的模型（具体实现未展示）
    auto_upgrade(src_path)
    # 从源路径加载模型，使用半精度浮点数格式，并优化内存使用
    src_model = AutoModelForCausalLM.from_pretrained(src_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # 从源路径加载tokenizer，不使用fast版本
    src_tokenizer = AutoTokenizer.from_pretrained(src_path, use_fast=False)
    # 将模型和tokenizer保存到目标路径
    src_model.save_pretrained(dst_path)
    src_tokenizer.save_pretrained(dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)

    args = parser.parse_args()

    consolidate_ckpt(args.src, args.dst)
