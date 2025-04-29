"""
Usage:
python3 -m mgm.model.consolidate --src ~/model_weights/llava-7b --dst ~/model_weights/llava-7b_consolidate
模型权重整合工具脚本
"""

import argparse  # 命令行参数解析

import torch  # PyTorch基础模块
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace模型加载
from mgm.model import *  # 导入自定义模型
from mgm.model.utils import auto_upgrade  # 自动升级工具

def consolidate_ckpt(src_path, dst_path):
    """整合模型检查点主函数
    参数：
        src_path: 源模型路径
        dst_path: 目标保存路径
    """
    print("Loading model")
    auto_upgrade(src_path)  # 自动升级旧版模型配置
    
    # 加载源模型（FP16精度，低内存模式）
    src_model = AutoModelForCausalLM.from_pretrained(
        src_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    
    # 加载源tokenizer
    src_tokenizer = AutoTokenizer.from_pretrained(src_path, use_fast=False)
    
    # 保存整合后的模型和tokenizer
    src_model.save_pretrained(dst_path)
    src_tokenizer.save_pretrained(dst_path)

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='模型权重整合工具')
    parser.add_argument("--src", type=str, required=True, help="源模型路径")
    parser.add_argument("--dst", type=str, required=True, help="目标保存路径")

    args = parser.parse_args()  # 解析参数

    consolidate_ckpt(args.src, args.dst)  # 执行整合
