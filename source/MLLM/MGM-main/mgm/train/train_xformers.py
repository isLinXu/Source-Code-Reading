# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.
# 中文：通过对 LLaMA 模型进行猴子补丁（monkey patch），使用 xformers 注意力机制以提高内存效率。

# Need to call this before importing transformers.
# 中文：必须在导入 transformers 之前调用此函数。
from mgm.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()
# 中文：执行替换操作，将 LlamaAttention.forward 方法替换为 xformers 加速版本

from mgm.train.train import train
# 中文：从 mgm.train.train 模块中导入 train 函数

if __name__ == "__main__":
    train()
    # 中文：当脚本作为主程序执行时，调用 train() 进入训练流程
