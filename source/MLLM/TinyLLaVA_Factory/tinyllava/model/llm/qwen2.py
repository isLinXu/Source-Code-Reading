from transformers import Qwen2ForCausalLM, AutoTokenizer

from . import register_llm

# 注册一个名为'qwen2'的LLM
@register_llm('qwen2')
def return_qwen2class():
    """
    返回Qwen2ForCausalLM模型类和相关的tokenizer配置函数。

    Returns:
        tuple: 包含模型类和tokenizer配置函数的元组。
    """

    # 定义一个函数，用于在加载tokenizer后进行配置
    def tokenizer_and_post_load(tokenizer):
        """
        配置tokenizer，将unk_token设置为pad_token。

        Args:
            tokenizer (AutoTokenizer): 要配置的tokenizer对象。

        Returns:
            AutoTokenizer: 配置后的tokenizer对象。
        """
        tokenizer.unk_token = tokenizer.pad_token
        # 注释掉的代码是将pad_token设置为unk_token，这里保留了unk_token的原始值
#        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    # 返回Qwen2ForCausalLM模型类和tokenizer配置函数的元组
    return Qwen2ForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
