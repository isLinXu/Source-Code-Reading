from transformers import LlamaForCausalLM, AutoTokenizer

from . import register_llm
# 使用register_llm装饰器注册一个名为'tinyllama'的LLM
@register_llm('tinyllama')
def return_tinyllamaclass():
    """
    返回tinyllama模型的类和相关的tokenizer处理函数。

    返回:
        tuple: 包含LlamaForCausalLM模型类和一个包含AutoTokenizer和tokenizer处理函数的元组。
    """
    def tokenizer_and_post_load(tokenizer):
        """
        设置tokenizer的pad_token为unk_token。

        参数:
            tokenizer (AutoTokenizer): 要处理的tokenizer对象。

        返回:
            AutoTokenizer: 处理后的tokenizer对象。
        """
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    # 返回LlamaForCausalLM模型类和包含AutoTokenizer及tokenizer处理函数的元组
    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
