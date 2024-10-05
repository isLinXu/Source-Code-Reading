from transformers import PhiForCausalLM, AutoTokenizer

from . import register_llm

# 注册一个名为'phi'的LLM
@register_llm('phi')
def return_phiclass():
    """
    返回PhiForCausalLM模型类和一个包含AutoTokenizer与tokenizer_and_post_load函数的元组。

    Returns:
        tuple: 包含模型类和tokenizer处理函数的元组。
    """
    def tokenizer_and_post_load(tokenizer):
        """
        设置tokenizer的pad_token为unk_token。

        Args:
            tokenizer (AutoTokenizer): 要处理的tokenizer对象。

        Returns:
            AutoTokenizer: 处理后的tokenizer对象。
        """
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    # 返回PhiForCausalLM模型类和包含AutoTokenizer与tokenizer_and_post_load函数的元组
    return (PhiForCausalLM, (AutoTokenizer, tokenizer_and_post_load))
