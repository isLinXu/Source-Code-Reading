from transformers import StableLmForCausalLM, AutoTokenizer

from . import register_llm

# 注册名为'stablelm'的语言模型
@register_llm('stablelm')
def return_phiclass():
    """
    返回一个包含StableLmForCausalLM模型和相应tokenizer的元组。

    Returns:
        tuple: 包含模型类和tokenizer处理函数的元组。
    """
    # 定义一个内部函数，用于在加载tokenizer后进行额外处理
    def tokenizer_and_post_load(tokenizer):
        """
        对加载的tokenizer进行处理。

        Args:
            tokenizer (AutoTokenizer): 自动加载的tokenizer对象。

        Returns:
            AutoTokenizer: 处理后的tokenizer对象。
        """
        return tokenizer

    # 返回模型类和包含tokenizer及其处理函数的元组
    return (StableLmForCausalLM, (AutoTokenizer, tokenizer_and_post_load))
