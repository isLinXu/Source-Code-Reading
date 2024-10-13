from transformers import GemmaForCausalLM, AutoTokenizer

from . import register_llm

# 注册一个名为'gemma'的LLM模型
@register_llm('gemma')
def return_gemmaclass():
    """
    返回Gemma模型的类和相关的tokenizer配置。

    Returns:
        tuple: 包含Gemma模型类和一个包含tokenizer类及后加载处理的元组。
    """
    def tokenizer_and_post_load(tokenizer):
        """
        设置tokenizer的pad_token为unk_token。

        Args:
            tokenizer (AutoTokenizer): 需要配置的tokenizer实例。

        Returns:
            AutoTokenizer: 配置后的tokenizer实例。
        """
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    # 返回Gemma模型类和包含AutoTokenizer类及后加载处理函数的元组
    return (GemmaForCausalLM, (AutoTokenizer, tokenizer_and_post_load))

