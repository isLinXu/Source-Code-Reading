import os

from ...utils import import_modules

# LLM_FACTORY 是一个字典，用于存储模型名称和对应的类
LLM_FACTORY = {}

def LLMFactory(model_name_or_path):
    """
    根据模型名称或路径返回模型实例和分词器及加载后处理函数。

    参数:
    model_name_or_path (str): 模型的名称或路径。

    返回:
    tuple: 包含模型实例和分词器及加载后处理函数的元组。

    异常:
    AssertionError: 如果模型名称或路径未注册，则抛出此异常。
    """
    model, tokenizer_and_post_load = None, None
    # 遍历 LLM_FACTORY 字典中的键
    for name in LLM_FACTORY.keys():
        # 如果模型名称或路径包含字典中的键，则获取对应的模型实例和分词器及加载后处理函数
        if name in model_name_or_path.lower():
            model, tokenizer_and_post_load = LLM_FACTORY[name]()
    # 确保找到了模型实例
    assert model, f"{model_name_or_path} is not registered"
    return model, tokenizer_and_post_load


def register_llm(name):
    """
    注册一个新的模型类。

    参数:
    name (str): 模型的名称。

    返回:
    function: 一个装饰器函数，用于注册模型类。
    """
    def register_llm_cls(cls):
        # 如果名称已存在于 LLM_FACTORY 中，则返回现有的类
        if name in LLM_FACTORY:
            return LLM_FACTORY[name]
        # 否则，将新的模型类注册到 LLM_FACTORY 字典中
        LLM_FACTORY[name] = cls
        return cls
    return register_llm_cls


# automatically import any Python files in the models/ directory
# 自动导入 models/ 目录下的所有 Python 文件
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.llm")
