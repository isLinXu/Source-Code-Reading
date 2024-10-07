from packaging import version
import pathlib

import tokenizers
import transformers

# 导入TinyLlava训练相关的模块和工具
from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

# 检查tokenizers版本是否大于等于0.14
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    """
    加载模型设置，将训练参数合并到模型参数中。

    :param model_arguments: 模型参数
    :param data_arguments: 数据参数
    :param training_arguments: 训练参数
    :return: 合并后的模型参数
    """
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args

def _load_llm_settings(model_arguments):
    """
    加载LLM（Large Language Model）的设置。

    :param model_arguments: 模型参数
    :return: LLM设置
    """
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    """
    加载视觉塔的设置。

    :param model_arguments: 模型参数
    :return: 视觉塔设置
    """
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    """
    加载连接器的设置。

    :param model_arguments: 模型参数
    :return: 连接器设置
    """
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


def train():
    # 加载参数
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    # 设置日志
    logger_setting(getattr(training_arguments, 'output_dir', None))
    # 创建训练配方
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    # 加载模型参数
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGeneration(model_config)
    # 加载预训练模型
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)

    # 记录可训练参数
    log_trainable_params(model)  # not work well with zero3
    # 创建训练器并开始训练
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    
    trainer.train()
    # 保存模型和训练结果
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
