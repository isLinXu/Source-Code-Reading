import tokenizers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor

from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

# 加载设置，将训练参数同步到模型参数中
def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio


# 训练函数
def train():
    # 加载参数
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    logger_setting(getattr(training_arguments, 'output_dir', None))                                 # 设置日志输出目录
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) # 创建训练配方
    load_settings(model_arguments, data_arguments, training_arguments)                              # 加载设置

    # load pretrained checkpoint
    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(training_arguments.pretrained_model_path, trust_remote_code=True)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(training_arguments.pretrained_model_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
    model.tokenizer = tokenizer                                             # 将tokenizer赋值给模型
    model = training_recipe(model)                                          # 应用训练配方
    model.config.use_cache = False                                          # 关闭缓存
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio     # 设置图像宽高比
    data_arguments.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name_or_path) # 加载图像处理器
    data_arguments.is_multimodal = True                                     # 设置数据模块为多模态
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)     # 创建数据模块
    log_trainable_params(model)  # not work well with zero3                 # 打印可训练参数（可能与zero3不兼容）
    # 创建并启动训练器
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    # 创建并启动训练器
    trainer.train()
    # 保存模型和训练结果
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
