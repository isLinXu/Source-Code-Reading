import os

from collections import OrderedDict

import torch
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.tuners.lora import LoraLayer

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils.train_utils import *
from ..utils import log
from ..model import TinyLlavaConfig, TinyLlavaForConditionalGeneration


@register_training_recipe('qlora_int8')
class QLoRAInt8TrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        """
        初始化QLoRAInt8训练配方。

        :param training_arguments: 训练参数
        """
        super().__init__(training_arguments)                                    # 调用父类的初始化方法
        self.training_arguments = training_arguments                            # 保存训练参数
        self.lora_skip_module = ['connector', 'vision_tower', 'language_model']

        
    def add_args(self, model_args):
        """
        向模型参数中添加必要的配置。

        :param model_args: 模型参数字典
        :return: 更新后的模型参数字典
        """
        # 根据训练参数设置llm的数据类型
        llm_dtype = (torch.float16 if self.training_arguments.fp16 else (torch.bfloat16 if self.training_arguments.bf16 else torch.float32))
        model_args['llm'].update(dict(torch_dtype=llm_dtype))                   # 更新llm的数据类型
        model_args['llm'].update(dict(low_cpu_mem_usage=True))                  # 设置低CPU内存使用

        # 定义量化配置
        quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,                                      # 加载8位
                        bnb_4bit_use_double_quant=True,                         # 使用双精度量化
                        bnb_4bit_quant_type="nf4",                              # 量化类型为nf4
                        bnb_4bit_compute_dtype=llm_dtype                        # 计算数据类型
                        )
        model_args['llm'].update(dict(quantization_config=quantization_config)) # 更新llm的量化配置

        # 如果预训练模型路径不为空，则更新模型参数中的预训练路径
        if self.training_arguments.pretrained_model_path is not None:
            model_args['llm'].update(dict(pretrained_llm_path=os.path.join(self.training_arguments.pretrained_model_path, 'language_model')))
            model_args['vision_tower'].update(dict(pretrained_vision_tower_path=os.path.join(self.training_arguments.pretrained_model_path, 'vision_tower')))
            model_args['connector'].update(dict(pretrained_connector_path=os.path.join(self.training_arguments.pretrained_model_path, 'connector')))
        return model_args


    def training_model_converse(self, model):
        """
        根据训练参数配置LoRA模型适配器。

        Args:
            model (torch.nn.Module): 需要配置LoRA适配器的模型。

        Returns:
            torch.nn.Module: 配置了LoRA适配器的模型。
        """
        # 如果调优类型为'qlora'，则移除'connector'模块
        if self.training_arguments.tune_type_connector == 'qlora':
            self.lora_skip_module.remove('connector')
        # 如果调优类型为'qlora'，则移除'language_model'模块
        if self.training_arguments.tune_type_llm == 'qlora':
            self.lora_skip_module.remove('language_model')
        # 如果调优类型为'qlora'，则移除'vision_tower'模块
        if self.training_arguments.tune_type_vision_tower == 'qlora':
            self.lora_skip_module.remove('vision_tower')

        # 配置LoRA参数
        lora_config = LoraConfig(
            r=self.training_arguments.lora_r,
            lora_alpha=self.training_arguments.lora_alpha,
            target_modules=find_all_linear_names(model, self.lora_skip_module),
            lora_dropout=self.training_arguments.lora_dropout,
            bias=self.training_arguments.lora_bias,
            task_type="CAUSAL_LM",
        )

        # 根据训练参数设置模型数据类型
        if self.training_arguments.bits == 16:
            if self.training_arguments.bf16:
                model.to(torch.bfloat16)
            if self.training_arguments.fp16:
                model.to(torch.float16)

        # 添加LoRA适配器
        log("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)  
        return model  # 返回配置了LoRA适配器的模型
        
        
    def save(self, model, trainer):
        """
        保存模型和训练器的状态。

        Args:
            model (Any): 需要保存的模型对象。
            trainer (Any): 训练器对象。
        """
        model.config.use_cache = True    # 启用缓存
        #save tokenizer                  # 保存分词器
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        #save entire model config        # 保存整个模型配置
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        #save trainer                    # 保存训练器状态
        trainer.save_state() 

        #save language model base params # 保存语言模型基础参数
        language_model_state_dict = get_peft_state_non_lora_maybe_zero_3(model.language_model.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)

        #save vision tower base params   # 保存视觉塔基础参数
        vision_tower_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(self.training_arguments.output_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            model.config.vision_config.save_pretrained(vision_tower_output_dir, from_pt=True)

        #save connector base params      # 保存连接器基础参数
        connector_state_dict = get_peft_state_non_lora_maybe_zero_3(model.connector.named_parameters(),  False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)

        # save lora params               # 保存lora参数
        lora_state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), self.training_arguments.lora_bias
        )
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            model.save_pretrained(self.training_arguments.output_dir, state_dict=lora_state_dict)
        

