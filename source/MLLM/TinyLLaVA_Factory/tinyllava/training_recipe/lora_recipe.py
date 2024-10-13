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


@register_training_recipe('lora')
class LoRATrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        """
        初始化LoRATrainingRecipe类。

        :param training_arguments: 训练参数对象，包含各种训练相关的配置。
        """
        super().__init__(training_arguments)                                        # 调用父类的初始化方法
        self.training_arguments = training_arguments                                # 保存训练参数对象
        self.lora_skip_module = ['connector', 'vision_tower', 'language_model']     # 初始化跳过的模块列表
        
        
    def training_model_converse(self, model):
        """
        根据训练参数配置LoRA模型，并返回配置后的模型。

        :param model: 需要配置的原始模型。
        :return: 配置后的LoRA模型。
        """
        # 根据不同的tune_type，从跳过模块列表中移除相应的模块
        if self.training_arguments.tune_type_connector == 'lora':
            self.lora_skip_module.remove('connector')
        if self.training_arguments.tune_type_llm == 'lora':
            self.lora_skip_module.remove('language_model')
        if self.training_arguments.tune_type_vision_tower == 'lora':
            self.lora_skip_module.remove('vision_tower')

        # 创建LoRA配置对象
        lora_config = LoraConfig(
            r=self.training_arguments.lora_r,
            lora_alpha=self.training_arguments.lora_alpha,
            target_modules=find_all_linear_names(model, self.lora_skip_module),
            lora_dropout=self.training_arguments.lora_dropout,
            bias=self.training_arguments.lora_bias,
            task_type="CAUSAL_LM",
        )

        # 根据训练参数设置模型的数据类型
        if self.training_arguments.bits == 16:
            if self.training_arguments.bf16:
                model.to(torch.bfloat16)
            if self.training_arguments.fp16:
                model.to(torch.float16)

        # 输出日志信息
        log("Adding LoRA adapters...")

        # 获取并返回配置后的LoRA模型
        model = get_peft_model(model, lora_config)  
        return model

    # 保存模型的函数，包括tokenizer、模型配置、trainer状态以及各个部分的参数
    def save(self, model, trainer):
        model.config.use_cache = True
        #save tokenizer                  # 保存tokenizer
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        #save entire model config        # 保存整个模型配置
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        #save trainer                    # 保存trainer状态
        trainer.save_state() 

        #save language model base params # 保存语言模型基础参数
        language_model_state_dict = get_peft_state_non_lora_maybe_zero_3(model.language_model.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:                     # 仅在主进程或无local_rank时执行
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)                             # 创建目录
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)                 # 保存参数
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True) # 保存文本配置
        #save vision tower base params   # 保存视觉塔基础参数，逻辑同语言模型
        vision_tower_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(self.training_arguments.output_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            model.config.vision_config.save_pretrained(vision_tower_output_dir, from_pt=True)
        #save connector base params     # 保存连接器基础参数，逻辑同上
        connector_state_dict = get_peft_state_non_lora_maybe_zero_3(model.connector.named_parameters(),  False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)
        # save lora params              # 保存lora参数
        lora_state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), self.training_arguments.lora_bias
        )
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            model.save_pretrained(self.training_arguments.output_dir, state_dict=lora_state_dict)
        

