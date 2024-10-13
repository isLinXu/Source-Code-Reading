import os
import torch

from ..utils import *
from ..model import *

class BaseTrainingRecipe:

    def __init__(self, training_arguments):
        """
        初始化训练配方，保存训练参数。

        :param training_arguments: 训练参数对象
        """
        self.training_arguments = training_arguments

    
    def __call__(self, model):
        """
        对模型进行一系列的调整和设置。

        :param model: 需要调整的模型对象
        :return: 调整后的模型对象
        """
        model = self.training_model_converse(model)                                                             # 调用模型对话函数
        model = self.tune_type_setting(model)                                                                   # 设置调整类型

        # 设置模型的配置参数
        model.config.tune_type_connector = self.training_arguments.tune_type_connector                          # 设置连接器微调类型
        model.config.tune_type_vision_tower = self.training_arguments.tune_type_vision_tower                    # 设置视觉塔微调类型
        model.config.tune_type_llm = self.training_arguments.tune_type_llm                                      # 设置LLM（大型语言模型）微调类型
        model.config.tune_vision_tower_from_layer = self.training_arguments.tune_vision_tower_from_layer        # 设置从哪一层开始微调视觉塔
        return model
    
 
    def add_args(self, model_args):
        """
        根据训练参数设置模型参数中的数据类型，并更新预训练模型的路径。

        :param model_args: 包含模型参数的字典
        :return: 更新后的模型参数字典
        """
        # 根据训练参数选择合适的数据类型
        llm_dtype = (torch.float16 if self.training_arguments.fp16 else (torch.bfloat16 if self.training_arguments.bf16 else torch.float32))
        # 更新模型参数中的数据类型
        model_args['llm'].update(dict(torch_dtype=llm_dtype))
        # 如果指定了预训练模型路径，则更新各部分的预训练模型路径
        if self.training_arguments.pretrained_model_path is not None:
            model_args['llm'].update(dict(pretrained_llm_path=os.path.join(self.training_arguments.pretrained_model_path, 'language_model')))
            model_args['vision_tower'].update(dict(pretrained_vision_tower_path=os.path.join(self.training_arguments.pretrained_model_path, 'vision_tower')))
            model_args['connector'].update(dict(pretrained_connector_path=os.path.join(self.training_arguments.pretrained_model_path, 'connector')))
        return model_args
            
    def tune_type_setting(self, model):
        """
        设置模型的微调类型。

        :param model: 需要设置微调类型的模型
        :return: 设置完成后的模型
        """
        # 分别对语言模型、视觉塔和连接器进行微调类型设置
        # 设置LLM（大型语言模型）的微调类型
        model = self._llm_tune_type_setting(model)
        # 设置视觉塔（vision tower）的微调类型
        model = self._vision_tower_tune_type_setting(model)
        # 设置连接器的微调类型
        model = self._connector_tune_type_setting(model)
        # 返回设置完成后的模型
        return model    
        
        
        
    def _llm_tune_type_setting(self, model):
        """
        设置模型的微调类型。

        参数:
            model (Model): 需要设置微调类型的模型。

        返回:
            Model: 设置好微调类型的模型。
        """
        # 获取训练参数中的微调类型并转换为小写
        tune_type = self.training_arguments.tune_type_llm.lower()
        # 断言微调类型是否在支持的范围内，tune_type in ('frozen', 'full', 'lora', 'qlora')
        assert tune_type in ('frozen', 'full', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'
        # 根据微调类型设置模型的语言模型层的梯度要求
        if tune_type == 'full':
            model.language_model.requires_grad_(True)
        elif tune_type == 'frozen':
            model.language_model.requires_grad_(False)
        # 支持梯度检查点
        self.support_gradient_checkpoint(model.language_model, self.training_arguments.gradient_checkpointing)
        # 返回设置好的模型
        return model
        
    def _vision_tower_tune_type_setting(self, model):
        """
        设置视觉塔的微调类型。

        Args:
            model: 需要设置微调类型的模型。

        Returns:
            model: 设置完成微调类型后的模型。
        """
        # 获取训练参数中的微调类型，并转换为小写
        tune_type = self.training_arguments.tune_type_vision_tower.lower()
        # 断言微调类型是否在支持的范围内
        assert tune_type in ('frozen', 'full', 'partially-tune', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'
        # 根据微调类型设置模型的requires_grad属性
        if tune_type == 'full':                         # 全部微调
            model.vision_tower.requires_grad_(True)
        elif tune_type == 'frozen':                     # 冻结不微调
            model.vision_tower.requires_grad_(False)         
        elif tune_type == 'partially-tune':             # 部分微调
            #--------------------------------------------
            #--------------------------------------------
            #TODO gradient checkpointing related???
            #--------------------------------------------
            #--------------------------------------------
            from_layer = self.training_arguments.tune_vision_tower_from_layer
            if from_layer > -1:
                log(f'Tune the vision tower from layer {from_layer}!')
                for n, p in model.vision_tower.named_parameters():
                    # 检查参数名称是否包含特定字符串，以确定是否属于需要微调的层
                    if 'vision_model.encoder.layers.' in n: #TODO not sure if other visual encoders contain 'vision_model.encoder.layers.'
                        layer_id = int(n.split('vision_model.encoder.layers.')[-1].split('.')[0])
                        # 根据from_layer参数设置层的requires_grad属性
                        if layer_id >= from_layer:
                            p.requires_grad = True
                        else:
                            p.requires_grad = False
                    else:
                        p.requires_grad = False
        #self.support_gradient_checkpoint(model.vision_tower._vision_tower, self.training_arguments.gradient_checkpointing)
        return model # 返回设置完成微调类型的模型
        
    def _connector_tune_type_setting(self, model):
        """
        根据训练参数设置模型连接器的微调类型。

        :param model: 需要设置微调类型的模型对象
        :return: 设置完成后的模型对象
        """
        # 获取训练参数中的连接器微调类型，并转换为小写
        tune_type = self.training_arguments.tune_type_connector.lower()
        # 断言微调类型是否在支持的范围内
        assert tune_type in ('frozen', 'full', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'   
        # 如果微调类型为'full'，则将模型连接器的所有参数设置为可训练
        if tune_type == 'full':
            for p in model.connector.parameters():
                p.requires_grad = True
        # 如果微调类型为'frozen'，则将模型连接器的所有参数设置为不可训练
        elif tune_type == 'frozen':
            for p in model.connector.parameters():
                p.requires_grad = False
        # 返回设置完成后的模型对象
        return model
    
    
        
    def training_model_converse(self, model):
        """
        对模型进行训练对话的函数。

        :param model: 需要进行训练对话的模型对象
        :return: 训练对话后的模型对象
        """
        # 这里可以添加具体的训练对话逻辑
        return model

    # 保存模型的函数，包括tokenizer、模型配置、trainer状态以及特定阶段的模型组件
    def save(self, model, trainer):
        model.config.use_cache = True                                                       # 设置模型使用缓存
        #save tokenizer                                                                     # 保存tokenizer
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        #save entire model config                                                           # 保存整个模型配置
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        #save trainer                                                                       # 保存trainer状态
        trainer.save_state()

        # 如果是微调阶段，并且有预训练模型路径
        if 'finetune' in self.training_arguments.output_dir and self.training_arguments.pretrained_model_path is not None: # for finetune stage
            # 获取vision tower的状态字典
            vision_tower_state_dict = get_state_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), [''],
                                                             False)
            # 如果是主进程（local_rank为0或-1）
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                # 创建vision tower的输出目录
                vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
                os.makedirs(vision_tower_output_dir, exist_ok=True)
                # 定义vision tower的输出路径
                vision_tower_output_path = os.path.join(self.training_arguments.output_dir,
                                                        'vision_tower/pytorch_model.bin')
                # 保存vision tower的状态字典到文件
                torch.save(vision_tower_state_dict, vision_tower_output_path)
                # 如果vision tower是预训练模型，则保存其配置
                if isinstance(model.vision_tower._vision_tower, PreTrainedModel):
                    model.vision_tower._vision_tower.config.save_pretrained(vision_tower_output_dir, from_pt=True)
            # save connector                                                                 # 获取connector的状态字典
            connector_state_dict = get_state_maybe_zero_3(model.connector.named_parameters(), [''], False)

            # 如果是主进程
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                # 创建connector的输出目录
                connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
                os.makedirs(connector_output_dir, exist_ok=True)
                # 定义connector的输出路径
                connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
                # 保存connector的状态字典到文件
                torch.save(connector_state_dict, connector_output_path)

            # 如果使用了deepspeed，则同步CUDA
            if trainer.deepspeed:
                torch.cuda.synchronize()
            # 保存模型
            trainer.save_model(self.training_arguments.output_dir)
            return
        
        #the followings are for pretrain stage 如果是预训练阶段
        #save language model 获取语言模型的状态字典
        language_model_state_dict = get_state_maybe_zero_3(model.language_model.named_parameters(), [''], False)
        # 如果是主进程
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            # 创建语言模型的输出目录
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            # 定义语言模型的输出路径
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            # 保存语言模型的状态字典到文件
            torch.save(language_model_state_dict, language_model_output_path)
            # 保存语言模型的配置
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)
        #save vision tower

    

    def load(self, model, model_args={}):
        """
        加载模型的函数，根据是否包含LoRA权重来决定加载方式。
        :param model: 需要加载的模型对象
        :param model_args: 包含模型加载所需参数的字典
        :return: 加载完成的模型对象
        """
        # 检查是否为非LoRA/非QLoRA预训练模型
        if not ('lora' in self.training_arguments.pretrained_model_path and os.path.exists(os.path.join(self.training_arguments.pretrained_model_path, 'adapter_config.json'))): # loading model for non-lora/non-qlora pretraining
            # 加载非LoRA模型的LLM、视觉塔和连接器部分
            model.load_llm(**model_args['llm'])
            model.load_vision_tower(**model_args['vision_tower'])
            model.load_connector(**model_args['connector'])
        else:
            # 加载LoRA模型的LLM部分，并指定注意力实现方式和数据类型
            model.language_model = model.language_model.from_pretrained(model_args['llm']['model_name_or_path'],attn_implementation='flash_attention_2',torch_dtype=model_args['llm']['torch_dtype'])
            # 加载视觉塔和连接器部分
            model.load_vision_tower(**model_args['vision_tower'])
            model.load_connector(**model_args['connector'])
            # 将模型转移到指定的数据类型
            model.to(model_args['llm']['torch_dtype'])

            # 从Peft导入PeftModel类
            from peft import PeftModel
            print('Loading LoRA weights...')

            # 加载LoRA权重
            model = PeftModel.from_pretrained(model, self.training_arguments.pretrained_model_path)
            print('Merging LoRA weights...')
            # 合并并卸载LoRA权重
            model = model.merge_and_unload()
            print('Model is loaded...')

        return model
        
    
    def support_gradient_checkpoint(self, model, gradient_checkpointing=False):
        """
        支持梯度检查点的功能，允许模型在反向传播时节省内存。

        :param model: 需要支持梯度检查点的模型对象。
        :param gradient_checkpointing: 是否启用梯度检查点功能的布尔值。
        """
        def make_inputs_require_grad(module, input, output):
            """
            设置模块的输出需要梯度计算。

            :param module: 当前处理的模块。
            :param input: 模块的输入。
            :param output: 模块的输出。
            """
            output.requires_grad_(True)

        # 如果启用了梯度检查点功能
        if gradient_checkpointing:
            # 如果模型有enable_input_require_grads方法，则调用它
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                # 否则，注册一个前向钩子函数，使得模型的输入嵌入层输出需要梯度计算
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        
       
