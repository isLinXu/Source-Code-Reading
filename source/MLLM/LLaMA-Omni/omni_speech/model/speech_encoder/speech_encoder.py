# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        """
        加载并返回一个修改后的Whisper模型编码器。

        Args:
            model_config (ModelConfig): 包含模型配置信息的对象。

        Returns:
            nn.Module: 修改后的Whisper模型编码器。
        """
        # 定义一个函数用于递归替换模型中的LayerNorm层
        def replace_layer_norm(module):
            """
            递归遍历模型的子模块，如果子模块是LayerNorm层，则用nn.LayerNorm替换它。

            Args:
                module (nn.Module): 要遍历的PyTorch模块。
            """
            from whisper.model import LayerNorm             # 导入whisper库中的LayerNorm类
            for name, child in module.named_children():     # 遍历模块的子模块
                if isinstance(child, LayerNorm):            # 如果子模块是LayerNorm层
                    old_params = child.state_dict()         # 获取旧LayerNorm层的状态字典
                    # 创建一个新的nn.LayerNorm层，参数与旧LayerNorm层相同
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)  # 将旧LayerNorm层的状态加载到新LayerNorm层
                    setattr(module, name, new_layer_norm)       # 在原模块中替换为新LayerNorm层
                else:
                    replace_layer_norm(child)

        import whisper                  # 导入whisper库
        encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder # 加载Whisper模型编码器
        replace_layer_norm(encoder)     # 调用replace_layer_norm函数替换编码器中的LayerNorm层
        return encoder                  # 返回修改后的编码器