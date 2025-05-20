"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可声明

import logging # 导入 logging 模块，用于记录日志
from typing import List # 从 typing 模块导入 List 类型提示

from torch import nn # 从 torch 库导入 nn 模块，用于构建神经网络

# 定义一个函数，用于绑定编码器和解码器的权重
def tie_encoder_decoder_weights(
    encoder: nn.Module, # 编码器模块，类型为 torch.nn.Module
    decoder: nn.Module, # 解码器模块，类型为 torch.nn.Module
    base_model_prefix: str, # 模型基础前缀，字符串类型
    skip_key: str # 需要跳过的键（权重名称），字符串类型
):
    # 初始化一个列表，用于存储未初始化的编码器权重名称
    uninitialized_encoder_weights: List[str] = []
    # 检查解码器和编码器的类是否相同
    if decoder.__class__ != encoder.__class__:
        # 如果不相同，记录一条信息日志
        logging.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
            # 提示解码器和编码器类不相等，在这种情况下确保所有编码器权重都已正确初始化。
        )

    # 定义一个递归函数，用于将编码器权重绑定到解码器
    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module, # 当前解码器模块指针
        encoder_pointer: nn.Module, # 当前编码器模块指针
        module_name: str, # 当前模块的名称（路径）
        uninitialized_encoder_weights: List[str], # 未初始化的编码器权重列表
        skip_key: str, # 需要跳过的键
        depth=0, # 递归深度，默认为0
    ):
        # 断言当前指针都是 torch.nn.Module 类型
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        # 如果当前解码器模块有 'weight' 属性且模块名称不包含 skip_key
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            # 断言当前编码器模块也有 'weight' 属性
            assert hasattr(encoder_pointer, "weight")
            # 将编码器的 weight 属性设置为解码器的 weight 属性（共享权重）
            encoder_pointer.weight = decoder_pointer.weight
            # 如果当前解码器模块有 'bias' 属性
            if hasattr(decoder_pointer, "bias"):
                # 断言当前编码器模块也有 'bias' 属性
                assert hasattr(encoder_pointer, "bias")
                # 将编码器的 bias 属性设置为解码器的 bias 属性（共享偏置）
                encoder_pointer.bias = decoder_pointer.bias
            # 打印信息，表示当前模块已绑定权重
            print(module_name + " is tied")
            return # 递归结束

        # 获取编码器和解码器的子模块字典
        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        # 如果解码器有子模块
        if len(decoder_modules) > 0:
            # 断言编码器也有子模块，且数量匹配
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            # 构建一个集合，包含所有编码器子模块的完整名称（路径）
            all_encoder_weights = set(
                [module_name + "/" + sub_name for sub_name in encoder_modules.keys()]
            )
            # 初始化编码器层的位置偏移量
            encoder_layer_pos = 0
            # 遍历解码器的子模块
            for name, module in decoder_modules.items():
                # 如果子模块名称是数字（表示在 nn.ModuleList 中的位置）
                if name.isdigit():
                    # 计算对应的编码器子模块名称（考虑偏移量）
                    encoder_name = str(int(name) + encoder_layer_pos)
                    # 解码器子模块名称就是当前名称
                    decoder_name = name
                    # 如果解码器子模块的类型与对应的编码器子模块类型不匹配，并且编码器和解码器的子模块数量不同
                    if not isinstance(
                        decoder_modules[decoder_name],
                        type(encoder_modules[encoder_name]),
                    ) and len(encoder_modules) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        # 这可能发生在名称对应于层模块列表中的位置时
                        # 在这种情况下，解码器添加了编码器没有的交叉注意力层
                        # 因此跳过这一步，并从编码器层位置减一
                        encoder_layer_pos -= 1
                        continue # 跳过当前子模块
                # 如果子模块名称不在编码器的子模块中
                elif name not in encoder_modules:
                    continue # 跳过当前子模块
                # 如果递归深度超过500，抛出错误
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                        # 递归函数 `tie_encoder_to_decoder` 达到最大深度。似乎您的模型中两个或多个 `nn.Modules` 之间存在循环依赖。
                    )
                # 否则，编码器和解码器子模块名称相同
                else:
                    decoder_name = encoder_name = name
                # 递归调用自身，处理子模块
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name], # 解码器子模块
                    encoder_modules[encoder_name], # 编码器子模块
                    module_name + "/" + name, # 更新模块名称路径
                    uninitialized_encoder_weights, # 传递未初始化权重列表
                    skip_key, # 传递跳过键
                    depth=depth + 1, # 增加递归深度
                )
                # 从所有编码器权重集合中移除已处理的编码器子模块名称
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            # 将剩余在集合中的编码器权重（即未绑定的）添加到未初始化权重列表中
            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    # 递归地绑定权重
    tie_encoder_to_decoder_recursively(
        decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key
        # 调用递归函数，从顶层模块开始绑定权重
    )
    