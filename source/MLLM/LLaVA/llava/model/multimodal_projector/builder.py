import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    """
    该类实现了一个恒等映射的神经网络模块，它在前向传播过程中不会对输入数据进行任何变换，直接返回输入数据。
    这种模块通常用于需要一个标准的模块接口但实际不需要改变数据的场景。
    """
    def __init__(self):
        """
        初始化IdentityMap类的实例。

        该构造函数通过调用父类nn.Module的构造函数初始化了一个神经网络模块实例。
        由于该类不需要维护任何参数，因此没有必要在构造函数中添加任何参数。
        """
        super().__init__()

    def forward(self, x, *args, **kwargs):
        """
        定义了该模块的前向传播行为。

        参数:
            x: 输入数据，可以是任意形状的张量。
            *args, **kwargs: 允许传递额外的未指定参数，以提供更高的灵活性，
                             但在这个实现中并未使用这些参数。

        返回:
            直接返回输入数据x，不做任何处理。
        """
        return x

    @property
    def config(self):
        """
        提供了一个只读属性，用于表示模块的配置信息。

        返回:
            一个字典，包含了该模块的具体配置信息。在这个实现中，
            它返回了一个包含'identity'类型标识的字典，表明了该模块
            是一个恒等映射模块。
        """
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    """
    简单的残差块类.

    该类实现了深度神经网络中的一个基本残差块，包含一个层归一化操作和两个全连接层，
    后接GELU激活函数. 输入和输出在经过层归一化后进行残差连接.

    参数:
    - channels (int): 输入和输出的通道数（特征维度）.
    """
    def __init__(self, channels):
        """
        初始化SimpleResBlock对象.

        参数:
        - channels (int): 输入和输出的通道数（特征维度）.
        """
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        """
        定义前向传播过程.

        参数:
        - x (Tensor): 输入的特征张量.

        返回:
        - Tensor: 经过残差块处理后的特征张量.
        """
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    """
    根据配置构建视觉投影器。

    该函数根据配置文件中指定的投影器类型，返回相应的视觉投影器模型。
    支持的投影器类型包括'linear'、'mlpNx_gelu'和'identity'。

    参数:
    - config: 配置对象，包含投影器类型和隐藏尺寸等信息。
    - delay_load: 延迟加载标志，目前未使用。
    - **kwargs: 附加关键字参数，目前未使用。

    返回:
    - 一个视觉投影器模型实例。

    异常:
    - 如果投影器类型未知，抛出ValueError异常。
    """
    # 获取配置中指定的投影器类型，默认为'linear'
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    # 如果投影器类型为'linear'，直接返回一个线性投影器模型
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    # 使用正则表达式匹配mlpNx_gelu类型的投影器，并构建相应的多层感知机模型
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        # 通过正则表达式获取mlp的深度
        mlp_depth = int(mlp_gelu_match.group(1))
        # 初始化模型模块列表，从输入尺寸映射到隐藏尺寸
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        # 循环添加GELU激活函数和线性变换层，构建mlp模型
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        # 返回顺序连接的多层感知机模型
        return nn.Sequential(*modules)

    # 如果投影器类型为'identity'，返回一个恒等映射对象
    if projector_type == 'identity':
        return IdentityMap()

    # 如果投影器类型不属于已知类型，抛出异常
    raise ValueError(f'Unknown projector type: {projector_type}')
