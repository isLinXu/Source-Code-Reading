# 导入EncoderProjectorConcat类，用于后续构建语音投影器
from .speech_projector import EncoderProjectorConcat

def build_speech_projector(config):
    """
    构建并返回一个语音投影器实例。

    根据配置文件中指定的投影器类型，选择相应的投影器实现。
    当配置文件中未指定或指定为'linear'时，使用EncoderProjectorConcat类实例化投影器。

    参数:
    config: 配置文件或对象，包含模型的相关配置信息，包括speech_projector_type属性。

    返回:
    语音投影器的实例。

    抛出:
    ValueError: 当speech_projector_type不是'linear'时抛出，表示投影器类型未知。
    """
    # 获取配置文件中指定的投影器类型，如果未指定则默认为'linear'
    projector_type = getattr(config, 'speech_projector_type', 'linear')

    # 当投影器类型为'linear'时，返回EncoderProjectorConcat类实例化的投影器
    if projector_type == 'linear':
        return EncoderProjectorConcat(config)

    # 如果投影器类型不是'linear'，抛出ValueError异常，指出未知的投影器类型
    raise ValueError(f'Unknown projector type: {projector_type}')
