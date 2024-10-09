from .speech_generator import SpeechGeneratorCTC

def build_speech_generator(config):
    """
    根据配置文件构建语音生成器实例。

    该函数根据配置文件中指定的语音生成器类型，返回相应的语音生成器实例。目前支持的类型包括'ctc'。

    参数:
        config: 配置文件，包含语音生成器类型等信息。

    返回:
        SpeechGeneratorCTC实例: 如果配置文件中指定的语音生成器类型为'ctc'。

    异常:
        ValueError: 如果配置文件中指定的语音生成器类型不被支持。

    """
    # 获取配置文件中指定的语音生成器类型，默认为'ctc'
    generator_type = getattr(config, 'speech_generator_type', 'ctc')

    # 如果生成器类型为'ctc'，则返回SpeechGeneratorCTC实例
    if generator_type == 'ctc':
        return SpeechGeneratorCTC(config)

    # 如果生成器类型不被支持，抛出ValueError异常
    raise ValueError(f'Unknown generator type: {generator_type}')
