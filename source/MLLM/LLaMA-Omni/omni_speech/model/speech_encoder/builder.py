from .speech_encoder import WhisperWrappedEncoder


def build_speech_encoder(config):
    """
    根据配置构建语音编码器。

    参数:
    config (object): 包含配置信息的对象。

    返回:
    object: 构建好的语音编码器对象。

    异常:
    ValueError: 如果配置中的语音编码器类型未知。
    """
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if "whisper" in speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')
