import torch
import torchaudio
from transformers import WhisperProcessor

class AudioPreprocess:
    def __init__(self, audio_processor, data_args={}):
        self.audio_processor = audio_processor
        self.sample_rate = getattr(data_args, 'audio_sample_rate', 16000)
        self.max_length = getattr(data_args, 'max_audio_length', 30)  # 最大音频长度（秒）

    def __call__(self, audio_path):
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 确保音频是单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到目标采样率
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # 裁剪或填充到最大长度
        target_length = int(self.max_length * self.sample_rate)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = torch.zeros(1, target_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)

        # 使用 Whisper 处理器进行特征提取
        inputs = self.audio_processor(waveform.squeeze().numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
        
        return inputs.input_features

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        audio_processor = WhisperProcessor.from_pretrained(model_name_or_path)
        return cls(audio_processor, **kwargs)