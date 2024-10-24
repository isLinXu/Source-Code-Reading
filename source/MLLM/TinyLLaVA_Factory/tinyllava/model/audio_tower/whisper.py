
# audio_tower/whisper.py
from transformers import WhisperModel, WhisperProcessor
from . import register_audio_tower
from .base import AudioTower

@register_audio_tower('whisper')
class WhisperAudioTower(AudioTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._audio_tower = WhisperModel(cfg)
        self._audio_processor = WhisperProcessor.from_pretrained(cfg.model_name_or_path)

    def forward(self, x):
        return self._audio_tower(x).last_hidden_state

    def load_model(self, model_path):
        self._audio_tower = WhisperModel.from_pretrained(model_path)
        self._audio_processor = WhisperProcessor.from_pretrained(model_path)