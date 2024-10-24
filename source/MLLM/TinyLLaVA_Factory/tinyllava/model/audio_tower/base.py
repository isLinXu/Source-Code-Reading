# audio_tower/base.py
import torch.nn as nn

class AudioTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._audio_tower = None
        self._audio_processor = None
        self.config = cfg

    def forward(self, x):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError