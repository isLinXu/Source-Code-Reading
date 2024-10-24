import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_hidden_size = config.audio_hidden_size
        self.llm_hidden_size = config.hidden_size
        self.linear = nn.Linear(self.audio_hidden_size, self.llm_hidden_size)

    def forward(self, x):
        return self.linear(x)