# 从 https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/projector.py 参考并修改
import torch
import torch.nn as nn

class EncoderProjectorConcat(nn.Module):
    """
    该类用于将编码器的输出投影到统一的语言模型维度空间。
    它首先对编码器输出进行重排和维度变换，然后通过全连接层将其映射到目标维度。

    参数:
        config: 配置类，包含演讲编码器的下采样率、隐藏状态维度以及目标语言模型的隐藏状态维度。
    """
    def __init__(self, config):
        super().__init__()
        # 演讲编码器的下采样率，用于后续计算中调整序列长度
        self.k = config.speech_encoder_ds_rate
        # 演讲编码器的隐藏状态维度
        self.encoder_dim = config.speech_encoder_hidden_size
        # 目标语言模型的隐藏状态维度
        self.llm_dim = config.hidden_size
        # 将编码器输出的维度变换至2048
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        # ReLU激活函数，用于引入非线性
        self.relu = nn.ReLU()
        # 将经过激活的输出映射到目标语言模型的维度空间
        self.linear2 = nn.Linear(2048, config.hidden_size)

    def forward(self, x):
        """
        前向传播函数，对输入的编码器输出进行维度变换和映射。

        参数:
            x: 输入张量，为编码器的输出，形状为(batch_size, seq_len, dim)。

        返回:
            经过维度变换和映射后的输出张量，形状为(batch_size, new_seq_len, llm_dim)。
        """
        # 获取输入张量的batch_size, seq_len和dim
        batch_size, seq_len, dim = x.size()
        # 计算并丢弃不能整除下采样率k的序列帧数，以保证后续重排不会改变信息
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        # 确保张量在内存中连续存放，避免潜在的梯度计算错误
        x = x.contiguous()
        # 重排张量，将连续的k帧合并为一个单元，适应后续全连接层的输入要求
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        # 通过第一层全连接层，将编码器输出映射到2048维空间
        x = self.linear1(x)
        # 应用ReLU激活函数，引入非线性变换
        x = self.relu(x)
        # 通过第二层全连接层，将经过非线性变换的输出映射到目标语言模型的维度空间
        x = self.linear2(x)
        return x
