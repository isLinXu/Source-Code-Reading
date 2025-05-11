# Modality Projection from Vision to Language # 模态投影：从视觉到语言
import torch.nn as nn  # 导入PyTorch神经网络模块

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数
        self.cfg = cfg  # 保存配置对象
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)  # 计算输入维度，视觉隐藏层维度乘以像素重排因子平方
        self.output_dim = cfg.lm_hidden_dim  # 输出维度，语言模型隐藏层维度
        self.scale_factor = cfg.mp_pixel_shuffle_factor  # 像素重排因子

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)  # 线性投影层，将视觉特征映射到语言模型维度
        
        self.apply(self._init_weights)  # 应用权重初始化函数

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)  # 初始化线性层权重为正态分布
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # 初始化线性层偏置为零（如果存在）

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()  # 获取批次大小、序列长度和嵌入维度
        seq_root = int(seq**0.5)  # 计算序列长度的平方根（假设输入是展平的图像块）
        assert seq_root**2 == seq # Sequence lenght must be a perfect square for pixel shuffle # 断言序列长度必须是完全平方数，以便进行像素重排
        assert seq_root % self.scale_factor == 0 # Sequence root must be dividible by scale factor # 断言序列长度的平方根必须能被缩放因子整除

        height = width = seq_root  # 假设输入是正方形的图像块序列
        x = x.view(bsz, height, width, embed_dim)  # 将序列展平的输入重塑回图像形状 [bsz, height, width, embed_dim]
        h_out = height // self.scale_factor  # 计算输出高度
        w_out = width // self.scale_factor  # 计算输出宽度
        
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)  # 重塑以准备像素重排
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # 调整维度顺序以实现像素重排效果
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)  # 将重排后的像素展平，并增加嵌入维度
        
        return x  # 返回像素重排后的张量

    def forward(self, x):
        x = self.pixel_shuffle(x)  # 对输入应用像素重排
        x = self.proj(x)  # 通过线性投影层

        return x  # 返回投影后的张量

    