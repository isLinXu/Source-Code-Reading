from collections import OrderedDict  # 从collections模块导入OrderedDict
from typing import Tuple, Union  # 从typing模块导入Tuple和Union

import numpy as np  # 导入numpy库
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的功能性API
from torch import nn  # 从PyTorch导入nn模块


class Bottleneck(nn.Module):  # 定义Bottleneck类，继承自nn.Module
    expansion = 4  # 扩展因子

    def __init__(self, inplanes, planes, stride=1):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        # 所有卷积层的步幅为1。当步幅大于1时，在第二个卷积后执行平均池化
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(planes)  # 第一个BatchNorm层
        self.relu1 = nn.ReLU(inplace=True)  # 第一个ReLU激活层

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(planes)  # 第二个BatchNorm层
        self.relu2 = nn.ReLU(inplace=True)  # 第二个ReLU激活层

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()  # 平均池化层，步幅大于1时使用

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)  # 第三个卷积层
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)  # 第三个BatchNorm层
        self.relu3 = nn.ReLU(inplace=True)  # 第三个ReLU激活层

        self.downsample = None  # 下采样层初始化为None
        self.stride = stride  # 保存步幅

        if stride > 1 or inplanes != planes * Bottleneck.expansion:  # 如果步幅大于1或输入通道数与扩展后的通道数不匹配
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            # 下采样层在平均池化层之前，后续卷积的步幅为1
            self.downsample = nn.Sequential(OrderedDict([  # 定义下采样层
                ("-1", nn.AvgPool2d(stride)),  # 平均池化层
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),  # 卷积层
                ("1", nn.BatchNorm2d(planes * self.expansion))  # BatchNorm层
            ]))

    def forward(self, x: torch.Tensor):  # 前向传播方法
        identity = x  # 保存输入以便于后续的残差连接

        out = self.relu1(self.bn1(self.conv1(x)))  # 经过第一个卷积、BatchNorm和ReLU激活
        out = self.relu2(self.bn2(self.conv2(out)))  # 经过第二个卷积、BatchNorm和ReLU激活
        out = self.avgpool(out)  # 经过平均池化层
        out = self.bn3(self.conv3(out))  # 经过第三个卷积和BatchNorm

        if self.downsample is not None:  # 如果存在下采样层
            identity = self.downsample(x)  # 对输入进行下采样

        out += identity  # 残差连接
        out = self.relu3(out)  # 经过第三个ReLU激活
        return out  # 返回输出


class AttentionPool2d(nn.Module):  # 定义AttentionPool2d类，继承自nn.Module
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)  # 位置嵌入
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # K投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # Q投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim)  # V投影层
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)  # 输出投影层
        self.num_heads = num_heads  # 头数

    def forward(self, x):  # 前向传播方法
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(  # 进行多头注意力计算
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)  # 返回输出


class ModifiedResNet(nn.Module):  # 定义ModifiedResNet类，继承自nn.Module
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    # 一个与torchvision的ResNet类似的ResNet类，但包含以下更改：
    # - 现在有3个“stem”卷积，而不是1个，使用平均池化而不是最大池化。
    # - 执行抗混叠的步幅卷积，其中在步幅大于1的卷积之前执行平均池化
    # - 最后的池化层是QKV注意力，而不是平均池化

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.output_dim = output_dim  # 输出维度
        self.input_resolution = input_resolution  # 输入分辨率

        # the 3-layer stem
        # 3层stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(width // 2)  # 第一个BatchNorm层
        self.relu1 = nn.ReLU(inplace=True)  # 第一个ReLU激活层
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(width // 2)  # 第二个BatchNorm层
        self.relu2 = nn.ReLU(inplace=True)  # 第二个ReLU激活层
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)  # 第三个卷积层
        self.bn3 = nn.BatchNorm2d(width)  # 第三个BatchNorm层
        self.relu3 = nn.ReLU(inplace=True)  # 第三个ReLU激活层
        self.avgpool = nn.AvgPool2d(2)  # 平均池化层

        # residual layers
        # 残差层
        self._inplanes = width  # 这是一个在构造期间使用的可变变量
        self.layer1 = self._make_layer(width, layers[0])  # 创建第一层
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  # 创建第二层
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)  # 创建第三层
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)  # 创建第四层

        embed_dim = width * 32  # ResNet特征维度
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)  # 注意力池化层

    def _make_layer(self, planes, blocks, stride=1):  # 创建层的方法
        layers = [Bottleneck(self._inplanes, planes, stride)]  # 添加Bottleneck模块

        self._inplanes = planes * Bottleneck.expansion  # 更新输入通道数
        for _ in range(1, blocks):  # 添加剩余的Bottleneck模块
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)  # 返回一个顺序容器

    def forward(self, x):  # 前向传播方法
        def stem(x):  # stem函数
            x = self.relu1(self.bn1(self.conv1(x)))  # 经过第一个卷积、BatchNorm和ReLU激活
            x = self.relu2(self.bn2(self.conv2(x)))  # 经过第二个卷积、BatchNorm和ReLU激活
            x = self.relu3(self.bn3(self.conv3(x)))  # 经过第三个卷积、BatchNorm和ReLU激活
            x = self.avgpool(x)  # 经过平均池化层
            return x  # 返回输出

        x = x.type(self.conv1.weight.dtype)  # 转换输入数据类型
        x = stem(x)  # 经过stem
        x = self.layer1(x)  # 经过第一层
        x = self.layer2(x)  # 经过第二层
        x = self.layer3(x)  # 经过第三层
        x = self.layer4(x)  # 经过第四层
        x = self.attnpool(x)  # 经过注意力池化层

        return x  # 返回输出


class LayerNorm(nn.LayerNorm):  # 定义LayerNorm类，继承自nn.LayerNorm
    """Subclass torch's LayerNorm to handle fp16."""  # 子类化torch的LayerNorm以处理fp16

    def forward(self, x: torch.Tensor):  # 前向传播方法
        orig_type = x.dtype  # 保存原始数据类型
        ret = super().forward(x.type(torch.float32))  # 调用父类的前向传播
        return ret.type(orig_type)  # 返回原始数据类型的输出


class QuickGELU(nn.Module):  # 定义QuickGELU类，继承自nn.Module
    def forward(self, x: torch.Tensor):  # 前向传播方法
        return x * torch.sigmoid(1.702 * x)  # 返回GELU激活值


class ResidualAttentionBlock(nn.Module):  # 定义ResidualAttentionBlock类，继承自nn.Module
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法

        self.attn = nn.MultiheadAttention(d_model, n_head)  # 多头注意力层
        self.ln_1 = LayerNorm(d_model)  # 第一层LayerNorm
        self.mlp = nn.Sequential(OrderedDict([  # MLP层
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # 全连接层
            ("gelu", QuickGELU()),  # GELU激活层
            ("c_proj", nn.Linear(d_model * 4, d_model))  # 输出全连接层
        ]))
        self.ln_2 = LayerNorm(d_model)  # 第二层LayerNorm
        self.attn_mask = attn_mask  # 注意力掩码

    def attention(self, x: torch.Tensor):  # 注意力计算方法
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None  # 转换注意力掩码的数据类型
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]  # 返回注意力输出

    def forward(self, x: torch.Tensor):  # 前向传播方法
        x = x + self.attention(self.ln_1(x))  # 残差连接
        x = x + self.mlp(self.ln_2(x))  # 残差连接
        return x  # 返回输出


class Transformer(nn.Module):  # 定义Transformer类，继承自nn.Module
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.width = width  # 宽度
        self.layers = layers  # 层数
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])  # 残差注意力块

    def forward(self, x: torch.Tensor):  # 前向传播方法
        return self.resblocks(x)  # 返回经过残差注意力块的输出


class VisionTransformer(nn.Module):  # 定义VisionTransformer类，继承自nn.Module
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.input_resolution = input_resolution  # 输入分辨率
        self.output_dim = output_dim  # 输出维度
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)  # 卷积层

        scale = width ** -0.5  # 缩放因子
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # 类别嵌入
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # 位置嵌入
        self.ln_pre = LayerNorm(width)  # 前置LayerNorm

        self.transformer = Transformer(width, layers, heads)  # Transformer层

        self.ln_post = LayerNorm(width)  # 后置LayerNorm
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))  # 输出投影

    def forward(self, x: torch.Tensor):  # 前向传播方法
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # 添加位置嵌入
        x = self.ln_pre(x)  # 经过前置LayerNorm

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)  # 经过Transformer
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])  # 经过后置LayerNorm

        if self.proj is not None:  # 如果存在投影层
            x = x @ self.proj  # 计算投影

        return x  # 返回输出


class CLIP(nn.Module):  # 定义CLIP类，继承自nn.Module
    def __init__(self,  # 初始化方法
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()  # 调用父类的初始化方法

        self.context_length = context_length  # 上下文长度

        if isinstance(vision_layers, (tuple, list)):  # 如果vision_layers是元组或列表
            vision_heads = vision_width * 32 // 64  # 计算头数
            self.visual = ModifiedResNet(  # 使用ModifiedResNet
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:  # 否则使用VisionTransformer
            vision_heads = vision_width // 64  # 计算头数
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(  # 定义Transformer
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # 构建注意力掩码
        )

        self.vocab_size = vocab_size  # 词汇表大小
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # 词嵌入层
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # 位置嵌入
        self.ln_final = LayerNorm(transformer_width)  # 最终LayerNorm

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # 文本投影
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 日志缩放因子

        self.initialize_parameters()  # 初始化参数

    def initialize_parameters(self):  # 初始化参数的方法
        nn.init.normal_(self.token_embedding.weight, std=0.02)  # 初始化词嵌入
        nn.init.normal_(self.positional_embedding, std=0.01)  # 初始化位置嵌入

        if isinstance(self.visual, ModifiedResNet):  # 如果视觉模型是ModifiedResNet
            if self.visual.attnpool is not None:  # 如果存在注意力池化层
                std = self.visual.attnpool.c_proj.in_features ** -0.5  # 计算标准差
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)  # 初始化Q投影
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)  # 初始化K投影
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)  # 初始化V投影
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)  # 初始化输出投影

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:  # 遍历ResNet的每一层
                for name, param in resnet_block.named_parameters():  # 遍历每个参数
                    if name.endswith("bn3.weight"):  # 如果是bn3的权重
                        nn.init.zeros_(param)  # 初始化为零

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)  # 计算投影标准差
        attn_std = self.transformer.width ** -0.5  # 计算注意力标准差
        fc_std = (2 * self.transformer.width) ** -0.5  # 计算全连接标准差
        for block in self.transformer.resblocks:  # 遍历Transformer的每个块
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)  # 初始化注意力输入权重
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)  # 初始化注意力输出权重
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)  # 初始化MLP的全连接权重
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)  # 初始化MLP的输出权重

        if self.text_projection is not None:  # 如果存在文本投影
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)  # 初始化文本投影

    def build_attention_mask(self):  # 构建注意力掩码的方法
        # lazily create causal attention mask, with full attention between the vision tokens
        # 懒惰地创建因果注意力掩码，视觉标记之间的全注意力
        # pytorch uses additive attention mask; fill with -inf
        # pytorch使用加法注意力掩码；填充为-inf
        mask = torch.empty(self.context_length, self.context_length)  # 创建掩码
        mask.fill_(float("-inf"))  # 填充为-inf
        mask.triu_(1)  # 零出下三角
        return mask  # 返回掩码

    @property
    def dtype(self):  # 数据类型属性
        return self.visual.conv1.weight.dtype  # 返回卷积层的权重数据类型

    def encode_image(self, image):  # 编码图像的方法
        return self.visual(image.type(self.dtype))  # 返回经过视觉模型的图像特征

    def encode_text(self, text):  # 编码文本的方法
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)  # 添加位置嵌入
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)  # 经过Transformer
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # 经过最终LayerNorm

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # 从eot嵌入中提取特征（eot_token是每个序列中的最大值）
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # 计算文本特征

        return x  # 返回文本特征

    def forward(self, image, text):  # 前向传播方法
        image_features = self.encode_image(image)  # 编码图像特征
        text_features = self.encode_text(text)  # 编码文本特征

        # normalized features
        # 归一化特征
        image_features = image_features / image_features.norm(dim=1, keepdim=True)  # 归一化图像特征
        text_features = text_features / text_features.norm(dim=1, keepdim=True)  # 归一化文本特征

        # cosine similarity as logits
        # 余弦相似度作为logits
        logit_scale = self.logit_scale.exp()  # 计算日志缩放因子
        logits_per_image = logit_scale * image_features @ text_features.t()  # 计算图像的logits
        logits_per_text = logits_per_image.t()  # 计算文本的logits

        # shape = [global_batch_size, global_batch_size]
        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text  # 返回图像和文本的logits


def convert_weights(model: nn.Module):  # 转换权重的方法
    """Convert applicable model parameters to fp16"""  # 将适用的模型参数转换为fp16

    def _convert_weights_to_fp16(l):  # 内部方法
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):  # 如果是卷积层或线性层
            l.weight.data = l.weight.data.half()  # 将权重转换为fp16
            if l.bias is not None:  # 如果存在偏置
                l.bias.data = l.bias.data.half()  # 将偏置转换为fp16

        if isinstance(l, nn.MultiheadAttention):  # 如果是多头注意力层
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:  # 遍历所有属性
                tensor = getattr(l, attr)  # 获取属性
                if tensor is not None:  # 如果属性存在
                    tensor.data = tensor.data.half()  # 将其转换为fp16

        for name in ["text_projection", "proj"]:  # 遍历文本投影和输出投影
            if hasattr(l, name):  # 如果模型具有该属性
                attr = getattr(l, name)  # 获取属性
                if attr is not None:  # 如果属性存在
                    attr.data = attr.data.half()  # 将其转换为fp16

    model.apply(_convert_weights_to_fp16)  # 应用权重转换


def build_model(state_dict: dict):  # 构建模型的方法
    vit = "visual.proj" in state_dict  # 检查是否为视觉模型

    if vit:  # 如果是视觉模型
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # 获取视觉宽度
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])  # 获取视觉层数
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # 获取视觉补丁大小
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # 计算网格大小
        image_resolution = vision_patch_size * grid_size  # 计算图像分辨率
    else:  # 否则
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]  # 计算视觉层数
        vision_layers = tuple(counts)  # 转换为元组
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]  # 获取视觉宽度
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)  # 计算输出宽度
        vision_patch_size = None  # 补丁大小为None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]  # 断言
        image_resolution = output_width * 32  # 计算图像分辨率

    embed_dim = state_dict["text_projection"].shape[1]  # 获取嵌入维度
    context_length = state_dict["positional_embedding"].shape[0]  # 获取上下文长度
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # 获取词汇表大小
    transformer_width = state_dict["ln_final.weight"].shape[0]  # 获取Transformer宽度
    transformer_heads = transformer_width // 64  # 计算头数
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))  # 计算Transformer层数

    model = CLIP(  # 创建CLIP模型
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:  # 遍历键
        if key in state_dict:  # 如果键在状态字典中
            del state_dict[key]  # 删除键

    convert_weights(model)  # 转换模型权重
    model.load_state_dict(state_dict)  # 加载状态字典
    return model.eval()  # 返回评估模式的模型