import math  # 导入数学库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数模块

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数

        self.img_size = cfg.vit_img_size  # 图像尺寸
        self.patch_size = cfg.vit_patch_size  # 图像块尺寸
        self.num_patches = (self.img_size // self.patch_size) ** 2  # 计算图像块数量
        self.cls_flag = cfg.vit_cls_flag  # 是否使用CLS token的标志
        self.embd_dim = cfg.vit_hidden_dim  # 嵌入维度（隐藏层维度）

        # Conv layer to extract the patches
        # 用于提取图像块的卷积层
        self.conv = nn.Conv2d(
            in_channels=3,  # 输入通道数（彩色图像为3）
            out_channels=self.embd_dim,  # 输出通道数（等于嵌入维度）
            kernel_size=self.patch_size,  # 卷积核大小等于图像块尺寸
            stride=self.patch_size,  # 步长等于图像块尺寸，实现不重叠的图像块提取
            padding="valid",  # 不使用填充
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))  # 如果使用CLS token，初始化一个可学习的CLS token
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))  # 如果使用CLS token，位置嵌入包含CLS token的位置
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))  # 如果不使用CLS token，位置嵌入只包含图像块的位置


    def forward(self, x):
        x = self.conv(x)  # extract patches # 通过卷积层提取图像块
        x = x.flatten(2)  # flatten the patches into a single dimension # 将图像块展平到单个维度 [batch_size, hidden_dim, num_patches_h, num_patches_w] -> [batch_size, hidden_dim, num_patches]
        x = x.transpose(1, 2)  # transpose to (batch_size, num_patches, hidden_dim) # 调换维度，使其形状变为 [batch_size, num_patches, hidden_dim]

        # Add CLS token (according to original ViT Paper) and position embeddings
        # 添加CLS token（根据原始ViT论文）和位置嵌入
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 扩展CLS token以匹配批次大小
            x = torch.cat((cls_token, x), dim=1)  # 将CLS token拼接到图像块嵌入的前面
        x = x + self.position_embedding  # 添加位置嵌入
        return x  # 返回包含位置信息的图像块嵌入（可能包含CLS token）

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L381
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数

        self.n_heads = cfg.vit_n_heads  # 注意力头数
        self.embd_dim = cfg.vit_hidden_dim  # 嵌入维度
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"  # 断言嵌入维度必须能被头数整除
        self.head_dim = self.embd_dim // self.n_heads  # 每个注意力头的维度
        self.dropout = cfg.vit_dropout  # dropout概率

        # Combined projections for all heads
        # 所有头的组合投影层
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)  # Q, K, V的线性投影层
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)  # 输出投影层

        # Dropout layers
        # Dropout层
        self.attn_dropout = nn.Dropout(self.dropout)  # 注意力权重上的dropout
        self.resid_dropout = nn.Dropout(self.dropout)  # 残差连接后的dropout

        # Use flash attention if available
        # 如果可用，使用Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # 检查是否支持Flash Attention
        if not self.flash:
            print("Warning: Flash attention not available. Using standard attention in ViT.")  # 如果不支持，打印警告并使用标准注意力

    def forward(self, x):
        B, T, C = x.size()  # 获取批次大小、序列长度和嵌入维度

        qkv = self.qkv_proj(x)  # 通过线性层计算Q, K, V的组合投影
        q, k, v = qkv.split(C, dim=2)  # 将组合投影沿最后一维分割成Q, K, V
        # Reshape  [B, T, C] -> [B, T, n_heads, head_dim] and transpose -> [B, n_heads, T, head_dim]
        # 重塑 [B, T, C] -> [B, T, n_heads, head_dim] 并转置 -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim) # 重塑并转置Q
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim) # 重塑并转置K
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim) # 重塑并转置V

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,  # 不使用额外的注意力掩码
                dropout_p=self.dropout if self.training else 0.0,  # 根据训练状态应用dropout
                is_causal=False # ViT attention is bidirectional # ViT的注意力是双向的，非因果
            )  # 使用Flash Attention计算注意力输出
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 计算Q和K的转置的点积，并进行缩放
            attn = F.softmax(attn, dim=-1)  # 应用softmax获取注意力权重
            attn = self.attn_dropout(attn)  # 在注意力权重上应用dropout
            y = attn @ v  # (B, n_heads, T, head_dim) x (B, n_heads, head_dim, T) -> (B, n_heads, T, head_dim) # 将注意力权重与V相乘得到输出
        
        # Transpose back from [B, n_heads, T, head_dim] to [B, T, n_heads * head_dim] and combine all heads to [B, T, C]
        # 从 [B, n_heads, T, head_dim] 转置回 [B, T, n_heads * head_dim] 并合并所有头到 [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 转置并重塑回原始形状
        y = self.out_proj(y)  # 通过输出投影层
        y = self.resid_dropout(y)  # 应用残差dropout

        return y  # 返回注意力层的输出

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L453
class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数
        self.activation_fn = nn.GELU(approximate='tanh')  # 激活函数，使用GELU
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)  # 第一个全连接层，从隐藏层维度到中间维度
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)  # 第二个全连接层，从中间维度到隐藏层维度
        self.dropout = nn.Dropout(cfg.vit_dropout)  # dropout层

    def forward(self, x):
        x = self.fc1(x)  # 通过第一个全连接层
        x = self.activation_fn(x)  # 应用激活函数
        x = self.fc2(x)  # 通过第二个全连接层
        x = self.dropout(x)  # 应用dropout
        return x  # 返回MLP的输出

# https://github.com/karpathy/nanoGPT/blob/master/model.py#L94    
class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)  # 第一个Layer Normalization层
        self.attn = ViTMultiHeadAttention(cfg)  # 多头注意力模块
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)  # 第二个Layer Normalization层
        self.mlp = ViTMLP(cfg)  # MLP模块
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 应用第一个LayerNorm，然后通过注意力模块，并添加残差连接
        x = x + self.mlp(self.ln2(x))  # 应用第二个LayerNorm，然后通过MLP模块，并添加残差连接
        return x  # 返回ViT块的输出
    

class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数
        self.cfg = cfg  # 保存配置对象
        self.patch_embedding = ViTPatchEmbeddings(cfg)  # 初始化图像块嵌入模块
        self.cls_flag = cfg.vit_cls_flag  # 是否使用CLS token的标志
        self.dropout = nn.Dropout(cfg.vit_dropout)  # dropout层
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])  # 初始化ViT块列表
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)  # 最终的Layer Normalization层

        self.apply(self._init_weights)  # 应用权重初始化函数

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化线性层权重为正态分布
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 初始化线性层偏置为零
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  # 初始化LayerNorm偏置为零
            module.weight.data.fill_(1.0)  # 初始化LayerNorm权重为1.0
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化卷积层权重为正态分布
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 初始化卷积层偏置为零

    def forward(self, x):
        x = self.patch_embedding(x)  # 通过图像块嵌入模块处理输入图像
        x = self.dropout(x)  # 应用dropout
        for block in self.blocks:
            x = block(x)  # 依次通过每个ViT块

        if self.cls_flag:
            x = self.layer_norm(x[:, 0])  # 如果使用CLS token，只对CLS token的输出进行LayerNorm
        else:
            x = self.layer_norm(x)  # 如果不使用CLS token，对整个序列的输出进行LayerNorm
            #x = x.mean(dim=1) # 原始代码中注释掉的行，表示可以对序列进行平均池化
        
        return x  # 返回ViT的输出特征

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Vision Backbone from scratch)
    # 从预训练的HuggingFace模型加载权重（我们不想从头开始训练视觉骨干网络）
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import SiglipVisionConfig  # 导入HuggingFace的SiglipVisionConfig
        from huggingface_hub import hf_hub_download  # 导入HuggingFace Hub的下载函数
        import safetensors  # 导入safetensors库

        hf_config = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)  # 从HuggingFace加载预训练模型的配置
        cfg.vit_dropout=hf_config.attention_dropout  # 更新配置中的dropout概率
        cfg.vit_hidden_dim=hf_config.hidden_size  # 更新配置中的隐藏层维度
        cfg.vit_img_size=hf_config.image_size  # 更新配置中的图像尺寸
        cfg.vit_inter_dim=hf_config.intermediate_size  # 更新配置中的中间层维度
        cfg.vit_ln_eps=hf_config.layer_norm_eps  # 更新配置中的LayerNorm epsilon
        cfg.vit_n_heads=hf_config.num_attention_heads  # 更新配置中的注意力头数
        cfg.vit_n_blocks=hf_config.num_hidden_layers  # 更新配置中的块数量
        cfg.vit_patch_size=hf_config.patch_size  # 更新配置中的图像块尺寸
        model = cls(cfg)  # 使用更新后的配置创建模型实例
        safetensors_file = hf_hub_download(repo_id=cfg.vit_model_type, filename="model.safetensors")  # 从HuggingFace Hub下载safetensors权重文件

        sd = model.state_dict()  # 获取当前模型的state_dict
        

        mapping = {
            'vision_model.embeddings.patch_embedding.weight': 'patch_embedding.conv.weight',  # 图像块嵌入卷积层权重映射
            'vision_model.embeddings.patch_embedding.bias': 'patch_embedding.conv.bias',  # 图像块嵌入卷积层偏置映射
            'vision_model.embeddings.position_embedding.weight': 'patch_embedding.position_embedding',  # 位置嵌入权重映射
            'vision_model.post_layernorm.weight': 'layer_norm.weight',  # 最终LayerNorm权重映射
            'vision_model.post_layernorm.bias': 'layer_norm.bias',  # 最终LayerNorm偏置映射
        }
        
        for i in range(cfg.vit_n_blocks):  # 遍历每个ViT块
            # Layer norms
            # Layer Norms映射
            mapping[f'vision_model.encoder.layers.{i}.layer_norm1.weight'] = f'blocks.{i}.ln1.weight'  # 第一个LayerNorm权重映射
            mapping[f'vision_model.encoder.layers.{i}.layer_norm1.bias'] = f'blocks.{i}.ln1.bias'  # 第一个LayerNorm偏置映射
            mapping[f'vision_model.encoder.layers.{i}.layer_norm2.weight'] = f'blocks.{i}.ln2.weight'  # 第二个LayerNorm权重映射
            mapping[f'vision_model.encoder.layers.{i}.layer_norm2.bias'] = f'blocks.{i}.ln2.bias'  # 第二个LayerNorm偏置映射
            
            # MLP
            # MLP映射
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc1.weight'] = f'blocks.{i}.mlp.fc1.weight'  # MLP第一个全连接层权重映射
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc1.bias'] = f'blocks.{i}.mlp.fc1.bias'  # MLP第一个全连接层偏置映射
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc2.weight'] = f'blocks.{i}.mlp.fc2.weight'  # MLP第二个全连接层权重映射
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc2.bias'] = f'blocks.{i}.mlp.fc2.bias'  # MLP第二个全连接层偏置映射
            
            # Output projection
            # 输出投影映射
            mapping[f'vision_model.encoder.layers.{i}.self_attn.out_proj.weight'] = f'blocks.{i}.attn.out_proj.weight'  # 注意力输出投影权重映射
            mapping[f'vision_model.encoder.layers.{i}.self_attn.out_proj.bias'] = f'blocks.{i}.attn.out_proj.bias'  # 注意力输出投影偏置映射
        
        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:  # 打开safetensors文件
            for hf_key, our_key in mapping.items():  # 遍历映射关系
                if hf_key in f.keys() and our_key in sd:  # 如果HF key和我们的 key都存在
                    tensor = f.get_tensor(hf_key)  # 获取HF模型的权重张量
                    if tensor.shape == sd[our_key].shape:  # 如果形状匹配
                        sd[our_key].copy_(tensor)  # 直接复制权重
                    else:
                        if 'position_embedding' in hf_key:  # 特殊处理位置嵌入，可能需要增加批次维度
                            sd[our_key].copy_(tensor.unsqueeze(0))  # 复制并增加批次维度
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")  # 打印形状不匹配警告
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")  # 警告：HF权重文件找不到对应的key
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")  # 警告：模型state dict找不到对应的key
            
            # Manually handle QKV concatenation since our implementation combines Q, K, V into one
            # 手动处理QKV拼接，因为我们的实现将Q, K, V合并为一个
            for i in range(model.cfg.vit_n_blocks):  # 遍历每个ViT块
                q_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.q_proj.weight')  # 获取HF模型的Q权重
                k_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.k_proj.weight')  # 获取HF模型的K权重
                v_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.v_proj.weight')  # 获取HF模型的V权重
                
                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)  # 沿维度0拼接Q, K, V权重
                sd[f'blocks.{i}.attn.qkv_proj.weight'].copy_(qkv_weight)  # 将拼接后的权重复制到我们的模型state_dict中
                
                q_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.q_proj.bias')  # 获取HF模型的Q偏置
                k_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.k_proj.bias')  # 获取HF模型的K偏置
                v_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.v_proj.bias')  # 获取HF模型的V偏置
                
                qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)  # 沿维度0拼接Q, K, V偏置
                sd[f'blocks.{i}.attn.qkv_proj.bias'].copy_(qkv_bias)  # 将拼接后的偏置复制到我们的模型state_dict中
        
        model.load_state_dict(sd)  # 将加载并处理好的state_dict加载到模型中
        print(f"Successfully loaded {cfg.vit_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")  # 打印加载成功信息和模型参数数量
        return model  # 返回加载了预训练权重的模型实例
