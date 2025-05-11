import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))  # 权重参数，初始化为1，形状为隐藏层维度
        self.eps = cfg.lm_rms_eps  # 极小值，防止除零

    def forward(self, x):
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepkeepdim=True) + self.eps) # inverse of RMS # 计算均方根的倒数，防止数值不稳定
        x = x * irms * self.weight  # 归一化并缩放

        return x

# Multiple derivates of Rotary Embeddings by now, this is a basic one with linear scaling to context length
# e.g. https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190
class RotaryEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"
        # 检查隐藏层维度能否被头数整除
        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads # dim of each head # 每个注意力头的维度
        self.base = cfg.lm_re_base  # 旋转嵌入的基数
        self.max_seq_len = cfg.lm_max_position_embeddings  # 最大序列长度
        
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        # 计算每个维度的倒数频率
        self.register_buffer("inv_freq", inv_freq)  # 注册为buffer，参与模型保存但不作为参数
        self.original_max_seq_len = cfg.lm_max_position_embeddings  # 原始最大序列长度
        self.attention_scaling = cfg.lm_attn_scaling  # 注意力缩放因子

    @torch.no_grad()
    def forward(self, position_ids):
        batch_size, seq_len = position_ids.shape  # 获取批次和序列长度
        
        # Dynamic scaling for longer sequences
        max_seq = position_ids.max() + 1  # 当前最大位置
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len  # 计算缩放比例
            inv_freq = self.inv_freq / scale  # 动态缩放频率
        else:
            inv_freq = self.inv_freq  # 不需要缩放
            
        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()  # 展平成一维，便于批量处理
        
        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        # 计算每个位置和频率的乘积
        
        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)  # 恢复批次和序列长度
        
        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1)  # 拼接，形成交错模式
        
        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling  # 计算cos并缩放
        sin = torch.sin(emb) * self.attention_scaling  # 计算sin并缩放
        
        return cos, sin  # 返回cos和sin嵌入

# Rotates half the hidden dims of the input by swapping and negating dimensions.
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)  # 沿最后一维分成两半
    return torch.cat((-x2, x1), dim=-1)  # 交换并对前半部分取负，拼接

# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(q, k, cos, sin, unsqeeze_dim=1):
    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqeeze_dim)  # [batch_size, 1, seq_len, head_dim] # 增加一个维度以便广播
    sin = sin.unsqueeze(unsqeeze_dim)  # [batch_size, 1, seq_len, head_dim] # 增加一个维度以便广播
    
    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 对q应用旋转位置嵌入
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 对k应用旋转位置嵌入
    
    return q_embed, k_embed  # 返回嵌入后的q和k

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.lm_n_heads  # 总头数
        self.n_kv_heads = cfg.lm_n_kv_heads  # k/v头数
        self.embd_dim = cfg.lm_hidden_dim  # 嵌入维度
        self.dropout = cfg.lm_dropout  # dropout概率

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        # 检查头数能否被k/v头数整除
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"
        # 检查嵌入维度能否被头数整除

        self.n_kv_groups = self.n_heads // self.n_kv_heads  # 每组k/v对应多少q头
        self.head_dim = self.embd_dim // self.n_heads  # 每个头的维度

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)  # q投影
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)  # k投影
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)  # v投影
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)  # 输出投影

        self.attn_dropout = nn.Dropout(self.dropout)  # 注意力dropout
        self.resid_dropout = nn.Dropout(self.dropout)  # 残差dropout

        # Use flash attention if available
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # 检查是否支持Flash Attention
        if not self.flash:
            print("Warning: Flash attention not available, using standard attention in LM.")
            # 如果不支持，打印警告

    def forward(self, x, cos, sin, attention_mask=None):
        B, T, C = x.size()  # 获取批次、序列长度和通道数

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        # 计算q并调整形状
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        # 计算k并调整形状
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        # 计算v并调整形状
        
        # Use precomputed positional embeddings
        q, k = apply_rotary_pos_embd(q, k, cos, sin)
        # 应用旋转位置嵌入

        k = k.repeat_interleave(self.n_kv_groups, dim=1)
        # 重复k以匹配q头数
        v = v.repeat_interleave(self.n_kv_groups, dim=1)
        # 重复v以匹配q头数

        # Process attention mask if provided
        if attention_mask is not None:
            # Create a 4D attention mask [batch_size, 1, 1, seq_length], In this format, 1 = attend, 0 = mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T] # 扩展mask维度
            padding_mask = (attention_mask == 0).transpose(-1, -2) # Use this for the manual path # 生成padding mask，用于后续输出置零
            # Convert to attention mask where 0 keeps values and -inf masks
            attention_mask = (1.0 - attention_mask) * torch.finfo(q.dtype).min
            # 将mask转换为0和-inf，0表示保留，-inf表示掩码

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True # LM attention is causal (masked) # 使用Flash Attention，自动处理掩码和因果性
            )
        else:
            attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            # 计算注意力分数并缩放
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            # 构造下三角因果掩码
            attn = attn.masked_fill(causal_mask == 0, float('-inf'))
            # 非因果位置填充为-inf
            if attention_mask is not None:
                attn = attn + attention_mask 
                # 加入外部掩码

            attn = F.softmax(attn, dim=-1)
            # softmax归一化
            attn = self.attn_dropout(attn)
            # dropout
            y = attn @ v
            # 计算加权和
            
            if attention_mask is not None:
                y = y.masked_fill(padding_mask, 0.0) # Zero out the padded positions in the output # 对填充位置输出置零

        y = y.transpose(1, 2).contiguous().view(B, T, C)  
        # 恢复输出形状
        y = self.out_proj(y)
        # 输出投影
        y = self.resid_dropout(y)
        # 残差dropout

        return y

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim  # 输入维度
        self.inter_dim = cfg.lm_inter_dim  # 中间维度

        self.activation_fn = F.silu  # 激活函数
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)  # 门控投影
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)  # 上升投影
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)  # 降维投影

    def forward(self, x):
        gate = self.activation_fn(self.gate_proj(x))  # 门控分支并激活
        x = self.up_proj(x)  # 上升分支
        x = self.down_proj(gate * x)  # 门控与上升分支相乘后降维

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)  # MLP模块
        self.attn = LanguageModelGroupedQueryAttention(cfg)  # 注意力模块
        self.norm1 = RMSNorm(cfg) # Input Norm # 输入归一化
        self.norm2 = RMSNorm(cfg) # Post Attention Norm # 注意力后归一化
    
    def forward(self, x, cos, sin, attention_mask=None):
        res = x  # 残差连接
        x = self.norm1(x)  # 输入归一化
        x = self.attn(x, cos, sin, attention_mask)  # 注意力
        x = res + x  # 残差加和

        res = x  # 残差连接
        x = self.norm2(x)  # 注意力后归一化
        x = self.mlp(x)  # MLP
        x = res + x  # 残差加和

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # 保存配置对象
        self.lm_use_tokens = cfg.lm_use_tokens  # 是否使用token作为输入/输出
        self.lm_tie_weights = cfg.lm_tie_weights  # 是否绑定token embedding和head权重

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)  # token嵌入层
        self.rotary_embd = RotaryEmbedding(cfg)  # 旋转位置嵌入模块
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
        ])  # 语言模型块列表
        self.norm = RMSNorm(cfg) # Final Norm # 最终归一化层
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)  # 语言模型头（输出层）
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight  # 如果绑定权重，共享token embedding和head的权重

        self.apply(self._init_weights)  # 应用权重初始化函数

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化线性层权重为正态分布
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 初始化线性层偏置为零
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化嵌入层权重为正态分布
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)  # 初始化RMSNorm权重为1.0

    def forward(self, x, attention_mask=None):
        if self.lm_use_tokens:
            x = self.token_embedding(x) # Only embed the inputs when using tokens # 如果使用token，对输入进行嵌入
        
        B , T, _ = x.size()  # 获取批次大小、序列长度和特征维度
        
        # Note: You could also cache these input embeddings if you want to avoid recomputing them
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1) # Create position ids [0, 1, 2, ..., seq_len-1] # 创建位置ID序列
        cos, sin = self.rotary_embd(position_ids) # Get rotary position embeddings # 获取旋转位置嵌入的cos和sin值

        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)  # 依次通过每个语言模型块
        x = self.norm(x)  # 最终归一化

        if self.lm_use_tokens:
            x = self.head(x) # Compute logits if we are using tokens, otherwise stay in the embedding space # 如果使用token，通过head计算logits，否则保持在嵌入空间

        return x  # 返回输出

    @torch.no_grad()  # 在生成过程中不计算梯度
    def generate(self, inputs, max_new_tokens=20):
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # 如果输入是一维的，增加批次维度
            
        generated = inputs.clone()  # 复制输入作为生成序列的起始
        
        for _ in range(max_new_tokens):
            # Forward pass through the model
            outputs = self.forward(generated)  # 前向传播获取模型输出
            last_output = outputs[:, -1, :]  # 获取最后一个时间步的输出

            if self.lm_use_tokens:
                # Now the model outputs logits
                next_token = torch.argmax(last_output, dim=-1, keepdim=True)  # 如果使用token，取logits最大值作为下一个token
                generated = torch.cat((generated, next_token), dim=-1)  # 将预测的下一个token拼接到生成序列
            else:
                # Now the model outputs embeddings
                next_token_embedding = last_output.unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim] # 如果输出是嵌入，获取最后一个时间步的嵌入
                generated = torch.cat((generated, next_token_embedding), dim=1)  # 将预测的下一个嵌入拼接到生成序列
            
            #Note: You could enable the generation to break earlier than max_new_tokens when it detects a eos token, but this does not work in batched generation (output tensors need to have the same size)
            # 注意：可以在检测到EOS token时提前停止生成，但这在批量生成中不起作用（输出张量需要大小一致）
    
        return generated  # 返回生成的序列

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    # 从预训练的HuggingFace模型加载权重（避免从头训练语言骨干）
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig  # 导入AutoConfig用于加载HF配置
        from huggingface_hub import hf_hub_download  # 导入hf_hub_download用于下载文件
        import safetensors  # 导入safetensors用于加载权重
        import torch.nn.init as init  # 导入初始化函数
                
        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)  # 从HuggingFace加载模型配置
        
        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size  # 存储原始HF词汇表大小
        print(f"Original vocabulary size from pretrained model: {original_vocab_size}")  # 打印原始词汇表大小
        
        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size  # 从HF配置设置隐藏层维度
        cfg.lm_inter_dim = hf_config.intermediate_size  # 从HF配置设置中间层维度
        cfg.lm_rms_eps = hf_config.rms_norm_eps  # 从HF配置设置RMSNorm的eps
        cfg.lm_re_base = hf_config.rope_theta  # 从HF配置设置RoPE的base
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings  # 从HF配置设置最大位置嵌入数
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        # 我们保留cfg中的词汇表大小，但检查它是否大于原始大小
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
                # 如果cfg中的词汇表大小小于原始大小，抛出错误
            print(f"Using extended vocabulary size: {cfg.lm_vocab_size}")  # 打印使用的扩展词汇表大小
        else:
            # If not specified, use the original
            # 如果未指定，使用原始大小
            cfg.lm_vocab_size = original_vocab_size  # 使用原始词汇表大小
            print(f"Using original vocabulary size: {cfg.lm_vocab_size}")  # 打印使用的原始词汇表大小
        
        cfg.lm_n_heads = hf_config.num_attention_heads  # 从HF配置设置注意力头数
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads  # 从HF配置设置k/v头数
        cfg.lm_dropout = hf_config.attention_dropout  # 从HF配置设置dropout概率
        cfg.lm_n_blocks = hf_config.num_hidden_layers  # 从HF配置设置语言模型块数
        
        # Create our model with potentially larger vocabulary
        model = cls(cfg)  # 创建我们的模型实例，可能使用更大的词汇表
        safetensors_file = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")  # 下载safetensors格式的权重文件
        
        sd = model.state_dict()  # 获取模型的state_dict
        
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',  # HF权重名到我们模型权重名的映射
            'model.norm.weight': 'norm.weight'  # HF权重名到我们模型权重名的映射
        }
        
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'  # HF层的前缀
            block_prefix = f'blocks.{i}.'  # 我们模型块的前缀
            
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",  # 注意力q投影权重映射
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",  # 注意力k投影权重映射
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",  # 注意力v投影权重映射
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",  # 注意力输出投影权重映射
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",  # MLP门控投影权重映射
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",  # MLP上升投影权重映射
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",  # MLP降维投影权重映射
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",  # 输入归一化权重映射
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"  # 注意力后归一化权重映射
            })
        
        # Special handling for token embeddings with extended vocabulary
        # 特殊处理扩展词汇表的token嵌入
        has_extended_embeddings = False  # 标记是否扩展了嵌入
        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)  # 获取HF模型的权重张量
                    
                    # Special handling for token embeddings if vocab sizes differ
                    # 如果词汇表大小不同，特殊处理token嵌入
                    if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                        has_extended_embeddings = True  # 标记已扩展嵌入
                        print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")  # 打印扩展信息
                        
                        # Copy existing embeddings to the beginning of our larger embedding matrix
                        # 将现有嵌入复制到我们更大嵌入矩阵的开头
                        sd[our_key][:tensor.shape[0]].copy_(tensor)
                        
                        # Initialize the new embeddings using the same approach as the original model
                        # 使用与原始模型相同的方法初始化新的嵌入
                        std = 0.02  # Common value, but you might want to adjust based on model # 常用标准差，可根据模型调整
                        init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)  # 初始化新增部分的嵌入
                        
                        print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")  # 打印初始化新增嵌入的数量
                        if 'head.weight' in sd: # Ensure head.weight exists before copying
                             sd['head.weight'].copy_(sd[our_key])  # Update the head weights as well # 如果head权重存在，更新head权重以匹配扩展后的嵌入
                    elif tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)  # 如果形状匹配，直接复制权重
                    else:
                        print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")  # 打印形状不匹配警告
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")  # 警告：HF权重文件找不到对应的key
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")  # 警告：模型state dict找不到对应的key
        
        # Load the state dict
        model.load_state_dict(sd)  # 加载state_dict到模型
        
        # Handle output projection / language modeling head
        # 处理输出投影/语言模型头
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            # 如果我们有单独的输出投影层并扩展了词汇表，应类似处理输入嵌入
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                if 'lm_head.weight' in f.keys():
                    lm_head = f.get_tensor('lm_head.weight')  # 获取HF模型的lm_head权重
                    if lm_head.shape[0] != sd['head.weight'].shape[0]:
                        print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")  # 打印扩展信息
                        # Copy existing weights
                        # 复制现有权重
                        sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                        # Initialize new weights
                        # 初始化新权重
                        std = 0.02
                        init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)  # 初始化新增部分的head权重
                        # Load updated weights
                        # 加载更新后的权重
                        model.load_state_dict(sd)
        
        # Handle weight tying (if needed)
        # 处理权重绑定（如果需要）
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight  # 绑定token embedding和head的权重
            print("Tied token embedding and LM head weights")  # 打印绑定信息
        
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")  # 打印加载成功信息和模型参数数量
        return model  # 返回加载权重的模型实例