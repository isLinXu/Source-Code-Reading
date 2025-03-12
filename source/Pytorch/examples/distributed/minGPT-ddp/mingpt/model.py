"""
Full definition of a GPT Language Model, all of it in this single file.
Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""  # GPT语言模型的完整定义，所有内容都在这个单一文件中。改编自 https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
import math  # 导入数学模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.nn import functional as F  # 从PyTorch的神经网络模块导入功能性API

@dataclass
class GPTConfig:
    model_type: str = 'gpt2'  # 模型类型，默认为'gpt2'
    # model configurations
    n_layer: int = None  # 网络层数
    n_head: int = None  # 注意力头数
    n_embd: int =  None  # 嵌入维度
    # openai's values for gpt2
    vocab_size: int = 50257  # 词汇表大小，GPT-2的值
    block_size: int = 1024  # 块大小
    # dropout hyperparameters
    embd_pdrop: float = 0.1  # 嵌入层的dropout比例
    resid_pdrop: float = 0.1  # 残差连接的dropout比例
    attn_pdrop: float = 0.1  # 注意力层的dropout比例

@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4  # 学习率
    weight_decay: float = 0.1  # 权重衰减

class MultiheadAttentionLayer(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end.
    """  # 一个多头掩蔽自注意力层，最后带有一个投影

    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()  # 调用父类构造函数
        assert config.n_embd % config.n_head == 0  # 确保嵌入维度能被头数整除
        self.resid_drop = nn.Dropout(config.resid_pdrop)  # 残差连接的dropout层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)  # 投影层
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))  # 创建下三角矩阵作为掩码
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=config.n_embd,  # 嵌入维度
            num_heads=config.n_head,  # 注意力头数
            dropout=config.attn_pdrop,  # 注意力层的dropout比例
            batch_first=True,  # 批量维度在前
            device=device,  # 设备
            dtype=dtype  # 数据类型
        )

    def forward(self, x):
        _, seq_size, _ = x.size()  # 获取输入的序列大小
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]  # 计算注意力输出
        y = self.resid_drop(self.c_proj(y))  # 应用投影和dropout
        return y  # 返回输出

class Block(nn.Module):
    """ an unassuming Transformer block """  # 一个不起眼的Transformer块
    def __init__(self, config: GPTConfig):
        super().__init__()  # 调用父类构造函数
        self.ln1 = nn.LayerNorm(config.n_embd)  # 第一层归一化
        self.ln2 = nn.LayerNorm(config.n_embd)  # 第二层归一化
        self.attn = MultiheadAttentionLayer(config)  # 多头注意力层
        self.mlp = nn.Sequential(  # 多层感知机
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 线性层
            nn.GELU(),  # 激活函数
            nn.Linear(4 * config.n_embd, config.n_embd),  # 线性层
            nn.Dropout(config.resid_pdrop),  # dropout层
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差连接与注意力层
        x = x + self.mlp(self.ln2(x))  # 残差连接与多层感知机
        return x  # 返回输出

class EmbeddingStem(nn.Module):
    def __init__(self, config: GPTConfig, device="cpu", dtype=torch.float32):
        super().__init__()  # 调用父类构造函数
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=device, dtype=dtype)  # 词嵌入层
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype))  # 位置嵌入
        self.drop = nn.Dropout(config.embd_pdrop)  # dropout层
        self.block_size = config.block_size  # 块大小

    def reset_parameters(self):
        self.tok_emb.reset_parameters()  # 重置词嵌入层的参数

    def forward(self, idx):
        b, t = idx.size()  # 获取批量大小和序列长度
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"  # 确保序列长度不超过块大小

        token_embeddings = self.tok_emb(idx)  # 每个索引映射到一个（可学习的）嵌入向量
        position_embeddings = self.pos_emb[:, :t, :]  # 每个位置映射到一个（可学习的）位置向量
        return self.drop(token_embeddings + position_embeddings)  # 返回嵌入向量与位置向量的和，经过dropout

class GPT(nn.Module):
    """ GPT Language Model """  # GPT语言模型

    def __init__(self, config: GPTConfig):
        super().__init__()  # 调用父类构造函数
        self.block_size = config.block_size  # 块大小
        config = self._set_model_config(config)  # 设置模型配置

        # input embedding stem
        self.emb_stem = EmbeddingStem(config)  # 输入嵌入层
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])  # Transformer块的序列
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)  # 最后的归一化层
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 解码头

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)  # 初始化所有权重
        for pn, p in self.named_parameters():  # 遍历所有参数
            if pn.endswith('c_proj.weight'):  # 如果参数是投影层的权重
                p.data.normal_(mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))  # 应用特殊的初始化

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.blocks.parameters())  # 计算参数总数
        print("number of parameters: %.2fM" % (n_params/1e6,))  # 打印参数数量

    def _set_model_config(self, config):
        type_given = config.model_type is not None  # 检查模型类型是否给定
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])  # 检查所有参数是否给定
        # assert type_given ^ params_given # exactly one of these (XOR)
        if type_given and not params_given:  # 如果给定了模型类型但没有给定参数
            # translate from model_type to detailed configuration
            config.__dict__.update({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])  # 根据模型类型更新配置
        return config  # 返回配置
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):  # 如果模块是线性层或嵌入层
            module.weight.data.normal_(mean=0.0, std=0.02)  # 初始化权重
            if isinstance(module, nn.Linear) and module.bias is not None:  # 如果是线性层且有偏置
                module.bias.data.zero_()  # 偏置初始化为零
        elif isinstance(module, nn.LayerNorm):  # 如果模块是层归一化
            module.bias.data.zero_()  # 偏置初始化为零
            module.weight.data.fill_(1.0)  # 权重初始化为1.0

    def forward(self, idx, targets=None):
        x = self.emb_stem(idx)  # 获取嵌入
        x = self.blocks(x)  # 通过Transformer块
        x = self.ln_f(x)  # 最后归一化
        logits = self.head(x)  # 获取logits

        # if we are given some desired targets also calculate the loss
        loss = None  # 初始化损失
        if targets is not None:  # 如果给定了目标
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)  # 计算交叉熵损失

        return logits, loss  # 返回logits和损失

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """  # 接受一个条件序列的索引idx（形状为(b,t)的LongTensor），并完成序列max_new_tokens次，每次将预测反馈到模型中。
        for _ in range(max_new_tokens):  # 循环生成新token
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]  # 如果序列过长，则裁剪到块大小
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)  # 前向传播以获取logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # 获取最后一步的logits并按温度缩放
            # optionally crop the logits to only the top k options
            if top_k is not None:  # 如果给定了top_k
                v, _ = torch.topk(logits, top_k)  # 获取logits的前k个值
                logits[logits < v[:, [-1]]] = -float('Inf')  # 将低于第k个的logits设为负无穷
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)  # 将logits转换为概率
            # either sample from the distribution or take the most likely element
            if do_sample:  # 如果需要采样
                idx_next = torch.multinomial(probs, num_samples=1)  # 从概率分布中采样
            else:  # 否则
                _, idx_next = torch.topk(probs, k=1, dim=-1)  # 取最可能的元素
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)  # 将采样的索引附加到序列中

        return idx  # 返回生成的序列


def create_optimizer(model: torch.nn.Module, opt_config: OptimizerConfig):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """  # 这个长函数实际上做的事情很简单，并且非常谨慎：我们将模型的所有参数分为两个桶：那些会经历权重衰减以进行正则化的参数和那些不会的（偏置、层归一化和嵌入权重）。然后返回PyTorch优化器对象。

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()  # 将经历权重衰减的参数集合
    no_decay = set()  # 不经历权重衰减的参数集合
    whitelist_weight_modules = (torch.nn.Linear, )  # 允许权重衰减的模块
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)  # 不允许权重衰减的模块
    for mn, m in model.named_modules():  # 遍历模型的所有模块
        for pn, p in m.named_parameters():  # 遍历模块的所有参数
            fpn = '%s.%s' % (mn, pn) if mn else pn  # 完整参数名称
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):  # 如果参数是偏置
                # all biases will not be decayed
                no_decay.add(fpn)  # 偏置不经历权重衰减
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):  # 如果参数是权重并且模块在白名单中
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)  # 权重经历权重衰减
            elif pn.endswith('in_proj_weight'):  # 多头注意力投影层的权重
                decay.add(fpn)  # 权重经历权重衰减
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):  # 如果参数是权重并且模块在黑名单中
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)  # 权重不经历权重衰减
            elif pn.endswith('pos_emb'):  # 位置嵌入权重
                no_decay.add(fpn)  # 位置嵌入不经历权重衰减

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}  # 获取所有参数的字典
    inter_params = decay & no_decay  # 权重衰减和不衰减的交集
    union_params = decay | no_decay  # 权重衰减和不衰减的并集
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )  # 确保没有参数同时在两个集合中
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )  # 确保所有参数都被分类

    # create the pytorch optimizer object
    optim_groups = [  # 优化器参数组
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": opt_config.weight_decay},  # 权重衰减组
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},  # 不衰减组
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=opt_config.learning_rate, betas=(0.9, 0.95))  # 创建AdamW优化器
    return optimizer  # 返回优化器