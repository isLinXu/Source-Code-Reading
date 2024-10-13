import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from omni_speech.constants import IGNORE_INDEX


def lengths_to_padding_mask(lens):
    """
    根据输入的长度张量生成一个填充掩码张量。

    参数:
    lens (torch.Tensor): 一个包含每个序列长度的1D张量，形状为 (batch_size,)。

    返回:
    torch.Tensor: 一个填充掩码张量，形状为 (batch_size, max_lens)，其中max_lens是输入序列的最大长度。
                  掩码中的True表示对应位置是填充（padding），False表示是有效数据。

    示例:
    >>> lens = torch.tensor([3, 5])
    >>> mask = lengths_to_padding_mask(lens)
    >>> print(mask)
    tensor([[False, False,  True],
            [False, False, False, False,  True]])
    """
    # 获取批量大小和最大序列长度
    bsz, max_lens = lens.size(0), torch.max(lens).item()

    # 创建一个从0到max_lens-1的张量，并将其扩展到(bsz, max_lens)的形状
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)

    # 将长度张量扩展到(bsz, 1)的形状，并将其与掩码张量进行比较
    # 结果是一个布尔张量，表示每个位置是否是填充
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def _uniform_assignment(src_lens, tgt_lens):
    """
    根据源序列长度和目标序列长度，计算一个均匀分配的索引矩阵。

    参数:
    src_lens (torch.Tensor): 源序列的长度列表。
    tgt_lens (torch.Tensor): 目标序列的长度列表。

    返回:
    torch.Tensor: 一个索引矩阵，用于将源序列均匀分配到目标序列中。
    """
    # 创建一个从0到最大目标序列长度的张量，并将其扩展为与目标序列长度相同的行数
    tgt_indices = torch.arange(torch.max(tgt_lens)).expand(len(tgt_lens), -1).to(tgt_lens.device)

    # 计算目标序列长度与源序列长度的比例
    ratio = tgt_lens / src_lens

    # 使用比例来计算目标序列中的索引位置
    index_t = (tgt_indices / ratio.view(-1, 1)).long()
    return index_t


class SpeechGeneratorCTC(nn.Module):
    # 初始化SpeechGeneratorCTC类
    def __init__(self, config):
        super().__init__()
        # 解析配置参数
        n_layers, n_dims, n_heads, n_inter_dims = list(map(int, config.ctc_decoder_config[1:-1].split(",")))
        # 创建一个新的配置对象，用于LlamaDecoderLayer
        _config = copy.deepcopy(config)
        _config.hidden_size = n_dims
        _config.num_hidden_layers = n_layers
        _config.num_attention_heads = n_heads
        _config.num_key_value_heads = n_heads
        _config.intermediate_size = n_inter_dims
        _config._attn_implementation = "flash_attention_2"
        # 初始化类的属性
        self.upsample_factor = config.ctc_upsample_factor
        self.input_proj = nn.Linear(config.hidden_size, n_dims)
        # 创建解码层列表
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(_config, layer_idx) for layer_idx in range(n_layers)]
        )
        self.unit_vocab_size = config.unit_vocab_size
        self.output_proj = nn.Linear(n_dims, config.unit_vocab_size + 1)

    # 上采样函数，用于增加序列的长度
    def upsample(self, reps, tgt_units=None):
        """
        Upsample the input sequences to a longer length.

        Args:
            reps (List[Tensor]): List of input sequences.
            tgt_units (Tensor, optional): Target units for masking. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Upsampled sequences, padding mask, and position ids.
        """
        # 计算输入序列的长度，并将其转移到相同的设备上
        src_lens = torch.LongTensor([len(rep) for rep in reps]).to(reps[0].device)
        # 计算上采样后的长度
        up_lens = src_lens * self.upsample_factor
        # 如果提供了目标单位，则使用目标单位的长度
        if tgt_units is not None:
            tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
            up_lens = torch.max(up_lens, tgt_lens)
        # 对输入序列进行填充，使其具有相同的长度
        reps = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True)
        # 创建填充掩码，并将其转换为布尔张量
        padding_mask = lengths_to_padding_mask(up_lens)
        # 创建映射输入，即将输入序列均匀分配到上采样后的长度
        mapped_inputs = _uniform_assignment(src_lens, up_lens).masked_fill(
            padding_mask, 0
        )
        # 根据映射输入复制序列，并将其填充到填充掩码中
        copied_reps = torch.gather(
            reps,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), reps.size(-1)
            ),
        )
        # 应用填充掩码
        copied_reps = copied_reps.masked_fill(padding_mask.unsqueeze(-1), 0)
        # 创建位置编码，即从0到最大长度减1的序列，并将其扩展到与输入序列相同的形状
        position_ids = torch.arange(0, max(up_lens)).unsqueeze(0).expand(len(reps), -1).to(device=copied_reps.device)
        # 返回上采样后的序列、填充掩码和位置编码
        return copied_reps, ~padding_mask, position_ids
    
    def forward(self, tgt_reps, labels, tgt_units):
        """
        前向传播函数，计算 CTC 损失。

        参数:
            tgt_reps (list of torch.Tensor): 目标表示的列表。
            labels (torch.Tensor): 标签张量。
            tgt_units (torch.Tensor): 目标单元张量。

        返回:
            torch.Tensor: 计算得到的 CTC 损失。
        """
        tgt_label_reps = []                                 # 存储过滤后的目标表示
        # 遍历目标表示和标签，过滤掉忽略索引对应的部分
        for tgt_rep, label in zip(tgt_reps, labels):
            tgt_label_reps.append(tgt_rep[label != IGNORE_INDEX])

        # 上采样并获取隐藏状态、注意力掩码和位置ID
        hidden_states, attention_mask, position_ids = self.upsample(tgt_label_reps, tgt_units)
        hidden_states = self.input_proj(hidden_states)      # 输入投影层

        # 通过每一层进行处理
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]                         # 更新隐藏状态
        ctc_logits = self.output_proj(hidden_states)                 # 输出投影得到 CTC 分类 logits
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32) # 计算 log 概率
        ctc_lens = attention_mask.long().sum(dim=-1)                 # 计算每个序列的实际长度
        ctc_tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1) # 计算目标序列的实际长度
        ctc_tgt_mask = ~lengths_to_padding_mask(ctc_tgt_lens)        # 生成目标序列的填充掩码
        ctc_tgt_flat = tgt_units.masked_select(ctc_tgt_mask)         # 选择非填充部分的目标单元

        # 计算 CTC 损失
        ctc_loss = F.ctc_loss(
            ctc_lprobs.transpose(0, 1),
            ctc_tgt_flat,
            ctc_lens,
            ctc_tgt_lens,
            reduction="sum",
            zero_infinity=True,
            blank=self.unit_vocab_size
        )
        ctc_loss /= ctc_tgt_lens.sum().item()                        # 损失归一化
        return ctc_loss
    
    def predict(self, tgt_reps):
        """
        预测给定输入的目标表示（tgt_reps）的输出。

        Args:
            tgt_reps (torch.Tensor): 目标表示的张量。

        Returns:
            torch.Tensor: 预测的CTC（Connectionist Temporal Classification）结果。
        """
        # 使用上采样方法处理输入的目标表示
        hidden_states, attention_mask, position_ids = self.upsample([tgt_reps])

        # 将隐藏状态通过输入投影层
        hidden_states = self.input_proj(hidden_states)

        # 遍历每一层，更新隐藏状态
        for layer in self.layers:
            # 每一层处理隐藏状态，注意力掩码和位置ID
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
        # 通过输出投影层得到CTC的logits
        ctc_logits = self.output_proj(hidden_states)

        # 对logits应用softmax函数，得到CTC的概率分布
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)

        # 获取概率分布中最大值的索引作为预测结果，并用注意力掩码填充无效位置
        ctc_pred = ctc_lprobs.argmax(dim=-1).masked_fill_(~attention_mask, self.unit_vocab_size)
        return ctc_pred