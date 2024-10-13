import os
import torch
from torch import nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
from typing import List, Optional

from ..utils.train_utils import *


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    将索引列表分割成`chunks`个大致相等长度的块。

    :param indices: 索引列表
    :param lengths: 每个索引对应的长度列表
    :param num_chunks: 要分割的块数
    :return: 分割后的索引块列表
    """
    # 如果索引总数不能被块数整除，则简单地按块数进行分割
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    # 计算每个块应包含的索引数
    num_indices_per_chunk = len(indices) // num_chunks

    # 初始化块列表和块长度列表
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]

    # 将索引分配到最短的块中，直到达到每个块的索引数上限
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    根据模态和长度对索引进行分组。

    :param lengths: 数据集的长度列表
    :param batch_size: 批量大小
    :param world_size: 世界大小（通常指GPU数量）
    :param generator: 随机数生成器
    :return: 分组后的索引列表
    """
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # 确保没有长度为0的数据
    assert all(l != 0 for l in lengths), "Should not have zero length."

    # 将数据分为多模态和语言两类
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # 确保至少有一个多模态和一个语言样本
    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    # 对多模态和语言索引进行随机打乱和分组
    # 根据mm_lengths和batch_size等参数，获取分组后的索引列表mm_shuffle
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    # 根据lang_lengths和batch_size等参数，获取分组后的索引列表lang_shuffle
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    # 计算一个超级批次的样本数量
    megabatch_size = world_size * batch_size
    # 将mm_shuffle和lang_shuffle分别分割成多个超级批次
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # 获取最后一个mm超级批次和lang超级批次
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]

    # 将最后一个mm超级批次和lang超级批次合并为一个额外的批次
    additional_batch = last_mm + last_lang
    # 去除最后一个超级批次后的所有超级批次
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]

    # 对超级批次列表进行随机排列
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # 如果额外的批次大小大于等于一个超级批次的大小，则将其作为一个新的超级批次添加到超级批次列表的开头，并更新额外的批次
    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    # 如果还有剩余的额外批次，则将其添加到超级批次列表的末尾
    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    # 将所有超级批次中的样本合并为一个列表并返回
    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    采样器，以一种方式采样索引，将数据集中大致相同长度的特征分组在一起，同时保持一定的随机性。
    """

    def __init__(
        self,
        batch_size: int,                        # 批量大小
        world_size: int,                        # 世界大小，通常用于分布式训练
        lengths: Optional[List[int]] = None,    # 数据集特征的长度列表
        generator=None,                         # 随机数生成器
        group_by_modality: bool = False,        # 是否按模态分组
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.") # 如果未提供长度，则抛出异常

        self.batch_size = batch_size                      # 初始化批量大小
        self.world_size = world_size                      # 初始化世界大小
        self.lengths = lengths                            # 初始化长度列表
        self.generator = generator                        # 初始化随机数生成器
        self.group_by_modality = group_by_modality        # 初始化是否按模态分组

    def __len__(self):
        return len(self.lengths)                          # 返回长度列表的长度

    def __iter__(self):
        if self.group_by_modality:
            # 如果按模态分组，则获取按模态长度分组的索引
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            # 否则，获取按长度分组的索引
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices) # 返回索引的迭代器


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        获取训练数据采样器。

        Returns:
            Optional[torch.utils.data.Sampler]: 训练数据采样器，如果无法创建则返回None。
        """
        # 检查训练数据集是否存在以及是否有长度
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # 如果按模态长度分组
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            # 创建LengthGroupedSampler，用于按模态长度分组采样
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,             # 批量大小
                world_size=self.args.world_size,        # 世界大小
                lengths=lengths,                        # 模态长度
                group_by_modality=True,                 # 按模态分组
            )
        else:
            # 否则使用父类的采样器
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        设置优化器。
        我们提供了一个合理的默认值，效果很好。如果你想使用其他的，可以通过Trainer的init传递一个元组到`optimizers`，
        或者子类化并重写这个方法。
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            # 获取需要衰减的参数名称
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                # 获取连接器参数名称
                connector_parameters = [name for name, _ in opt_model.named_parameters() if "connector" in name]
                # 根据参数类型分组
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
                ]
            # 如果启用了moe，将参数分组到不同的moe组中
            if getattr(self.args, "moe_enable", False):
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
            # 获取优化器类和参数
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            # 创建优化器实例
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            # 如果使用的是Adam8bit优化器，注册Embedding模块的权重优化位数为32
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer




