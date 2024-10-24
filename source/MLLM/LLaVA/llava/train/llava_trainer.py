import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    """
    根据参数的状况，决定是否使用ZeroParamStatus处理参数数据。

    参数:
    - param: 待处理的参数，可以是一个深度学习模型的参数。
    - ignore_status: 布尔值，指示是否忽略参数的可用状态，默认为False。
    - name: 参数的名称，用于调试信息。

    返回:
    - param的副本，如果param有ds_id属性且其状态为NOT_AVAILABLE，则返回其在CPU上的副本，否则原样返回。
    """
    # 导入deepSpeed的zero模块和ZeroParamStatus类，用于处理参数的分区和状态检查。
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    # 检查param是否具有ds_id属性，这是使用ZeroParamStatus处理的先决条件。
    if hasattr(param, "ds_id"):
        # 如果参数的状态为NOT_AVAILABLE且不忽略状态，则打印调试信息。
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        # 使用GatheredParameters上下文管理器收集和释放参数，然后创建参数数据的CPU副本。
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        # 如果param没有ds_id属性，则直接创建其CPU副本。
        param = param.detach().cpu().clone()
    # 返回处理后的参数副本。
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """
    根据指定的键匹配参数并将其通过maybe_zero_3函数处理后返回

    该函数从named_params中筛选出包含keys_to_match的参数，然后对这些参数应用maybe_zero_3函数，
    最后将处理后的参数及其名称以字典形式返回

    参数:
    named_params (Iterable[Tuple[str, Tensor]]): 模型的命名参数迭代器，通常来源于模型的named_parameters方法
    keys_to_match (Iterable[str]): 需要匹配的键列表，用于筛选参数

    返回:
    Dict[str, Tensor]: 处理后的参数字典，键为参数名，值为处理后的参数张量
    """
    # 筛选包含keys_to_match的参数
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}

    # 对筛选后的参数应用maybe_zero_3函数，并将结果移到CPU上
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}

    # 返回处理后的参数字典
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    将索引列表分割成大致相等长度的多个块。

    参数:
    indices (list): 要分割的索引列表。
    lengths (list): 每个索引对应的长度列表。
    num_chunks (int): 分割成的块数。

    返回:
    list: 包含分割后的块的列表。
    """

    # 如果索引列表不能被均匀地分成 num_chunks 块，则使用更简单的切片方法
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    # 计算每个块应包含的索引数量
    num_indices_per_chunk = len(indices) // num_chunks

    # 初始化块列表和块长度列表
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    # 遍历索引列表，尽可能均匀地将索引分配到每个块中
    for index in indices:
        # 找到当前长度最短的块
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        # 将索引添加到最短的块中
        chunks[shortest_chunk].append(index)
        # 更新最短块的总长度
        chunks_lengths[shortest_chunk] += lengths[index]
        # 如果当前块达到了预期的索引数量，将其长度标记为无穷大以防止进一步分配
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    # 返回包含分割后块的列表


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    按模态和长度分组索引，确保不同模态之间的批次平衡。

    该函数适用于数据样本具有不同模态（例如，多模态和单模态样本）的场景，通过正负长度值来区分。
    它确保每个批次包含来自不同模态的样本，这对于分布式训练环境非常有益。

    参数:
    lengths (list of int): 样本长度列表，多模态样本为正值，单模态样本为负值。
    batch_size (int): 每个批次的大小。
    world_size (int): 分布式训练中的进程总数。
    generator (torch.Generator, 可选): 用于可重复性的随机数生成器。

    返回:
    list: 按模态和长度分组并排序的索引列表，适用于批量数据加载。
    """
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # 需要使用 torch 进行随机部分，因为分布式采样器会设置 torch 的随机种子。
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # 所有样本都在同一模态
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    # 分离多模态和单模态样本
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # 按长度对每种模态的索引进行打乱，然后按 Megabatch 分组
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]

    # 计算分布式训练的Megabatch大小
    megabatch_size = world_size * batch_size

    # 将打乱后的索引分成Megabatchs
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # 准备每种模态的最后一个Megabatchs，并将其与其他Megabatchs合并
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]

    # 从每种模态的最后一个Megabatchs创建一个额外的批次
    additional_batch = last_mm + last_lang

    # 将两种模态的Megabatchs合并并打乱
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # 如果额外的批次包含任何样本，则添加到Megabatchs列表中
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    # 展平Megabatchs列表并返回
    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    根据序列长度生成数据样本的索引。

    该函数首先随机打乱所有数据样本的索引。然后，将数据样本按序列长度降序排序，分成多个megabatch（每个megabatch包含world_size * batch_size个数据样本）。
    最后，将每个megabatch分割成均匀大小的块，以便于分布式处理。

    参数:
    lengths (list): 包含所有数据样本序列长度的列表。
    batch_size (int): 每个批次中的数据样本数量。
    world_size (int): 参与分布式训练的进程数。
    generator (torch.Generator, 可选): 用于确保结果可重复性的随机数生成器。
    merge (bool, 可选): 是否将相同长度的连续序列合并成一个批次。当前函数中未使用此参数。

    返回:
    list: 一个索引列表，表示数据样本应按此顺序处理。
    """
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # 使用torch进行随机部分，因为分布式采样器会设置torch的随机种子。
    indices = torch.randperm(len(lengths), generator=generator)

    # 计算megabatch的大小，即world_size和batch_size的乘积。
    megabatch_size = world_size * batch_size

    # 将随机化的索引分成多个megabatch。
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]

    # 按序列长度降序对每个megabatch内的索引进行排序。
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]

    # 将每个megabatch分割成均匀大小的块，以便于分布式处理。
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # 展平megabatch列表并返回索引列表。
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    LengthGroupedSampler是一个Sampler类，它通过将数据集中长度大致相同的元素分组在一起来采样索引，
    同时保持一定程度的随机性。这在处理序列数据时特别有用，比如在自然语言处理任务中，可以将长度
    相似的文本放在一起处理，以提高计算效率。

    参数:
        batch_size (int): 每个批次的大小。
        world_size (int): 并行处理的进程数，用于分布式训练。
        lengths (Optional[List[int]]): 数据集中每个元素的长度列表。默认为None。
        generator: 随机数生成器，用于确保结果的可重复性。默认为None。
        group_by_modality (bool): 是否按模态分组。如果为True，则使用get_modality_length_grouped_indices方法；
                                  否则使用get_length_grouped_indices方法。默认为False。
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        # 检查lengths参数是否提供，这是为了确保数据集的长度信息被正确提供，否则会引发ValueError异常。
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        # 返回数据集的长度，即有多少个元素
        return len(self.lengths)

    def __iter__(self):
        # 根据是否按模态分组，选择不同的方法生成索引
        if self.group_by_modality:
            # 如果按模式分组，则调用专门处理模态长度分组的函数
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            # 如果不按模式分组，则调用通用的长度分组函数
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        # 返回生成的索引迭代器
        return iter(indices)


class LLaVATrainer(Trainer):
    """
    LLaVA训练器，继承自Trainer类，专用于LLaVA模型的训练。
    """
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        获取训练数据集的采样器。

        Returns:
            Optional[torch.utils.data.Sampler]: 返回训练数据集的采样器，如果没有训练数据集或数据集没有指定长度，则返回None。
        """
        # 检查训练数据集是否存在且有指定长度，否则返回None
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        # 根据模态长度分组采样
        if self.args.group_by_modality_length:
            # 获取训练数据集的模态长度
            lengths = self.train_dataset.modality_lengths
            # 返回按模态长度分组的采样器
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            # 如果不按模态长度分组，则调用父类方法获取采样器
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        设置优化器。
        我们提供了一个合理的默认设置，如果你想要使用其他优化器，
        可以通过在 Trainer 的初始化中传递一个元组 `optimizers`，或者通过子类化并覆盖此方法来实现。
        """
        # 如果启用了 SageMaker 模型并行化，调用父类的 create_optimizer 方法
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        # 设置要优化的模型
        opt_model = self.model

        # 只有在优化器尚未设置的情况下才进行设置
        if self.optimizer is None:
            # 获取需要权重衰减的参数名称
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # 如果指定了多模态投影层的单独学习率，准备相应的参数
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                # 如果没有指定多模态投影层的单独学习率，正常准备参数
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            # 根据训练参数获取优化器类和关键字参数
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # 创建优化器实例
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # 如果使用 Adam8bit 优化器，注册模块覆盖以进行 fp32 优化
            if optimizer_cls.__name__ == "Adam8bit":
                # 导入bitsandbytes库，用于支持8位优化器
                import bitsandbytes

                # 获取全局优化管理器实例
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                # 初始化跳过的参数数量
                skipped = 0
                # 遍历模型的所有模块
                for module in opt_model.modules():
                    # 检查模块是否为Embedding层
                    if isinstance(module, nn.Embedding):
                        # 计算并累加跳过的参数数量
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        # 记录跳过的模块及其参数量
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        # 在优化管理器中注册模块权重的特殊优化位数
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        # 记录将以fp32精度优化的模块
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                # 记录总共跳过的参数量
                logger.info(f"skipped: {skipped/2**20}M params")
        # 返回优化器
        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
