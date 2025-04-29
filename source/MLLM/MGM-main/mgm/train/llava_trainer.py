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
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

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
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.args.lr_multi is not None:
            lr_multi_dict = {}
            for _dict in self.args.lr_multi.split(','):
                _key_val = _dict.split(':')
                print("_key_val:", _key_val)
                lr_multi_dict[_key_val[0]] = float(_key_val[1])

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
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
            elif self.args.lr_multi is not None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and not any([_key in n for _key in lr_multi_dict.keys()]))
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and not any([_key in n for _key in lr_multi_dict.keys()]))
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for _key in lr_multi_dict:
                    _key_decay = [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and _key in n)
                        ]
                    _key_no_decay = [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and _key in n)
                        ]
                    print("Params LR Change:", _key, "NUM:", len(_key_decay), len(_key_no_decay))
                    if len(_key_decay) > 0:
                        optimizer_grouped_parameters.append(
                            {
                                "params": _key_decay,
                                "lr": self.args.learning_rate * lr_multi_dict[_key],
                                "weight_decay": self.args.weight_decay,
                            },
                        )
                    if len(_key_no_decay) > 0:
                        optimizer_grouped_parameters.append(
                            {
                                "params": _key_no_decay,
                                "lr": self.args.learning_rate * lr_multi_dict[_key],
                                "weight_decay": 0.0,
                            },
                        )
            else:
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

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
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

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = seimport os  # 导入 os 模块，用于文件和目录操作
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块并别名为 nn

from torch.utils.data import Sampler  # 导入数据采样器基类

from transformers import Trainer  # 从 transformers 库导入基类 Trainer
from transformers.trainer import (  # 导入 transformers.trainer 中的工具函数和常量
    is_sagemaker_mp_enabled,  # 用于判断是否启用了 SageMaker 的模型并行
    get_parameter_names,  # 获取模型参数名称列表
    has_length,  # 检查数据集是否实现 __len__ 方法
    ALL_LAYERNORM_LAYERS,  # 表示所有 LayerNorm 层名称的集合
    logger,  # transformers 内部的日志记录器
)
from typing import List, Optional  # 导入类型提示：List 和 Optional


def maybe_zero_3(param, ignore_status=False, name=None):  # 定义函数 maybe_zero_3
    from deepspeed import zero  # 动态导入 deepspeed.zero 模块
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus  # 导入 ZeroParamStatus 枚举
    if hasattr(param, "ds_id"):  # 如果参数具有 deepspeed 标记，则认为它参与了零冗余优化
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:  # 如果当前参数尚未加载
            if not ignore_status:
                print(name, 'no ignore status')  # 打印提示信息
        with zero.GatheredParameters([param]):  # 在上下文中收集参数到主 GPU
            param = param.data.detach().cpu().clone()  # 拷贝并分离出 CPU 上的张量副本
    else:
        param = param.detach().cpu().clone()  # 否则直接拷贝并分离张量
    return param  # 返回转换后的参数张量


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):  # 定义函数：筛选并获取 adapter 权重
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}  # 过滤出名称包含指定关键字的参数
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}  # 对每个参数应用 maybe_zero_3 并移到 CPU
    return to_return  # 返回一个 {name: tensor} 的字典


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    # 英文注释保留：Split a list of indices into `chunks` chunks of roughly equal lengths.
    # 中文：将索引列表分割成 num_chunks 个长度大致相等的块

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]  # 若无法整除，则按交错方式分配索引

    num_indices_per_chunk = len(indices) // num_chunks  # 计算每个块应该包含的索引数

    chunks = [[] for _ in range(num_chunks)]  # 初始化每个块的索引列表
    chunks_lengths = [0 for _ in range(num_chunks)]  # 初始化每个块的总长度
    for index in indices:  # 遍历所有索引
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))  # 找到当前长度最短的块
        chunks[shortest_chunk].append(index)  # 将该索引添加到最短块
        chunks_lengths[shortest_chunk] += lengths[index]  # 累加该索引对应的长度
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")  # 达到容量后将长度标记为无限，不再往此块中添加

    return chunks  # 返回分块后的索引列表


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # 英文注释保留：We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # 中文：由于分布式采样器会设置 torch 随机种子，因此我们需要用 torch 来生成随机数
    assert all(l != 0 for l in lengths), "Should not have zero length."  # 确保所有长度都不为零
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        # 英文注释保留：all samples are in the same modality
        # 中文：所有样本都属于同一模态时
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)  # 直接调用单一模态分组函数

    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])  # 图文模态索引及其长度
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])  # 语言模态索引及其长度（取正值）

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    # 按长度分组后获取图文模态的乱序索引
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    # 按长度分组后获取语言模态的乱序索引
    megabatch_size = world_size * batch_size  # 定义 mega-batch 的大小
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    # 将图文模态拆分为多个 mega-batch
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    # 将语言模态拆分为多个 mega-batch

    last_mm = mm_megabatches[-1]  # 最后一个图文 mega-batch
    last_lang = lang_megabatches[-1]  # 最后一个语言 mega-batch
    additional_batch = last_mm + last_lang  # 将两者合并，形成额外的一个小 batch
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]  # 剔除最后一个各自的 mega-batch 组合
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)  # 随机打乱 mega-batch 顺序
    megabatches = [megabatches[i] for i in megabatch_indices]  # 重排序

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))  # 若有额外 batch，则排序后追加

    return [i for megabatch in megabatches for i in megabatch]  # 展平所有 megabatch，返回索引列表


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # 英文注释保留：We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    # 中文：由于分布式采样器会设置 torch 随机种子，因此我们需要用 torch 来生成随机数
    indices = torch.randperm(len(lengths), generator=generator)  # 随机打乱所有样本索引
    megabatch_size = world_size * batch_size  # 计算 mega-batch 大小
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    # 将乱序索引按 mega-batch 大小切分
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    # 在每个 mega-batch 内按样本长度降序排序
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    # 将每个 mega-batch 再分割为 world_size 个小 batch

    return [i for megabatch in megabatches for batch in megabatch for i in batch]
    # 展平所有层级后的索引列表并返回


class LengthGroupedSampler(Sampler):  # 定义按长度分组采样器类
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """
    # 英文注释保留：Sampler that samples indices in a way that groups together features of the dataset...
    # 中文：按长度将数据集样本大致分组，同时保持一定随机性的采样器

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")  # 若未提供 lengths，抛出异常
        self.batch_size = batch_size  # 每个设备的 batch size
        self.world_size = world_size  # 分布式训练中的设备数
        self.lengths = lengths  # 样本长度列表
        self.generator = generator  # 随机数生成器
        self.group_by_modality = group_by_modality  # 是否按模态分组

    def __len__(self):
        return len(self.lengths)  # 返回数据集大小

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )  # 若按模态分组，调用相应函数获取索引
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )  # 否则按单一模态分组
        return iter(indices)  # 返回索引迭代器


class LLaVATrainer(Trainer):  # 定义 LLaVATrainer 类，继承自 transformers.Trainer

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None  # 如果没有训练集或其不支持 __len__，返回 None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths  # 获取每个样本的"模态长度"
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )  # 当需要按模态长度分组时，返回自定义采样器
        else:
            return super()._get_train_sampler()  # 否则调用父类方法


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # 英文注释保留：Setup the optimizer...
        # 中文：设置优化器，提供了一个合理的默认方案，若需自定义可通过传入 optimizers 或重写此方法
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()  # 若启用了 SageMaker 模型并行，调用父类逻辑

        opt_model = self.model  # 待优化的模型

        if self.args.lr_multi is not None:
            lr_multi_dict = {}
            for _dict in self.args.lr_multi.split(','):
                _key_val = _dict.split(':')
                print("_key_val:", _key_val)  # 打印 lr_multi 配置
                lr_multi_dict[_key_val[0]] = float(_key_val[1])  # 构建多学习率字典

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]  # 去掉 bias
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]  # 找到 mm_projector 参数
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
                ]  # 针对 mm_projector 使用单独学习率
            elif self.args.lr_multi is not None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and not any([_key in n for _key in lr_multi_dict.keys()]))
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and not any([_key in n for _key in lr_multi_dict.keys()]))
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for _key in lr_multi_dict:
                    _key_decay = [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and _key in n)
                        ]
                    _key_no_decay = [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and _key in n)
                        ]
                    print("Params LR Change:", _key, "NUM:", len(_key_decay), len(_key_no_decay))  # 打印每组参数数量
                    if len(_key_decay) > 0:
                        optimizer_grouped_parameters.append(
                            {
                                "params": _key_decay,
                                "lr": self.args.learning_rate * lr_multi_dict[_key],
                                "weight_decay": self.args.weight_decay,
                            },
                        )
                    if len(_key_no_decay) > 0:
                        optimizer_grouped_parameters.append(
                            {
                                "params": _key_no_decay,
                                "lr": self.args.learning_rate * lr_multi_dict[_key],
                                "weight_decay": 0.0,
                            },
                        )  # 针对指定关键字参数设置多学习率
            else:
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
                ]  # 默认参数分组

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            # 获取优化器类及其初始化参数

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            # 实例化优化器

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
                # 8bit 优化时跳过 embedding 并以 fp32 优化

        return self.optimizer  # 返回优化器实例


    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"  # 构建 checkpoint 文件夹名

            run_dir = self._get_output_dir(trial=trial)  # 获取运行目录
            output_dir = os.path.join(run_dir, checkpoint_folder)  # 拼接输出目录

            # Only save Adapter
            # 英文注释保留：Only save Adapter
            # 中文：仅保存 Adapter 权重
            keys_to_match = ['mm_projector', 'vision_resampler']
            keys_to_match.extend(['vlm_att', 'vlm_uni'])
            keys_to_match.extend(['vision_fpn', 'vision_stages', 'vision_tower'])
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])  # 根据配置可能扩展额外键

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            # 获取需保存的权重字典

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)  # 保存模型配置
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))  # 保存权重文件
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)  # 否则使用父类保存逻辑


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass  # 若仅调优 Adapter，则不执行默认保存
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)  # 否则使用父类保存逻辑 lf._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            keys_to_match.extend(['vlm_att', 'vlm_uni'])
            keys_to_match.extend(['vision_fpn', 'vision_stages', 'vision_tower'])
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