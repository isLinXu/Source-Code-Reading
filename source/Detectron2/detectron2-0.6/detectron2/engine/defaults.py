# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.
# 本文件包含了一些用户在训练/测试中可能需要的默认样板逻辑组件。这些组件可能不适用于所有人，但许多用户可能会发现它们很有用。

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
# 本文件中函数/类的行为可能会发生变化，因为它们旨在表示人们在项目中需要的"通用默认行为"。
"""

import argparse
import logging
import os
import sys
import weakref
from collections import OrderedDict
from typing import Optional
import torch
from fvcore.nn.precise_bn import get_bn_modules
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from . import hooks
from .train_loop import AMPTrainer, SimpleTrainer, TrainerBase

__all__ = [  # 定义模块的公共接口，包含以下类和函数
    "create_ddp_model",
    "default_argument_parser",
    "default_setup",
    "default_writers",
    "DefaultPredictor",
    "DefaultTrainer",
]


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    # 如果有多个进程，创建一个分布式数据并行(DistributedDataParallel)模型

    Args:
        model: a torch.nn.Module
        # 输入的模型，必须是torch.nn.Module类型
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        # 是否添加fp16压缩钩子到ddp对象，可以减少通信开销
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
        # DistributedDataParallel的其他参数
    """  # noqa
    if comm.get_world_size() == 1:  # 如果只有一个进程，直接返回原始模型
        return model
    if "device_ids" not in kwargs:  # 如果没有指定device_ids，使用当前进程的本地rank作为device_id
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)  # 创建DistributedDataParallel模型
    if fp16_compression:  # 如果启用fp16压缩
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        # 注册fp16压缩钩子，用于减少通信开销
        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    # 创建一个包含detectron2用户常用参数的解析器

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
        # 传递给ArgumentParser的epilog参数，用于描述用法

    Returns:
        argparse.ArgumentParser:
        # 返回参数解析器对象
    """
    parser = argparse.ArgumentParser(  # 创建参数解析器
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",  # 设置使用示例说明
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")  # 配置文件路径
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )  # 是否从检查点目录恢复训练
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")  # 是否仅执行评估
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")  # 每台机器的GPU数量
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")  # 机器总数
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )  # 当前机器的rank（每台机器唯一）

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # PyTorch在多GPU训练中可能会留下孤立进程。因此我们使用确定性方式获取端口，这样用户可以通过查看端口占用来发现孤立进程。
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )  # 分布式训练的初始化URL
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),  # 在命令末尾修改配置选项。对于Yacs配置，使用空格分隔的"PATH.KEY VALUE"对；对于基于Python的LazyConfig，使用"path.key=value"
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    # 尝试从配置中选择键，直到找到第一个存在的键。否则返回默认值。
    """
    if isinstance(cfg, CfgNode):  # 如果配置是CfgNode类型
        cfg = OmegaConf.create(cfg.dump())  # 转换为OmegaConf格式
    for k in keys:  # 遍历所有键
        none = object()
        p = OmegaConf.select(cfg, k, default=none)  # 尝试获取键对应的值
        if p is not none:  # 如果找到值
            return p  # 返回找到的值
    return default  # 如果没有找到任何值，返回默认值


def _highlight(code, filename):
    """
    # 对代码进行语法高亮处理
    """
    try:
        import pygments  # 尝试导入pygments库
    except ImportError:
        return code  # 如果导入失败，直接返回原始代码

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()  # 根据文件扩展名选择合适的词法分析器
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))  # 使用monokai风格进行代码高亮
    return code


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    # 在作业开始时执行一些基本的通用设置，包括：

    1. Set up the detectron2 logger
    # 1. 设置detectron2日志记录器
    2. Log basic information about environment, cmdline arguments, and config
    # 2. 记录环境、命令行参数和配置的基本信息
    3. Backup the config to the output directory
    # 3. 将配置备份到输出目录

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        # 要使用的完整配置
        args (argparse.NameSpace): the command line arguments to be logged
        # 要记录的命令行参数
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")  # 获取输出目录
    if comm.is_main_process() and output_dir:  # 如果是主进程且指定了输出目录
        PathManager.mkdirs(output_dir)  # 创建输出目录

    rank = comm.get_rank()  # 获取当前进程的rank
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")  # 设置fvcore的日志记录器
    logger = setup_logger(output_dir, distributed_rank=rank)  # 设置主日志记录器

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))  # 记录当前进程的rank和总进程数
    logger.info("Environment info:\n" + collect_env_info())  # 记录环境信息

    logger.info("Command line arguments: " + str(args))  # 记录命令行参数
    if hasattr(args, "config_file") and args.config_file != "":  # 如果指定了配置文件
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )  # 记录配置文件内容

    if comm.is_main_process() and output_dir:  # 如果是主进程且指定了输出目录
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        # 注意：我们的一些脚本可能需要输出目录中存在config.yaml文件
        path = os.path.join(output_dir, "config.yaml")  # 配置文件保存路径
        if isinstance(cfg, CfgNode):  # 如果配置是CfgNode类型
            logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))  # 记录完整配置
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())  # 将配置保存到文件
        else:
            LazyConfig.save(cfg, path)  # 如果是LazyConfig类型，使用其保存方法
        logger.info("Full config saved to {}".format(path))  # 记录配置保存路径

    # make sure each worker has a different, yet deterministic seed if specified
    # 确保每个worker有不同但确定的随机种子（如果指定了的话）
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)  # 获取随机种子
    seed_all_rng(None if seed < 0 else seed + rank)  # 设置随机种子

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    # cudnn benchmark有较大开销。考虑到典型验证集的小规模，不应该使用它。
    if not (hasattr(args, "eval_only") and args.eval_only):  # 如果不是仅评估模式
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )  # 设置cudnn benchmark


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.
    # 构建要使用的EventWriter列表，包括CommonMetricPrinter、TensorboardXWriter和JSONWriter

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        # 存储JSON指标和tensorboard事件的目录
        max_iter: the total number of iterations
        # 总迭代次数

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
        # 返回EventWriter对象列表
    """
    PathManager.mkdirs(output_dir)  # 创建输出目录
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        # 可能不会总是打印你想看到的内容，因为它只打印"通用"指标
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    # 使用给定配置创建一个简单的端到端预测器，在单个设备上处理单个输入图像

    Compared to using the model directly, this class does the following additions:
    # 与直接使用模型相比，这个类做了以下额外工作：

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    # 1. 从cfg.MODEL.WEIGHTS加载检查点
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    # 2. 始终将BGR图像作为输入，并应用cfg.INPUT.FORMAT定义的转换
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    # 3. 应用cfg.INPUT.{MIN,MAX}_SIZE_TEST定义的调整大小操作
    4. Take one input image and produce a single output, instead of a batch.
    # 4. 处理单个输入图像并产生单个输出，而不是批处理

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    # 这是为了简单演示目的而设计的，所以它会自动执行上述步骤。
    # 这不适用于基准测试或运行复杂的推理逻辑。
    # 如果你想做更复杂的事情，请参考其源代码作为示例来手动构建和使用模型。

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
        # metadata：底层数据集的元数据，从cfg.DATASETS.TEST获取

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model  # 克隆配置，因为配置可能会被模型修改
        self.model = build_model(self.cfg)  # 构建模型
        self.model.eval()  # 设置为评估模式
        if len(cfg.DATASETS.TEST):  # 如果有测试数据集
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])  # 获取第一个测试数据集的元数据

        checkpointer = DetectionCheckpointer(self.model)  # 创建检查点加载器
        checkpointer.load(cfg.MODEL.WEIGHTS)  # 加载模型权重

        self.aug = T.ResizeShortestEdge(  # 创建图像增强器
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT  # 设置输入格式
        assert self.input_format in ["RGB", "BGR"], self.input_format  # 确保输入格式是RGB或BGR

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            # 原始图像：形状为(H, W, C)的图像（BGR顺序）

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
            # 预测结果：模型对单个图像的输出。有关格式的详细信息，请参见/tutorials/models
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            # 对图像进行预处理
            if self.input_format == "RGB":  # 如果输入格式是RGB
                # whether the model expects BGR inputs or RGB
                # 判断模型期望的是BGR输入还是RGB输入
                original_image = original_image[:, :, ::-1]  # 转换颜色通道顺序
            height, width = original_image.shape[:2]  # 获取图像高度和宽度
            image = self.aug.get_transform(original_image).apply_image(original_image)  # 应用图像变换
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # 转换为张量格式

            inputs = {"image": image, "height": height, "width": width}  # 构建输入字典
            predictions = self.model([inputs])[0]  # 获取模型预测结果
            return predictions


class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:
    一个具有默认训练逻辑的训练器，它执行以下操作：

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
       使用配置文件定义的模型、优化器和数据加载器创建一个SimpleTrainer，并创建学习率调度器。
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
       当调用resume_or_load时，加载最后的检查点或cfg.MODEL.WEIGHTS（如果存在）。
    3. Register a few common hooks defined by the config.
       注册配置文件中定义的一些常用钩子。

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.
    创建此类是为了简化标准模型训练工作流程，并为只需要标准训练工作流程和标准功能的用户减少样板代码。
    这意味着该类对训练逻辑做出了许多假设，这些假设在新的研究中可能变得无效。事实上，超出SimpleTrainer所做假设的任何假设对研究来说都太多了。

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:
    该类的代码已经注释了它所做的限制性假设。当这些假设不适合你时，建议你：

    1. Overwrite methods of this class, OR:
       重写此类的方法，或者：
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
       使用SimpleTrainer，它只进行最小的SGD训练，不做其他事情。如果需要，你可以添加自己的钩子。或者：
    3. Write your own training loop similar to `tools/plain_train_net.py`.
       编写类似于tools/plain_train_net.py的自己的训练循环。

    See the :doc:`/tutorials/training` tutorials for more details.
    更多详细信息请参见/tutorials/training教程。

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.
    注意，此类的行为与此文件中的其他函数/类一样不稳定，因为它旨在表示"通用默认行为"。
    它只保证在detectron2的标准模型和训练工作流程中工作良好。要获得更稳定的行为，请使用其他公共API编写自己的训练逻辑。

    Examples:
    示例：
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
    属性：
        scheduler: 学习率调度器
        checkpointer (DetectionCheckpointer): 检查点保存和加载器
        cfg (CfgNode): 配置节点
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): 配置节点，包含了训练所需的所有配置参数
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()  # 如果日志记录器未设置，则进行设置
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())  # 根据当前进程数自动调整配置参数

        # Assume these objects must be constructed in this order.
        # 假定这些对象必须按照以下顺序构造
        model = self.build_model(cfg)  # 构建模型
        optimizer = self.build_optimizer(cfg, model)  # 构建优化器
        data_loader = self.build_train_loader(cfg)  # 构建训练数据加载器

        model = create_ddp_model(model, broadcast_buffers=False)  # 创建分布式数据并行模型
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(  # 根据是否启用AMP选择训练器
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)  # 构建学习率调度器
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            # 假定你想将检查点与日志/统计信息一起保存
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),  # 使用弱引用代理避免循环引用
        )
        self.start_iter = 0  # 初始化起始迭代次数
        self.max_iter = cfg.SOLVER.MAX_ITER  # 设置最大迭代次数
        self.cfg = cfg  # 保存配置

        self.register_hooks(self.build_hooks())  # 注册训练钩子

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        如果resume=True且cfg.OUTPUT_DIR包含最后的检查点（由last_checkpoint文件定义），则从该文件恢复。
        恢复意味着加载所有可用状态（如优化器和调度器）并从检查点更新迭代计数器。此时不会使用cfg.MODEL.WEIGHTS。

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        否则，这将被视为独立的训练。该方法将从cfg.MODEL.WEIGHTS文件加载模型权重（但不会加载其他状态），并从迭代0开始。

        Args:
            resume (bool): whether to do resume or not
                          是否进行恢复训练
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)  # 调用检查点管理器的恢复或加载方法
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            # 检查点存储了刚刚完成的训练迭代，因此我们从下一次迭代开始
            self.start_iter = self.iter + 1

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        构建默认钩子列表，包括计时、评估、检查点保存、学习率调度、精确批归一化和事件写入。

        Returns:
            list[HookBase]: 钩子列表
        """
        cfg = self.cfg.clone()  # 克隆配置对象
        cfg.defrost()  # 解冻配置使其可修改
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
                                       # 为精确批归一化节省内存和时间

        ret = [
            hooks.IterationTimer(),  # 迭代计时器，用于记录训练时间
            hooks.LRScheduler(),  # 学习率调度器，用于动态调整学习率
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                # 与评估同频率运行（但在评估之前）
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                # 构建新的数据加载器以不影响训练
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)  # 如果启用了精确批归一化且模型包含BN层
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        # 在检查点保存之前执行精确批归一化，因为它会更新模型，需要被检查点保存。
        # 这并不总是最好的：如果检查点保存频率不同，某些检查点可能比其他检查点有更精确的统计信息。
        if comm.is_main_process():  # 如果是主进程
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))  # 添加周期性检查点保存器

        def test_and_save_results():  # 测试并保存结果的辅助函数
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # 在检查点保存之后进行评估，这样如果评估失败，我们可以使用保存的检查点进行调试
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))  # 添加评估钩子

        if comm.is_main_process():  # 如果是主进程
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            # 这里使用每个写入器的默认打印/日志频率
            # 在最后运行写入器，以便写入评估指标
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))  # 添加周期性写入器，用于记录训练日志
        return ret  # 返回构建的钩子列表

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        使用default_writers()构建要使用的写入器列表。
        如果你想要不同的写入器列表，可以在你的训练器中重写此方法。

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
                             一个EventWriter对象列表。
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)  # 返回默认的事件写入器列表

    def train(self):
        """
        Run training.
        运行训练。

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
            如果启用了评估，返回结果的有序字典。否则返回None。
        """
        super().train(self.start_iter, self.max_iter)  # 调用父类的训练方法
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():  # 如果有预期结果且是主进程
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"  # 确保训练过程中获得了评估结果
            verify_results(self.cfg, self._last_eval_results)  # 验证评估结果
            return self._last_eval_results  # 返回最后的评估结果

    def run_step(self):
        self._trainer.iter = self.iter  # 更新训练器的迭代计数
        self._trainer.run_step()  # 执行一步训练

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module: 返回构建的模型

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        现在调用detectron2.modeling.build_model函数。
        如果你想要不同的模型，可以重写此方法。
        """
        model = build_model(cfg)  # 使用配置构建模型
        logger = logging.getLogger(__name__)  # 获取日志记录器
        logger.info("Model:\n{}".format(model))  # 记录模型结构信息
        return model  # 返回构建的模型

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer: 返回构建的优化器

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        现在调用detectron2.solver.build_optimizer函数。
        如果你想要不同的优化器，可以重写此方法。
        """
        return build_optimizer(cfg, model)  # 使用配置和模型构建优化器

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        现在调用detectron2.solver.build_lr_scheduler函数。
        如果你想要不同的学习率调度器，可以重写此方法。
        """
        return build_lr_scheduler(cfg, optimizer)  # 使用配置和优化器构建学习率调度器

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable: 返回可迭代的训练数据加载器

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        现在调用detectron2.data.build_detection_train_loader函数。
        如果你想要不同的数据加载器，可以重写此方法。
        """
        return build_detection_train_loader(cfg)  # 构建训练数据加载器

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable: 返回可迭代的测试数据加载器

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        现在调用detectron2.data.build_detection_test_loader函数。
        如果你想要不同的数据加载器，可以重写此方法。
        """
        return build_detection_test_loader(cfg, dataset_name)  # 构建测试数据加载器

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None: 返回数据集评估器或None

        It is not implemented by default.
        默认情况下未实现此方法。
        """
        raise NotImplementedError(
            """
If you want DefaultTrainer to automatically run evaluation,
please implement `build_evaluator()` in subclasses (see train_net.py for example).
Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
如果你想要DefaultTrainer自动运行评估，
请在子类中实现build_evaluator()方法（参见train_net.py示例）。
或者，你可以自己调用评估函数（参见Colab气球教程示例）。
"""
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        评估给定的模型。给定的模型应该已经包含要评估的权重。

        Args:
            cfg (CfgNode): 配置节点
            model (nn.Module): 要评估的模型
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
                如果为None，将调用build_evaluator。否则，必须与cfg.DATASETS.TEST具有相同的长度。

        Returns:
            dict: a dict of result metrics
                 包含结果指标的字典
        """
        logger = logging.getLogger(__name__)  # 获取日志记录器
        if isinstance(evaluators, DatasetEvaluator):  # 如果evaluators是单个评估器，转换为列表
            evaluators = [evaluators]
        if evaluators is not None:  # 如果提供了评估器，验证其数量与测试数据集数量匹配
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()  # 用于存储评估结果的有序字典
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):  # 遍历所有测试数据集
            data_loader = cls.build_test_loader(cfg, dataset_name)  # 构建测试数据加载器
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            # 当评估器作为参数传入时，隐式假设评估器可以在数据加载器之前创建
            if evaluators is not None:
                evaluator = evaluators[idx]  # 使用提供的评估器
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)  # 尝试构建评估器
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )  # 如果没有找到评估器，记录警告
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)  # 在数据集上进行推理
            results[dataset_name] = results_i  # 保存当前数据集的评估结果
            if comm.is_main_process():  # 如果是主进程
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )  # 确保评估结果是字典类型
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))  # 记录评估结果
                print_csv_format(results_i)  # 以CSV格式打印结果

        if len(results) == 1:  # 如果只有一个数据集的结果
            results = list(results.values())[0]  # 直接返回该结果而不是字典
        return results  # 返回评估结果

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.
        当配置文件中定义的工作进程数（由cfg.SOLVER.REFERENCE_WORLD_SIZE指定）与当前使用的工作进程数不同时，
        返回一个新的配置，其中总批量大小被缩放，使得每个GPU的批量大小保持与原始的IMS_PER_BATCH // REFERENCE_WORLD_SIZE相同。

        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.
        其他配置选项也相应进行缩放：
        * 训练步数和预热步数按比例反向缩放
        * 学习率按比例正向缩放，遵循ImageNet in 1h论文的方法

        For example, with the original config like the following:
        例如，原始配置如下：

        .. code-block:: yaml

            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000

        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:
        当这个配置在16个GPU上使用（而不是参考数量8）时，调用此方法将返回一个新的配置：

        .. code-block:: yaml

            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500

        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).
        注意，原始配置和新配置都可以在16个GPU上训练。
        是否启用此功能由用户决定（通过设置REFERENCE_WORLD_SIZE）。

        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
            返回：CfgNode类型，一个新的配置。如果cfg.SOLVER.REFERENCE_WORLD_SIZE为0，则返回原始配置。
        """
        # 获取原始的工作进程数
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        # 如果原始工作进程数为0或与当前工作进程数相同，直接返回原始配置
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        # 克隆配置对象以避免修改原始配置
        cfg = cfg.clone()
        # 记录配置是否被冻结
        frozen = cfg.is_frozen()
        # 解冻配置以进行修改
        cfg.defrost()

        # 确保每个GPU的批量大小是整数
        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        # 计算缩放比例
        scale = num_workers / old_world_size
        # 按比例缩放批量大小
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        # 按比例缩放基础学习率
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        # 反向缩放最大迭代次数
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        # 反向缩放预热迭代次数
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        # 反向缩放学习率调整步骤
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        # 反向缩放评估周期
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        # 反向缩放检查点保存周期
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        # 更新工作进程数以保持不变量
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        # 获取日志记录器
        logger = logging.getLogger(__name__)
        # 记录自动缩放的配置信息
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        # 如果原配置是冻结的，重新冻结配置
        if frozen:
            cfg.freeze()
        # 返回更新后的配置
        return cfg


# Access basic attributes from the underlying trainer
# 访问底层训练器的基本属性
for _attr in ["model", "data_loader", "optimizer"]:
    setattr(
        DefaultTrainer,
        _attr,
        property(
            # getter
            # 获取器：从_trainer实例中获取对应属性
            lambda self, x=_attr: getattr(self._trainer, x),
            # setter
            # 设置器：设置_trainer实例的对应属性
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )
