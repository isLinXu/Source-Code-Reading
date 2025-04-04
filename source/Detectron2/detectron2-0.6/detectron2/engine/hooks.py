# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import datetime
import itertools
import logging
import math
import operator
import os
import tempfile
import time
import warnings
from collections import Counter
import torch
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.timer import Timer
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import detectron2.utils.comm as comm
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.solver import LRMultiplier
from detectron2.utils.events import EventStorage, EventWriter
from detectron2.utils.file_io import PathManager

from .train_loop import HookBase

__all__ = [
    "CallbackHook",      # 回调钩子
    "IterationTimer",    # 迭代计时器
    "PeriodicWriter",    # 周期性写入器
    "PeriodicCheckpointer",  # 周期性检查点保存器
    "BestCheckpointer",      # 最佳检查点保存器
    "LRScheduler",          # 学习率调度器
    "AutogradProfiler",     # 自动梯度分析器
    "EvalHook",            # 评估钩子
    "PreciseBN",           # 精确批归一化
    "TorchProfiler",       # PyTorch性能分析器
    "TorchMemoryStats",    # PyTorch内存统计
]


"""
Implement some common hooks.
实现一些常用的钩子。
"""


class CallbackHook(HookBase):
    """
    Create a hook using callback functions provided by the user.
    使用用户提供的回调函数创建一个钩子。
    """

    def __init__(self, *, before_train=None, after_train=None, before_step=None, after_step=None):
        """
        Each argument is a function that takes one argument: the trainer.
        每个参数都是一个接收trainer作为唯一参数的函数。
        """
        self._before_train = before_train  # 训练开始前的回调函数
        self._before_step = before_step   # 每步迭代前的回调函数
        self._after_step = after_step     # 每步迭代后的回调函数
        self._after_train = after_train   # 训练结束后的回调函数

    def before_train(self):
        if self._before_train:  # 如果存在训练前回调函数，则执行
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:   # 如果存在训练后回调函数，则执行
            self._after_train(self.trainer)
        # The functions may be closures that hold reference to the trainer
        # Therefore, delete them to avoid circular reference.
        # 这些函数可能是闭包，持有trainer的引用
        # 因此，删除它们以避免循环引用
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_step(self):
        if self._before_step:    # 如果存在步骤前回调函数，则执行
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:     # 如果存在步骤后回调函数，则执行
            self._after_step(self.trainer)


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.
    跟踪每次迭代所花费的时间（训练器中的每个run_step调用）。
    在训练结束时打印摘要。

    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    该钩子使用其before_step和after_step方法之间的时间。
    根据约定，所有钩子的before_step方法应该只占用可忽略的时间，
    因此IterationTimer钩子应该放在钩子列表的开头以获得准确的计时。
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
                在开始时要从计时中排除的迭代次数。
        """
        self._warmup_iter = warmup_iter  # 预热迭代次数
        self._step_timer = Timer()       # 步骤计时器
        self._start_time = time.perf_counter()  # 开始时间
        self._total_timer = Timer()      # 总计时器

    def before_train(self):
        self._start_time = time.perf_counter()  # 记录训练开始时间
        self._total_timer.reset()   # 重置总计时器
        self._total_timer.pause()    # 暂停总计时器

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time  # 计算总训练时间
        total_time_minus_hooks = self._total_timer.seconds()  # 获取不包含钩子执行时间的总时间
        hook_time = total_time - total_time_minus_hooks      # 计算钩子执行时间

        num_iter = self.trainer.storage.iter + 1 - self.trainer.start_iter - self._warmup_iter  # 计算实际迭代次数

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            # 只有在预热后，速度才有意义
            # 注意：这个格式会被一些脚本用grep解析
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()    # 重置步骤计时器
        self._total_timer.resume()  # 恢复总计时器

    def after_step(self):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        # +1是因为我们在after_step中，当前步骤已完成但还未计数
        iter_done = self.trainer.storage.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:  # 如果已完成预热阶段
            sec = self._step_timer.seconds()  # 获取当前步骤耗时
            self.trainer.storage.put_scalars(time=sec)  # 记录耗时
        else:  # 在预热阶段
            self._start_time = time.perf_counter()  # 重置开始时间
            self._total_timer.reset()  # 重置总计时器

        self._total_timer.pause()  # 暂停总计时器


class PeriodicWriter(HookBase):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.
    通过定期调用writer.write()将事件写入EventStorage。

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    它在每个period迭代和最后一次迭代后执行。
    注意，period不会影响每个写入器如何平滑数据。
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
                                        EventWriter对象列表
            period (int): 写入周期
        """
        self._writers = writers  # 保存写入器列表
        for w in writers:  # 确保所有写入器都是EventWriter实例
            assert isinstance(w, EventWriter), w
        self._period = period  # 设置写入周期

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (  # 每period次迭代或最后一次迭代时执行
            self.trainer.iter == self.trainer.max_iter - 1
        ):
            for writer in self._writers:  # 遍历所有写入器进行写入操作
                writer.write()

    def after_train(self):
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            # 如果发现任何新数据（例如由其他after_train产生的），
            # 在关闭前写入它们
            writer.write()
            writer.close()  # 关闭写入器


class PeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        self.step(self.trainer.iter)


class BestCheckpointer(HookBase):
    """
    Checkpoints best weights based off given metric.
    根据给定的评估指标保存最佳权重的检查点。

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    该钩子应该与产生评估指标的钩子（如EvalHook）一起使用，并在其之后执行。
    """

    def __init__(
        self,
        eval_period: int,
        checkpointer: Checkpointer,
        val_metric: str,
        mode: str = "max",
        file_prefix: str = "model_best",
    ) -> None:
        """
        Args:
            eval_period (int): the period `EvalHook` is set to run.
                              EvalHook的运行周期
            checkpointer: the checkpointer object used to save checkpoints.
                         用于保存检查点的检查点对象
            val_metric (str): validation metric to track for best checkpoint, e.g. "bbox/AP50"
                             用于跟踪最佳检查点的验证指标，例如"bbox/AP50"
            mode (str): one of {'max', 'min'}. controls whether the chosen val metric should be
                maximized or minimized, e.g. for "bbox/AP50" it should be "max"
                       'max'或'min'之一，控制所选验证指标是应该最大化还是最小化
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
                              检查点文件名的前缀，默认为"model_best"
        """
        self._logger = logging.getLogger(__name__)  # 获取日志记录器
        self._period = eval_period  # 评估周期
        self._val_metric = val_metric  # 验证指标名称
        assert mode in [
            "max",
            "min",
        ], f'Mode "{mode}" to `BestCheckpointer` is unknown. It should be one of {"max", "min"}.'
        if mode == "max":  # 根据模式选择比较操作符
            self._compare = operator.gt  # 大于操作符
        else:
            self._compare = operator.lt  # 小于操作符
        self._checkpointer = checkpointer  # 检查点保存器
        self._file_prefix = file_prefix  # 文件前缀
        self.best_metric = None  # 最佳指标值
        self.best_iter = None  # 最佳指标对应的迭代次数

    def _update_best(self, val, iteration):
        if math.isnan(val) or math.isinf(val):  # 检查值是否为NaN或无穷大
            return False
        self.best_metric = val  # 更新最佳指标值
        self.best_iter = iteration  # 更新最佳迭代次数
        return True

    def _best_checking(self):
        metric_tuple = self.trainer.storage.latest().get(self._val_metric)  # 获取最新的评估指标
        if metric_tuple is None:  # 如果指标不存在
            self._logger.warning(
                f"Given val metric {self._val_metric} does not seem to be computed/stored."
                "Will not be checkpointing based on it."
            )
            return
        else:
            latest_metric, metric_iter = metric_tuple  # 解包最新指标值和对应的迭代次数

        if self.best_metric is None:  # 如果是第一次保存检查点
            if self._update_best(latest_metric, metric_iter):
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(f"{self._file_prefix}", **additional_state)  # 保存首个模型
                self._logger.info(
                    f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                )
        elif self._compare(latest_metric, self.best_metric):  # 如果当前指标更好
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save(f"{self._file_prefix}", **additional_state)  # 保存更好的模型
            self._logger.info(
                f"Saved best model as latest eval score for {self._val_metric} is"
                f"{latest_metric:0.5f}, better than last best score "
                f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
            )
            self._update_best(latest_metric, metric_iter)  # 更新最佳指标
        else:  # 如果当前指标不够好
            self._logger.info(
                f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
            )

    def after_step(self):
        # same conditions as `EvalHook`
        # 与EvalHook相同的条件
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0  # 每个评估周期
            and next_iter != self.trainer.max_iter  # 且不是最后一次迭代
        ):
            self._best_checking()  # 执行最佳检查点检查

    def after_train(self):
        # same conditions as `EvalHook`
        # 与EvalHook相同的条件
        if self.trainer.iter + 1 >= self.trainer.max_iter:  # 在训练结束时
            self._best_checking()  # 执行最后一次最佳检查点检查


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    一个执行PyTorch内置学习率调度器并总结学习率的钩子。
    它在每次迭代后执行。
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer): 优化器对象
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.
                如果是ParamScheduler对象，它定义了优化器中基础学习率的乘数。

        If any argument is not given, will try to obtain it from the trainer.
        如果没有提供任何参数，将尝试从训练器中获取。
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id = LRScheduler.get_best_param_group_id(self._optimizer)

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        # 注意：关于要总结哪个学习率的一些启发式方法
        # 总结具有最多参数的参数组
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            # 如果所有组都只有一个参数，
            # 那么找到最常见的初始学习率，并用它来做总结
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)


class TorchProfiler(HookBase):
    """
    A hook which runs `torch.profiler.profile`.
    一个运行PyTorch性能分析器的钩子。

    Examples:
    ::
        hooks.TorchProfiler(
             lambda trainer: 10 < trainer.iter < 20, self.cfg.OUTPUT_DIR
        )

    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We did not profile the first few iterations
    because they are typically slower than the rest.
    上面的例子将在第10~20次迭代时运行性能分析器，并将结果保存到``OUTPUT_DIR``。
    我们没有对前几次迭代进行分析，因为它们通常比其他迭代慢。

    The result files can be loaded in the ``chrome://tracing`` page in chrome browser,
    and the tensorboard visualizations can be visualized using
    ``tensorboard --logdir OUTPUT_DIR/log``
    结果文件可以在Chrome浏览器的``chrome://tracing``页面中加载，
    并且可以使用``tensorboard --logdir OUTPUT_DIR/log``来查看TensorBoard可视化结果。
    """

    def __init__(self, enable_predicate, output_dir, *, activities=None, save_tensorboard=True):
        """
        Args:
            enable_predicate (callable[trainer -> bool]): a function which takes a trainer,
                and returns whether to enable the profiler.
                It will be called once every step, and can be used to select which steps to profile.
                一个接收trainer作为参数的函数，返回是否启用性能分析器。
                它会在每一步调用一次，可以用来选择要分析哪些步骤。
            output_dir (str): the output directory to dump tracing files.
                             用于保存跟踪文件的输出目录。
            activities (iterable): same as in `torch.profiler.profile`.
                                  与`torch.profiler.profile`中的参数相同。
            save_tensorboard (bool): whether to save tensorboard visualizations at (output_dir)/log/
                                    是否在(output_dir)/log/目录下保存TensorBoard可视化结果。
        """
        self._enable_predicate = enable_predicate
        self._activities = activities
        self._output_dir = output_dir
        self._save_tensorboard = save_tensorboard

    def before_step(self):
        if self._enable_predicate(self.trainer):
            if self._save_tensorboard:
                on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    os.path.join(
                        self._output_dir,
                        "log",
                        "profiler-tensorboard-iter{}".format(self.trainer.iter),
                    ),
                    f"worker{comm.get_rank()}",
                )
            else:
                on_trace_ready = None
            self._profiler = torch.profiler.profile(
                activities=self._activities,
                on_trace_ready=on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
            self._profiler.__enter__()
        else:
            self._profiler = None

    def after_step(self):
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        if not self._save_tensorboard:
            PathManager.mkdirs(self._output_dir)
            out_file = os.path.join(
                self._output_dir, "profiler-trace-iter{}.json".format(self.trainer.iter)
            )
            if "://" not in out_file:
                self._profiler.export_chrome_trace(out_file)
            else:
                # Support non-posix filesystems
                with tempfile.TemporaryDirectory(prefix="detectron2_profiler") as d:
                    tmp_file = os.path.join(d, "tmp.json")
                    self._profiler.export_chrome_trace(tmp_file)
                    with open(tmp_file) as f:
                        content = f.read()
                with PathManager.open(out_file, "w") as f:
                    f.write(content)


class AutogradProfiler(TorchProfiler):
    """
    A hook which runs `torch.autograd.profiler.profile`.
    一个运行PyTorch自动求导性能分析器的钩子。

    Examples:
    ::
        hooks.AutogradProfiler(
             lambda trainer: 10 < trainer.iter < 20, self.cfg.OUTPUT_DIR
        )

    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We did not profile the first few iterations
    because they are typically slower than the rest.
    上面的例子将在第10~20次迭代时运行性能分析器，并将结果保存到``OUTPUT_DIR``。
    我们没有对前几次迭代进行分析，因为它们通常比其他迭代慢。

    The result files can be loaded in the ``chrome://tracing`` page in chrome browser.
    结果文件可以在Chrome浏览器的``chrome://tracing``页面中加载。

    Note:
        When used together with NCCL on older version of GPUs,
        autograd profiler may cause deadlock because it unnecessarily allocates
        memory on every device it sees. The memory management calls, if
        interleaved with NCCL calls, lead to deadlock on GPUs that do not
        support ``cudaLaunchCooperativeKernelMultiDevice``.
        注意：
            当在旧版本的GPU上与NCCL一起使用时，
            自动求导性能分析器可能会导致死锁，因为它会在每个看到的设备上不必要地分配内存。
            如果内存管理调用与NCCL调用交错，在不支持``cudaLaunchCooperativeKernelMultiDevice``的GPU上会导致死锁。
    """

    def __init__(self, enable_predicate, output_dir, *, use_cuda=True):
        """
        Args:
            enable_predicate (callable[trainer -> bool]): a function which takes a trainer,
                and returns whether to enable the profiler.
                It will be called once every step, and can be used to select which steps to profile.
                一个接收trainer作为参数的函数，返回是否启用性能分析器。
                它会在每一步调用一次，可以用来选择要分析哪些步骤。
            output_dir (str): the output directory to dump tracing files.
                             用于保存跟踪文件的输出目录。
            use_cuda (bool): same as in `torch.autograd.profiler.profile`.
                            与`torch.autograd.profiler.profile`中的参数相同。
        """
        warnings.warn("AutogradProfiler has been deprecated in favor of TorchProfiler.")
        self._enable_predicate = enable_predicate
        self._use_cuda = use_cuda
        self._output_dir = output_dir

    def before_step(self):
        if self._enable_predicate(self.trainer):
            self._profiler = torch.autograd.profiler.profile(use_cuda=self._use_cuda)
            self._profiler.__enter__()
        else:
            self._profiler = None


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    定期运行评估函数，并在训练结束时运行。

    It is executed every ``eval_period`` iterations and after the last iteration.
    它在每``eval_period``次迭代和最后一次迭代后执行。
    """

    def __init__(self, eval_period, eval_function):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
                运行`eval_function`的周期。设置为0表示不定期评估（但仍在最后一次迭代后评估）。
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
                一个不接受参数的函数，返回一个嵌套的评估指标字典。

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
            # 注意：这个钩子必须在所有工作进程中都启用或都不启用。
            # 如果只想让特定工作进程执行评估，
            # 需要给其他工作进程一个空操作函数（`eval_function=lambda: None`）。
        """
        # 初始化评估周期和评估函数
        self._period = eval_period
        self._func = eval_function
        self._period = eval_period
        self._func = eval_function

    def _do_eval(self):
        # 执行评估函数并获取结果
        results = self._func()

        if results:
            # 确保评估函数返回的是字典类型
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            # 将嵌套字典展平为一层字典
            flattened_results = flatten_results_dict(results)
            # 验证所有值都可以转换为浮点数
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            # 将评估结果存储到trainer的storage中，不进行平滑处理
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        # 由于不同工作进程的评估时间可能不同，
        # 使用同步障碍确保所有工作进程一起开始下一次迭代
        comm.synchronize()

    def after_step(self):
        # 计算下一次迭代的编号
        next_iter = self.trainer.iter + 1
        # 如果设置了评估周期且到达评估时间点
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            # 如果不是最后一次迭代，执行评估
            # 最后一次迭代的评估将在after_train中进行
            if next_iter != self.trainer.max_iter:
                self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        # 这个条件用于防止在训练失败后执行评估
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        # 由于评估函数可能是一个闭包，持有trainer的引用
        # 因此在结束时删除它以避免循环引用
        del self._func


class PreciseBN(HookBase):
    """
    The standard implementation of BatchNorm uses EMA in inference, which is
    sometimes suboptimal.
    This class computes the true average of statistics rather than the moving average,
    and put true averages to every BN layer in the given model.
    # 标准的BatchNorm实现在推理时使用指数移动平均(EMA)，这有时并不是最优的。
    # 这个类计算真实的统计平均值而不是移动平均值，
    # 并将真实平均值应用到给定模型的每个BN层。

    It is executed every ``period`` iterations and after the last iteration.
    # 它在每个"period"迭代和最后一次迭代后执行。
    """

    def __init__(self, period, model, data_loader, num_iter):
        """
        Args:
            period (int): the period this hook is run, or 0 to not run during training.
                The hook will always run in the end of training.
                # 运行此钩子的周期，设为0表示在训练期间不运行。
                # 钩子将始终在训练结束时运行。
            model (nn.Module): a module whose all BN layers in training mode will be
                updated by precise BN.
                Note that user is responsible for ensuring the BN layers to be
                updated are in training mode when this hook is triggered.
                # 一个模块，其中所有处于训练模式的BN层将被精确BN更新。
                # 注意：用户负责确保在触发此钩子时，要更新的BN层处于训练模式。
            data_loader (iterable): it will produce data to be run by `model(data)`.
                # 数据加载器，用于产生将由model(data)运行的数据。
            num_iter (int): number of iterations used to compute the precise
                statistics.
                # 用于计算精确统计信息的迭代次数。
        """
        # 初始化日志记录器
        self._logger = logging.getLogger(__name__)
        # 检查模型中是否有处于训练模式的BN层
        if len(get_bn_modules(model)) == 0:
            self._logger.info(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
                # PreciseBN被禁用，因为模型中没有处于训练模式的BN层
            )
            self._disabled = True
            return

        # 保存初始化参数
        self._model = model
        self._data_loader = data_loader
        self._num_iter = num_iter
        self._period = period
        self._disabled = False

        # 数据迭代器初始化为None
        self._data_iter = None

    def after_step(self):
        # 计算下一次迭代的编号
        next_iter = self.trainer.iter + 1
        # 检查是否是最后一次迭代
        is_final = next_iter == self.trainer.max_iter
        # 在最后一次迭代或达到更新周期时更新统计信息
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self.update_stats()

    def update_stats(self):
        """
        Update the model with precise statistics. Users can manually call this method.
        # 使用精确的统计信息更新模型。用户可以手动调用此方法。
        """
        # 如果PreciseBN被禁用，直接返回
        if self._disabled:
            return

        # 如果数据迭代器未初始化，则创建一个新的迭代器
        if self._data_iter is None:
            self._data_iter = iter(self._data_loader)

        # 定义一个数据加载生成器函数
        def data_loader():
            for num_iter in itertools.count(1):
                # 每100次迭代打印一次进度信息
                if num_iter % 100 == 0:
                    self._logger.info(
                        "Running precise-BN ... {}/{} iterations.".format(num_iter, self._num_iter)
                    )
                # This way we can reuse the same iterator
                # 这样可以重用同一个迭代器
                yield next(self._data_iter)

        # 使用新的EventStorage来捕获事件并丢弃它们
        with EventStorage():  # capture events in a new storage to discard them
            self._logger.info(
                "Running precise-BN for {} iterations...  ".format(self._num_iter)
                + "Note that this could produce different statistics every time."
            )
            update_bn_stats(self._model, data_loader(), self._num_iter)


class TorchMemoryStats(HookBase):
    """
    Writes pytorch's cuda memory statistics periodically.
    # 定期记录PyTorch的CUDA内存统计信息。
    """

    def __init__(self, period=20, max_runs=10):
        """
        Args:
            period (int): Output stats each 'period' iterations
                # 每隔'period'次迭代输出一次统计信息
            max_runs (int): Stop the logging after 'max_runs'
                # 在达到'max_runs'次数后停止日志记录
        """

        # 初始化日志记录器和计数器
        self._logger = logging.getLogger(__name__)
        self._period = period        # 统计周期
        self._max_runs = max_runs    # 最大运行次数
        self._runs = 0               # 当前运行次数

    def after_step(self):
        # 如果已达到最大运行次数，直接返回
        if self._runs > self._max_runs:
            return

        # 在达到周期或最后一次迭代时记录内存统计信息
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
        ):
            # 检查是否可以使用CUDA
            if torch.cuda.is_available():
                # 计算各种内存使用情况（单位：MB）
                max_reserved_mb = torch.cuda.max_memory_reserved() / 1024.0 / 1024.0  # 最大预留内存
                reserved_mb = torch.cuda.memory_reserved() / 1024.0 / 1024.0         # 当前预留内存
                max_allocated_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0  # 最大分配内存
                allocated_mb = torch.cuda.memory_allocated() / 1024.0 / 1024.0         # 当前分配内存

                # 记录内存使用信息
                self._logger.info(
                    (
                        " iter: {} "
                        " max_reserved_mem: {:.0f}MB "
                        " reserved_mem: {:.0f}MB "
                        " max_allocated_mem: {:.0f}MB "
                        " allocated_mem: {:.0f}MB "
                    ).format(
                        self.trainer.iter,
                        max_reserved_mb,
                        reserved_mb,
                        max_allocated_mb,
                        allocated_mb,
                    )
                )

                # 更新运行次数
                self._runs += 1
                # 在最后一次运行时输出详细的内存摘要
                if self._runs == self._max_runs:
                    mem_summary = torch.cuda.memory_summary()
                    self._logger.info("\n" + mem_summary)

                # 重置峰值内存统计信息
                torch.cuda.reset_peak_memory_stats()
