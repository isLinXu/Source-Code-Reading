# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import time
import weakref
from typing import List, Mapping, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm  # 导入通信相关的工具函数，用于分布式训练
from detectron2.utils.events import EventStorage, get_event_storage  # 导入事件存储相关的类和函数，用于记录训练过程中的各种指标
from detectron2.utils.logger import _log_api_usage  # 导入日志记录工具

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]  # 定义模块的公开接口


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    可以注册到TrainerBase类的钩子的基类。

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    每个钩子可以实现4个方法。它们的调用方式如下所示：
    ::
        hook.before_train()  # 训练开始前调用
        for iter in range(start_iter, max_iter):  # 训练循环
            hook.before_step()  # 每次迭代前调用
            trainer.run_step()  # 执行训练步骤
            hook.after_step()   # 每次迭代后调用
        iter += 1
        hook.after_train()  # 训练结束后调用

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).
           在钩子方法中，用户可以通过self.trainer访问更多上下文属性（例如模型、当前迭代次数或配置）。

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           在before_step中执行的钩子通常也可以在after_step中实现。
           如果钩子执行需要较长时间，强烈建议在after_step而不是before_step中实现。
           约定是before_step应该只占用很少的时间。

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
           遵循这个约定将使那些关心before_step和after_step之间差异的钩子（例如计时器）能够正常工作。
    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    对训练器对象的弱引用。当钩子被注册时由训练器设置。
    """

    def before_train(self):
        """
        Called before the first iteration.
        在第一次迭代之前调用。
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        在最后一次迭代之后调用。
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        在每次迭代之前调用。
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        在每次迭代之后调用。
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        钩子默认是无状态的，但可以通过实现state_dict和load_state_dict使其可以保存检查点。
        """
        return {}


class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    带有钩子系统的迭代式训练器的基类。

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.
    这里我们只做一个假设：训练在一个循环中运行。
    子类可以实现具体的循环内容。
    我们不对数据加载器、优化器、模型等的存在做任何假设。

    Attributes:
        iter(int): the current iteration.
                   当前迭代次数。

        start_iter(int): The iteration to start with.
                         开始训练的迭代次数。
            By convention the minimum possible value is 0.
            按照惯例，最小可能值为0。

        max_iter(int): The iteration to end training.
                       结束训练的迭代次数。

        storage(EventStorage): An EventStorage that's opened during the course of training.
                              在训练过程中打开的事件存储器。
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []  # 存储注册的钩子列表
        self.iter: int = 0  # 当前迭代次数
        self.start_iter: int = 0  # 开始迭代次数
        self.max_iter: int  # 最大迭代次数
        self.storage: EventStorage  # 事件存储器
        _log_api_usage("trainer." + self.__class__.__name__)  # 记录API使用情况

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        向训练器注册钩子。钩子按照注册的顺序执行。

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
                                              钩子列表
        """
        hooks = [h for h in hooks if h is not None]  # 过滤掉None值
        for h in hooks:
            assert isinstance(h, HookBase)  # 确保钩子是HookBase的实例
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            # 为了避免循环引用，钩子和训练器不能相互持有对方。
            # 这通常无关紧要，但如果涉及的对象包含__del__方法，会导致内存泄漏。
            h.trainer = weakref.proxy(self)  # 使用弱引用代理
        self._hooks.extend(hooks)  # 将钩子添加到列表中

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
                                        见上面的文档说明
        """
        logger = logging.getLogger(__name__)  # 获取日志记录器
        logger.info("Starting training from iteration {}".format(start_iter))  # 记录开始训练的信息

        self.iter = self.start_iter = start_iter  # 设置开始迭代次数
        self.max_iter = max_iter  # 设置最大迭代次数

        with EventStorage(start_iter) as self.storage:  # 创建事件存储器的上下文
            try:
                self.before_train()  # 调用训练前的钩子
                for self.iter in range(start_iter, max_iter):  # 训练循环
                    self.before_step()  # 调用步骤前的钩子
                    self.run_step()  # 执行训练步骤
                    self.after_step()  # 调用步骤后的钩子
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                # self.iter == max_iter 可以被after_train用来判断训练是成功完成还是因异常而失败
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")  # 记录训练过程中的异常
                raise
            finally:
                self.after_train()  # 确保调用训练后的钩子

    def before_train(self):
        for h in self._hooks:  # 遍历所有钩子
            h.before_train()  # 调用每个钩子的before_train方法

    def after_train(self):
        self.storage.iter = self.iter  # 更新存储器中的迭代次数
        for h in self._hooks:  # 遍历所有钩子
            h.after_train()  # 调用每个钩子的after_train方法

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        # 维护storage.iter == trainer.iter的不变性
        # 在每个步骤的整个执行过程中
        self.storage.iter = self.iter  # 更新存储器中的迭代次数

        for h in self._hooks:  # 遍历所有钩子
            h.before_step()  # 调用每个钩子的before_step方法

    def after_step(self):
        for h in self._hooks:  # 遍历所有钩子
            h.after_step()  # 调用每个钩子的after_step方法

    def run_step(self):
        raise NotImplementedError  # 子类必须实现此方法

    def state_dict(self):
        ret = {"iteration": self.iter}  # 保存当前迭代次数
        hooks_state = {}  # 保存钩子的状态
        for h in self._hooks:
            sd = h.state_dict()  # 获取钩子的状态字典
            if sd:  # 如果钩子有状态
                name = type(h).__qualname__  # 获取钩子的限定名
                if name in hooks_state:  # 如果已存在同名钩子
                    # TODO handle repetitive stateful hooks
                    continue  # 暂时跳过重复的钩子
                hooks_state[name] = sd  # 保存钩子的状态
        if hooks_state:  # 如果有钩子状态
            ret["hooks"] = hooks_state  # 将钩子状态添加到返回字典中
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)  # 获取日志记录器
        self.iter = state_dict["iteration"]  # 恢复迭代次数
        for key, value in state_dict.get("hooks", {}).items():  # 遍历钩子状态
            for h in self._hooks:  # 遍历当前的钩子
                try:
                    name = type(h).__qualname__  # 获取钩子的限定名
                except AttributeError:
                    continue
                if name == key:  # 如果找到匹配的钩子
                    h.load_state_dict(value)  # 恢复钩子的状态
                    break
            else:  # 如果没有找到匹配的钩子
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")  # 记录警告


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    用于最常见任务类型的简单训练器：
    单损失函数、单优化器、单数据源的迭代优化，
    可选择使用数据并行。

    It assumes that every step, you:
    假设每一步你都会：

    1. Compute the loss with a data from the data_loader.
       使用来自数据加载器的数据计算损失。
    2. Compute the gradients with the above loss.
       使用上述损失计算梯度。
    3. Update the model with the optimizer.
       使用优化器更新模型。

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.
    训练期间的所有其他任务（检查点、日志记录、评估、学习率调度）
    都由钩子维护，这些钩子可以通过TrainerBase.register_hooks方法注册。

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    如果你想做更复杂的事情，
    可以继承TrainerBase并实现自己的run_step方法，
    或者编写自己的训练循环。
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
                一个torch模块。从data_loader获取数据并返回损失字典。
            data_loader: an iterable. Contains data to be used to call model.
                一个可迭代对象。包含用于调用模型的数据。
            optimizer: a torch optimizer.
                一个torch优化器。
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        在训练器中将模型设置为训练模式。
        但是在eval模式下训练模型也是有效的。
        如果你希望你的模型（或其子模块）在训练期间表现得像评估模式，
        你可以重写它的train()方法。
        """
        model.train()  # 将模型设置为训练模式

        self.model = model  # 保存模型实例
        self.data_loader = data_loader  # 保存数据加载器
        self._data_loader_iter = iter(data_loader)  # 创建数据加载器的迭代器
        self.optimizer = optimizer  # 保存优化器实例

    def run_step(self):
        """
        Implement the standard training logic described above.
        实现上述标准训练逻辑。
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"  # 确保模型处于训练模式
        start = time.perf_counter()  # 记录开始时间
        """
        If you want to do something with the data, you can wrap the dataloader.
        如果你想对数据做一些处理，你可以包装数据加载器。
        """
        data = next(self._data_loader_iter)  # 获取下一批数据
        data_time = time.perf_counter() - start  # 计算数据加载时间

        """
        If you want to do something with the losses, you can wrap the model.
        如果你想对损失做一些处理，你可以包装模型。
        """
        loss_dict = self.model(data)  # 前向传播，计算损失
        if isinstance(loss_dict, torch.Tensor):  # 如果返回的是张量
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}  # 将其转换为字典格式
        else:
            losses = sum(loss_dict.values())  # 计算所有损失的总和

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        如果你需要累积梯度或做类似的事情，你可以用自定义的zero_grad()方法包装优化器。
        """
        self.optimizer.zero_grad()  # 清零梯度
        losses.backward()  # 反向传播

        self._write_metrics(loss_dict, data_time)  # 记录训练指标

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        如果你需要梯度裁剪/缩放或其他处理，你可以用自定义的step()方法包装优化器。
        但正如https://arxiv.org/abs/2006.15704 Sec 3.2.4中解释的那样，这是次优的。
        """
        self.optimizer.step()  # 更新模型参数

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)  # 调用静态方法记录指标

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
                标量损失的字典
            data_time (float): time taken by the dataloader iteration
                数据加载器迭代所需的时间
            prefix (str): prefix for logging keys
                日志键的前缀
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}  # 将损失转换为CPU上的标量值
        metrics_dict["data_time"] = data_time  # 添加数据加载时间到指标字典

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        # 收集所有工作进程的指标用于日志记录
        # 这假设我们使用DDP风格的训练，这是目前detectron2唯一支持的方法
        all_metrics_dict = comm.gather(metrics_dict)  # 收集所有进程的指标

        if comm.is_main_process():  # 如果是主进程
            storage = get_event_storage()  # 获取事件存储器

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            # 工作进程之间的data_time可能有很大的差异。
            # 实际的延迟是工作进程中的最大值。
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])  # 计算最大数据加载时间
            storage.put_scalar("data_time", data_time)  # 记录数据加载时间

            # average the rest metrics
            # 平均其余的指标
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }  # 计算所有指标的平均值
            total_losses_reduced = sum(metrics_dict.values())  # 计算总损失
            if not np.isfinite(total_losses_reduced):  # 检查损失是否为有限值
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)  # 记录总损失
            if len(metrics_dict) > 1:  # 如果有多个指标
                storage.put_scalars(**metrics_dict)  # 记录所有指标

    def state_dict(self):
        ret = super().state_dict()  # 获取父类的状态字典
        ret["optimizer"] = self.optimizer.state_dict()  # 添加优化器的状态
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)  # 加载父类的状态
        self.optimizer.load_state_dict(state_dict["optimizer"])  # 加载优化器的状态


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    类似于SimpleTrainer，但在训练循环中使用PyTorch的原生自动混合精度。
    """

    def __init__(self, model, data_loader, optimizer, grad_scaler=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
                与SimpleTrainer中的参数相同。
            grad_scaler: torch GradScaler to automatically scale gradients.
                torch的GradScaler，用于自动缩放梯度。
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"  # 不支持单进程多设备训练的错误信息
        if isinstance(model, DistributedDataParallel):  # 如果模型是DistributedDataParallel类型
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported  # 确保不是多设备训练
        assert not isinstance(model, DataParallel), unsupported  # 确保不是DataParallel类型

        super().__init__(model, data_loader, optimizer)  # 调用父类的初始化方法

        if grad_scaler is None:  # 如果没有提供梯度缩放器
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()  # 创建一个新的梯度缩放器
        self.grad_scaler = grad_scaler  # 保存梯度缩放器_loader, optimizer, grad_scaler=None):
        

    def run_step(self):
        """
        Implement the AMP training logic.
        实现自动混合精度(AMP)训练的逻辑。
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"  # 确保模型处于训练模式
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"  # 确保CUDA可用，因为AMP训练需要GPU支持
        from torch.cuda.amp import autocast  # 导入自动混合精度的autocast上下文管理器

        start = time.perf_counter()  # 记录数据加载开始时间
        data = next(self._data_loader_iter)  # 从数据加载器获取下一批数据
        data_time = time.perf_counter() - start  # 计算数据加载所需时间

        with autocast():  # 使用autocast上下文，在此上下文中自动进行FP16/FP32混合精度计算
            loss_dict = self.model(data)  # 前向传播，计算模型输出和损失
            if isinstance(loss_dict, torch.Tensor):  # 如果损失是单个张量
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}  # 将单个损失转换为字典格式
            else:
                losses = sum(loss_dict.values())  # 如果是多个损失，计算总损失

        self.optimizer.zero_grad()  # 清除所有参数的梯度
        self.grad_scaler.scale(losses).backward()  # 使用梯度缩放器缩放损失，然后进行反向传播

        self._write_metrics(loss_dict, data_time)  # 记录训练指标

        self.grad_scaler.step(self.optimizer)  # 使用梯度缩放器更新模型参数
        self.grad_scaler.update()  # 根据本次迭代是否出现梯度溢出来更新缩放器的缩放因子

    def state_dict(self):
        ret = super().state_dict()  # 获取父类的状态字典
        ret["grad_scaler"] = self.grad_scaler.state_dict()  # 将梯度缩放器的状态添加到状态字典中
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)  # 加载父类的状态
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])  # 加载梯度缩放器的状态
