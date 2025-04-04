# Copyright (c) Facebook, Inc. and its affiliates.
import datetime  # 导入日期时间模块
import json  # 导入JSON处理模块
import logging  # 导入日志模块
import os  # 导入操作系统接口模块
import time  # 导入时间处理模块
from collections import defaultdict  # 导入默认字典，处理键不存在的情况
from contextlib import contextmanager  # 导入上下文管理器
from typing import Optional  # 导入Optional类型提示
import torch  # 导入PyTorch库
from fvcore.common.history_buffer import HistoryBuffer  # 导入历史缓冲区

from detectron2.utils.file_io import PathManager  # 导入路径管理器

__all__ = [  # 定义公开的类和函数
    "get_event_storage",
    "JSONWriter",
    "TensorboardXWriter",
    "CommonMetricPrinter",
    "EventStorage",
]

_CURRENT_STORAGE_STACK = []  # 当前存储栈，用于存储EventStorage实例


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
        
    返回：
        当前正在使用的:class:`EventStorage`对象。
        如果当前没有启用:class:`EventStorage`，则抛出错误。
    """
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"  # 确保在EventStorage上下文中调用
    return _CURRENT_STORAGE_STACK[-1]  # 返回栈顶的EventStorage对象


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    
    基类，用于从:class:`EventStorage`获取事件并处理它们的写入器。
    """

    def write(self):
        raise NotImplementedError  # 抽象方法，需要子类实现

    def close(self):
        pass  # 关闭方法，默认不执行任何操作


class JSONWriter(EventWriter):
    """
    Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 19,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 39,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
       将标量写入json文件。
    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New scalars will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
                
        参数：
            json_file (str)：json文件的路径。如果文件存在，新的标量将被追加。
            window_size (int)：对`smoothing_hint`为True的标量进行中值平滑的窗口大小。
        """
        self._file_handle = PathManager.open(json_file, "a")  # 打开json文件用于追加
        self._window_size = window_size  # 存储窗口大小
        self._last_write = -1  # 上次写入的迭代次数，初始为-1

    def write(self):
        storage = get_event_storage()  # 获取当前的事件存储
        to_save = defaultdict(dict)  # 创建一个嵌套的默认字典，用于存储要保存的数据

        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            # keep scalars that have not been written
            # 保留尚未写入的标量
            if iter <= self._last_write:  # 如果该迭代已经写入
                continue  # 跳过
            to_save[iter][k] = v  # 存储到要保存的字典中
        if len(to_save):  # 如果有数据需要保存
            all_iters = sorted(to_save.keys())  # 排序所有迭代次数
            self._last_write = max(all_iters)  # 更新最后写入的迭代次数

        for itr, scalars_per_iter in to_save.items():  # 遍历每个迭代及其标量
            scalars_per_iter["iteration"] = itr  # 添加迭代次数信息
            self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")  # 写入JSON行
        self._file_handle.flush()  # 刷新文件缓冲区
        try:
            os.fsync(self._file_handle.fileno())  # 尝试同步文件到磁盘
        except AttributeError:  # 如果文件对象不支持fileno
            pass  # 忽略错误

    def close(self):
        self._file_handle.close()  # 关闭文件句柄


class TensorboardXWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    
    将所有标量写入tensorboard文件。
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
            
        参数：
            log_dir (str)：保存输出事件的目录
            window_size (int)：标量将通过此窗口大小进行中值平滑处理

            kwargs：传递给`torch.utils.tensorboard.SummaryWriter(...)`的其他参数
        """
        self._window_size = window_size  # 存储窗口大小
        from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)  # 创建SummaryWriter实例
        self._last_write = -1  # 上次写入的迭代次数，初始为-1

    def write(self):
        storage = get_event_storage()  # 获取当前的事件存储
        new_last_write = self._last_write  # 初始化新的最后写入迭代次数
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter <= self._last_write:  # 如果该迭代已经写入
                continue  # 跳过
            self._writer.add_scalar(k, v, iter)  # 添加标量到tensorboard
            new_last_write = max(new_last_write, iter)  # 更新最后写入的迭代次数
        self._last_write = new_last_write  # 保存新的最后写入迭代次数

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        # storage.put_{image,histogram}只供tensorboard写入器使用。
        # 所以我们从这里直接访问其内部字段。
        if len(storage._vis_data) >= 1:  # 如果存储中有可视化数据（图像）
            for img_name, img, step_num in storage._vis_data:  # 遍历所有图像数据
                self._writer.add_image(img_name, img, step_num)  # 将图像添加到tensorboard
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            # 存储保存所有图像数据，并依赖这个写入器来清除它们。
            # 因此它假设只有一个写入器会使用其图像数据。
            # 另一种设计是让存储只保存有限的最近数据
            # （例如，只保存最近的图像）供所有写入器访问。
            # 在这种情况下，如果写入器的周期较长，它可能看不到所有图像数据。
            storage.clear_images()  # 清除存储中的所有图像数据

        if len(storage._histograms) >= 1:  # 如果存储中有直方图数据
            for params in storage._histograms:  # 遍历所有直方图参数
                self._writer.add_histogram_raw(**params)  # 将原始直方图数据添加到tensorboard
            storage.clear_histograms()  # 清除存储中的所有直方图数据

    def close(self):
        if hasattr(self, "_writer"):  # 如果有_writer属性
            self._writer.close()  # 关闭写入器


class CommonMetricPrinter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    
    将**常见**指标打印到终端，包括
    迭代时间、预计完成时间(ETA)、内存、所有损失和学习率。
    它还使用20个元素的窗口进行平滑处理。

    它旨在以通用方式打印常见指标。
    要以更自定义的方式打印内容，请自行实现类似的打印器。
    """

    def __init__(self, max_iter: Optional[int] = None, window_size: int = 20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
            
        参数：
            max_iter：训练的最大迭代次数。
                用于计算预计完成时间(ETA)。如果未提供，将不会打印ETA。
            window_size (int)：损失将通过此窗口大小进行中值平滑处理
        """
        self.logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
        self._max_iter = max_iter  # 存储最大迭代次数
        self._window_size = window_size  # 存储窗口大小
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA  # 上次调用write()的(步骤,时间)，用于计算ETA

    def _get_eta(self, storage) -> Optional[str]:
        if self._max_iter is None:  # 如果未设置最大迭代次数
            return ""  # 返回空字符串
        iteration = storage.iter  # 获取当前迭代次数
        try:
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)  # 尝试使用时间历史计算剩余秒数
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)  # 存储预计完成时间(秒)
            return str(datetime.timedelta(seconds=int(eta_seconds)))  # 返回格式化的预计完成时间
        except KeyError:  # 如果无法获取时间历史
            # estimate eta on our own - more noisy
            # 自行估计预计完成时间 - 会更嘈杂
            eta_string = None  # 初始化预计完成时间字符串
            if self._last_write is not None:  # 如果有上次写入记录
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )  # 估计每次迭代时间
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)  # 计算剩余秒数
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))  # 格式化预计完成时间
            self._last_write = (iteration, time.perf_counter())  # 更新上次写入记录
            return eta_string  # 返回预计完成时间字符串

    def write(self):
        storage = get_event_storage()  # 获取当前的事件存储
        iteration = storage.iter  # 获取当前迭代次数
        if iteration == self._max_iter:  # 如果达到最大迭代次数
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            # 此钩子仅报告训练进度（损失、ETA等），而不报告其他数据，
            # 因此在训练成功后不写入任何内容，即使调用了此方法。
            return  # 直接返回

        try:
            data_time = storage.history("data_time").avg(20)  # 尝试获取数据加载时间平均值
        except KeyError:  # 如果无法获取数据加载时间
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            # 在前几次迭代中可能不存在（由于预热）
            # 或者当不使用SimpleTrainer时
            data_time = None  # 设为None
        try:
            iter_time = storage.history("time").global_avg()  # 尝试获取迭代时间全局平均值
        except KeyError:  # 如果无法获取迭代时间
            iter_time = None  # 设为None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())  # 尝试获取最新的学习率
        except KeyError:  # 如果无法获取学习率
            lr = "N/A"  # 设为"N/A"

        eta_string = self._get_eta(storage)  # 获取预计完成时间

        if torch.cuda.is_available():  # 如果CUDA可用
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0  # 计算最大分配内存(MB)
        else:
            max_mem_mb = None  # 设为None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        # 注意：max_mem被"dev/parse_results.sh"中的grep解析
        self.logger.info(
            " {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",  # 格式化预计完成时间
                iter=iteration,  # 当前迭代次数
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(self._window_size))
                        for k, v in storage.histories().items()
                        if "loss" in k  # 所有包含"loss"的项
                    ]
                ),  # 格式化所有损失
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",  # 格式化迭代时间
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",  # 格式化数据加载时间
                lr=lr,  # 学习率
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",  # 格式化最大内存
            )
        )  # 打印训练信息


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.

    In the future we may add support for storing / logging other types of data if needed.
    
    面向用户的类，提供指标存储功能。

    将来，如果需要，我们可能会添加对存储/记录其他类型数据的支持。
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
            
        参数：
            start_iter (int)：起始迭代次数
        """
        self._history = defaultdict(HistoryBuffer)  # 存储所有指标的历史记录
        self._smoothing_hints = {}  # 存储平滑提示
        self._latest_scalars = {}  # 存储最新的标量值
        self._iter = start_iter  # 当前迭代次数
        self._current_prefix = ""  # 当前前缀，用于分组指标
        self._vis_data = []  # 可视化数据列表
        self._histograms = []  # 直方图数据列表

    def put_image(self, img_name, img_tensor):
        """
        Add an `img_tensor` associated with `img_name`, to be shown on
        tensorboard.

        Args:
            img_name (str): The name of the image to put into tensorboard.
            img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is
                3. The image format should be RGB. The elements in img_tensor
                can either have values in [0, 1] (float32) or [0, 255] (uint8).
                The `img_tensor` will be visualized in tensorboard.
                
        添加与`img_name`关联的`img_tensor`，以便在tensorboard上显示。

        参数：
            img_name (str)：要放入tensorboard的图像名称。
            img_tensor (torch.Tensor或numpy.array)：形状为`[channel, height, width]`的`uint8`或`float`
                张量，其中`channel`为3。图像格式应为RGB。img_tensor中的元素
                可以是[0, 1]（float32）或[0, 255]（uint8）范围内的值。
                `img_tensor`将在tensorboard中可视化。
        """
        self._vis_data.append((img_name, img_tensor, self._iter))  # 将图像数据添加到可视化数据列表

    def put_scalar(self, name, value, smoothing_hint=True):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.

                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
                
        将标量`value`添加到与`name`关联的`HistoryBuffer`中。

        参数：
            smoothing_hint (bool)：关于此标量是否有噪声并在记录时应平滑的"提示"。
                该提示可通过:meth:`EventStorage.smoothing_hints`访问。写入器可能
                会忽略该提示并应用自定义平滑规则。

                默认为True，因为我们保存的大多数标量需要平滑处理才能
                提供任何有用的信号。
        """
        name = self._current_prefix + name  # 添加当前前缀到名称
        history = self._history[name]  # 获取或创建历史缓冲区
        value = float(value)  # 确保值是浮点数
        history.update(value, self._iter)  # 更新历史缓冲区
        self._latest_scalars[name] = (value, self._iter)  # 更新最新标量值

        existing_hint = self._smoothing_hints.get(name)  # 获取现有的平滑提示
        if existing_hint is not None:  # 如果已有平滑提示
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)  # 确保平滑提示一致
        else:
            self._smoothing_hints[name] = smoothing_hint  # 存储平滑提示

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        """
        Put multiple scalars from keyword arguments.

        Examples:

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
            
        从关键字参数中放入多个标量。

        示例：

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():  # 遍历所有关键字参数
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)  # 放入单个标量

    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        """
        Create a histogram from a tensor.

        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
            
        从张量创建直方图。

        参数：
            hist_name (str)：要放入tensorboard的直方图名称。
            hist_tensor (torch.Tensor)：要转换为直方图的任意形状的张量。
            bins (int)：直方图箱数。
        """
        ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()  # 获取张量的最小值和最大值

        # Create a histogram with PyTorch
        # 使用PyTorch创建直方图
        hist_counts = torch.histc(hist_tensor, bins=bins)  # 计算直方图计数
        hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)  # 计算直方图边界

        # Parameter for the add_histogram_raw function of SummaryWriter
        # SummaryWriter的add_histogram_raw函数的参数
        hist_params = dict(
            tag=hist_name,  # 标签
            min=ht_min,  # 最小值
            max=ht_max,  # 最大值
            num=len(hist_tensor),  # 元素数量
            sum=float(hist_tensor.sum()),  # 总和
            sum_squares=float(torch.sum(hist_tensor ** 2)),  # 平方和
            bucket_limits=hist_edges[1:].tolist(),  # 箱边界
            bucket_counts=hist_counts.tolist(),  # 箱计数
            global_step=self._iter,  # 全局步骤
        )
        self._histograms.append(hist_params)  # 将直方图参数添加到直方图列表

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
            
        返回：
            HistoryBuffer：name的标量历史
        """
        ret = self._history.get(name, None)  # 获取指定名称的历史缓冲区
        if ret is None:  # 如果不存在
            raise KeyError("No history metric available for {}!".format(name))  # 抛出KeyError异常
        return ret  # 返回历史缓冲区

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
            
        返回：
            dict[name -> HistoryBuffer]：所有标量的HistoryBuffer
        """
        return self._history  # 返回所有历史缓冲区

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
                
        返回：
            dict[str -> (float, int)]：从每个标量的名称到其
                最近值和添加它的迭代次数的映射。
        """
        return self._latest_scalars  # 返回最新标量值

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.

        This provides a default behavior that other writers can use.
        
        类似于:meth:`latest`，但返回的值
        要么是未经平滑的原始最新值，
        要么是给定window_size的中值，
        取决于smoothing_hint是否为True。

        这提供了其他写入器可以使用的默认行为。
        """
        result = {}  # 初始化结果字典
        for k, (v, itr) in self._latest_scalars.items():  # 遍历所有最新标量
            result[k] = (
                self._history[k].median(window_size) if self._smoothing_hints[k] else v,  # 根据平滑提示选择值
                itr,  # 迭代次数
            )
        return result  # 返回结果

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
                
        返回：
            dict[name -> bool]：用户提供的关于标量是否
                有噪声并需要平滑的提示。
        """
        return self._smoothing_hints  # 返回所有平滑提示

    def step(self):
        """
        User should either: (1) Call this function to increment storage.iter when needed. Or
        (2) Set `storage.iter` to the correct iteration number before each iteration.

        The storage will then be able to associate the new data with an iteration number.
        
        用户应该：(1) 在需要时调用此函数来增加storage.iter。或者
        (2) 在每次迭代前将`storage.iter`设置为正确的迭代次数。

        然后，存储将能够将新数据与迭代次数相关联。
        """
        self._iter += 1  # 增加迭代次数

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
                
        返回：
            int: 当前迭代次数。当与训练器一起使用时，
                确保与trainer.iter相同。
        """
        return self._iter  # 返回当前迭代次数

    @iter.setter
    def iter(self, val):
        self._iter = int(val)  # 设置迭代次数

    @property
    def iteration(self):
        # for backward compatibility
        # 为了向后兼容
        return self._iter  # 返回当前迭代次数

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)  # 将自身添加到存储栈
        return self  # 返回自身

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self  # 确保自身在栈顶
        _CURRENT_STORAGE_STACK.pop()  # 从栈中移除自身

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
            
        产生：
            一个上下文，在该上下文中添加到此存储的所有事件
            将以名称范围为前缀。
        """
        old_prefix = self._current_prefix  # 保存当前前缀
        self._current_prefix = name.rstrip("/") + "/"  # 设置新前缀
        yield  # 暂停执行，让with块中的代码执行
        self._current_prefix = old_prefix  # 恢复旧前缀

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        
        删除所有存储的可视化图像。应在图像
        写入tensorboard后调用此方法。
        """
        self._vis_data = []  # 清空可视化数据列表

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []
