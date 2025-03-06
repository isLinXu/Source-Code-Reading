def is_parallel(model):  # 检查模型是否为 DP 或 DDP 类型
    """Returns True if model is of type DP or DDP."""  # 如果模型是 DP 或 DDP 类型则返回 True
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))  # 判断模型是否为 DataParallel 或 DistributedDataParallel 类型


def de_parallel(model):  # 定义去并行化模型的函数
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""  # 去除模型的并行化：如果模型是 DP 或 DDP 类型，则返回单 GPU 模型
    return model.module if is_parallel(model) else model  # 如果模型是并行的，则返回其模块，否则返回模型本身


def one_cycle(y1=0.0, y2=1.0, steps=100):  # 定义一个从 y1 到 y2 的正弦波形函数
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""  # 返回一个从 y1 到 y2 的正弦波形的 lambda 函数
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1  # 返回正弦波形的计算公式


def init_seeds(seed=0, deterministic=False):  # 初始化随机数生成器的种子
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""  # 初始化随机数生成器（RNG）种子
    random.seed(seed)  # 设置 Python 随机数生成器的种子
    np.random.seed(seed)  # 设置 NumPy 随机数生成器的种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机数生成器的种子
    torch.cuda.manual_seed(seed)  # 设置当前 GPU 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机数种子，确保异常安全
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:  # 如果需要确定性训练
        if TORCH_2_0:  # 如果 PyTorch 版本为 2.0
            torch.use_deterministic_algorithms(True, warn_only=True)  # 仅在不可能实现确定性时发出警告
            torch.backends.cudnn.deterministic = True  # 设置 cuDNN 为确定性模式
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 设置 CUBLAS 工作空间配置
            os.environ["PYTHONHASHSEED"] = str(seed)  # 设置 Python 哈希种子
        else:  # 如果 PyTorch 版本低于 2.0
            LOGGER.warning("WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.")  # 警告用户升级 PyTorch 版本
    else:  # 如果不需要确定性训练
        unset_deterministic()  # 取消设置确定性训练的配置


def unset_deterministic():  # 定义取消确定性训练设置的函数
    """Unsets all the configurations applied for deterministic training."""  # 取消所有应用于确定性训练的配置
    torch.use_deterministic_algorithms(False)  # 关闭确定性算法
    torch.backends.cudnn.deterministic = False  # 设置 cuDNN 为非确定性模式
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)  # 移除 CUBLAS 工作空间配置
    os.environ.pop("PYTHONHASHSEED", None)  # 移除 Python 哈希种子


class ModelEMA:  # 定义模型的更新指数移动平均类
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models. Keeps a moving
    average of everything in the model state_dict (parameters and buffers).

    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To disable EMA set the `enabled` attribute to `False`.
    """  # 更新的指数移动平均（EMA）类，保持模型状态字典（参数和缓冲区）的移动平均

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):  # 初始化 EMA 类
        """Initialize EMA for 'model' with given arguments."""  # 使用给定参数初始化模型的 EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA  # 深拷贝去并行化模型并设置为评估模式
        self.updates = updates  # number of EMA updates  # EMA 更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)  # 指数衰减函数（帮助早期训练）
        for p in self.ema.parameters():  # 遍历 EMA 模型的参数
            p.requires_grad_(False)  # 不计算 EMA 模型参数的梯度
        self.enabled = True  # 设置 EMA 为启用状态

    def update(self, model):  # 定义更新 EMA 参数的函数
        """Update EMA parameters."""  # 更新 EMA 参数
        if self.enabled:  # 如果 EMA 被启用
            self.updates += 1  # 增加 EMA 更新次数
            d = self.decay(self.updates)  # 计算当前更新的衰减值

            msd = de_parallel(model).state_dict()  # model state_dict  # 获取去并行化模型的状态字典
            for k, v in self.ema.state_dict().items():  # 遍历 EMA 模型的状态字典
                if v.dtype.is_floating_point:  # true for FP16 and FP32  # 如果参数是浮点类型（FP16 或 FP32）
                    v *= d  # 更新 EMA 参数
                    v += (1 - d) * msd[k].detach()  # 更新 EMA 参数为当前模型参数的加权平均
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):  # 定义更新模型属性的函数
        """Updates attributes and saves stripped model with optimizer removed."""  # 更新属性并保存去除优化器的模型
        if self.enabled:  # 如果 EMA 被启用
            copy_attr(self.ema, model, include, exclude)  # 复制模型属性到 EMA 模型


def strip_optimizer(f: Union[str, Path] = "best.pt", s: str = "", updates: dict = None) -> dict:  # 定义去除优化器的函数
    """Strip optimizer from 'f' to finalize training, optionally save as 's'."""  # 从 'f' 中去除优化器以完成训练，可选择保存为 's'
    """
    Args:
        f (str): file path to model to strip the optimizer from. Default is 'best.pt'.  # 要去除优化器的模型文件路径，默认为 'best.pt'
        s (str): file path to save the model with stripped optimizer to. If not provided, 'f' will be overwritten.  # 保存去除优化器后的模型文件路径，如果未提供，则覆盖 'f'
        updates (dict): a dictionary of updates to overlay onto the checkpoint before saving.  # 在保存之前要叠加到检查点上的更新字典

    Returns:
        (dict): The combined checkpoint dictionary.  # 返回合并后的检查点字典
    
    Example:
        ```python
        from pathlib import Path
        from ultralytics.utils.torch_utils import strip_optimizer

        for f in Path("path/to/model/checkpoints").rglob("*.pt"):
            strip_optimizer(f)
        ```

    Note:
        Use `ultralytics.nn.torch_safe_load` for missing modules with `x = torch_safe_load(f)[0]`
    """  # 示例：使用 strip_optimizer 函数去除优化器
    try:
        x = torch.load(f, map_location=torch.device("cpu"))  # 加载模型检查点
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"  # 确保检查点是字典类型
        assert "model" in x, "'model' missing from checkpoint"  # 确保检查点中包含模型
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ Skipping {f}, not a valid Ultralytics model: {e}")  # 警告用户跳过无效的模型
        return {}  # 返回空字典

    metadata = {
        "date": datetime.now().isoformat(),  # 当前日期
        "version": __version__,  # 模型版本
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",  # 许可证信息
        "docs": "https://docs.ultralytics.com",  # 文档链接
    }

    # Update model
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA  # 用 EMA 替换模型
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # convert from IterableSimpleNamespace to dict  # 将参数从 IterableSimpleNamespace 转换为字典
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # strip loss criterion  # 去除损失标准
    x["model"].half()  # to FP16  # 转换模型为 FP16
    for p in x["model"].parameters():
        p.requires_grad = False  # 不计算模型参数的梯度

    # Update other keys
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # combine args  # 合并参数
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None  # 将优化器、最佳适应度、EMA 和更新次数设置为 None
    x["epoch"] = -1  # 设置 epoch 为 -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys  # 去除非默认键
    # x['model'].args = x['train_args']  # 更新模型参数

    # Save
    combined = {**metadata, **x, **(updates or {})}  # combine dicts (prefer to the right)  # 合并字典（优先右侧的值）
    torch.save(combined, s or f)  # combine dicts (prefer to the right)
    mb = os.path.getsize(s or f) / 1e6  # file size  # 获取文件大小
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")  # 记录去除优化器的操作
    return combined  # 返回合并后的字典


def convert_optimizer_state_dict_to_fp16(state_dict):  # 定义将优化器状态字典转换为 FP16 的函数
    """Converts the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions."""  # 将给定优化器的状态字典转换为 FP16，专注于 'state' 键的张量转换
    for state in state_dict["state"].values():  # 遍历状态字典中的状态
        for k, v in state.items():  # 遍历状态中的每个键值对
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:  # 如果不是步长且是浮点张量
                state[k] = v.half()  # 转换为 FP16

    return state_dict  # 返回转换后的状态字典


@contextmanager
def cuda_memory_usage(device=None):  # 定义 CUDA 内存使用情况监控的上下文管理器
    """Monitor and manage CUDA memory usage."""  # 监控和管理 CUDA 内存使用情况
    cuda_info = dict(memory=0)  # 初始化 CUDA 内存信息字典
    if torch.cuda.is_available():  # 如果 CUDA 可用
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        try:
            yield cuda_info  # 生成 CUDA 内存信息字典
        finally:
            cuda_info["memory"] = torch.cuda.memory_reserved(device)  # 更新 CUDA 内存信息
    else:
        yield cuda_info  # 如果 CUDA 不可用，直接生成 CUDA 内存信息字典


def profile(input, ops, n=10, device=None, max_num_obj=0):  # 定义性能分析函数
    """Ultralytics speed, memory and FLOPs profiler."""  # Ultralytics 的速度、内存和 FLOPs 性能分析器
    """
    Example:
        ```python
        from ultralytics.utils.torch_utils import profile

        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
        ```
    """  # 示例：使用 profile 函数进行性能分析
    results = []  # 初始化结果列表
    if not isinstance(device, torch.device):  # 如果 device 不是 torch.device 类型
        device = select_device(device)  # 选择设备
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )  # 打印表头
    gc.collect()  # 尝试释放未使用的内存
    torch.cuda.empty_cache()  # 清空 CUDA 缓存
    for x in input if isinstance(input, list) else [input]:  # 遍历输入
        x = x.to(device)  # 将输入移至指定设备
        x.requires_grad = True  # 使输入需要梯度
        for m in ops if isinstance(ops, list) else [ops]:  # 遍历操作
            m = m.to(device) if hasattr(m, "to") else m  # device  # 将操作移至指定设备
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # 如果输入为 FP16 则转换操作为 FP16
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward  # 初始化前向和后向时间
            try:
                flops = thop.profile(deepcopy(m), inputs=[x], verbose=False)[0] / 1e9 * 2  # GFLOPs  # 计算 GFLOPs
            except Exception:
                flops = 0  # 如果计算失败则将 flops 设置为 0

            try:
                mem = 0  # 初始化内存
                for _ in range(n):  # 循环 n 次
                    with cuda_memory_usage(device) as cuda_info:  # 监控 CUDA 内存使用情况
                        t[0] = time_sync()  # 记录前向开始时间
                        y = m(x)  # 执行操作
                        t[1] = time_sync()  # 记录前向结束时间
                        try:
                            (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()  # 反向传播
                            t[2] = time_sync()  # 记录后向结束时间
                        except Exception:  # no backward method
                            # print(e)  # for debug
                            t[2] = float("nan")  # 如果没有反向传播方法，则设置时间为 NaN
                    mem += cuda_info["memory"] / 1e9  # (GB)  # 更新内存使用情况
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward  # 计算每次前向的时间
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward  # 计算每次后向的时间
                    if max_num_obj:  # simulate training with predictions per image grid (for AutoBatch)  # 模拟训练以每个图像网格的预测（用于 AutoBatch）
                        with cuda_memory_usage(device) as cuda_info:  # 监控 CUDA 内存使用情况
                            torch.randn(
                                x.shape[0],
                                max_num_obj,
                                int(sum((x.shape[-1] / s) * (x.shape[-2] / s) for s in m.stride.tolist())),
                                device=device,
                                dtype=torch.float32,
                            )  # 创建随机张量以模拟训练
                        mem += cuda_info["memory"] / 1e9  # (GB)  # 更新内存使用情况
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # 获取输入和输出的形状
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # 获取参数数量
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")  # 记录性能信息
                results.append([p, flops, mem, tf, tb, s_in, s_out])  # 将结果添加到结果列表
            except Exception as e:
                LOGGER.info(e)  # 记录异常信息
                results.append(None)  # 添加 None 到结果列表
            finally:
                gc.collect()  # 尝试释放未使用的内存
                torch.cuda.empty_cache()  # 清空 CUDA 缓存
    return results  # 返回结果列表


class EarlyStopping:  # 定义早停类
    """Early stopping class that stops training when a specified number of epochs have passed without improvement."""  # 早停类，当指定的 epoch 数量没有改进时停止训练

    def __init__(self, patience=50):  # 初始化早停对象
        """Initialize early stopping object."""  # 初始化早停对象
        """
        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.  # 等待的 epoch 数量
        """  # 参数说明
        self.best_fitness = 0.0  # i.e. mAP  # 最佳适应度
        self.best_epoch = 0  # 最佳 epoch
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop  # 等待的 epoch 数量
        self.possible_stop = False  # possible stop may occur next epoch  # 可能在下一个 epoch 停止

    def __call__(self, epoch, fitness):  # 定义调用方法
        """Check whether to stop training."""  # 检查是否停止训练
        """
        Args:
            epoch (int): Current epoch of training  # 当前训练的 epoch
            fitness (float): Fitness value of current epoch  # 当前 epoch 的适应度值

        Returns:
            (bool): True if training should stop, False otherwise  # 如果应该停止训练则返回 True，否则返回 False
        """  # 返回值说明
        if fitness is None:  # check if fitness=None (happens when val=False)  # 检查适应度是否为 None（当 val=False 时会发生）
            return False  # 如果适应度为 None，返回 False

        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training  # 允许在训练的早期阶段适应度为零
            self.best_epoch = epoch  # 更新最佳 epoch
            self.best_fitness = fitness  # 更新最佳适应度
        delta = epoch - self.best_epoch  # epochs without improvement  # 没有改进的 epoch 数量
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch  # 可能在下一个 epoch 停止
        stop = delta >= self.patience  # stop training if patience exceeded  # 如果超过耐心值则停止训练
        if stop:  # 如果决定停止训练
            prefix = colorstr("EarlyStopping: ")  # 设置前缀
            LOGGER.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "  # 记录停止训练的原因
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"  # 记录最佳模型信息
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "  # 记录更新早停的方式
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )  # 记录更新早停的方式
        return stop  # 返回停止训练的标志


class FXModel(nn.Module):  # 定义 FXModel 类
    """
    A custom model class for torch.fx compatibility.

    This class extends `torch.nn.Module` and is designed to ensure compatibility with torch.fx for tracing and graph manipulation.
    It copies attributes from an existing model and explicitly sets the model attribute to ensure proper copying.

    Args:
        model (torch.nn.Module): The original model to wrap for torch.fx compatibility.
    """  # 自定义模型类，确保与 torch.fx 的兼容性

    def __init__(self, model):  # 初始化 FXModel 类
        """Initialize the FXModel."""  # 初始化 FXModel
        """
        Args:
            model (torch.nn.Module): The original model to wrap for torch.fx compatibility.
        """  # 参数说明
        super().__init__()  # 调用父类初始化
        copy_attr(self, model)  # 复制模型属性
        # Explicitly set `model` since `copy_attr` somehow does not copy it.  # 显式设置 `model` 属性，因为 `copy_attr` 可能未能复制它
        self.model = model.model  # 设置模型属性

    def forward(self, x):  # 定义前向传播方法
        """Forward pass through the model."""  # 前向传播方法
        """
        This method performs the forward pass through the model, handling the dependencies between layers and saving intermediate outputs.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The output tensor from the model.
        """  # 返回值说明
        y = []  # outputs  # 初始化输出列表
        for m in self.model:  # 遍历模型中的每个层
            if m.f != -1:  # if not from previous layer  # 如果不是来自上一层
                # from earlier layers  # 从早期层获取输出
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 获取当前层的输入
            x = m(x)  # run  # 执行当前层
            y.append(x)  # save output  # 保存输出
        return x  # 返回最终输出
