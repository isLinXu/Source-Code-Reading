# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir  # 从 ultralytics.cfg 导入相关配置
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks  # 从 ultralytics.utils 导入默认配置、日志记录器、线程数和检查工具


def run_ray_tune(  # 定义 run_ray_tune 函数
    model,  # 要调优的模型
    space: dict = None,  # 超参数搜索空间，默认为 None
    grace_period: int = 10,  # ASHA 调度器的宽限期，默认为 10
    gpu_per_trial: int = None,  # 每个试验分配的 GPU 数量，默认为 None
    max_samples: int = 10,  # 最大试验次数，默认为 10
    **train_args,  # 其他传递给 train() 方法的参数
):
    """
    Runs hyperparameter tuning using Ray Tune.  # 使用 Ray Tune 进行超参数调优

    Args:  # 参数说明
        model (YOLO): Model to run the tuner on.  # 要调优的模型
        space (dict, optional): The hyperparameter search space. Defaults to None.  # 超参数搜索空间，默认为 None
        grace_period (int, optional): The grace period in epochs of the ASHA scheduler. Defaults to 10.  # ASHA 调度器的宽限期，默认为 10
        gpu_per_trial (int, optional): The number of GPUs to allocate per trial. Defaults to None.  # 每个试验分配的 GPU 数量，默认为 None
        max_samples (int, optional): The maximum number of trials to run. Defaults to 10.  # 最大试验次数，默认为 10
        train_args (dict, optional): Additional arguments to pass to the `train()` method. Defaults to {}.  # 额外参数，默认为空字典

    Returns:  # 返回值说明
        (dict): A dictionary containing the results of the hyperparameter search.  # 包含超参数搜索结果的字典

    Example:  # 示例
        ```python
        from ultralytics import YOLO

        # Load a YOLO11n model
        model = YOLO("yolo11n.pt")

        # Start tuning hyperparameters for YOLO11n training on the COCO8 dataset
        result_grid = model.tune(data="coco8.yaml", use_ray=True)
        ```
    """
    LOGGER.info("💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune")  # 记录 RayTune 的学习链接
    if train_args is None:  # 如果没有提供训练参数
        train_args = {}  # 初始化为空字典

    try:
        checks.check_requirements("ray[tune]")  # 检查是否满足 ray[tune] 的要求

        import ray  # 导入 ray
        from ray import tune  # 从 ray 导入 tune
        from ray.air import RunConfig  # 从 ray.air 导入 RunConfig
        from ray.air.integrations.wandb import WandbLoggerCallback  # 从 ray.air.integrations.wandb 导入 WandbLoggerCallback
        from ray.tune.schedulers import ASHAScheduler  # 从 ray.tune.schedulers 导入 ASHAScheduler
    except ImportError:  # 如果导入失败
        raise ModuleNotFoundError('Ray Tune required but not found. To install run: pip install "ray[tune]"')  # 抛出模块未找到异常

    try:
        import wandb  # 尝试导入 wandb

        assert hasattr(wandb, "__version__")  # 确保 wandb 有版本属性
    except (ImportError, AssertionError):  # 如果导入失败或没有版本属性
        wandb = False  # 将 wandb 设置为 False

    checks.check_version(ray.__version__, ">=2.0.0", "ray")  # 检查 ray 的版本
    default_space = {  # 定义默认的超参数搜索空间
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": tune.uniform(1e-5, 1e-1),  # 学习率范围
        "lrf": tune.uniform(0.01, 1.0),  # 最终 OneCycleLR 学习率 (lr0 * lrf)
        "momentum": tune.uniform(0.6, 0.98),  # SGD 动量/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # 优化器权重衰减
        "warmup_epochs": tune.uniform(0.0, 5.0),  # 预热周期
        "warmup_momentum": tune.uniform(0.0, 0.95),  # 预热初始动量
        "box": tune.uniform(0.02, 0.2),  # box 损失增益
        "cls": tune.uniform(0.2, 4.0),  # cls 损失增益
        "hsv_h": tune.uniform(0.0, 0.1),  # 图像 HSV-Hue 增强
        "hsv_s": tune.uniform(0.0, 0.9),  # 图像 HSV-Saturation 增强
        "hsv_v": tune.uniform(0.0, 0.9),  # 图像 HSV-Value 增强
        "degrees": tune.uniform(0.0, 45.0),  # 图像旋转范围
        "translate": tune.uniform(0.0, 0.9),  # 图像平移范围
        "scale": tune.uniform(0.0, 0.9),  # 图像缩放范围
        "shear": tune.uniform(0.0, 10.0),  # 图像剪切范围
        "perspective": tune.uniform(0.0, 0.001),  # 图像透视范围
        "flipud": tune.uniform(0.0, 1.0),  # 图像上下翻转概率
        "fliplr": tune.uniform(0.0, 1.0),  # 图像左右翻转概率
        "bgr": tune.uniform(0.0, 1.0),  # 图像通道 BGR 概率
        "mosaic": tune.uniform(0.0, 1.0),  # 图像混合概率
        "mixup": tune.uniform(0.0, 1.0),  # 图像混合概率
        "copy_paste": tune.uniform(0.0, 1.0),  # 段落复制粘贴概率
    }

    # Put the model in ray store  # 将模型放入 ray 存储
    task = model.task  # 获取模型任务
    model_in_store = ray.put(model)  # 将模型放入 ray 存储

    def _tune(config):  # 定义调优函数
        """
        Trains the YOLO model with the specified hyperparameters and additional arguments.  # 使用指定的超参数和额外参数训练 YOLO 模型

        Args:  # 参数说明
            config (dict): A dictionary of hyperparameters to use for training.  # 用于训练的超参数字典

        Returns:  # 返回值说明
            None  # 无返回值
        """
        model_to_train = ray.get(model_in_store)  # 从 ray 存储中获取模型进行调优
        model_to_train.reset_callbacks()  # 重置回调
        config.update(train_args)  # 更新配置
        results = model_to_train.train(**config)  # 训练模型并获取结果
        return results.results_dict  # 返回结果字典

    # Get search space  # 获取搜索空间
    if not space:  # 如果没有提供搜索空间
        space = default_space  # 使用默认搜索空间
        LOGGER.warning("WARNING ⚠️ search space not provided, using default search space.")  # 记录警告

    # Get dataset  # 获取数据集
    data = train_args.get("data", TASK2DATA[task])  # 从训练参数中获取数据，默认为任务对应的数据
    space["data"] = data  # 将数据添加到搜索空间
    if "data" not in train_args:  # 如果训练参数中没有数据
        LOGGER.warning(f'WARNING ⚠️ data not provided, using default "data={data}".')  # 记录警告

    # Define the trainable function with allocated resources  # 定义可训练函数及其资源
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})  # 分配 CPU 和 GPU 资源

    # Define the ASHA scheduler for hyperparameter search  # 定义 ASHA 调度器进行超参数搜索
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",  # 时间属性
        metric=TASK2METRIC[task],  # 任务对应的度量
        mode="max",  # 最大化模式
        max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,  # 最大训练周期
        grace_period=grace_period,  # 宽限期
        reduction_factor=3,  # 降低因子
    )

    # Define the callbacks for the hyperparameter search  # 定义超参数搜索的回调
    tuner_callbacks = [WandbLoggerCallback(project="YOLOv8-tune")] if wandb else []  # 如果 wandb 可用，则添加回调

    # Create the Ray Tune hyperparameter search tuner  # 创建 Ray Tune 超参数搜索调优器
    tune_dir = get_save_dir(
        get_cfg(DEFAULT_CFG, train_args), name=train_args.pop("name", "tune")  # 获取保存目录
    ).resolve()  # 必须是绝对目录
    tune_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    tuner = tune.Tuner(
        trainable_with_resources,  # 可训练函数
        param_space=space,  # 参数空间
        tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),  # 调优配置
        run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir),  # 运行配置
    )

    # Run the hyperparameter search  # 运行超参数搜索
    tuner.fit()  # 进行调优

    # Get the results of the hyperparameter search  # 获取超参数搜索结果
    results = tuner.get_results()  # 获取结果

    # Shut down Ray to clean up workers  # 关闭 Ray 清理工作者
    ray.shutdown()  # 关闭 Ray

    return results  # 返回结果