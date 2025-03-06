# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, checks  # 从 ultralytics.utils 导入 LOGGER、SETTINGS、TESTS_RUNNING 和 checks

try:
    assert not TESTS_RUNNING  # do not log pytest  # 确保不在 pytest 中记录
    assert SETTINGS["dvc"] is True  # verify integration is enabled  # 验证集成是否启用
    import dvclive  # 导入 dvclive 库

    assert checks.check_version("dvclive", "2.11.0", verbose=True)  # 验证 dvclive 版本

    import os  # 导入 os 模块
    import re  # 导入正则表达式模块
    from pathlib import Path  # 从 pathlib 导入 Path 类

    # DVCLive logger instance  # DVCLive 日志记录实例
    live = None  # 初始化 live 为 None
    _processed_plots = {}  # 初始化已处理图表字典

    # [on_fit_epoch_end](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/utils/callbacks/base.py:53:0-55:54) is called on final validation (probably need to be fixed) for now this is the way we  # [on_fit_epoch_end](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/utils/callbacks/base.py:53:0-55:54) 在最终验证时被调用（可能需要修复），这是我们区分最佳模型的最终评估与最后一个周期验证的方法
    # distinguish final evaluation of the best model vs last epoch validation  # 区分最佳模型的最终评估与最后一个周期验证
    _training_epoch = False  # 初始化训练周期标志为 False

except (ImportError, AssertionError, TypeError):  # 捕获导入错误、断言错误和类型错误
    dvclive = None  # 如果导入失败或断言失败，则将 dvclive 设置为 None


def _log_images(path, prefix=""):
    """Logs images at specified path with an optional prefix using DVCLive.  # 使用 DVCLive 在指定路径记录图像，并可选地添加前缀。"""
    if live:  # 如果 live 实例存在
        name = path.name  # 获取文件名

        # Group images by batch to enable sliders in UI  # 按批次分组图像以启用 UI 中的滑块
        if m := re.search(r"_batch(\d+)", name):  # 在文件名中搜索批次编号
            ni = m[1]  # 获取批次编号
            new_stem = re.sub(r"_batch(\d+)", "_batch", path.stem)  # 替换文件名中的批次编号
            name = (Path(new_stem) / ni).with_suffix(path.suffix)  # 生成新的文件名

        live.log_image(os.path.join(prefix, name), path)  # 记录图像


def _log_plots(plots, prefix=""):
    """Logs plot images for training progress if they have not been previously processed.  # 记录训练进度的图像，如果之前未处理过。"""
    for name, params in plots.items():  # 遍历图表字典
        timestamp = params["timestamp"]  # 获取时间戳
        if _processed_plots.get(name) != timestamp:  # 如果图表未处理或时间戳不同
            _log_images(name, prefix)  # 记录图像
            _processed_plots[name] = timestamp  # 更新已处理图表字典


def _log_confusion_matrix(validator):
    """Logs the confusion matrix for the given validator using DVCLive.  # 使用 DVCLive 记录给定验证器的混淆矩阵。"""
    targets = []  # 初始化目标列表
    preds = []  # 初始化预测列表
    matrix = validator.confusion_matrix.matrix  # 获取混淆矩阵
    names = list(validator.names.values())  # 获取类别名称
    if validator.confusion_matrix.task == "detect":  # 如果任务是检测
        names += ["background"]  # 添加背景类别

    for ti, pred in enumerate(matrix.T.astype(int)):  # 遍历混淆矩阵的转置
        for pi, num in enumerate(pred):  # 遍历每个预测
            targets.extend([names[ti]] * num)  # 扩展目标列表
            preds.extend([names[pi]] * num)  # 扩展预测列表

    live.log_sklearn_plot("confusion_matrix", targets, preds, name="cf.json", normalized=True)  # 记录混淆矩阵


def on_pretrain_routine_start(trainer):
    """Initializes DVCLive logger for training metadata during pre-training routine.  # 在预训练例程中初始化 DVCLive 日志记录器以记录训练元数据。"""
    try:
        global live  # 声明全局变量
        live = dvclive.Live(save_dvc_exp=True, cache_images=True)  # 创建 DVCLive 实例
        LOGGER.info("DVCLive is detected and auto logging is enabled (run 'yolo settings dvc=False' to disable).")  # 记录信息
    except Exception as e:  # 捕获异常
        LOGGER.warning(f"WARNING ⚠️ DVCLive installed but not initialized correctly, not logging this run. {e}")  # 记录警告信息


def on_pretrain_routine_end(trainer):
    """Logs plots related to the training process at the end of the pretraining routine.  # 在预训练例程结束时记录与训练过程相关的图表。"""
    _log_plots(trainer.plots, "train")  # 记录训练图表


def on_train_start(trainer):
    """Logs the training parameters if DVCLive logging is active.  # 如果 DVCLive 日志记录处于活动状态，则记录训练参数。"""
    if live:  # 如果 live 实例存在
        live.log_params(trainer.args)  # 记录参数


def on_train_epoch_start(trainer):
    """Sets the global variable _training_epoch value to True at the start of training each epoch.  # 在每个训练周期开始时将全局变量 _training_epoch 的值设置为 True。"""
    global _training_epoch  # 声明全局变量
    _training_epoch = True  # 设置为 True


def on_fit_epoch_end(trainer):
    """Logs training metrics and model info, and advances to next step on the end of each fit epoch.  # 在每个拟合周期结束时记录训练指标和模型信息，并推进到下一步。"""
    global _training_epoch  # 声明全局变量
    if live and _training_epoch:  # 如果 live 实例存在且当前为训练周期
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}  # 获取所有指标
        for metric, value in all_metrics.items():  # 遍历所有指标
            live.log_metric(metric, value)  # 记录指标

        if trainer.epoch == 0:  # 如果是第一个周期
            from ultralytics.utils.torch_utils import model_info_for_loggers  # 从 ultralytics.utils.torch_utils 导入 model_info_for_loggers

            for metric, value in model_info_for_loggers(trainer).items():  # 遍历模型信息
                live.log_metric(metric, value, plot=False)  # 记录模型信息

        _log_plots(trainer.plots, "train")  # 记录训练图表
        _log_plots(trainer.validator.plots, "val")  # 记录验证图表

        live.next_step()  # 进入下一步
        _training_epoch = False  # 设置为 False


def on_train_end(trainer):
    """Logs the best metrics, plots, and confusion matrix at the end of training if DVCLive is active.  # 如果 DVCLive 活动，则在训练结束时记录最佳指标、图表和混淆矩阵。"""
    if live:  # 如果 live 实例存在
        # At the end log the best metrics. It runs validator on the best model internally.  # 在结束时记录最佳指标。它在最佳模型上运行验证器。
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}  # 获取所有指标
        for metric, value in all_metrics.items():  # 遍历所有指标
            live.log_metric(metric, value, plot=False)  # 记录指标

        _log_plots(trainer.plots, "val")  # 记录验证图表
        _log_plots(trainer.validator.plots, "val")  # 记录验证图表
        _log_confusion_matrix(trainer.validator)  # 记录混淆矩阵

        if trainer.best.exists():  # 如果最佳模型存在
            live.log_artifact(trainer.best, copy=True, type="model")  # 记录模型

        live.end()  # 结束日志记录


callbacks = (  # 定义回调函数
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,  # 预训练例程开始时的回调
        "on_pretrain_routine_end": on_pretrain_routine_end,  # 预训练例程结束时的回调
        "on_train_start": on_train_start,  # 训练开始时的回调
        "on_train_epoch_start": on_train_epoch_start,  # 训练周期开始时的回调
        "on_fit_epoch_end": on_fit_epoch_end,  # 拟合周期结束时的回调
        "on_train_end": on_train_end,  # 训练结束时的回调
    }
    if dvclive  # 如果 dvclive 可用
    else {}  # 否则为空字典
)