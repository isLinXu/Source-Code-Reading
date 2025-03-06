# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING  # 从 ultralytics.utils 导入 LOGGER、SETTINGS 和 TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # do not log pytest  # 不记录 pytest
    assert SETTINGS["neptune"] is True  # verify integration is enabled  # 验证集成是否启用
    import neptune  # 导入 neptune 模块
    from neptune.types import File  # 从 neptune.types 导入 File 类

    assert hasattr(neptune, "__version__")  # verify package is not directory  # 验证包不是目录

    run = None  # NeptuneAI experiment logger instance  # NeptuneAI 实验日志记录实例

except (ImportError, AssertionError):  # 捕获导入错误和断言错误
    neptune = None  # 如果导入失败，则将 neptune 设置为 None


def _log_scalars(scalars, step=0):
    """Log scalars to the NeptuneAI experiment logger.  # 将标量记录到 NeptuneAI 实验日志记录器。"""
    if run:  # 如果 run 实例存在
        for k, v in scalars.items():  # 遍历标量字典
            run[k].append(value=v, step=step)  # 记录标量值和步骤


def _log_images(imgs_dict, group=""):
    """Log scalars to the NeptuneAI experiment logger.  # 将图像记录到 NeptuneAI 实验日志记录器。"""
    if run:  # 如果 run 实例存在
        for k, v in imgs_dict.items():  # 遍历图像字典
            run[f"{group}/{k}"].upload(File(v))  # 上传图像文件


def _log_plot(title, plot_path):
    """
    Log plots to the NeptuneAI experiment logger.  # 将图表记录到 NeptuneAI 实验日志记录器。

    Args:  # 参数：
        title (str): Title of the plot.  # title (str): 图表的标题。
        plot_path (PosixPath | str): Path to the saved image file.  # plot_path (PosixPath | str): 保存的图像文件的路径。
    """
    import matplotlib.image as mpimg  # 导入 matplotlib.image 作为 mpimg
    import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 作为 plt

    img = mpimg.imread(plot_path)  # 读取图像文件
    fig = plt.figure()  # 创建图形
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # no ticks  # 添加坐标轴，不显示刻度
    ax.imshow(img)  # 显示图像
    run[f"Plots/{title}"].upload(fig)  # 上传图表


def on_pretrain_routine_start(trainer):
    """Callback function called before the training routine starts.  # 在训练例程开始前调用的回调函数。"""
    try:
        global run  # 声明全局变量 run
        run = neptune.init_run(  # 初始化 Neptune 运行
            project=trainer.args.project or "Ultralytics",  # 设置项目名称
            name=trainer.args.name,  # 设置运行名称
            tags=["Ultralytics"],  # 设置标签
        )
        run["Configuration/Hyperparameters"] = {k: "" if v is None else v for k, v in vars(trainer.args).items()}  # 记录超参数配置
    except Exception as e:  # 捕获异常
        LOGGER.warning(f"WARNING ⚠️ NeptuneAI installed but not initialized correctly, not logging this run. {e}")  # 记录警告信息


def on_train_epoch_end(trainer):
    """Callback function called at end of each training epoch.  # 在每个训练周期结束时调用的回调函数。"""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)  # 记录训练损失项
    _log_scalars(trainer.lr, trainer.epoch + 1)  # 记录学习率
    if trainer.epoch == 1:  # 如果是第一个周期
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob("train_batch*.jpg")}, "Mosaic")  # 记录训练批次图像


def on_fit_epoch_end(trainer):
    """Callback function called at end of each fit (train+val) epoch.  # 在每个拟合（训练+验证）周期结束时调用的回调函数。"""
    if run and trainer.epoch == 0:  # 如果 run 实例存在且当前为第一个周期
        from ultralytics.utils.torch_utils import model_info_for_loggers  # 从 ultralytics.utils.torch_utils 导入 model_info_for_loggers

        run["Configuration/Model"] = model_info_for_loggers(trainer)  # 记录模型信息
    _log_scalars(trainer.metrics, trainer.epoch + 1)  # 记录当前指标


def on_val_end(validator):
    """Callback function called at end of each validation.  # 在每次验证结束时调用的回调函数。"""
    if run:  # 如果 run 实例存在
        # Log val_labels and val_pred  # 记录 val_labels 和 val_pred
        _log_images({f.stem: str(f) for f in validator.save_dir.glob("val*.jpg")}, "Validation")  # 记录验证图像


def on_train_end(trainer):
    """Callback function called at end of training.  # 在训练结束时调用的回调函数。"""
    if run:  # 如果 run 实例存在
        # Log final results, CM matrix + PR plots  # 记录最终结果，混淆矩阵 + PR 图
        files = [  # 定义要记录的文件列表
            "results.png",  # 结果图
            "confusion_matrix.png",  # 混淆矩阵图
            "confusion_matrix_normalized.png",  # 归一化混淆矩阵图
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),  # F1、PR、P、R 曲线图
        ]
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]  # 过滤存在的文件
        for f in files:  # 遍历文件列表
            _log_plot(title=f.stem, plot_path=f)  # 记录图表
        # Log the final model  # 记录最终模型
        run[f"weights/{trainer.args.name or trainer.args.task}/{trainer.best.name}"].upload(File(str(trainer.best)))  # 上传最佳模型


callbacks = (  # 定义回调函数
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,  # 预训练例程开始时的回调
        "on_train_epoch_end": on_train_epoch_end,  # 训练周期结束时的回调
        "on_fit_epoch_end": on_fit_epoch_end,  # 拟合周期结束时的回调
        "on_val_end": on_val_end,  # 验证结束时的回调
        "on_train_end": on_train_end,  # 训练结束时的回调
    }
    if neptune  # 如果 neptune 可用
    else {}
)