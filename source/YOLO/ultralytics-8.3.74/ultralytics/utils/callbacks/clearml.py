# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING  # 从 ultralytics.utils 导入 LOGGER、SETTINGS 和 TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # do not log pytest  # 确保不在 pytest 中记录
    assert SETTINGS["clearml"] is True  # verify integration is enabled  # 验证集成是否启用
    import clearml  # 导入 clearml 库
    from clearml import Task  # 从 clearml 导入 Task 类

    assert hasattr(clearml, "__version__")  # verify package is not directory  # 验证包不是目录

except (ImportError, AssertionError):
    clearml = None  # 如果导入失败或断言失败，则将 clearml 设置为 None


def _log_debug_samples(files, title="Debug Samples") -> None:
    """
    Log files (images) as debug samples in the ClearML task.  # 将文件（图像）记录为 ClearML 任务中的调试样本。

    Args:
        files (list): A list of file paths in PosixPath format.  # 文件路径列表，格式为 PosixPath。
        title (str): A title that groups together images with the same values.  # 将具有相同值的图像分组的标题。
    """
    import re  # 导入正则表达式模块

    if task := Task.current_task():  # 获取当前任务
        for f in files:  # 遍历文件列表
            if f.exists():  # 如果文件存在
                it = re.search(r"_batch(\d+)", f.name)  # 在文件名中搜索批次编号
                iteration = int(it.groups()[0]) if it else 0  # 获取批次编号，默认为 0
                task.get_logger().report_image(  # 使用 ClearML 日志记录图像
                    title=title, series=f.name.replace(it.group(), ""), local_path=str(f), iteration=iteration
                )


def _log_plot(title, plot_path) -> None:
    """
    Log an image as a plot in the plot section of ClearML.  # 将图像记录为 ClearML 中图表部分的图形。

    Args:
        title (str): The title of the plot.  # 图表的标题。
        plot_path (str): The path to the saved image file.  # 保存的图像文件路径。
    """
    import matplotlib.image as mpimg  # 从 matplotlib 导入图像模块
    import matplotlib.pyplot as plt  # 从 matplotlib 导入 pyplot 模块

    img = mpimg.imread(plot_path)  # 读取图像文件
    fig = plt.figure()  # 创建一个新的图形
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # 创建坐标轴，不显示刻度
    ax.imshow(img)  # 显示图像

    Task.current_task().get_logger().report_matplotlib_figure(  # 使用 ClearML 日志记录 matplotlib 图形
        title=title, series="", figure=fig, report_interactive=False
    )


def on_pretrain_routine_start(trainer):
    """Runs at start of pretraining routine; initializes and connects/ logs task to ClearML.  # 在预训练例程开始时运行；初始化并连接/记录任务到 ClearML."""
    try:
        if task := Task.current_task():  # 获取当前任务
            # WARNING: make sure the automatic pytorch and matplotlib bindings are disabled!  # 警告：确保自动的 pytorch 和 matplotlib 绑定被禁用！
            # We are logging these plots and model files manually in the integration  # 我们在集成中手动记录这些图表和模型文件
            from clearml.binding.frameworks.pytorch_bind import PatchPyTorchModelIO  # 从 clearml.binding.frameworks.pytorch_bind 导入 PatchPyTorchModelIO
            from clearml.binding.matplotlib_bind import PatchedMatplotlib  # 从 clearml.binding.matplotlib_bind 导入 PatchedMatplotlib

            PatchPyTorchModelIO.update_current_task(None)  # 更新当前任务
            PatchedMatplotlib.update_current_task(None)  # 更新当前任务
        else:
            task = Task.init(  # 初始化任务
                project_name=trainer.args.project or "Ultralytics",  # 项目名称
                task_name=trainer.args.name,  # 任务名称
                tags=["Ultralytics"],  # 标签
                output_uri=True,  # 是否输出 URI
                reuse_last_task_id=False,  # 是否重用最后的任务 ID
                auto_connect_frameworks={"pytorch": False, "matplotlib": False},  # 禁用自动连接的框架
            )
            LOGGER.warning(  # 记录警告信息
                "ClearML Initialized a new task. If you want to run remotely, "
                "please add clearml-init and connect your arguments before initializing YOLO."
            )
        task.connect(vars(trainer.args), name="General")  # 连接任务参数
    except Exception as e:  # 捕获异常
        LOGGER.warning(f"WARNING ⚠️ ClearML installed but not initialized correctly, not logging this run. {e}")  # 记录警告信息


def on_train_epoch_end(trainer):
    """Logs debug samples for the first epoch of YOLO training and report current training progress.  # 记录 YOLO 训练第一个周期的调试样本并报告当前训练进度."""
    if task := Task.current_task():  # 获取当前任务
        # Log debug samples  # 记录调试样本
        if trainer.epoch == 1:  # 如果是第一个周期
            _log_debug_samples(sorted(trainer.save_dir.glob("train_batch*.jpg")), "Mosaic")  # 记录马赛克图像
        # Report the current training progress  # 报告当前训练进度
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():  # 遍历训练损失项
            task.get_logger().report_scalar("train", k, v, iteration=trainer.epoch)  # 记录标量损失
        for k, v in trainer.lr.items():  # 遍历学习率
            task.get_logger().report_scalar("lr", k, v, iteration=trainer.epoch)  # 记录学习率


def on_fit_epoch_end(trainer):
    """Reports model information to logger at the end of an epoch.  # 在每个周期结束时向日志记录模型信息."""
    if task := Task.current_task():  # 获取当前任务
        # You should have access to the validation bboxes under jdict  # 您应该可以访问 jdict 下的验证边界框
        task.get_logger().report_scalar(  # 记录周期时间
            title="Epoch Time", series="Epoch Time", value=trainer.epoch_time, iteration=trainer.epoch
        )
        for k, v in trainer.metrics.items():  # 遍历指标
            task.get_logger().report_scalar("val", k, v, iteration=trainer.epoch)  # 记录验证指标
        if trainer.epoch == 0:  # 如果是第一个周期
            from ultralytics.utils.torch_utils import model_info_for_loggers  # 从 ultralytics.utils.torch_utils 导入 model_info_for_loggers

            for k, v in model_info_for_loggers(trainer).items():  # 遍历模型信息
                task.get_logger().report_single_value(k, v)  # 记录单个值


def on_val_end(validator):
    """Logs validation results including labels and predictions.  # 记录验证结果，包括标签和预测。"""
    if Task.current_task():  # 如果有当前任务
        # Log val_labels and val_pred  # 记录验证标签和预测
        _log_debug_samples(sorted(validator.save_dir.glob("val*.jpg")), "Validation")  # 记录验证图像


def on_train_end(trainer):
    """Logs final model and its name on training completion.  # 在训练完成时记录最终模型及其名称。"""
    if task := Task.current_task():  # 获取当前任务
        # Log final results, CM matrix + PR plots  # 记录最终结果，混淆矩阵 + 精确率-召回率图
        files = [
            "results.png",  # 结果图
            "confusion_matrix.png",  # 混淆矩阵图
            "confusion_matrix_normalized.png",  # 归一化混淆矩阵图
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),  # 精确率、召回率、F1 曲线
        ]
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]  # 过滤存在的文件
        for f in files:  # 遍历文件
            _log_plot(title=f.stem, plot_path=f)  # 记录图像
        # Report final metrics  # 报告最终指标
        for k, v in trainer.validator.metrics.results_dict.items():  # 遍历验证指标
            task.get_logger().report_single_value(k, v)  # 记录单个值
        # Log the final model  # 记录最终模型
        task.update_output_model(model_path=str(trainer.best), model_name=trainer.args.name, auto_delete_file=False)  # 更新输出模型


callbacks = (  # 定义回调函数
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,  # 预训练例程开始时的回调
        "on_train_epoch_end": on_train_epoch_end,  # 训练周期结束时的回调
        "on_fit_epoch_end": on_fit_epoch_end,  # 拟合周期结束时的回调
        "on_val_end": on_val_end,  # 验证结束时的回调
        "on_train_end": on_train_end,  # 训练结束时的回调
    }
    if clearml  # 如果 clearml 可用
    else {}  # 否则为空字典
)