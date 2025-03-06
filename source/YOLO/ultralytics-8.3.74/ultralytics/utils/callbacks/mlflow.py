# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MLflow Logging for Ultralytics YOLO.  # Ultralytics YOLO 的 MLflow 日志记录。

This module enables MLflow logging for Ultralytics YOLO. It logs metrics, parameters, and model artifacts.  # 此模块启用 Ultralytics YOLO 的 MLflow 日志记录。它记录指标、参数和模型工件。
For setting up, a tracking URI should be specified. The logging can be customized using environment variables.  # 设置时，应指定跟踪 URI。日志记录可以使用环境变量进行自定义。

Commands:  # 命令：
    1. To set a project name:  # 设置项目名称：
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument  # `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` 或使用 project=<project> 参数

    2. To set a run name:  # 设置运行名称：
        `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument  # `export MLFLOW_RUN=<your_run_name>` 或使用 name=<name> 参数

    3. To start a local MLflow server:  # 启动本地 MLflow 服务器：
        mlflow server --backend-store-uri runs/mlflow  # mlflow server --backend-store-uri runs/mlflow
       It will by default start a local server at http://127.0.0.1:5000.  # 默认情况下，它将在 http://127.0.0.1:5000 启动本地服务器。
       To specify a different URI, set the MLFLOW_TRACKING_URI environment variable.  # 要指定不同的 URI，请设置 MLFLOW_TRACKING_URI 环境变量。

    4. To kill all running MLflow server instances:  # 杀死所有正在运行的 MLflow 服务器实例：
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9  # ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr  # 从 ultralytics.utils 导入 LOGGER、RUNS_DIR、SETTINGS、TESTS_RUNNING 和 colorstr

try:
    import os  # 导入 os 模块

    assert not TESTS_RUNNING or "test_mlflow" in os.environ.get("PYTEST_CURRENT_TEST", "")  # do not log pytest  # 不记录 pytest
    assert SETTINGS["mlflow"] is True  # verify integration is enabled  # 验证集成是否启用
    import mlflow  # 导入 mlflow 模块

    assert hasattr(mlflow, "__version__")  # verify package is not directory  # 验证包不是目录
    from pathlib import Path  # 从 pathlib 导入 Path 类

    PREFIX = colorstr("MLflow: ")  # 设置前缀为 "MLflow: "

except (ImportError, AssertionError):  # 捕获导入错误和断言错误
    mlflow = None  # 如果导入失败，则将 mlflow 设置为 None


def sanitize_dict(x):
    """Sanitize dictionary keys by removing parentheses and converting values to floats.  # 清理字典键，移除括号并将值转换为浮点数。"""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}  # 返回清理后的字典


def on_pretrain_routine_end(trainer):
    """
    Log training parameters to MLflow at the end of the pretraining routine.  # 在预训练例程结束时将训练参数记录到 MLflow。

    This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,  # 此函数根据环境变量和训练参数设置 MLflow 日志记录。它设置跟踪 URI，
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters  # 实验名称和运行名称，然后在未激活的情况下启动 MLflow 运行。最后记录参数
    from the trainer.  # 从训练器记录参数。

    Args:  # 参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.  # trainer (ultralytics.engine.trainer.BaseTrainer): 包含要记录的参数和参数的训练对象。

    Global:  # 全局变量：
        mlflow: The imported mlflow module to use for logging.  # mlflow: 用于记录的导入的 mlflow 模块。

    Environment Variables:  # 环境变量：
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.  # MLFLOW_TRACKING_URI: MLflow 跟踪的 URI。如果未设置，则默认为 'runs/mlflow'。
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.  # MLFLOW_EXPERIMENT_NAME: MLflow 实验的名称。如果未设置，则默认为 trainer.args.project。
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.  # MLFLOW_RUN: MLflow 运行的名称。如果未设置，则默认为 trainer.args.name。
        MLFLOW_KEEP_RUN_ACTIVE: Boolean indicating whether to keep the MLflow run active after the end of training.  # MLFLOW_KEEP_RUN_ACTIVE: 布尔值，指示在训练结束后是否保持 MLflow 运行处于活动状态。
    """
    global mlflow  # 声明全局变量 mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")  # 获取 MLflow 跟踪 URI，如果未设置则使用默认值
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")  # 记录跟踪 URI
    mlflow.set_tracking_uri(uri)  # 设置 MLflow 跟踪 URI

    # Set experiment and run names  # 设置实验和运行名称
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/Ultralytics"  # 获取实验名称
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name  # 获取运行名称
    mlflow.set_experiment(experiment_name)  # 设置实验

    mlflow.autolog()  # 启用自动日志记录
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)  # 获取当前活动运行或启动新的运行
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")  # 记录运行 ID
        if Path(uri).is_dir():  # 如果 URI 是目录
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")  # 记录查看链接
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")  # 记录禁用提示
        mlflow.log_params(dict(trainer.args))  # 记录训练参数
    except Exception as e:  # 捕获异常
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n{PREFIX}WARNING ⚠️ Not tracking this run")  # 记录警告信息


def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow.  # 在每个训练周期结束时将训练指标记录到 MLflow。"""
    if mlflow:  # 如果 mlflow 可用
        mlflow.log_metrics(  # 记录指标
            metrics={
                **sanitize_dict(trainer.lr),  # 记录学习率
                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),  # 记录训练损失项
            },
            step=trainer.epoch,  # 记录当前周期
        )


def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow.  # 在每个拟合周期结束时将训练指标记录到 MLflow。"""
    if mlflow:  # 如果 mlflow 可用
        mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch)  # 记录当前指标


def on_train_end(trainer):
    """Log model artifacts at the end of the training.  # 在训练结束时记录模型工件。"""
    if not mlflow:  # 如果 mlflow 不可用
        return  # 直接返回
    mlflow.log_artifact(str(trainer.best.parent))  # 记录保存目录，包含 best.pt 和 last.pt
    for f in trainer.save_dir.glob("*"):  # 记录保存目录中的所有其他文件
        if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:  # 如果文件扩展名符合条件
            mlflow.log_artifact(str(f))  # 记录文件
    keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"  # 获取是否保持运行活跃的设置
    if keep_run_active:  # 如果保持运行活跃
        LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")  # 记录保持活跃提示
    else:
        mlflow.end_run()  # 结束 mlflow 运行
        LOGGER.debug(f"{PREFIX}mlflow run ended")  # 记录结束信息

    LOGGER.info(  # 记录结果信息
        f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n{PREFIX}disable with 'yolo settings mlflow=False'"  # 记录结果 URI
    )


callbacks = (  # 定义回调函数
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,  # 预训练例程结束时的回调
        "on_train_epoch_end": on_train_epoch_end,  # 训练周期结束时的回调
        "on_fit_epoch_end": on_fit_epoch_end,  # 拟合周期结束时的回调
        "on_train_end": on_train_end,  # 训练结束时的回调
    }
    if mlflow  # 如果 mlflow 可用
    else {}
)  # 验证启用