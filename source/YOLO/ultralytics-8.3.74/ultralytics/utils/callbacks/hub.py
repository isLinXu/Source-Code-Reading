# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import json  # 导入 json 模块
from time import time  # 从 time 模块导入 time 函数

from ultralytics.hub import HUB_WEB_ROOT, PREFIX, HUBTrainingSession, events  # 从 ultralytics.hub 导入相关组件
from ultralytics.utils import LOGGER, RANK, SETTINGS  # 从 ultralytics.utils 导入 LOGGER、RANK 和 SETTINGS


def on_pretrain_routine_start(trainer):
    """Create a remote Ultralytics HUB session to log local model training.  # 创建一个远程 Ultralytics HUB 会话以记录本地模型训练。"""
    if RANK in {-1, 0} and SETTINGS["hub"] is True and SETTINGS["api_key"] and trainer.hub_session is None:  # 如果当前进程是主进程且启用了 HUB 集成且有 API 密钥且没有现有会话
        trainer.hub_session = HUBTrainingSession.create_session(trainer.args.model, trainer.args)  # 创建 HUBTrainingSession 实例


def on_pretrain_routine_end(trainer):
    """Logs info before starting timer for upload rate limit.  # 在开始上传速率限制计时器之前记录信息。"""
    if session := getattr(trainer, "hub_session", None):  # 获取当前训练会话
        # Start timer for upload rate limit  # 启动上传速率限制计时器
        session.timers = {"metrics": time(), "ckpt": time()}  # 在 session.rate_limit 上启动计时器


def on_fit_epoch_end(trainer):
    """Uploads training progress metrics at the end of each epoch.  # 在每个周期结束时上传训练进度指标。"""
    if session := getattr(trainer, "hub_session", None):  # 获取当前训练会话
        # Upload metrics after val end  # 在验证结束后上传指标
        all_plots = {  # 收集所有指标
            **trainer.label_loss_items(trainer.tloss, prefix="train"),  # 获取训练损失项
            **trainer.metrics,  # 获取其他指标
        }
        if trainer.epoch == 0:  # 如果是第一个周期
            from ultralytics.utils.torch_utils import model_info_for_loggers  # 从 ultralytics.utils.torch_utils 导入 model_info_for_loggers

            all_plots = {**all_plots, **model_info_for_loggers(trainer)}  # 添加模型信息到指标

        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)  # 将指标转换为 JSON 格式并存入队列

        # If any metrics fail to upload, add them to the queue to attempt uploading again.  # 如果任何指标上传失败，将其添加到队列以尝试再次上传。
        if session.metrics_upload_failed_queue:  # 如果有上传失败的指标
            session.metrics_queue.update(session.metrics_upload_failed_queue)  # 更新指标队列

        if time() - session.timers["metrics"] > session.rate_limits["metrics"]:  # 如果超过速率限制
            session.upload_metrics()  # 上传指标
            session.timers["metrics"] = time()  # 重置计时器
            session.metrics_queue = {}  # 重置队列


def on_model_save(trainer):
    """Saves checkpoints to Ultralytics HUB with rate limiting.  # 以速率限制将检查点保存到 Ultralytics HUB。"""
    if session := getattr(trainer, "hub_session", None):  # 获取当前训练会话
        # Upload checkpoints with rate limiting  # 以速率限制上传检查点
        is_best = trainer.best_fitness == trainer.fitness  # 判断当前模型是否为最佳模型
        if time() - session.timers["ckpt"] > session.rate_limits["ckpt"]:  # 如果超过速率限制
            LOGGER.info(f"{PREFIX}Uploading checkpoint {HUB_WEB_ROOT}/models/{session.model.id}")  # 记录上传信息
            session.upload_model(trainer.epoch, trainer.last, is_best)  # 上传模型
            session.timers["ckpt"] = time()  # 重置计时器


def on_train_end(trainer):
    """Upload final model and metrics to Ultralytics HUB at the end of training.  # 在训练结束时将最终模型和指标上传到 Ultralytics HUB。"""
    if session := getattr(trainer, "hub_session", None):  # 获取当前训练会话
        # Upload final model and metrics with exponential standoff  # 以指数间隔上传最终模型和指标
        LOGGER.info(f"{PREFIX}Syncing final model...")  # 记录同步信息
        session.upload_model(  # 上传最终模型
            trainer.epoch,
            trainer.best,
            map=trainer.metrics.get("metrics/mAP50-95(B)", 0),  # 获取 mAP 指标
            final=True,  # 标记为最终模型
        )
        session.alive = False  # 停止心跳
        LOGGER.info(f"{PREFIX}Done ✅\n{PREFIX}View model at {session.model_url} 🚀")  # 记录完成信息


def on_train_start(trainer):
    """Run events on train start.  # 在训练开始时运行事件。"""
    events(trainer.args)  # 触发事件


def on_val_start(validator):
    """Runs events on validation start.  # 在验证开始时运行事件。"""
    events(validator.args)  # 触发事件


def on_predict_start(predictor):
    """Run events on predict start.  # 在预测开始时运行事件。"""
    events(predictor.args)  # 触发事件


def on_export_start(exporter):
    """Run events on export start.  # 在导出开始时运行事件。"""
    events(exporter.args)  # 触发事件


callbacks = (  # 定义回调函数
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,  # 预训练例程开始时的回调
        "on_pretrain_routine_end": on_pretrain_routine_end,  # 预训练例程结束时的回调
        "on_fit_epoch_end": on_fit_epoch_end,  # 拟合周期结束时的回调
        "on_model_save": on_model_save,  # 保存模型时的回调
        "on_train_end": on_train_end,  # 训练结束时的回调
        "on_train_start": on_train_start,  # 训练开始时的回调
        "on_val_start": on_val_start,  # 验证开始时的回调
        "on_predict_start": on_predict_start,  # 预测开始时的回调
        "on_export_start": on_export_start,  # 导出开始时的回调
    }
    if SETTINGS["hub"] is True  # 如果启用了 hub 设置
    else {}
)  # 验证启用