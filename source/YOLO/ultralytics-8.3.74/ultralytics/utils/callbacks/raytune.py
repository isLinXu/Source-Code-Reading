# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import SETTINGS  # 从 ultralytics.utils 导入 SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled  # 验证集成是否启用
    import ray  # 导入 ray 模块
    from ray import tune  # 从 ray 导入 tune
    from ray.air import session  # 从 ray.air 导入 session

except (ImportError, AssertionError):  # 捕获导入错误和断言错误
    tune = None  # 如果导入失败，则将 tune 设置为 None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch.  # 在每个周期结束时将训练指标发送到 Ray Tune。"""
    if ray.train._internal.session.get_session():  # replacement for deprecated ray.tune.is_session_enabled()  # 替换已弃用的 ray.tune.is_session_enabled()
        metrics = trainer.metrics  # 获取训练指标
        session.report({**metrics, **{"epoch": trainer.epoch + 1}})  # 上报当前指标和周期


callbacks = (  # 定义回调函数
    {
        "on_fit_epoch_end": on_fit_epoch_end,  # 训练周期结束时的回调
    }
    if tune  # 如果 tune 可用
    else {}
)