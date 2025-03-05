# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license
"""Base callbacks."""  # 基础回调函数。

from collections import defaultdict  # 从 collections 模块导入 defaultdict 类
from copy import deepcopy  # 从 copy 模块导入 deepcopy 函数

# Trainer callbacks ----------------------------------------------------------------------------------------------------  # 训练器回调函数

def on_pretrain_routine_start(trainer):
    """Called before the pretraining routine starts.  # 在预训练例程开始之前调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_pretrain_routine_end(trainer):
    """Called after the pretraining routine ends.  # 在预训练例程结束后调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_train_start(trainer):
    """Called when the training starts.  # 当训练开始时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_train_epoch_start(trainer):
    """Called at the start of each training epoch.  # 在每个训练周期开始时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_train_batch_start(trainer):
    """Called at the start of each training batch.  # 在每个训练批次开始时调用."""
    pass  # 占位符函数，暂不执行任何操作


def optimizer_step(trainer):
    """Called when the optimizer takes a step.  # 当优化器进行一步时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_before_zero_grad(trainer):
    """Called before the gradients are set to zero.  # 在梯度被置为零之前调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_train_batch_end(trainer):
    """Called at the end of each training batch.  # 在每个训练批次结束时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_train_epoch_end(trainer):
    """Called at the end of each training epoch.  # 在每个训练周期结束时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_fit_epoch_end(trainer):
    """Called at the end of each fit epoch (train + val).  # 在每个拟合周期结束时调用（训练 + 验证）."""
    pass  # 占位符函数，暂不执行任何操作


def on_model_save(trainer):
    """Called when the model is saved.  # 当模型被保存时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_train_end(trainer):
    """Called when the training ends.  # 当训练结束时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_params_update(trainer):
    """Called when the model parameters are updated.  # 当模型参数被更新时调用."""
    pass  # 占位符函数，暂不执行任何操作


def teardown(trainer):
    """Called during the teardown of the training process.  # 在训练过程的清理阶段调用."""
    pass  # 占位符函数，暂不执行任何操作


# Validator callbacks --------------------------------------------------------------------------------------------------  # 验证器回调函数

def on_val_start(validator):
    """Called when the validation starts.  # 当验证开始时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_val_batch_start(validator):
    """Called at the start of each validation batch.  # 在每个验证批次开始时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_val_batch_end(validator):
    """Called at the end of each validation batch.  # 在每个验证批次结束时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_val_end(validator):
    """Called when the validation ends.  # 当验证结束时调用."""
    pass  # 占位符函数，暂不执行任何操作


# Predictor callbacks --------------------------------------------------------------------------------------------------  # 预测器回调函数

def on_predict_start(predictor):
    """Called when the prediction starts.  # 当预测开始时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_predict_batch_start(predictor):
    """Called at the start of each prediction batch.  # 在每个预测批次开始时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_predict_batch_end(predictor):
    """Called at the end of each prediction batch.  # 在每个预测批次结束时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_predict_postprocess_end(predictor):
    """Called after the post-processing of the prediction ends.  # 在预测后处理结束后调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_predict_end(predictor):
    """Called when the prediction ends.  # 当预测结束时调用."""
    pass  # 占位符函数，暂不执行任何操作


# Exporter callbacks ---------------------------------------------------------------------------------------------------  # 导出器回调函数

def on_export_start(exporter):
    """Called when the model export starts.  # 当模型导出开始时调用."""
    pass  # 占位符函数，暂不执行任何操作


def on_export_end(exporter):
    """Called when the model export ends.  # 当模型导出结束时调用."""
    pass  # 占位符函数，暂不执行任何操作


default_callbacks = {
    # Run in trainer  # 在训练器中运行
    "on_pretrain_routine_start": [on_pretrain_routine_start],  # 预训练例程开始时的回调
    "on_pretrain_routine_end": [on_pretrain_routine_end],  # 预训练例程结束时的回调
    "on_train_start": [on_train_start],  # 训练开始时的回调
    "on_train_epoch_start": [on_train_epoch_start],  # 每个训练周期开始时的回调
    "on_train_batch_start": [on_train_batch_start],  # 每个训练批次开始时的回调
    "optimizer_step": [optimizer_step],  # 优化器步骤时的回调
    "on_before_zero_grad": [on_before_zero_grad],  # 在梯度置为零之前的回调
    "on_train_batch_end": [on_train_batch_end],  # 每个训练批次结束时的回调
    "on_train_epoch_end": [on_train_epoch_end],  # 每个训练周期结束时的回调
    "on_fit_epoch_end": [on_fit_epoch_end],  # 拟合周期结束时的回调（训练 + 验证）
    "on_model_save": [on_model_save],  # 模型保存时的回调
    "on_train_end": [on_train_end],  # 训练结束时的回调
    "on_params_update": [on_params_update],  # 模型参数更新时的回调
    "teardown": [teardown],  # 清理阶段的回调
    # Run in validator  # 在验证器中运行
    "on_val_start": [on_val_start],  # 验证开始时的回调
    "on_val_batch_start": [on_val_batch_start],  # 每个验证批次开始时的回调
    "on_val_batch_end": [on_val_batch_end],  # 每个验证批次结束时的回调
    "on_val_end": [on_val_end],  # 验证结束时的回调
    # Run in predictor  # 在预测器中运行
    "on_predict_start": [on_predict_start],  # 预测开始时的回调
    "on_predict_batch_start": [on_predict_batch_start],  # 每个预测批次开始时的回调
    "on_predict_postprocess_end": [on_predict_postprocess_end],  # 预测后处理结束时的回调
    "on_predict_batch_end": [on_predict_batch_end],  # 每个预测批次结束时的回调
    "on_predict_end": [on_predict_end],  # 预测结束时的回调
    # Run in exporter  # 在导出器中运行
    "on_export_start": [on_export_start],  # 导出开始时的回调
    "on_export_end": [on_export_end],  # 导出结束时的回调
}


def get_default_callbacks():
    """
    Return a copy of the default_callbacks dictionary with lists as default values.  # 返回 default_callbacks 字典的副本，列表作为默认值。

    Returns:
        (defaultdict): A defaultdict with keys from default_callbacks and empty lists as default values.  # 返回一个 defaultdict，键来自 default_callbacks，值为默认空列表。
    """
    return defaultdict(list, deepcopy(default_callbacks))  # 返回带有默认值的 defaultdict


def add_integration_callbacks(instance):
    """
    Add integration callbacks from various sources to the instance's callbacks.  # 将来自不同源的集成回调添加到实例的回调中。

    Args:
        instance (Trainer, Predictor, Validator, Exporter): An object with a 'callbacks' attribute that is a dictionary  # 一个具有 'callbacks' 属性的对象，该属性是一个字典
            of callback lists.  # 回调列表的字典。
    """
    # Load HUB callbacks  # 加载 HUB 回调
    from .hub import callbacks as hub_cb  # 从 hub 模块导入回调

    callbacks_list = [hub_cb]  # 初始化回调列表

    # Load training callbacks  # 加载训练回调
    if "Trainer" in instance.__class__.__name__:  # 如果实例是 Trainer 类
        from .clearml import callbacks as clear_cb  # 从 clearml 模块导入回调
        from .comet import callbacks as comet_cb  # 从 comet 模块导入回调
        from .dvc import callbacks as dvc_cb  # 从 dvc 模块导入回调
        from .mlflow import callbacks as mlflow_cb  # 从 mlflow 模块导入回调
        from .neptune import callbacks as neptune_cb  # 从 neptune 模块导入回调
        from .raytune import callbacks as tune_cb  # 从 raytune 模块导入回调
        from .tensorboard import callbacks as tb_cb  # 从 tensorboard 模块导入回调
        from .wb import callbacks as wb_cb  # 从 wb 模块导入回调

        callbacks_list.extend([clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])  # 扩展回调列表

    # Add the callbacks to the callbacks dictionary  # 将回调添加到回调字典中
    for callbacks in callbacks_list:  # 遍历回调列表
        for k, v in callbacks.items():  # 遍历每个回调
            if v not in instance.callbacks[k]:  # 如果回调不在实例的回调中
                instance.callbacks[k].append(v)  # 添加回调