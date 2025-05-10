import os
import shutil
from typing import Callable

import lightning
from lightning.pytorch.callbacks import Callback

from maestro.trainer.common.training import MaestroTrainer, TModel, TProcessor


class SaveCheckpoint(Callback):
    def __init__(self, result_path: str, save_model_callback: Callable[[str, TProcessor, TModel], None]):
        """
        初始化保存检查点的回调类。

        Args:
            result_path (str): 保存检查点的路径。
            save_model_callback (Callable[[str, TProcessor, TModel], None]): 保存模型的回调函数。
        """
        self.result_path = result_path  # 保存检查点的路径
        self.save_model_callback = save_model_callback  # 保存模型的回调函数

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: MaestroTrainer):
        """
        在每个训练周期结束时调用，保存最新的检查点。

        Args:
            trainer (lightning.Trainer): PyTorch Lightning的训练器对象。
            pl_module (MaestroTrainer): 训练模块对象。
        """
        # 定义最新检查点的路径
        checkpoint_path = f"{self.result_path}/latest"
        # 如果路径已存在，删除旧检查点
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        # 调用保存模型的回调函数，保存最新检查点
        self.save_model_callback(checkpoint_path, pl_module.processor, pl_module.model)
        # 打印保存信息
        print(f"Saved latest checkpoint to {checkpoint_path}")

        # TODO: 从训练器中获取当前指标值
        # TODO: 与最佳值进行比较，如果更好则保存
        # TODO: 如果指标有所改进，将最佳模型保存到{self.result_path}/best

    def on_train_end(self, trainer: lightning.Trainer, pl_module: MaestroTrainer):
        """
        在训练结束时调用，目前未实现具体功能。

        Args:
            trainer (lightning.Trainer): PyTorch Lightning的训练器对象。
            pl_module (MaestroTrainer): 训练模块对象。
        """
        pass
