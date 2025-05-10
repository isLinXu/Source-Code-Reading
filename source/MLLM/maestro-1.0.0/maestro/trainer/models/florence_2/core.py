# 导入必要的模块
import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Optional

import dacite
import lightning
import torch
from torch.optim import AdamW

# 导入自定义模块
from maestro.trainer.common.callbacks import SaveCheckpoint
from maestro.trainer.common.datasets import create_data_loaders
from maestro.trainer.common.metrics import BaseMetric, MetricsTracker, parse_metrics, save_metric_plots
from maestro.trainer.common.training import MaestroTrainer
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec
from maestro.trainer.common.utils.path import create_new_run_directory
from maestro.trainer.common.utils.seed import ensure_reproducibility
from maestro.trainer.models.florence_2.checkpoints import (
    DEFAULT_FLORENCE2_MODEL_ID,
    DEFAULT_FLORENCE2_MODEL_REVISION,
    OptimizationStrategy,
    load_model,
    save_model,
)
from maestro.trainer.models.florence_2.inference import predict_with_inputs
from maestro.trainer.models.florence_2.loaders import evaluation_collate_fn, train_collate_fn

# 使用dataclass装饰器定义Florence2Configuration类
@dataclass()
class Florence2Configuration:
    """
    Configuration for training the Florence-2 model.

    Attributes:
        dataset (str):
            Path to the dataset used for training.
        model_id (str):
            Identifier for the Florence-2 model.
        revision (str):
            Model revision to use.
        device (str | torch.device):
            Device to run training on. Can be a ``torch.device`` or a string such as
            "auto", "cpu", "cuda", or "mps". If "auto", the code will pick the best
            available device.
        optimization_strategy (Literal["lora", "qlora", "freeze", "none"]):
            Strategy for optimizing the model parameters.
        cache_dir (Optional[str]):
            Directory to cache the model weights locally.
        epochs (int):
            Number of training epochs.
        lr (float):
            Learning rate for training.
        batch_size (int):
            Training batch size.
        accumulate_grad_batches (int):
            Number of batches to accumulate before performing a gradient update.
        val_batch_size (Optional[int]):
            Validation batch size. If None, defaults to the training batch size.
        num_workers (int):
            Number of workers for data loading.
        val_num_workers (Optional[int]):
            Number of workers for validation data loading. If None, defaults to num_workers.
        output_dir (str):
            Directory to store training outputs.
        metrics (list[BaseMetric] | list[str]):
            Metrics to track during training. Can be a list of metric objects or metric names.
        max_new_tokens (int):
            Maximum number of new tokens generated during inference.
        random_seed (Optional[int]):
            Random seed for ensuring reproducibility. If None, no seeding is applied.
    """

    # 定义类的属性
    dataset: str  # 训练数据集路径
    model_id: str = DEFAULT_FLORENCE2_MODEL_ID  # 模型ID，默认使用DEFAULT_FLORENCE2_MODEL_ID
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION  # 模型版本，默认使用DEFAULT_FLORENCE2_MODEL_REVISION
    device: str | torch.device = "auto"  # 训练设备，默认自动选择最佳设备
    optimization_strategy: Literal["lora", "freeze", "none"] = "lora"  # 优化策略，默认为"lora"
    cache_dir: Optional[str] = None  # 模型权重缓存目录，默认为None
    epochs: int = 10  # 训练轮数，默认为10
    lr: float = 1e-5  # 学习率，默认为1e-5
    batch_size: int = 4  # 训练批次大小，默认为4
    accumulate_grad_batches: int = 8  # 梯度累积的批次数量，默认为8
    val_batch_size: Optional[int] = None  # 验证批次大小，默认为None
    num_workers: int = 0  # 数据加载的worker数量，默认为0
    val_num_workers: Optional[int] = None  # 验证数据加载的worker数量，默认为None
    output_dir: str = "./training/florence_2"  # 训练输出目录，默认为"./training/florence_2"
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)  # 训练期间跟踪的指标，默认为空列表
    max_new_tokens: int = 1024  # 推理期间生成的最大新token数量，默认为1024
    random_seed: Optional[int] = None  # 随机种子，默认为None

    # 定义__post_init__方法，用于初始化后的处理
    def __post_init__(self):
        # 如果验证批次大小为None，则设置为训练批次大小
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size

        # 如果验证数据加载的worker数量为None，则设置为训练数据加载的worker数量
        if self.val_num_workers is None:
            self.val_num_workers = self.num_workers

        # 如果metrics是字符串列表，则解析为指标对象
        if isinstance(self.metrics, list) and all(isinstance(m, str) for m in self.metrics):
            self.metrics = parse_metrics(self.metrics)

        # 解析设备规格
        self.device = parse_device_spec(self.device)
        # 检查设备是否可用，如果不可用则抛出错误
        if not device_is_available(self.device):
            raise ValueError(f"Requested device '{self.device}' is not available.")


# 定义Florence2Trainer类，继承自MaestroTrainer
class Florence2Trainer(MaestroTrainer):
    """
    Trainer for fine-tuning the Florence-2 model.

    Attributes:
        processor (AutoProcessor): Processor for model inputs.
        model (AutoModelForCausalLM): The Florence-2 model.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        config (Florence2Configuration): Configuration object with training parameters.
    """

    # 初始化方法
    def __init__(self, processor, model, train_loader, valid_loader, config):
        # 调用父类的初始化方法
        super().__init__(processor, model, train_loader, valid_loader)
        # 保存配置对象
        self.config = config

        # TODO: Redesign metric tracking system
        # 初始化训练指标跟踪器，默认跟踪"loss"指标
        self.train_metrics_tracker = MetricsTracker.init(metrics=["loss"])
        # 初始化验证指标列表，默认包含"loss"指标
        metrics = ["loss"]
        # 遍历配置中的指标，如果是BaseMetric类型，则添加到指标列表中
        for metric in config.metrics:
            if isinstance(metric, BaseMetric):
                metrics += metric.describe()
        # 初始化验证指标跟踪器
        self.valid_metrics_tracker = MetricsTracker.init(metrics=metrics)

    # 定义训练步骤
    def training_step(self, batch, batch_idx):
        # 解包批次数据
        input_ids, pixel_values, labels = batch
        # 调用模型进行前向传播
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        # 获取损失值
        loss = outputs.loss
        # 记录训练损失
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.config.batch_size)
        # 在训练指标跟踪器中注册损失值
        self.train_metrics_tracker.register("loss", epoch=self.current_epoch, step=batch_idx, value=loss.item())
        # 返回损失值
        return loss

    # 定义验证步骤
    def validation_step(self, batch, batch_idx):
        # 解包批次数据
        input_ids, pixel_values, prefixes, suffixes = batch
        # 使用模型进行推理，生成后缀
        generated_suffixes = predict_with_inputs(
            model=self.model,
            processor=self.processor,
            input_ids=input_ids,
            pixel_values=pixel_values,
            device=self.config.device,
            max_new_tokens=self.config.max_new_tokens,
        )
        # 遍历配置中的指标，计算并记录指标值
        for metric in self.config.metrics:
            result = metric.compute(predictions=generated_suffixes, targets=suffixes)
            for key, value in result.items():
                self.valid_metrics_tracker.register(
                    metric=key,
                    epoch=self.current_epoch,
                    step=batch_idx,
                    value=value,
                )
                # 记录指标值
                self.log(key, value, prog_bar=True, logger=True)

    # 配置优化器
    def configure_optimizers(self):
        # 使用AdamW优化器
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer

    # 在训练结束时调用
    def on_fit_end(self) -> None:
        # 定义保存指标的路径
        save_metrics_path = os.path.join(self.config.output_dir, "metrics")
        # 保存训练和验证指标的图表
        save_metric_plots(
            training_tracker=self.train_metrics_tracker,
            validation_tracker=self.valid_metrics_tracker,
            output_dir=save_metrics_path,
        )


# 定义训练函数
def train(config: Florence2Configuration | dict) -> None:
    # 如果配置是字典类型，则转换为Florence2Configuration对象
    if isinstance(config, dict):
        config = dacite.from_dict(data_class=Florence2Configuration, data=config)
    # 确保配置是Florence2Configuration类型
    assert isinstance(config, Florence2Configuration)  # ensure mypy understands it's not a dict

    # 确保训练的可重复性
    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)
    # 创建新的运行目录
    run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    # 更新配置中的输出目录
    config = replace(config, output_dir=run_dir)

    # 加载模型和处理器
    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),
        cache_dir=config.cache_dir,
    )
    # 创建数据加载器
    train_loader, valid_loader, test_loader = create_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(train_collate_fn, processor=processor),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor),
        test_num_workers=config.val_num_workers,
    )
    # 创建Florence2Trainer实例
    pl_module = Florence2Trainer(
        processor=processor, model=model, train_loader=train_loader, valid_loader=valid_loader, config=config
    )
    # 定义保存检查点的路径
    save_checkpoints_path = os.path.join(config.output_dir, "checkpoints")
    # 创建保存检查点的回调
    save_checkpoint_callback = SaveCheckpoint(result_path=save_checkpoints_path, save_model_callback=save_model)
    # 创建Trainer实例
    trainer = lightning.Trainer(
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        check_val_every_n_epoch=1,
        limit_val_batches=1,
        log_every_n_steps=10,
        callbacks=[save_checkpoint_callback],
    )
    # 开始训练
    trainer.fit(pl_module)