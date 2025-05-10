# 导入操作系统接口模块
import os
# 数据类装饰器和字段定义
from dataclasses import dataclass, field, replace
# 函数式编程工具（部分参数应用）
from functools import partial
# 类型提示相关
from typing import Literal, Optional

# 数据类反序列化库
import dacite
# PyTorch Lightning深度学习框架
import lightning
# PyTorch深度学习框架
import torch
# AdamW优化器
from torch.optim import AdamW
# 数据加载器
from torch.utils.data import DataLoader
# 预训练模型和处理器
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# 训练回调函数
from maestro.trainer.common.callbacks import SaveCheckpoint
# 数据集加载工具
from maestro.trainer.common.datasets import create_data_loaders
# 指标跟踪相关
from maestro.trainer.common.metrics import BaseMetric, MetricsTracker, parse_metrics, save_metric_plots
# 基础训练类
from maestro.trainer.common.training import MaestroTrainer
# 设备管理工具
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec
# 路径管理工具
from maestro.trainer.common.utils.path import create_new_run_directory
# 可复现性工具
from maestro.trainer.common.utils.seed import ensure_reproducibility
# 模型检查点相关
from maestro.trainer.models.qwen_2_5_vl.checkpoints import (
    DEFAULT_QWEN2_5_VL_MODEL_ID,
    DEFAULT_QWEN2_5_VL_MODEL_REVISION,
    OptimizationStrategy,
    load_model,
    save_model,
)
# 推理工具
from maestro.trainer.models.qwen_2_5_vl.inference import predict_with_inputs
# 数据整理函数
from maestro.trainer.models.qwen_2_5_vl.loaders import evaluation_collate_fn, train_collate_fn


@dataclass()
class Qwen25VLConfiguration:
    """
    Configuration for training the Qwen2.5-VL model.
    Qwen2.5-VL模型训练配置类

    Attributes:
        dataset (str): Path to the dataset in Roboflow JSONL format.
        model_id (str): Identifier for the Qwen2.5-VL model from HuggingFace Hub.
        revision (str): Model revision to use.
        device (torch.device): Device to run training on.
        optimization_strategy (Literal["lora", "qlora", "none"]): Optimization strategy.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for training.
        batch_size (int): Training batch size.
        accumulate_grad_batches (int): Number of batches to accumulate for gradient updates.
        val_batch_size (Optional[int]): Validation batch size.
        num_workers (int): Number of workers for data loading.
        val_num_workers (Optional[int]): Number of workers for validation data loading.
        output_dir (str): Directory to store training outputs.
        metrics (list[BaseMetric] | list[str]): Metrics to track during training.
        system_message (Optional[str]): System message used during data loading.
        min_pixels (int): Minimum number of pixels for input images.
        max_pixels (int): Maximum number of pixels for input images.
        max_new_tokens (int): Maximum number of new tokens generated during inference.
        random_seed (Optional[int]): Random seed for ensuring reproducibility. If `None`, no seed is set.
    """

    # 数据集路径（必须参数）
    dataset: str
    # 模型标识（默认使用预定义值）
    model_id: str = DEFAULT_QWEN2_5_VL_MODEL_ID
    # 模型版本（默认使用main分支）
    revision: str = DEFAULT_QWEN2_5_VL_MODEL_REVISION
    # 训练设备（自动检测）
    device: str | torch.device = "auto"
    # 优化策略（默认使用LoRA）
    optimization_strategy: Literal["lora", "qlora", "none"] = "lora"
    # 训练轮次（默认10轮）
    epochs: int = 10
    # 学习率（默认2e-4）
    lr: float = 2e-4
    # 训练批次大小（默认4）
    batch_size: int = 4
    # 梯度累积步数（默认8）
    accumulate_grad_batches: int = 8
    # 验证批次大小（可选，默认同训练批次）
    val_batch_size: Optional[int] = None
    # 数据加载工作线程数（默认0）
    num_workers: int = 0
    # 验证数据加载工作线程数（可选）
    val_num_workers: Optional[int] = None
    # 输出目录（默认"./training/qwen_2_5_vl"）
    output_dir: str = "./training/qwen_2_5_vl"
    # 评估指标列表（默认为空列表）
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)
    # 系统提示信息（可选）
    system_message: Optional[str] = None
    # 图像最小像素数（默认256*28*28）
    min_pixels: int = 256 * 28 * 28
    # 图像最大像素数（默认1280*28*28）
    max_pixels: int = 1280 * 28 * 28
    # 生成最大新token数（默认1024）
    max_new_tokens: int = 1024
    # 随机种子（可选）
    random_seed: Optional[int] = None

    def __post_init__(self):
        """后初始化处理，设置默认值和类型转换"""
        # 设置默认验证批次大小
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size

        # 设置默认验证数据加载线程数
        if self.val_num_workers is None:
            self.val_num_workers = self.num_workers

        # 转换字符串指标为Metric对象
        if isinstance(self.metrics, list) and all(isinstance(m, str) for m in self.metrics):
            self.metrics = parse_metrics(self.metrics)

        # 解析设备规格
        self.device = parse_device_spec(self.device)
        # 检查设备可用性
        if not device_is_available(self.device):
            raise ValueError(f"Requested device '{self.device}' is not available.")


class Qwen25VLTrainer(MaestroTrainer):
    """
    Trainer for fine-tuning the Qwen2.5-VL model.
    Qwen2.5-VL模型微调训练器

    Attributes:
        processor (Qwen2_5_VLProcessor): Tokenizer and processor for model inputs.
        model (Qwen2_5_VLForConditionalGeneration): Pre-trained Qwen2.5-VL model.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        config (Qwen25VLConfiguration): Configuration object containing training parameters.
    """

    def __init__(
        self,
        processor: Qwen2_5_VLProcessor,
        model: Qwen2_5_VLForConditionalGeneration,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: Qwen25VLConfiguration,
    ):
        # 初始化父类
        super().__init__(processor, model, train_loader, valid_loader)
        self.config = config

        # TODO: Redesign metric tracking system
        # 初始化训练指标跟踪器（仅跟踪loss）
        self.train_metrics_tracker = MetricsTracker.init(metrics=["loss"])
        # 初始化验证指标（包含loss和其他配置的指标）
        metrics = ["loss"]
        for metric in config.metrics:
            if isinstance(metric, BaseMetric):
                metrics += metric.describe()  # 确保类型检查器理解这是BaseMetric
        self.valid_metrics_tracker = MetricsTracker.init(metrics=metrics)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 解包批次数据
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        # 计算损失
        loss = outputs.loss
        # 记录训练损失
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.config.batch_size)
        # 更新指标跟踪器
        self.train_metrics_tracker.register("loss", epoch=self.current_epoch, step=batch_idx, value=loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 解包验证批次数据
        input_ids, attention_mask, pixel_values, image_grid_thw, prefixes, suffixes = batch
        # 生成预测结果
        generated_suffixes = predict_with_inputs(
            model=self.model,
            processor=self.processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            device=self.config.device,
        )

        # 计算每个指标
        for metric in self.config.metrics:
            result = metric.compute(predictions=generated_suffixes, targets=suffixes)
            # 记录指标结果
            for key, value in result.items():
                self.valid_metrics_tracker.register(
                    metric=key,
                    epoch=self.current_epoch,
                    step=batch_idx,
                    value=value,
                )
                self.log(key, value, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """配置优化器"""
        # 使用AdamW优化器
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def on_fit_end(self) -> None:
        """训练结束时回调"""
        # 保存指标路径
        save_metrics_path = os.path.join(self.config.output_dir, "metrics")
        # 保存指标图表
        save_metric_plots(
            training_tracker=self.train_metrics_tracker,
            validation_tracker=self.valid_metrics_tracker,
            output_dir=save_metrics_path,
        )


def train(config: Qwen25VLConfiguration | dict) -> None:
    """
    Trains the Qwen2.5-VL model based on the given configuration.
    根据给定配置训练Qwen2.5-VL模型

    Args:
        config (Qwen25VLConfiguration | dict): Training configuration or dictionary with configuration parameters.

    Returns:
        None
    """
    # 处理字典类型的配置
    if isinstance(config, dict):
        config = dacite.from_dict(data_class=Qwen25VLConfiguration, data=config)
    assert isinstance(config, Qwen25VLConfiguration)  # 确保类型检查器理解这是配置对象

    # 确保实验可复现性
    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)
    # 创建新的运行目录
    run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    config = replace(config, output_dir=run_dir)

    # 加载模型和处理器
    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    # 创建数据加载器
    train_loader, valid_loader, test_loader = create_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(train_collate_fn, processor=processor, system_message=config.system_message),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor, system_message=config.system_message),
        test_num_workers=config.val_num_workers,
    )
    # 初始化训练模块
    pl_module = Qwen25VLTrainer(
        processor=processor, model=model, train_loader=train_loader, valid_loader=valid_loader, config=config
    )
    # 设置检查点保存路径
    save_checkpoints_path = os.path.join(config.output_dir, "checkpoints")
    save_checkpoint_callback = SaveCheckpoint(result_path=save_checkpoints_path, save_model_callback=save_model)
    # 初始化PyTorch Lightning训练器
    trainer = lightning.Trainer(
        max_epochs=config.epochs,  # 最大训练轮次
        accumulate_grad_batches=config.accumulate_grad_batches,  # 梯度累积步数
        check_val_every_n_epoch=1,  # 每轮验证一次
        limit_val_batches=1,  # 限制验证批次数量
        log_every_n_steps=10,  # 每10步记录日志
        callbacks=[save_checkpoint_callback],  # 使用检查点保存回调
    )
    # 开始训练
    trainer.fit(pl_module)
