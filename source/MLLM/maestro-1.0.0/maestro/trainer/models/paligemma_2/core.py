import os  # 导入操作系统接口模块
from dataclasses import dataclass, field, replace  # 从dataclasses模块导入dataclass, field和replace，用于创建数据类
from functools import partial  # 从functools模块导入partial，用于创建偏函数
from typing import Literal, Optional  # 从typing模块导入Literal和Optional，用于类型注解

import dacite  # 导入dacite库，用于从字典创建数据类实例
import lightning  # 导入lightning库，一个PyTorch的轻量级包装器
import torch  # 导入PyTorch库
from torch.optim import AdamW  # 从torch.optim导入AdamW优化器
from torch.utils.data import DataLoader  # 从torch.utils.data导入DataLoader，用于加载数据
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor  # 从transformers库导入PaliGemma模型和处理器

from maestro.trainer.common.callbacks import SaveCheckpoint  # 从maestro.trainer.common.callbacks导入SaveCheckpoint回调
from maestro.trainer.common.datasets import create_data_loaders  # 从maestro.trainer.common.datasets导入create_data_loaders函数
from maestro.trainer.common.metrics import BaseMetric, MetricsTracker, parse_metrics, save_metric_plots  # 从maestro.trainer.common.metrics导入相关类和函数
from maestro.trainer.common.training import MaestroTrainer  # 从maestro.trainer.common.training导入MaestroTrainer基类
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec  # 从maestro.trainer.common.utils.device导入设备相关工具函数
from maestro.trainer.common.utils.path import create_new_run_directory  # 从maestro.trainer.common.utils.path导入创建运行目录的函数
from maestro.trainer.common.utils.seed import ensure_reproducibility  # 从maestro.trainer.common.utils.seed导入确保可复现性的函数
from maestro.trainer.models.paligemma_2.checkpoints import (  # 从maestro.trainer.models.paligemma_2.checkpoints导入相关内容
    DEFAULT_PALIGEMMA2_MODEL_ID,  # 默认PaliGemma2模型ID
    DEFAULT_PALIGEMMA2_MODEL_REVISION,  # 默认PaliGemma2模型修订版本
    OptimizationStrategy,  # 优化策略枚举
    load_model,  # 加载模型的函数
    save_model,  # 保存模型的函数
)
from maestro.trainer.models.paligemma_2.inference import predict_with_inputs  # 从maestro.trainer.models.paligemma_2.inference导入预测函数
from maestro.trainer.models.paligemma_2.loaders import evaluation_collate_fn, train_collate_fn  # 从maestro.trainer.models.paligemma_2.loaders导入数据整理函数


@dataclass()  # 定义一个数据类
class PaliGemma2Configuration:
    """
    Configuration for training the PaliGemma2 model.
    用于训练PaliGemma2模型的配置。

    Attributes:
        dataset (str):
            Path to the dataset used for training.
            用于训练的数据集的路径。
        model_id (str):
            Identifier for the PaliGemma2 model.
            PaliGemma2模型的标识符。
        revision (str):
            Model revision to use.
            要使用的模型修订版本。
        device (str | torch.device):
            Device to run training on. Can be a ``torch.device`` or a string such as
            "auto", "cpu", "cuda", or "mps". If "auto", the code will pick the best
            available device.
            运行训练的设备。可以是 ``torch.device`` 或字符串，例如
            "auto"、"cpu"、"cuda" 或 "mps"。如果是 "auto"，代码将选择最佳
            可用设备。
        optimization_strategy (Literal["lora", "qlora", "freeze", "none"]):
            Strategy for optimizing the model parameters.
            优化模型参数的策略。
        cache_dir (Optional[str]):
            Directory to cache the model weights locally.
            在本地缓存模型权重的目录。
        epochs (int):
            Number of training epochs.
            训练轮数。
        lr (float):
            Learning rate for training.
            训练的学习率。
        batch_size (int):
            Training batch size.
            训练批次大小。
        accumulate_grad_batches (int):
            Number of batches to accumulate before performing a gradient update.
            在执行梯度更新之前累积的批次数。
        val_batch_size (Optional[int]):
            Validation batch size. If None, defaults to the training batch size.
            验证批次大小。如果为 None，则默认为训练批次大小。
        num_workers (int):
            Number of workers for data loading.
            用于数据加载的工作进程数。
        val_num_workers (Optional[int]):
            Number of workers for validation data loading. If None, defaults to num_workers.
            用于验证数据加载的工作进程数。如果为 None，则默认为 num_workers。
        output_dir (str):
            Directory to store training outputs.
            存储训练输出的目录。
        metrics (list[BaseMetric] | list[str]):
            Metrics to track during training. Can be a list of metric objects or metric names.
            训练期间要跟踪的指标。可以是指标对象列表或指标名称列表。
        max_new_tokens (int):
            Maximum number of new tokens generated during inference.
            推理期间生成的最大新标记数。
        random_seed (Optional[int]):
            Random seed for ensuring reproducibility. If None, no seeding is applied.
            用于确保可复现性的随机种子。如果为 None，则不应用种子。
    """

    dataset: str  # 数据集路径
    model_id: str = DEFAULT_PALIGEMMA2_MODEL_ID  # 模型ID，默认为预定义的常量
    revision: str = DEFAULT_PALIGEMMA2_MODEL_REVISION  # 模型修订版本，默认为预定义的常量
    device: str | torch.device = "auto"  # 训练设备，默认为 "auto"
    optimization_strategy: Literal["lora", "qlora", "freeze", "none"] = "lora"  # 优化策略，默认为 "lora"
    cache_dir: Optional[str] = None  # 模型缓存目录，默认为 None
    epochs: int = 10  # 训练轮数，默认为 10
    lr: float = 1e-5  # 学习率，默认为 1e-5
    batch_size: int = 4  # 训练批次大小，默认为 4
    accumulate_grad_batches: int = 8  # 梯度累积批次数，默认为 8
    val_batch_size: Optional[int] = None  # 验证批次大小，默认为 None
    num_workers: int = 0  # 数据加载工作进程数，默认为 0
    val_num_workers: Optional[int] = None  # 验证数据加载工作进程数，默认为 None
    output_dir: str = "./training/paligemma_2"  # 输出目录，默认为 "./training/paligemma_2"
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)  # 评估指标列表，默认为空列表
    max_new_tokens: int = 512  # 推理时最大新生成token数，默认为 512
    random_seed: Optional[int] = None  # 随机种子，默认为 None

    def __post_init__(self):  # 数据类初始化后执行的方法
        if self.val_batch_size is None:  # 如果验证批次大小未设置
            self.val_batch_size = self.batch_size  # 将验证批次大小设置为训练批次大小

        if self.val_num_workers is None:  # 如果验证数据加载工作进程数未设置
            self.val_num_workers = self.num_workers  # 将验证数据加载工作进程数设置为训练数据加载工作进程数

        if isinstance(self.metrics, list) and all(isinstance(m, str) for m in self.metrics):  # 如果指标是字符串列表
            self.metrics = parse_metrics(self.metrics)  # 解析字符串列表为指标对象列表

        self.device = parse_device_spec(self.device)  # 解析设备规范字符串
        if not device_is_available(self.device):  # 如果指定的设备不可用
            raise ValueError(f"Requested device '{self.device}' is not available.")  # 抛出值错误


class PaliGemma2Trainer(MaestroTrainer):  # 定义PaliGemma2训练器类，继承自MaestroTrainer
    """
    Trainer for fine-tuning the PaliGemma-2 model.
    用于微调PaliGemma-2模型的训练器。

    Attributes:
        processor (PaliGemmaProcessor): Tokenizer and processor for model inputs.
                                        用于模型输入的Tokenizer和处理器。
        model (PaliGemmaForConditionalGeneration): Pre-trained PaliGemma-2 model.
                                                   预训练的PaliGemma-2模型。
        train_loader (DataLoader): DataLoader for training data.
                                   训练数据的DataLoader。
        valid_loader (DataLoader): DataLoader for validation data.
                                   验证数据的DataLoader。
        config (PaliGemma2Configuration): Configuration object containing training parameters.
                                          包含训练参数的配置对象。
    """

    def __init__(  # 初始化方法
        self,
        processor: PaliGemmaProcessor,  # PaliGemma处理器实例
        model: PaliGemmaForConditionalGeneration,  # PaliGemma模型实例
        train_loader: DataLoader,  # 训练数据加载器
        valid_loader: DataLoader,  # 验证数据加载器
        config: PaliGemma2Configuration,  # 训练配置
    ):
        super().__init__(processor, model, train_loader, valid_loader)  # 调用父类的初始化方法
        self.config = config  # 保存配置

        # TODO: Redesign metric tracking system  # TODO: 重新设计指标跟踪系统
        self.train_metrics_tracker = MetricsTracker.init(metrics=["loss"])  # 初始化训练指标跟踪器，默认跟踪 "loss"
        metrics = ["loss"]  # 初始化验证指标列表，包含 "loss"
        for metric in config.metrics:  # 遍历配置中的指标
            if isinstance(metric, BaseMetric):  # 如果指标是BaseMetric的实例
                metrics += metric.describe()  # ensure mypy understands it's BaseMetric # 将指标描述添加到列表中（确保mypy理解它是BaseMetric）
        self.valid_metrics_tracker = MetricsTracker.init(metrics=metrics)  # 初始化验证指标跟踪器

    def training_step(self, batch, batch_idx):  # 定义训练步骤
        input_ids, attention_mask, token_type_ids, pixel_values, labels = batch  # 解包批次数据
        outputs = self.model(  # 模型前向传播
            input_ids=input_ids,  # 输入ID
            attention_mask=attention_mask,  # 注意力掩码
            token_type_ids=token_type_ids,  # Token类型ID
            pixel_values=pixel_values,  # 像素值
            labels=labels,  # 标签
        )
        loss = outputs.loss  # 获取损失值
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.config.batch_size)  # 记录训练损失
        self.train_metrics_tracker.register("loss", epoch=self.current_epoch, step=batch_idx, value=loss.item())  # 在训练指标跟踪器中注册损失
        return loss  # 返回损失

    def validation_step(self, batch, batch_idx):  # 定义验证步骤
        input_ids, attention_mask, pixel_values, prefixes, suffixes = batch  # 解包批次数据
        generated_suffixes = predict_with_inputs(  # 使用模型进行预测
            model=self.model,  # 模型
            processor=self.processor,  # 处理器
            input_ids=input_ids,  # 输入ID
            attention_mask=attention_mask,  # 注意力掩码
            pixel_values=pixel_values,  # 像素值
            device=self.config.device,  # 设备
            max_new_tokens=self.config.max_new_tokens,  # 最大新生成token数
        )
        for metric in self.config.metrics:  # 遍历配置中的指标
            result = metric.compute(predictions=generated_suffixes, targets=suffixes)  # 计算指标
            for key, value in result.items():  # 遍历指标结果
                self.valid_metrics_tracker.register(  # 在验证指标跟踪器中注册指标值
                    metric=key,  # 指标名称
                    epoch=self.current_epoch,  # 当前轮数
                    step=batch_idx,  # 当前步骤
                    value=value,  # 指标值
                )
                self.log(key, value, prog_bar=True, logger=True)  # 记录指标值

    def configure_optimizers(self):  # 配置优化器
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)  # 创建AdamW优化器
        return optimizer  # 返回优化器

    def on_fit_end(self) -> None:  # 训练结束时调用的钩子函数
        save_metrics_path = os.path.join(self.config.output_dir, "metrics")  # 定义保存指标图表的路径
        save_metric_plots(  # 保存指标图表
            training_tracker=self.train_metrics_tracker,  # 训练指标跟踪器
            validation_tracker=self.valid_metrics_tracker,  # 验证指标跟踪器
            output_dir=save_metrics_path,  # 输出目录
        )


def train(config: PaliGemma2Configuration | dict) -> None:  # 定义训练函数
    if isinstance(config, dict):  # 如果配置是字典类型
        config = dacite.from_dict(data_class=PaliGemma2Configuration, data=config)  # 从字典创建PaliGemma2Configuration实例
    assert isinstance(config, PaliGemma2Configuration)  # ensure mypy understands it's not a dict # 断言配置是PaliGemma2Configuration实例（确保mypy理解它不是字典）

    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)  # 设置随机种子以确保可复现性
    run_dir = create_new_run_directory(base_output_dir=config.output_dir)  # 创建新的运行目录
    config = replace(config, output_dir=run_dir)  # 更新配置中的输出目录

    processor, model = load_model(  # 加载模型和处理器
        model_id_or_path=config.model_id,  # 模型ID或路径
        revision=config.revision,  # 模型修订版本
        device=config.device,  # 设备
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),  # 优化策略
        cache_dir=config.cache_dir,  # 缓存目录
    )
    train_loader, valid_loader, test_loader = create_data_loaders(  # 创建数据加载器
        dataset_location=config.dataset,  # 数据集位置
        train_batch_size=config.batch_size,  # 训练批次大小
        train_collect_fn=partial(train_collate_fn, processor=processor, max_length=config.max_new_tokens),  # 训练数据整理函数
        train_num_workers=config.num_workers,  # 训练数据加载工作进程数
        test_batch_size=config.val_batch_size,  # 测试/验证批次大小
        test_collect_fn=partial(evaluation_collate_fn, processor=processor),  # 测试/验证数据整理函数
        test_num_workers=config.val_num_workers,  # 测试/验证数据加载工作进程数
    )

    pl_module = PaliGemma2Trainer(  # 创建PaliGemma2Trainer实例
        processor=processor, model=model, train_loader=train_loader, valid_loader=valid_loader, config=config
    )
    save_checkpoints_path = os.path.join(config.output_dir, "checkpoints")  # 定义保存检查点的路径
    save_checkpoint_callback = SaveCheckpoint(result_path=save_checkpoints_path, save_model_callback=save_model)  # 创建保存检查点的回调
    trainer = lightning.Trainer(  # 创建Lightning Trainer实例
        max_epochs=config.epochs,  # 最大训练轮数
        accumulate_grad_batches=config.accumulate_grad_batches,  # 梯度累积批次数
        check_val_every_n_epoch=1,  # 每隔多少轮进行一次验证
        limit_val_batches=1,  # 限制验证批次数
        log_every_n_steps=10,  # 每隔多少步记录一次日志
        callbacks=[save_checkpoint_callback],  # 使用的回调列表
    )
    trainer.fit(pl_module)  # 开始训练
