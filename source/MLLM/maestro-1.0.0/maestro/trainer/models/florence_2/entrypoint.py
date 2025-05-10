# 导入必要的模块
import dataclasses
from typing import Annotated, Optional

import rich
import typer

# 导入默认模型ID和版本
from maestro.trainer.models.florence_2.checkpoints import DEFAULT_FLORENCE2_MODEL_ID, DEFAULT_FLORENCE2_MODEL_REVISION
# 导入Florence2Configuration配置类和训练函数
from maestro.trainer.models.florence_2.core import Florence2Configuration
from maestro.trainer.models.florence_2.core import train as florence_2_train

# 创建Typer应用实例，用于命令行交互
florence_2_app = typer.Typer(help="Fine-tune and evaluate Florence-2 model")

# 定义train命令，用于训练Florence-2模型
@florence_2_app.command(
    help="Train Florence-2 model",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(
    # 定义命令行选项，并添加帮助信息
    dataset: Annotated[str, typer.Option("--dataset", help="Path to the dataset used for training")],
    model_id: Annotated[
        str, typer.Option("--model_id", help="Identifier for the Florence-2 model")
    ] = DEFAULT_FLORENCE2_MODEL_ID,  # 默认使用DEFAULT_FLORENCE2_MODEL_ID
    revision: Annotated[
        str, typer.Option("--revision", help="Model revision to use")
    ] = DEFAULT_FLORENCE2_MODEL_REVISION,  # 默认使用DEFAULT_FLORENCE2_MODEL_REVISION
    device: Annotated[str, typer.Option("--device", help="Device to use for training")] = "auto",  # 默认自动选择设备
    optimization_strategy: Annotated[
        str, typer.Option("--optimization_strategy", help="Optimization strategy: lora, freeze, or none")
    ] = "lora",  # 默认优化策略为"lora"
    cache_dir: Annotated[
        Optional[str], typer.Option("--cache_dir", help="Directory to cache the model weights locally")
    ] = None,  # 默认缓存目录为None
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs")] = 10,  # 默认训练轮数为10
    lr: Annotated[float, typer.Option("--lr", help="Learning rate for training")] = 1e-5,  # 默认学习率为1e-5
    batch_size: Annotated[int, typer.Option("--batch_size", help="Training batch size")] = 4,  # 默认批次大小为4
    accumulate_grad_batches: Annotated[
        int, typer.Option("--accumulate_grad_batches", help="Number of batches to accumulate for gradient updates")
    ] = 8,  # 默认梯度累积批次为8
    val_batch_size: Annotated[Optional[int], typer.Option("--val_batch_size", help="Validation batch size")] = None,  # 默认验证批次大小为None
    num_workers: Annotated[int, typer.Option("--num_workers", help="Number of workers for data loading")] = 0,  # 默认数据加载worker数量为0
    val_num_workers: Annotated[
        Optional[int], typer.Option("--val_num_workers", help="Number of workers for validation data loading")
    ] = None,  # 默认验证数据加载worker数量为None
    output_dir: Annotated[
        str, typer.Option("--output_dir", help="Directory to store training outputs")
    ] = "./training/florence_2",  # 默认输出目录为"./training/florence_2"
    metrics: Annotated[list[str], typer.Option("--metrics", help="List of metrics to track during training")] = [],  # 默认指标列表为空
    max_new_tokens: Annotated[
        int,
        typer.Option("--max_new_tokens", help="Maximum number of new tokens generated during inference"),
    ] = 1024,  # 默认最大新token数量为1024
    random_seed: Annotated[
        Optional[int],
        typer.Option("--random_seed", help="Random seed for ensuring reproducibility. If None, no seed is set"),
    ] = None,  # 默认随机种子为None
) -> None:
    # 创建Florence2Configuration配置对象
    config = Florence2Configuration(
        dataset=dataset,
        model_id=model_id,
        revision=revision,
        device=device,
        optimization_strategy=optimization_strategy,  # type: ignore
        cache_dir=cache_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        val_num_workers=val_num_workers,
        output_dir=output_dir,
        metrics=metrics,
        max_new_tokens=max_new_tokens,
        random_seed=random_seed,
    )
    # 输出训练配置信息
    typer.echo(typer.style("Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
    # 使用rich库打印配置字典
    rich.print(dataclasses.asdict(config))
    # 调用训练函数，开始训练
    florence_2_train(config=config)