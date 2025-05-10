import dataclasses  # 导入数据类处理模块
from typing import Annotated, Optional  # 从typing模块导入Annotated和Optional，用于类型注解

import rich  # 导入rich库，用于美化控制台输出
import typer  # 导入typer库，用于构建命令行界面

from maestro.trainer.models.paligemma_2.checkpoints import (  # 从checkpoints模块导入相关内容
    DEFAULT_PALIGEMMA2_MODEL_ID,  # 默认PaliGemma2模型ID
    DEFAULT_PALIGEMMA2_MODEL_REVISION,  # 默认PaliGemma2模型修订版本
)
from maestro.trainer.models.paligemma_2.core import PaliGemma2Configuration  # 导入配置类
from maestro.trainer.models.paligemma_2.core import train as paligemma_2_train  # 导入训练函数并重命名

# 创建Typer应用实例，用于构建命令行界面
paligemma_2_app = typer.Typer(help="Fine-tune and evaluate PaliGemma-2 model")


@paligemma_2_app.command(  # 注册train命令到Typer应用
    help="Train PaliGemma-2 model",  # 命令帮助信息
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}  # 允许额外参数
)
def train(
    # 数据集路径参数（必需）
    dataset: Annotated[str, typer.Option("--dataset", help="Path to the dataset used for training")],
    # 模型ID参数（带默认值）
    model_id: Annotated[
        str, typer.Option("--model_id", help="Identifier for the PaliGemma-2 model")
    ] = DEFAULT_PALIGEMMA2_MODEL_ID,
    # 模型版本参数（带默认值）
    revision: Annotated[
        str, typer.Option("--revision", help="Model revision to use")
    ] = DEFAULT_PALIGEMMA2_MODEL_REVISION,
    # 训练设备参数（带默认值）
    device: Annotated[str, typer.Option("--device", help="Device to use for training")] = "auto",
    # 优化策略参数（带默认值）
    optimization_strategy: Annotated[
        str, typer.Option("--optimization_strategy", help="Optimization strategy: lora, qlora, freeze, or none")
    ] = "lora",
    # 模型缓存目录参数
    cache_dir: Annotated[
        Optional[str], typer.Option("--cache_dir", help="Directory to cache the model weights locally")
    ] = None,
    # 训练轮数参数（带默认值）
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs")] = 10,
    # 学习率参数（带默认值）
    lr: Annotated[float, typer.Option("--lr", help="Learning rate for training")] = 1e-5,
    # 训练批次大小参数（带默认值）
    batch_size: Annotated[int, typer.Option("--batch_size", help="Training batch size")] = 4,
    # 梯度累积步数参数（带默认值）
    accumulate_grad_batches: Annotated[
        int, typer.Option("--accumulate_grad_batches", help="Number of batches to accumulate for gradient updates")
    ] = 8,
    # 验证批次大小参数
    val_batch_size: Annotated[Optional[int], typer.Option("--val_batch_size", help="Validation batch size")] = None,
    # 数据加载工作进程数参数（带默认值）
    num_workers: Annotated[int, typer.Option("--num_workers", help="Number of workers for data loading")] = 0,
    # 验证数据加载工作进程数参数
    val_num_workers: Annotated[
        Optional[int], typer.Option("--val_num_workers", help="Number of workers for validation data loading")
    ] = None,
    # 输出目录参数（带默认值）
    output_dir: Annotated[
        str, typer.Option("--output_dir", help="Directory to store training outputs")
    ] = "./training/paligemma_2",
    # 评估指标列表参数
    metrics: Annotated[list[str], typer.Option("--metrics", help="List of metrics to track during training")] = [],
    # 最大生成token数参数（带默认值）
    max_new_tokens: Annotated[
        int, typer.Option("--max_new_tokens", help="Maximum number of new tokens generated during inference")
    ] = 512,
    # 随机种子参数
    random_seed: Annotated[
        Optional[int],
        typer.Option("--random_seed", help="Random seed for ensuring reproducibility. If None, no seed is set"),
    ] = None,
) -> None:
    # 创建训练配置对象
    config = PaliGemma2Configuration(
        dataset=dataset,
        model_id=model_id,
        revision=revision,
        device=device,
        optimization_strategy=optimization_strategy,  # type: ignore  # 忽略类型检查
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
    
    # 打印配置信息
    typer.echo(typer.style(text="Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))  # 绿色标题
    rich.print(dataclasses.asdict(config))  # 使用rich美化输出配置字典
    
    # 启动训练流程
    paligemma_2_train(config=config)
