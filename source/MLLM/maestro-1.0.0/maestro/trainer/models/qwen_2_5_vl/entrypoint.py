# 导入数据类库
import dataclasses
# 类型提示相关
from typing import Annotated, Optional

# 富文本显示库
import rich
# 命令行应用框架
import typer

# 模型检查点默认配置
from maestro.trainer.models.qwen_2_5_vl.checkpoints import (
    DEFAULT_QWEN2_5_VL_MODEL_ID,    # 默认模型ID
    DEFAULT_QWEN2_5_VL_MODEL_REVISION,  # 默认模型版本
)
# 模型核心配置类
from maestro.trainer.models.qwen_2_5_vl.core import Qwen25VLConfiguration
# 训练核心函数
from maestro.trainer.models.qwen_2_5_vl.core import train as qwen_2_5_vl_train

# 创建Typer命令行应用实例
qwen_2_5_vl_app = typer.Typer(help="Fine-tune and evaluate Qwen2.5-VL model")


@qwen_2_5_vl_app.command(
    help="Train Qwen2.5-VL model",  # 命令帮助信息
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}  # 允许额外参数
)
def train(
    # 数据集路径参数
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Path to the dataset in Roboflow JSONL format"),  # JSONL格式数据集路径
    ],
    # 模型ID参数（默认使用预定义值）
    model_id: Annotated[
        str,
        typer.Option("--model_id", help="Identifier for the Qwen2.5-VL model from HuggingFace Hub"),
    ] = DEFAULT_QWEN2_5_VL_MODEL_ID,
    # 模型版本参数（默认使用main分支）
    revision: Annotated[
        str,
        typer.Option("--revision", help="Model revision to use"),
    ] = DEFAULT_QWEN2_5_VL_MODEL_REVISION,
    # 训练设备参数（默认自动检测）
    device: Annotated[
        str,
        typer.Option("--device", help="Device to use for training"),
    ] = "auto",
    # 优化策略参数（默认LoRA）
    optimization_strategy: Annotated[
        str,
        typer.Option("--optimization_strategy", help="Optimization strategy: lora, qlora, or none"),
    ] = "lora",
    # 训练轮次参数（默认10轮）
    epochs: Annotated[
        int,
        typer.Option("--epochs", help="Number of training epochs"),
    ] = 10,
    # 学习率参数（默认2e-4）
    lr: Annotated[
        float,
        typer.Option("--lr", help="Learning rate for training"),
    ] = 2e-4,
    # 批次大小参数（默认4）
    batch_size: Annotated[
        int,
        typer.Option("--batch_size", help="Training batch size"),
    ] = 4,
    # 梯度累积步数参数（默认8）
    accumulate_grad_batches: Annotated[
        int,
        typer.Option("--accumulate_grad_batches", help="Number of batches to accumulate for gradient updates"),
    ] = 8,
    # 验证批次大小参数（可选）
    val_batch_size: Annotated[
        Optional[int],
        typer.Option("--val_batch_size", help="Validation batch size"),
    ] = None,
    # 数据加载工作线程数（默认0）
    num_workers: Annotated[
        int,
        typer.Option("--num_workers", help="Number of workers for data loading"),
    ] = 0,
    # 验证数据加载工作线程数（可选）
    val_num_workers: Annotated[
        Optional[int],
        typer.Option("--val_num_workers", help="Number of workers for validation data loading"),
    ] = None,
    # 输出目录参数（默认"./training/qwen_2_5_vl"）
    output_dir: Annotated[
        str,
        typer.Option("--output_dir", help="Directory to store training outputs"),
    ] = "./training/qwen_2_5_vl",
    # 评估指标参数（默认为空列表）
    metrics: Annotated[
        list[str],
        typer.Option("--metrics", help="List of metrics to track during training"),
    ] = [],
    # 系统消息参数（可选）
    system_message: Annotated[
        Optional[str],
        typer.Option("--system_message", help="System message used during data loading"),
    ] = None,
    # 图像最小像素参数（默认256*28*28）
    min_pixels: Annotated[
        int,
        typer.Option("--min_pixels", help="Minimum number of pixels for input images"),
    ] = 256 * 28 * 28,
    # 图像最大像素参数（默认1280*28*28）
    max_pixels: Annotated[
        int,
        typer.Option("--max_pixels", help="Maximum number of pixels for input images"),
    ] = 1280 * 28 * 28,
    # 生成最大新token数参数（默认1024）
    max_new_tokens: Annotated[
        int,
        typer.Option("--max_new_tokens", help="Maximum number of new tokens generated during inference"),
    ] = 1024,
    # 随机种子参数（可选）
    random_seed: Annotated[
        Optional[int],
        typer.Option("--random_seed", help="Random seed for ensuring reproducibility. If None, no seed is set"),
    ] = None,
) -> None:
    """训练任务入口函数"""
    # 创建训练配置对象
    config = Qwen25VLConfiguration(
        dataset=dataset,
        model_id=model_id,
        revision=revision,
        device=device,
        optimization_strategy=optimization_strategy,  # type: ignore  # 忽略类型检查
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        val_num_workers=val_num_workers,
        output_dir=output_dir,
        metrics=metrics,  # 指标列表会自动转换为Metric对象
        system_message=system_message,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        random_seed=random_seed,
    )
    
    # 输出带样式的配置信息
    typer.echo(typer.style(text="Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
    # 使用rich库漂亮打印配置字典
    rich.print(dataclasses.asdict(config))
    
    # 调用核心训练函数
    qwen_2_5_vl_train(config=config)