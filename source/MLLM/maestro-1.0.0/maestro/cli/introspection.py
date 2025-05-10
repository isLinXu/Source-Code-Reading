import os

import typer

from maestro.cli.env import DEFAULT_DISABLE_RECIPE_IMPORTS_WARNINGS_ENV, DISABLE_RECIPE_IMPORTS_WARNINGS_ENV
from maestro.cli.utils import str2bool


def find_training_recipes(app: typer.Typer) -> None:
    """查找并添加训练模型的CLI命令。
    
    Args:
        app (typer.Typer): Typer应用实例
    """
    try:
        from maestro.trainer.models.florence_2.entrypoint import florence_2_app
        # 尝试导入Florence-2模型的CLI应用

        app.add_typer(florence_2_app, name="florence_2")  # 将Florence-2命令添加到主应用
    except Exception:
        _warn_about_recipe_import_error(model_name="Florence-2")  # 导入失败时发出警告

    try:
        from maestro.trainer.models.paligemma_2.entrypoint import paligemma_2_app
        # 尝试导入PaliGemma 2模型的CLI应用

        app.add_typer(paligemma_2_app, name="paligemma_2")  # 将PaliGemma 2命令添加到主应用
    except Exception:
        _warn_about_recipe_import_error(model_name="PaliGemma 2")  # 导入失败时发出警告

    try:
        from maestro.trainer.models.qwen_2_5_vl.entrypoint import qwen_2_5_vl_app
        # 尝试导入Qwen2.5-VL模型的CLI应用

        app.add_typer(qwen_2_5_vl_app, name="qwen_2_5_vl")  # 将Qwen2.5-VL命令添加到主应用
    except Exception:
        _warn_about_recipe_import_error(model_name="Qwen2.5-VL")  # 导入失败时发出警告


def _warn_about_recipe_import_error(model_name: str) -> None:
    """处理模型导入失败时的警告信息。
    
    Args:
        model_name (str): 模型名称
    """
    disable_warnings = str2bool(
        os.getenv(
            DISABLE_RECIPE_IMPORTS_WARNINGS_ENV,
            DEFAULT_DISABLE_RECIPE_IMPORTS_WARNINGS_ENV,
        )
    )  # 从环境变量中获取是否禁用警告的设置
    if disable_warnings:
        return None  # 如果禁用警告，则直接返回

    warning = typer.style("WARNING", fg=typer.colors.RED, bold=True)  # 设置警告文本样式
    message = "🚧 " + warning + f" cannot import recipe for {model_name}"  # 构建警告信息
    typer.echo(message)  # 输出警告信息
