# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json  # 导入JSON处理模块
import math  # 导入数学计算模块
import os  # 导入操作系统接口模块
from typing import Any, Dict, List  # 导入类型提示

from transformers.trainer import TRAINER_STATE_NAME  # 导入Trainer状态文件名常量

from . import logging  # 导入日志模块
from .packages import is_matplotlib_available  # 导入matplotlib可用性检查函数


# 如果matplotlib可用，则导入相关模块
if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


logger = logging.get_logger(__name__)  # 获取日志记录器


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    基于TensorBoard的指数移动平均(EMA)实现
    """
    if len(scalars) == 0:  # 如果输入列表为空
        return []  # 返回空列表

    last = scalars[0]  # 初始化last为第一个值
    smoothed = []  # 存储平滑后的值
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function，使用sigmoid函数计算权重
    for next_val in scalars:  # 遍历所有值
        smoothed_val = last * weight + (1 - weight) * next_val  # 计算平滑值
        smoothed.append(smoothed_val)  # 添加到结果列表
        last = smoothed_val  # 更新last值
    return smoothed


def gen_loss_plot(trainer_log: List[Dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""
    Plots loss curves in LlamaBoard.
    在LlamaBoard中绘制损失曲线
    """
    plt.close("all")  # 关闭所有图形
    plt.switch_backend("agg")  # 切换到非交互式后端
    fig = plt.figure()  # 创建新图形
    ax = fig.add_subplot(111)  # 添加子图
    steps, losses = [], []  # 初始化步数和损失列表
    for log in trainer_log:  # 遍历训练日志
        if log.get("loss", None):  # 如果日志中包含loss
            steps.append(log["current_steps"])  # 添加步数
            losses.append(log["loss"])  # 添加损失值

    # 绘制原始损失曲线
    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    # 绘制平滑后的损失曲线
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()  # 添加图例
    ax.set_xlabel("step")  # 设置x轴标签
    ax.set_ylabel("loss")  # 设置y轴标签
    return fig


def plot_loss(save_dictionary: str, keys: List[str] = ["loss"]) -> None:
    r"""
    Plots loss curves and saves the image.
    绘制损失曲线并保存图像
    """
    plt.switch_backend("agg")  # 切换到非交互式后端
    # 读取训练状态文件
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:  # 遍历所有需要绘制的指标
        steps, metrics = [], []  # 初始化步数和指标列表
        for i in range(len(data["log_history"])):  # 遍历日志历史
            if key in data["log_history"][i]:  # 如果日志中包含当前指标
                steps.append(data["log_history"][i]["step"])  # 添加步数
                metrics.append(data["log_history"][i][key])  # 添加指标值

        if len(metrics) == 0:  # 如果没有找到指标数据
            logger.warning_rank0(f"No metric {key} to plot.")  # 输出警告
            continue

        plt.figure()  # 创建新图形
        # 绘制原始指标曲线
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        # 绘制平滑后的指标曲线
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title(f"training {key} of {save_dictionary}")  # 设置图形标题
        plt.xlabel("step")  # 设置x轴标签
        plt.ylabel(key)  # 设置y轴标签
        plt.legend()  # 添加图例
        # 构造图像保存路径
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)  # 保存图像
        print("Figure saved at:", figure_path)  # 打印保存路径
