from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import supervision as sv
from evaluate import load
from nltk import edit_distance
from supervision.metrics.mean_average_precision import MeanAveragePrecision


class BaseMetric(ABC):
    """Abstract base class for custom metrics. Subclasses must implement
    the 'describe' and 'compute' methods.
    自定义指标的抽象基类。子类必须实现'describe'和'compute'方法。
    """

    @abstractmethod
    def describe(self) -> list[str]:
        """Describe the names of the metrics that this class will compute.

        Returns:
            List[str]: A list of metric names that will be computed.
        描述该类将计算的指标名称。

        返回:
            List[str]: 将计算的指标名称列表
        """
        pass

    @abstractmethod
    def compute(self, targets: list[Any], predictions: list[Any]) -> dict[str, float]:
        """Compute the metric based on the targets and predictions.

        Args:
            targets (List[Any]): The ground truth.
            predictions (List[Any]): The prediction result.

        Returns:
            Dict[str, float]: A dictionary of computed metrics with metric names as
                keys and their values.
        根据目标和预测计算指标。

        参数:
            targets (List[Any]): 真实值
            predictions (List[Any]): 预测结果

        返回:
            Dict[str, float]: 计算得到的指标字典，键为指标名称，值为指标值
        """
        pass


class MeanAveragePrecisionMetric(BaseMetric):
    """A class used to compute the Mean Average Precision (mAP) metric.

    mAP is a popular metric for object detection tasks, measuring the average precision
    across all classes and IoU thresholds.
    用于计算平均精度(mAP)指标的类。

    mAP是目标检测任务中常用的指标，用于衡量所有类别和IoU阈值下的平均精度。
    """

    name = "mean_average_precision"

    def describe(self) -> list[str]:
        """Returns a list of metric names that this class will compute.

        Returns:
            List[str]: A list of metric names.
        返回该类将计算的指标名称列表。

        返回:
            List[str]: 指标名称列表
        """
        return ["map50:95", "map50", "map75"]

    def compute(self, targets: list[sv.Detections], predictions: list[sv.Detections]) -> dict[str, float]:
        """Computes the mAP metrics based on the targets and predictions.

        Args:
            targets (List[sv.Detections]): The ground truth detections.
            predictions (List[sv.Detections]): The predicted detections.

        Returns:
            Dict[str, float]: A dictionary of computed mAP metrics with metric names as
                keys and their values.
        根据目标和预测计算mAP指标。

        参数:
            targets (List[sv.Detections]): 真实检测结果
            predictions (List[sv.Detections]): 预测检测结果

        返回:
            Dict[str, float]: 计算得到的mAP指标字典，键为指标名称，值为指标值
        """
        result = MeanAveragePrecision().update(targets=targets, predictions=predictions).compute()
        return {"map50:95": result.map50_95, "map50": result.map50, "map75": result.map75}


class BLEUMetric(BaseMetric):
    """A class used to compute the BLEU (Bilingual Evaluation Understudy) metric.

    BLEU is a popular metric for evaluating the quality of text predictions in natural
    language processing tasks, particularly machine translation. It measures the
    similarity between the predicted text and one or more reference texts based on
    n-gram precision, brevity penalty, and other factors.
    
    用于计算BLEU（双语评估替代）指标的类。
    BLEU是评估自然语言处理任务中文本预测质量的流行指标，特别是在机器翻译中。
    它基于n-gram精度、简短惩罚和其他因素来衡量预测文本与一个或多个参考文本之间的相似性。
    """

    bleu = load("bleu")  # 加载BLEU指标计算器
    name = "bleu"  # 指标名称

    def describe(self) -> list[str]:
        """Returns a list of metric names that this class will compute.

        Returns:
            List[str]: A list of metric names.
        
        返回该类将计算的指标名称列表。
        返回值：
            List[str]: 指标名称列表。
        """
        return ["bleu"]  # 返回包含"bleu"的列表

    def compute(self, targets: list[str], predictions: list[str]) -> dict[str, float]:
        """Computes the BLEU metric based on the targets and predictions.

        Args:
            targets (List[str]): The ground truth texts (references), where each element
                represents the reference text for the corresponding prediction.
            predictions (List[str]): The predicted texts (hypotheses) to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the computed BLEU score, with the
                metric name ("bleu") as the key and its value as the score.
        
        基于目标和预测计算BLEU指标。
        参数：
            targets (List[str]): 真实文本（参考），每个元素代表对应预测的参考文本。
            predictions (List[str]): 要评估的预测文本（假设）。
        返回值：
            Dict[str, float]: 包含计算出的BLEU分数的字典，以指标名称("bleu")为键，其值为分数。
        """
        if len(targets) != len(predictions):
            raise ValueError("The number of targets and predictions must be the same.")
            # 如果目标和预测的数量不一致，抛出错误

        try:
            results = self.bleu.compute(predictions=predictions, references=targets)
            return {"bleu": results["bleu"]}  # 计算并返回BLEU分数
        except ZeroDivisionError:
            return {"bleu": 0.0}  # 如果出现除零错误，返回0.0


class EditDistanceMetric(BaseMetric):
    """A class used to compute the normalized Edit Distance metric.

    Edit Distance measures the minimum number of single-character edits required to change
    one string into another. This implementation normalizes the score by the length of the
    longer string to produce a value between 0 and 1.
    
    用于计算归一化编辑距离指标的类。
    编辑距离衡量将一个字符串转换为另一个字符串所需的最少单字符编辑次数。
    此实现通过较长字符串的长度对分数进行归一化，以生成0到1之间的值。
    """

    name = "edit_distance"  # 指标名称

    def describe(self) -> list[str]:
        """Returns a list of metric names that this class will compute.

        Returns:
            List[str]: A list of metric names.
        
        返回该类将计算的指标名称列表。
        返回值：
            List[str]: 指标名称列表。
        """
        return ["edit_distance"]  # 返回包含"edit_distance"的列表

    def compute(self, targets: list[str], predictions: list[str]) -> dict[str, float]:
        """Computes the normalized Edit Distance metric based on the targets and predictions.

        Args:
            targets (List[str]): The ground truth texts.
            predictions (List[str]): The predicted texts to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the computed normalized Edit Distance,
                with the metric name ("edit_distance") as the key and its value as the score.
        
        基于目标和预测计算归一化编辑距离指标。
        参数：
            targets (List[str]): 真实文本。
            predictions (List[str]): 要评估的预测文本。
        返回值：
            Dict[str, float]: 包含计算出的归一化编辑距离的字典，以指标名称("edit_distance")为键，其值为分数。
        """
        if len(targets) != len(predictions):
            raise ValueError("The number of targets and predictions must be the same.")
            # 如果目标和预测的数量不一致，抛出错误

        scores = []
        for prediction, target in zip(predictions, targets):
            score = edit_distance(prediction, target)  # 计算编辑距离
            score = score / max(len(prediction), len(target))  # 归一化处理
            scores.append(score)  # 将分数加入列表

        average_score = sum(scores) / len(scores)  # 计算平均分数
        return {"edit_distance": average_score}  # 返回平均分数


class MetricsTracker:
    @classmethod
    def init(cls, metrics: list[str]) -> MetricsTracker:
        """初始化MetricsTracker类，创建包含指定指标的字典。
        
        Args:
            metrics (list[str]): 指标名称列表
        Returns:
            MetricsTracker: 初始化后的MetricsTracker实例
        """
        return cls(metrics={metric: [] for metric in metrics})  # 为每个指标创建空列表

    def __init__(self, metrics: dict[str, list[tuple[int, int, float]]]) -> None:
        """构造函数，初始化指标跟踪器。
        
        Args:
            metrics (dict[str, list[tuple[int, int, float]]]): 指标字典，键为指标名称，值为(epoch, step, value)元组列表
        """
        self._metrics = metrics  # 存储指标数据

    def register(self, metric: str, epoch: int, step: int, value: float) -> None:
        """注册一个新的指标值。
        
        Args:
            metric (str): 指标名称
            epoch (int): 当前epoch
            step (int): 当前step
            value (float): 指标值
        """
        self._metrics[metric].append((epoch, step, value))  # 将新值添加到对应指标的列表中

    def describe_metrics(self) -> list[str]:
        """返回当前跟踪的所有指标名称列表。
        
        Returns:
            list[str]: 指标名称列表
        """
        return list(self._metrics.keys())  # 返回所有指标名称

    def get_metric_values(
        self,
        metric: str,
        with_index: bool = True,
    ) -> list:
        """获取指定指标的值。
        
        Args:
            metric (str): 指标名称
            with_index (bool): 是否返回包含epoch和step的完整元组
        Returns:
            list: 指标值列表，格式取决于with_index参数
        """
        if with_index:
            return self._metrics[metric]  # 返回完整元组
        return [value[2] for value in self._metrics[metric]]  # 仅返回值

    def as_json(
        self, output_dir: Optional[str] = None, filename: Optional[str] = None
    ) -> dict[str, list[dict[str, float]]]:
        """将指标数据转换为JSON格式，并可选择保存到文件。
        
        Args:
            output_dir (Optional[str]): 输出目录
            filename (Optional[str]): 文件名
        Returns:
            dict[str, list[dict[str, float]]]: JSON格式的指标数据
        """
        metrics_data = {}
        for metric, values in self._metrics.items():
            metrics_data[metric] = [{"epoch": epoch, "step": step, "value": value} for epoch, step, value in values]
            # 将数据转换为字典格式

        if output_dir and filename:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  # 创建输出目录
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as file:
                json.dump(metrics_data, file, indent=4)  # 将数据写入文件

        return metrics_data


def aggregate_by_epoch(metric_values: list[tuple[int, int, float]]) -> dict[int, float]:
    """按epoch聚合指标值，计算每个epoch的平均值。

    Args:
        metric_values (List[Tuple[int, int, float]]): 包含(epoch, step, value)的元组列表

    Returns:
        Dict[int, float]: 以epoch为键，平均指标值为值的字典
    """
    epoch_data = defaultdict(list)
    for epoch, step, value in metric_values:
        epoch_data[epoch].append(value)  # 按epoch分组
    avg_per_epoch = {epoch: sum(values) / len(values) for epoch, values in epoch_data.items()}
    return avg_per_epoch  # 计算每个epoch的平均值


def save_metric_plots(training_tracker: MetricsTracker, validation_tracker: MetricsTracker, output_dir: str) -> None:
    """保存训练和验证指标随epoch变化的图表。

    Args:
        training_tracker (MetricsTracker): 包含训练指标的跟踪器
        validation_tracker (MetricsTracker): 包含验证指标的跟踪器
        output_dir (str): 保存生成图表的目录

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出目录

    training_metrics = training_tracker.describe_metrics()
    validation_metrics = validation_tracker.describe_metrics()
    all_metrics = set(training_metrics + validation_metrics)  # 获取所有指标

    for metric in all_metrics:
        plt.figure(figsize=(8, 6))  # 创建新图表

        if metric in training_metrics:
            training_values = training_tracker.get_metric_values(metric=metric, with_index=True)
            training_avg_values = aggregate_by_epoch(training_values)  # 计算训练指标平均值
            training_epochs = sorted(training_avg_values.keys())
            training_vals = [training_avg_values[epoch] for epoch in training_epochs]
            plt.plot(
                training_epochs, training_vals, label=f"Training {metric}", marker="o", linestyle="-", color="blue"
            )  # 绘制训练曲线

        if metric in validation_metrics:
            validation_values = validation_tracker.get_metric_values(metric=metric, with_index=True)
            validation_avg_values = aggregate_by_epoch(validation_values)  # 计算验证指标平均值
            validation_epochs = sorted(validation_avg_values.keys())
            validation_vals = [validation_avg_values[epoch] for epoch in validation_epochs]
            plt.plot(
                validation_epochs,
                validation_vals,
                label=f"Validation {metric}",
                marker="o",
                linestyle="--",
                color="orange",
            )  # 绘制验证曲线

        plt.title(f"{metric.capitalize()} over Epochs")  # 设置图表标题
        plt.xlabel("Epoch")  # 设置x轴标签
        plt.ylabel(f"{metric.capitalize()} Value")  # 设置y轴标签
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格
        plt.savefig(f"{output_dir}/{metric}_plot.png")  # 保存图表
        plt.close()  # 关闭图表


METRIC_CLASSES: dict[str, type[BaseMetric]] = {
    MeanAveragePrecisionMetric.name: MeanAveragePrecisionMetric,
    BLEUMetric.name: BLEUMetric,
    EditDistanceMetric.name: EditDistanceMetric,
}  # 指标类映射字典


def parse_metrics(metrics: list[str]) -> list[BaseMetric]:
    """Parse metric names into metric objects.

    Args:
        metrics (List[str]): List of metric names.

    Returns:
        List[BaseMetric]: List of metric objects.

    Raises:
        ValueError: If an unsupported metric is provided.
    将指标名称解析为指标对象。

    参数:
        metrics (List[str]): 指标名称列表

    返回:
        List[BaseMetric]: 指标对象列表

    抛出:
        ValueError: 如果提供了不支持的指标
    """
    metric_objects = []
    for metric_name in metrics:
        metric_class = METRIC_CLASSES.get(metric_name.lower())
        if metric_class:
            metric_objects.append(metric_class())
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
    return metric_objects
