# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from functools import partial  # 从 functools 模块导入 partial 函数
from pathlib import Path  # 从 pathlib 模块导入 Path 类

import torch  # 导入 PyTorch 库

from ultralytics.utils import IterableSimpleNamespace, yaml_load  # 从 ultralytics.utils 导入 IterableSimpleNamespace 和 yaml_load
from ultralytics.utils.checks import check_yaml  # 从 ultralytics.utils.checks 导入 check_yaml

from .bot_sort import BOTSORT  # 从当前模块导入 BOTSORT 类
from .byte_tracker import BYTETracker  # 从当前模块导入 BYTETracker 类

# A mapping of tracker types to corresponding tracker classes  # 跟踪器类型与相应跟踪器类的映射
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}  # 定义跟踪器类型与类的映射字典


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    Initialize trackers for object tracking during prediction.  # 在预测过程中初始化对象跟踪器。

    Args:
        predictor (object): The predictor object to initialize trackers for.  # 要初始化跟踪器的预测器对象。
        persist (bool): Whether to persist the trackers if they already exist.  # 如果跟踪器已存在，是否保持它们。

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.  # 如果 tracker_type 不是 'bytetrack' 或 'botsort'，则引发 AssertionError。

    Examples:
        Initialize trackers for a predictor object:  # 初始化预测器对象的跟踪器：
        >>> predictor = SomePredictorClass()  # 创建预测器实例
        >>> on_predict_start(predictor, persist=True)  # 初始化跟踪器
    """
    if predictor.args.task == "classify":  # 如果任务是分类
        raise ValueError("❌ Classification doesn't support 'mode=track'")  # 抛出错误，分类不支持跟踪模式

    if hasattr(predictor, "trackers") and persist:  # 如果预测器已有跟踪器且需要保持
        return  # 直接返回

    tracker = check_yaml(predictor.args.tracker)  # 检查 YAML 配置文件
    cfg = IterableSimpleNamespace(**yaml_load(tracker))  # 加载 YAML 配置为可迭代的命名空间

    if cfg.tracker_type not in {"bytetrack", "botsort"}:  # 如果跟踪器类型不是 'bytetrack' 或 'botsort'
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")  # 抛出错误

    trackers = []  # 初始化跟踪器列表
    for _ in range(predictor.dataset.bs):  # 遍历数据集的批量大小
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)  # 根据配置创建跟踪器实例
        trackers.append(tracker)  # 将跟踪器添加到列表
        if predictor.dataset.mode != "stream":  # 如果不是流模式
            break  # 只需一个跟踪器

    predictor.trackers = trackers  # 将跟踪器列表赋值给预测器
    predictor.vid_path = [None] * predictor.dataset.bs  # 初始化视频路径列表，用于确定何时在新视频上重置跟踪器


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.  # 后处理检测到的边界框并更新对象跟踪。

    Args:
        predictor (object): The predictor object containing the predictions.  # 包含预测结果的预测器对象。
        persist (bool): Whether to persist the trackers if they already exist.  # 如果跟踪器已存在，是否保持它们。

    Examples:
        Postprocess predictions and update with tracking  # 后处理预测并更新跟踪
        >>> predictor = YourPredictorClass()  # 创建预测器实例
        >>> on_predict_postprocess_end(predictor, persist=True)  # 后处理预测
    """
    path, im0s = predictor.batch[:2]  # 获取批次的路径和图像

    is_obb = predictor.args.task == "obb"  # 判断任务是否为 OBB
    is_stream = predictor.dataset.mode == "stream"  # 判断数据集模式是否为流模式
    for i in range(len(im0s)):  # 遍历每个图像
        tracker = predictor.trackers[i if is_stream else 0]  # 根据模式选择跟踪器
        vid_path = predictor.save_dir / Path(path[i]).name  # 获取视频路径
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:  # 如果不保持且视频路径不同
            tracker.reset()  # 重置跟踪器
            predictor.vid_path[i if is_stream else 0] = vid_path  # 更新视频路径

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()  # 获取检测结果
        if len(det) == 0:  # 如果没有检测结果
            continue  # 跳过
        tracks = tracker.update(det, im0s[i])  # 更新跟踪器
        if len(tracks) == 0:  # 如果没有跟踪结果
            continue  # 跳过
        idx = tracks[:, -1].astype(int)  # 获取跟踪的索引
        predictor.results[i] = predictor.results[i][idx]  # 更新预测结果

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}  # 创建更新参数
        predictor.results[i].update(**update_args)  # 更新预测结果


def register_tracker(model: object, persist: bool) -> None:
    """
    Register tracking callbacks to the model for object tracking during prediction.  # 在预测过程中为模型注册跟踪回调。

    Args:
        model (object): The model object to register tracking callbacks for.  # 要为其注册跟踪回调的模型对象。
        persist (bool): Whether to persist the trackers if they already exist.  # 如果跟踪器已存在，是否保持它们。

    Examples:
        Register tracking callbacks to a YOLO model  # 为 YOLO 模型注册跟踪回调
        >>> model = YOLOModel()  # 创建 YOLO 模型实例
        >>> register_tracker(model, persist=True)  # 注册跟踪器
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))  # 注册预测开始时的回调
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))  # 注册后处理结束时的回调