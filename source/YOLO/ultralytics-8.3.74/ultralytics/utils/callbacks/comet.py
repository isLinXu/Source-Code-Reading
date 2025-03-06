# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, ops  # 从 ultralytics.utils 导入 LOGGER、RANK、SETTINGS、TESTS_RUNNING 和 ops
from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics, OBBMetrics, PoseMetrics, SegmentMetrics  # 从 ultralytics.utils.metrics 导入多种指标类

try:
    assert not TESTS_RUNNING  # do not log pytest  # 确保不在 pytest 中记录
    assert SETTINGS["comet"] is True  # verify integration is enabled  # 验证集成是否启用
    import comet_ml  # 导入 comet_ml 库

    assert hasattr(comet_ml, "__version__")  # verify package is not directory  # 验证包不是目录

    import os  # 导入 os 模块
    from pathlib import Path  # 从 pathlib 导入 Path 类

    # Ensures certain logging functions only run for supported tasks  # 确保某些日志记录功能仅在支持的任务中运行
    COMET_SUPPORTED_TASKS = ["detect"]  # 支持的任务列表

    # Names of plots created by Ultralytics that are logged to Comet  # Ultralytics 创建的图表名称，将记录到 Comet
    CONFUSION_MATRIX_PLOT_NAMES = "confusion_matrix", "confusion_matrix_normalized"  # 混淆矩阵图名称
    EVALUATION_PLOT_NAMES = "F1_curve", "P_curve", "R_curve", "PR_curve"  # 评估图名称
    LABEL_PLOT_NAMES = "labels", "labels_correlogram"  # 标签图名称
    SEGMENT_METRICS_PLOT_PREFIX = "Box", "Mask"  # 分割指标图前缀
    POSE_METRICS_PLOT_PREFIX = "Box", "Pose"  # 姿态指标图前缀

    _comet_image_prediction_count = 0  # 初始化图像预测计数

except (ImportError, AssertionError):  # 捕获导入错误或断言错误
    comet_ml = None  # 如果导入失败或断言失败，则将 comet_ml 设置为 None


def _get_comet_mode():
    """Returns the mode of comet set in the environment variables, defaults to 'online' if not set.  # 返回环境变量中设置的 comet 模式，如果未设置则默认为 'online'。"""
    return os.getenv("COMET_MODE", "online")  # 获取 COMET_MODE 环境变量的值


def _get_comet_model_name():
    """Returns the model name for Comet from the environment variable COMET_MODEL_NAME or defaults to 'Ultralytics'.  # 从环境变量 COMET_MODEL_NAME 返回 Comet 的模型名称，默认为 'Ultralytics'。"""
    return os.getenv("COMET_MODEL_NAME", "Ultralytics")  # 获取 COMET_MODEL_NAME 环境变量的值


def _get_eval_batch_logging_interval():
    """Get the evaluation batch logging interval from environment variable or use default value 1.  # 从环境变量获取评估批次日志记录间隔，默认为 1。"""
    return int(os.getenv("COMET_EVAL_BATCH_LOGGING_INTERVAL", 1))  # 获取 COMET_EVAL_BATCH_LOGGING_INTERVAL 环境变量的值


def _get_max_image_predictions_to_log():
    """Get the maximum number of image predictions to log from the environment variables.  # 从环境变量获取要记录的最大图像预测数量。"""
    return int(os.getenv("COMET_MAX_IMAGE_PREDICTIONS", 100))  # 获取 COMET_MAX_IMAGE_PREDICTIONS 环境变量的值


def _scale_confidence_score(score):
    """Scales the given confidence score by a factor specified in an environment variable.  # 根据环境变量中指定的因子缩放给定的置信度分数。"""
    scale = float(os.getenv("COMET_MAX_CONFIDENCE_SCORE", 100.0))  # 获取 COMET_MAX_CONFIDENCE_SCORE 环境变量的值
    return score * scale  # 返回缩放后的置信度分数


def _should_log_confusion_matrix():
    """Determines if the confusion matrix should be logged based on the environment variable settings.  # 根据环境变量设置确定是否应记录混淆矩阵。"""
    return os.getenv("COMET_EVAL_LOG_CONFUSION_MATRIX", "false").lower() == "true"  # 检查环境变量


def _should_log_image_predictions():
    """Determines whether to log image predictions based on a specified environment variable.  # 根据指定的环境变量确定是否记录图像预测。"""
    return os.getenv("COMET_EVAL_LOG_IMAGE_PREDICTIONS", "true").lower() == "true"  # 检查环境变量


def _get_experiment_type(mode, project_name):
    """Return an experiment based on mode and project name.  # 根据模式和项目名称返回实验。"""
    if mode == "offline":  # 如果模式为离线
        return comet_ml.OfflineExperiment(project_name=project_name)  # 返回离线实验

    return comet_ml.Experiment(project_name=project_name)  # 返回在线实验


def _create_experiment(args):
    """Ensures that the experiment object is only created in a single process during distributed training.  # 确保在分布式训练期间仅在单个进程中创建实验对象。"""
    if RANK not in {-1, 0}:  # 如果当前进程不是主进程
        return  # 退出函数
    try:
        comet_mode = _get_comet_mode()  # 获取当前的 comet 模式
        _project_name = os.getenv("COMET_PROJECT_NAME", args.project)  # 获取项目名称
        experiment = _get_experiment_type(comet_mode, _project_name)  # 创建实验对象
        experiment.log_parameters(vars(args))  # 记录参数
        experiment.log_others(  # 记录其他信息
            {
                "eval_batch_logging_interval": _get_eval_batch_logging_interval(),  # 记录评估批次日志记录间隔
                "log_confusion_matrix_on_eval": _should_log_confusion_matrix(),  # 记录评估时的混淆矩阵
                "log_image_predictions": _should_log_image_predictions(),  # 记录图像预测
                "max_image_predictions": _get_max_image_predictions_to_log(),  # 记录最大图像预测数量
            }
        )
        experiment.log_other("Created from", "ultralytics")  # 记录来源

    except Exception as e:  # 捕获异常
        LOGGER.warning(f"WARNING ⚠️ Comet installed but not initialized correctly, not logging this run. {e}")  # 记录警告信息


def _fetch_trainer_metadata(trainer):
    """Returns metadata for YOLO training including epoch and asset saving status.  # 返回 YOLO 训练的元数据，包括周期和资产保存状态。"""
    curr_epoch = trainer.epoch + 1  # 当前周期

    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size  # 每个周期的训练步数
    curr_step = curr_epoch * train_num_steps_per_epoch  # 当前步数
    final_epoch = curr_epoch == trainer.epochs  # 是否为最后一个周期

    save = trainer.args.save  # 保存参数
    save_period = trainer.args.save_period  # 保存周期
    save_interval = curr_epoch % save_period == 0  # 是否在保存周期内
    save_assets = save and save_period > 0 and save_interval and not final_epoch  # 是否保存资产

    return dict(curr_epoch=curr_epoch, curr_step=curr_step, save_assets=save_assets, final_epoch=final_epoch)  # 返回元数据字典


def _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad):
    """
    YOLO resizes images during training and the label values are normalized based on this resized shape.  # YOLO 在训练期间调整图像大小，标签值基于此调整后的形状进行归一化。

    This function rescales the bounding box labels to the original image shape.  # 此函数将边界框标签重新缩放到原始图像形状。
    """
    resized_image_height, resized_image_width = resized_image_shape  # 获取调整后图像的高度和宽度

    # Convert normalized xywh format predictions to xyxy in resized scale format  # 将归一化的 xywh 格式预测转换为调整后比例的 xyxy 格式
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)  # 转换边界框格式
    # Scale box predictions from resized image scale back to original image scale  # 将边界框预测从调整后图像比例缩放回原始图像比例
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)  # 缩放边界框
    # Convert bounding box format from xyxy to xywh for Comet logging  # 将边界框格式从 xyxy 转换为 xywh 以便记录到 Comet
    box = ops.xyxy2xywh(box)  # 转换格式
    # Adjust xy center to correspond top-left corner  # 调整 xy 中心以对应左上角
    box[:2] -= box[2:] / 2  # 调整中心点
    box = box.tolist()  # 转换为列表

    return box  # 返回边界框


def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map=None):
    """Format ground truth annotations for detection.  # 格式化用于检测的真实标签注释。"""
    indices = batch["batch_idx"] == img_idx  # 获取当前图像的索引
    bboxes = batch["bboxes"][indices]  # 获取当前图像的边界框
    if len(bboxes) == 0:  # 如果没有边界框
        LOGGER.debug(f"COMET WARNING: Image: {image_path} has no bounding boxes labels")  # 记录警告信息
        return None  # 返回 None

    cls_labels = batch["cls"][indices].squeeze(1).tolist()  # 获取类别标签
    if class_name_map:  # 如果有类别名称映射
        cls_labels = [str(class_name_map[label]) for label in cls_labels]  # 将标签映射为名称

    original_image_shape = batch["ori_shape"][img_idx]  # 获取原始图像形状
    resized_image_shape = batch["resized_shape"][img_idx]  # 获取调整后图像形状
    ratio_pad = batch["ratio_pad"][img_idx]  # 获取填充比例

    data = []  # 初始化数据列表
    for box, label in zip(bboxes, cls_labels):  # 遍历边界框和标签
        box = _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad)  # 缩放边界框
        data.append(  # 添加到数据列表
            {
                "boxes": [box],  # 边界框
                "label": f"gt_{label}",  # 标签
                "score": _scale_confidence_score(1.0),  # 置信度分数
            }
        )

    return {"name": "ground_truth", "data": data}  # 返回格式化的真实标签


def _format_prediction_annotations_for_detection(image_path, metadata, class_label_map=None):
    """Format YOLO predictions for object detection visualization.  # 格式化 YOLO 预测以便于对象检测可视化。"""
    stem = image_path.stem  # 获取图像文件名
    image_id = int(stem) if stem.isnumeric() else stem  # 如果文件名是数字，则转换为整数

    predictions = metadata.get(image_id)  # 获取当前图像的预测
    if not predictions:  # 如果没有预测
        LOGGER.debug(f"COMET WARNING: Image: {image_path} has no bounding boxes predictions")  # 记录警告信息
        return None  # 返回 None

    data = []  # 初始化数据列表
    for prediction in predictions:  # 遍历预测
        boxes = prediction["bbox"]  # 获取边界框
        score = _scale_confidence_score(prediction["score"])  # 缩放置信度分数
        cls_label = prediction["category_id"]  # 获取类别标签
        if class_label_map:  # 如果有类别标签映射
            cls_label = str(class_label_map[cls_label])  # 将标签映射为名称

        data.append({"boxes": [boxes], "label": cls_label, "score": score})  # 添加到数据列表

    return {"name": "prediction", "data": data}  # 返回格式化的预测


def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map, class_label_map):
    """Join the ground truth and prediction annotations if they exist.  # 如果存在，则合并真实标签和预测注释。"""
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(  # 格式化真实标签
        img_idx, image_path, batch, class_label_map
    )
    prediction_annotations = _format_prediction_annotations_for_detection(  # 格式化预测
        image_path, prediction_metadata_map, class_label_map
    )

    annotations = [  # 合并注释
        annotation for annotation in [ground_truth_annotations, prediction_annotations] if annotation is not None
    ]
    return [annotations] if annotations else None  # 返回注释


def _create_prediction_metadata_map(model_predictions):
    """Create metadata map for model predictions by groupings them based on image ID.  # 根据图像 ID 创建模型预测的元数据映射。"""
    pred_metadata_map = {}  # 初始化预测元数据映射
    for prediction in model_predictions:  # 遍历模型预测
        pred_metadata_map.setdefault(prediction["image_id"], [])  # 初始化图像 ID 的列表
        pred_metadata_map[prediction["image_id"]].append(prediction)  # 添加预测到列表

    return pred_metadata_map  # 返回预测元数据映射


def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch):
    """Log the confusion matrix to Comet experiment.  # 将混淆矩阵记录到 Comet 实验中。"""
    conf_mat = trainer.validator.confusion_matrix.matrix  # 获取混淆矩阵
    names = list(trainer.data["names"].values()) + ["background"]  # 获取类别名称
    experiment.log_confusion_matrix(  # 记录混淆矩阵
        matrix=conf_mat, labels=names, max_categories=len(names), epoch=curr_epoch, step=curr_step
    )


def _log_images(experiment, image_paths, curr_step, annotations=None):
    """Logs images to the experiment with optional annotations.  # 将图像记录到实验中，并可选地添加注释。"""
    if annotations:  # 如果有注释
        for image_path, annotation in zip(image_paths, annotations):  # 遍历图像路径和注释
            experiment.log_image(image_path, name=image_path.stem, step=curr_step, annotations=annotation)  # 记录图像

    else:  # 如果没有注释
        for image_path in image_paths:  # 遍历图像路径
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)  # 记录图像


def _log_image_predictions(experiment, validator, curr_step):
    """Logs predicted boxes for a single image during training.  # 在训练期间记录单个图像的预测边界框。"""
    global _comet_image_prediction_count  # 声明全局变量

    task = validator.args.task  # 获取任务
    if task not in COMET_SUPPORTED_TASKS:  # 如果任务不在支持的任务列表中
        return  # 退出函数

    jdict = validator.jdict  # 获取验证器的字典
    if not jdict:  # 如果字典为空
        return  # 退出函数

    predictions_metadata_map = _create_prediction_metadata_map(jdict)  # 创建预测元数据映射
    dataloader = validator.dataloader  # 获取数据加载器
    class_label_map = validator.names  # 获取类别标签映射

    batch_logging_interval = _get_eval_batch_logging_interval()  # 获取批次日志记录间隔
    max_image_predictions = _get_max_image_predictions_to_log()  # 获取最大图像预测数量

    for batch_idx, batch in enumerate(dataloader):  # 遍历数据加载器
        if (batch_idx + 1) % batch_logging_interval != 0:  # 如果当前批次不是记录批次
            continue  # 跳过当前批次

        image_paths = batch["im_file"]  # 获取图像路径
        for img_idx, image_path in enumerate(image_paths):  # 遍历图像路径
            if _comet_image_prediction_count >= max_image_predictions:  # 如果达到最大预测数量
                return  # 退出函数

            image_path = Path(image_path)  # 转换为 Path 对象
            annotations = _fetch_annotations(  # 获取注释
                img_idx,
                image_path,
                batch,
                predictions_metadata_map,
                class_label_map,
            )
            _log_images(  # 记录图像
                experiment,
                [image_path],
                curr_step,
                annotations=annotations,
            )
            _comet_image_prediction_count += 1  # 增加预测计数


def _log_plots(experiment, trainer):
    """Logs evaluation plots and label plots for the experiment.  # 记录评估图和标签图到实验中。"""
    plot_filenames = None  # 初始化图像文件名
    if isinstance(trainer.validator.metrics, SegmentMetrics) and trainer.validator.metrics.task == "segment":  # 如果是分割任务
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"  # 获取图像文件名
            for plots in EVALUATION_PLOT_NAMES  # 遍历评估图名称
            for prefix in SEGMENT_METRICS_PLOT_PREFIX  # 遍历分割指标前缀
        ]
    elif isinstance(trainer.validator.metrics, PoseMetrics):  # 如果是姿态任务
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"  # 获取图像文件名
            for plots in EVALUATION_PLOT_NAMES  # 遍历评估图名称
            for prefix in POSE_METRICS_PLOT_PREFIX  # 遍历姿态指标前缀
        ]
    elif isinstance(trainer.validator.metrics, (DetMetrics, OBBMetrics)):  # 如果是检测任务
        plot_filenames = [trainer.save_dir / f"{plots}.png" for plots in EVALUATION_PLOT_NAMES]  # 获取图像文件名

    if plot_filenames is not None:  # 如果有图像文件名
        _log_images(experiment, plot_filenames, None)  # 记录图像

    confusion_matrix_filenames = [trainer.save_dir / f"{plots}.png" for plots in CONFUSION_MATRIX_PLOT_NAMES]  # 获取混淆矩阵图像文件名
    _log_images(experiment, confusion_matrix_filenames, None)  # 记录混淆矩阵图像

    if not isinstance(trainer.validator.metrics, ClassifyMetrics):  # 如果不是分类任务
        label_plot_filenames = [trainer.save_dir / f"{labels}.jpg" for labels in LABEL_PLOT_NAMES]  # 获取标签图像文件名
        _log_images(experiment, label_plot_filenames, None)  # 记录标签图像


def _log_model(experiment, trainer):
    """Log the best-trained model to Comet.ml.  # 将最佳训练模型记录到 Comet.ml。"""
    model_name = _get_comet_model_name()  # 获取模型名称
    experiment.log_model(model_name, file_or_folder=str(trainer.best), file_name="best.pt", overwrite=True)  # 记录模型


def on_pretrain_routine_start(trainer):
    """Creates or resumes a CometML experiment at the start of a YOLO pre-training routine.  # 在 YOLO 预训练例程开始时创建或恢复 CometML 实验。"""
    experiment = comet_ml.get_global_experiment()  # 获取全局实验
    is_alive = getattr(experiment, "alive", False)  # 检查实验是否存活
    if not experiment or not is_alive:  # 如果实验不存在或未存活
        _create_experiment(trainer.args)  # 创建实验


def on_train_epoch_end(trainer):
    """Log metrics and save batch images at the end of training epochs.  # 在训练周期结束时记录指标和保存批次图像。"""
    experiment = comet_ml.get_global_experiment()  # 获取全局实验
    if not experiment:  # 如果实验不存在
        return  # 退出函数

    metadata = _fetch_trainer_metadata(trainer)  # 获取训练元数据
    curr_epoch = metadata["curr_epoch"]  # 当前周期
    curr_step = metadata["curr_step"]  # 当前步数

    experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix="train"), step=curr_step, epoch=curr_epoch)  # 记录训练损失


def on_fit_epoch_end(trainer):
    """Logs model assets at the end of each epoch.  # 在每个周期结束时记录模型资产。"""
    experiment = comet_ml.get_global_experiment()  # 获取全局实验
    if not experiment:  # 如果实验不存在
        return  # 退出函数

    metadata = _fetch_trainer_metadata(trainer)  # 获取训练元数据
    curr_epoch = metadata["curr_epoch"]  # 当前周期
    curr_step = metadata["curr_step"]  # 当前步数
    save_assets = metadata["save_assets"]  # 是否保存资产

    experiment.log_metrics(trainer.metrics, step=curr_step, epoch=curr_epoch)  # 记录指标
    experiment.log_metrics(trainer.lr, step=curr_step, epoch=curr_epoch)  # 记录学习率
    if curr_epoch == 1:  # 如果是第一个周期
        from ultralytics.utils.torch_utils import model_info_for_loggers  # 从 ultralytics.utils.torch_utils 导入 model_info_for_loggers

        experiment.log_metrics(model_info_for_loggers(trainer), step=curr_step, epoch=curr_epoch)  # 记录模型信息

    if not save_assets:  # 如果不保存资产
        return  # 退出函数

    _log_model(experiment, trainer)  # 记录模型
    if _should_log_confusion_matrix():  # 如果应记录混淆矩阵
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)  # 记录混淆矩阵
    if _should_log_image_predictions():  # 如果应记录图像预测
        _log_image_predictions(experiment, trainer.validator, curr_step)  # 记录图像预测


def on_train_end(trainer):
    """Perform operations at the end of training.  # 在训练结束时执行操作。"""
    experiment = comet_ml.get_global_experiment()  # 获取全局实验
    if not experiment:  # 如果实验不存在
        return  # 退出函数

    metadata = _fetch_trainer_metadata(trainer)  # 获取训练元数据
    curr_epoch = metadata["curr_epoch"]  # 当前周期
    curr_step = metadata["curr_step"]  # 当前步数
    plots = trainer.args.plots  # 获取绘图参数

    _log_model(experiment, trainer)  # 记录模型
    if plots:  # 如果有绘图参数
        _log_plots(experiment, trainer)  # 记录绘图

    _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)  # 记录混淆矩阵
    _log_image_predictions(experiment, trainer.validator, curr_step)  # 记录图像预测
    _log_images(experiment, trainer.save_dir.glob("train_batch*.jpg"), curr_step)  # 记录训练批次图像
    _log_images(experiment, trainer.save_dir.glob("val_batch*.jpg"), curr_step)  # 记录验证批次图像
    experiment.end()  # 结束实验

    global _comet_image_prediction_count  # 声明全局变量
    _comet_image_prediction_count = 0  # 重置图像预测计数


callbacks = (  # 定义回调函数
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,  # 预训练例程开始时的回调
        "on_train_epoch_end": on_train_epoch_end,  # 训练周期结束时的回调
        "on_fit_epoch_end": on_fit_epoch_end,  # 拟合周期结束时的回调
        "on_train_end": on_train_end,  # 训练结束时的回调
    }
    if comet_ml  # 如果 comet_ml 可用
    else {}  # 否则为空字典
)