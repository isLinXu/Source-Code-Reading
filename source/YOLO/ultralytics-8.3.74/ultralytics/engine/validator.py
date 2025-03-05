# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Check a model's accuracy on a test or val split of a dataset.
检查模型在数据集的测试或验证分割上的准确性。

Usage:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640
使用方法：
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""

import json  # 导入json模块，用于处理JSON数据
import time  # 导入time模块，用于时间相关操作
from pathlib import Path  # 从pathlib模块导入Path类，用于处理路径

import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习

from ultralytics.cfg import get_cfg, get_save_dir  # 从ultralytics.cfg模块导入get_cfg和get_save_dir函数
from ultralytics.data.utils import check_cls_dataset, check_det_dataset  # 从ultralytics.data.utils模块导入数据集检查函数
from ultralytics.nn.autobackend import AutoBackend  # 从ultralytics.nn.autobackend模块导入AutoBackend类
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis  # 从ultralytics.utils模块导入工具函数
from ultralytics.utils.checks import check_imgsz  # 从ultralytics.utils.checks模块导入检查图像大小的函数
from ultralytics.utils.ops import Profile  # 从ultralytics.utils.ops模块导入Profile类
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode  # 导入PyTorch相关工具函数

class BaseValidator:
    """
    BaseValidator.
    基础验证器。

    A base class for creating validators.
    用于创建验证器的基类。

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.
        初始化BaseValidator实例。

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)  # 获取配置
        self.dataloader = dataloader  # 设置数据加载器
        self.pbar = pbar  # 设置进度条
        self.stride = None  # 初始化步幅为None
        self.data = None  # 初始化数据为None
        self.device = None  # 初始化设备为None
        self.batch_i = None  # 初始化批次索引为None
        self.training = True  # 初始化训练模式为True
        self.names = None  # 初始化类名为None
        self.seen = None  # 初始化已见图像数量为None
        self.stats = None  # 初始化统计信息为None
        self.confusion_matrix = None  # 初始化混淆矩阵为None
        self.nc = None  # 初始化类别数量为None
        self.iouv = None  # 初始化IoU阈值为None
        self.jdict = None  # 初始化JSON验证结果字典为None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}  # 初始化处理速度字典

        self.save_dir = save_dir or get_save_dir(self.args)  # 设置保存结果的目录
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # 创建保存标签的目录
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001 默认置信度为0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)  # 检查图像大小

        self.plots = {}  # 初始化绘图字典
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 获取默认回调函数

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics.
        执行验证过程，在数据加载器上运行推理并计算性能指标。"""
        self.training = trainer is not None  # 判断是否在训练模式
        augment = self.args.augment and (not self.training)  # 判断是否进行数据增强
        if self.training:
            self.device = trainer.device  # 获取训练设备
            self.data = trainer.data  # 获取训练数据
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp  # 在训练期间强制使用FP16
            model = trainer.ema.ema or trainer.model  # 获取模型
            model = model.half() if self.args.half else model.float()  # 根据是否使用FP16转换模型
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)  # 初始化损失
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)  # 更新绘图参数
            model.eval()  # 设置模型为评估模式
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")  # 警告：验证未训练的模型将导致0 mAP
            callbacks.add_integration_callbacks(self)  # 添加集成回调
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),  # 选择设备
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )  # 创建AutoBackend实例
            # self.model = model
            self.device = model.device  # 更新设备
            self.args.half = model.fp16  # 更新FP16参数
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine  # 获取模型的步幅和其他参数
            imgsz = check_imgsz(self.args.imgsz, stride=stride)  # 检查图像大小
            if engine:
                self.args.batch = model.batch_size  # 更新批次大小
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py模型默认批次大小为1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")  # 日志记录当前批次大小

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)  # 检查检测数据集
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)  # 检查分类数据集
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))  # 抛出数据集未找到的异常

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # 在CPU或MPS上进行更快的验证，因为时间主要由推理而不是数据加载决定
            if not pt:
                self.args.rect = False  # 设置长方形输入
            self.stride = model.stride  # 在get_dataloader()中用于填充
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)  # 获取数据加载器

            model.eval()  # 设置模型为评估模式
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # 预热模型

        self.run_callbacks("on_val_start")  # 运行验证开始的回调
        dt = (
            Profile(device=self.device),  # 记录预处理时间
            Profile(device=self.device),  # 记录推理时间
            Profile(device=self.device),  # 记录损失计算时间
            Profile(device=self.device),  # 记录后处理时间
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))  # 创建进度条
        self.init_metrics(de_parallel(model))  # 初始化性能指标
        self.jdict = []  # 在每次验证前清空JSON结果字典
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")  # 运行每个批次开始的回调
            self.batch_i = batch_i  # 更新当前批次索引
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)  # 预处理批次数据

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)  # 进行推理

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]  # 计算损失并累加

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)  # 后处理预测结果

            self.update_metrics(preds, batch)  # 更新指标
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)  # 绘制验证样本
                self.plot_predictions(batch, preds, batch_i)  # 绘制预测结果

            self.run_callbacks("on_val_batch_end")  # 运行每个批次结束的回调
        stats = self.get_stats()  # 获取统计信息
        self.check_stats(stats)  # 检查统计信息
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))  # 计算处理速度
        self.finalize_metrics()  # 完成指标计算
        self.print_results()  # 打印结果
        self.run_callbacks("on_val_end")  # 运行验证结束的回调
        if self.training:
            model.float()  # 将模型转换为浮点模式
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}  # 返回结果
            return {k: round(float(v), 5) for k, v in results.items()}  # 将结果四舍五入到5位小数
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )  # 日志记录每张图像的处理速度
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")  # 日志记录保存文件
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # 更新统计信息
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")  # 日志记录结果保存位置
            return stats  # 返回统计信息

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.
        使用IoU将预测与真实对象（pred_classes，true_classes）匹配。

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)  # 初始化正确匹配矩阵
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes  # 判断预测类别与真实类别是否匹配
        iou = iou * correct_class  # zero out the wrong classes 将错误类别的IoU置为0
        iou = iou.cpu().numpy()  # 转换为numpy数组
        for i, threshold in enumerate(self.iouv.cpu().tolist()):  # 遍历每个IoU阈值
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)  # 计算成本矩阵
                if cost_matrix.any():  # 如果成本矩阵中有值
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)  # 使用匈牙利算法进行匹配
                    valid = cost_matrix[labels_idx, detections_idx] > 0  # 判断有效匹配
                    if valid.any():  # 如果有有效匹配
                        correct[detections_idx[valid], i] = True  # 更新正确匹配矩阵
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T  # 转换为数组
                if matches.shape[0]:  # 如果有匹配
                    if matches.shape[0] > 1:  # 如果有多个匹配
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]  # 根据IoU排序
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 去重
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 去重
                    correct[matches[:, 1].astype(int), i] = True  # 更新正确匹配矩阵
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)  # 返回正确匹配的张量

    def add_callback(self, event: str, callback):
        """Appends the given callback.
        添加给定的回调函数。"""
        self.callbacks[event].append(callback)  # 将回调函数添加到指定事件的回调列表中

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event.
        运行与指定事件相关的所有回调函数。"""
        for callback in self.callbacks.get(event, []):  # 遍历事件的回调函数
            callback(self)  # 执行回调函数

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size.
        从数据集路径和批次大小获取数据加载器。"""
        raise NotImplementedError("get_dataloader function not implemented for this validator")  # 抛出未实现异常

    def build_dataset(self, img_path):
        """Build dataset.
        构建数据集。"""
        raise NotImplementedError("build_dataset function not implemented in validator")  # 抛出未实现异常

    def preprocess(self, batch):
        """Preprocesses an input batch.
        对输入批次进行预处理。"""
        return batch  # 返回原始批次

    def postprocess(self, preds):
        """Preprocesses the predictions.
        对预测结果进行后处理。"""
        return preds  # 返回原始预测结果

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model.
        初始化YOLO模型的性能指标。"""
        pass  # 占位符

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch.
        根据预测结果和批次更新指标。"""
        pass  # 占位符

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics.
        完成并返回所有指标。"""
        pass  # 占位符

    def get_stats(self):
        """Returns statistics about the model's performance.
        返回模型性能的统计信息。"""
        return {}  # 返回空字典

    def check_stats(self, stats):
        """Checks statistics.
        检查统计信息。"""
        pass  # 占位符

    def print_results(self):
        """Prints the results of the model's predictions.
        打印模型预测结果。"""
        pass  # 占位符

    def get_desc(self):
        """Get description of the YOLO model.
        获取YOLO模型的描述。"""
        pass  # 占位符

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation.
        返回YOLO训练/验证中使用的指标键。"""
        return []  # 返回空列表

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks).
        注册绘图（例如，供回调函数使用）。"""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}  # 将绘图数据和时间戳存储在字典中

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training.
        在训练期间绘制验证样本。"""
        pass  # 占位符

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images.
        在批次图像上绘制YOLO模型的预测结果。"""
        pass  # 占位符

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format.
        将预测结果转换为JSON格式。"""
        pass  # 占位符

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics.
        评估并返回预测统计信息的JSON格式。"""
        pass  # 占位符