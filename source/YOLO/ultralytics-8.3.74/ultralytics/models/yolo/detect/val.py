# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


# class DetectionValidator(BaseValidator):
#     """
#     A class extending the BaseValidator class for validation based on a detection model.

#     Example:
#         ```python
#         from ultralytics.models.yolo.detect import DetectionValidator

#         args = dict(model="yolo11n.pt", data="coco8.yaml")
#         validator = DetectionValidator(args=args)
#         validator()
#         ```
#     """

#     def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
#         """Initialize detection model with necessary variables and settings."""
#         super().__init__(dataloader, save_dir, pbar, args, _callbacks)
#         self.nt_per_class = None
#         self.nt_per_image = None
#         self.is_coco = False
#         self.is_lvis = False
#         self.class_map = None
#         self.args.task = "detect"
#         self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
#         self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
#         self.niou = self.iouv.numel()
#         self.lb = []  # for autolabelling
#         if self.args.save_hybrid:
#             LOGGER.warning(
#                 "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
#                 "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
#             )

#     def preprocess(self, batch):
#         """Preprocesses batch of images for YOLO training."""
#         batch["img"] = batch["img"].to(self.device, non_blocking=True)
#         batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
#         for k in ["batch_idx", "cls", "bboxes"]:
#             batch[k] = batch[k].to(self.device)

#         if self.args.save_hybrid:
#             height, width = batch["img"].shape[2:]
#             nb = len(batch["img"])
#             bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
#             self.lb = [
#                 torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
#                 for i in range(nb)
#             ]

#         return batch

#     def init_metrics(self, model):
#         """Initialize evaluation metrics for YOLO."""
#         val = self.data.get(self.args.split, "")  # validation path
#         self.is_coco = (
#             isinstance(val, str)
#             and "coco" in val
#             and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
#         )  # is COCO
#         self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
#         self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
#         self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
#         self.names = model.names
#         self.nc = len(model.names)
#         self.end2end = getattr(model, "end2end", False)
#         self.metrics.names = self.names
#         self.metrics.plot = self.args.plots
#         self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
#         self.seen = 0
#         self.jdict = []
#         self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

#     def get_desc(self):
#         """Return a formatted string summarizing class metrics of YOLO model."""
#         return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

#     def postprocess(self, preds):
#         """Apply Non-maximum suppression to prediction outputs."""
#         return ops.non_max_suppression(
#             preds,
#             self.args.conf,
#             self.args.iou,
#             labels=self.lb,
#             nc=self.nc,
#             multi_label=True,
#             agnostic=self.args.single_cls or self.args.agnostic_nms,
#             max_det=self.args.max_det,
#             end2end=self.end2end,
#             rotated=self.args.task == "obb",
#         )

#     def _prepare_batch(self, si, batch):
#         """Prepares a batch of images and annotations for validation."""
#         idx = batch["batch_idx"] == si
#         cls = batch["cls"][idx].squeeze(-1)
#         bbox = batch["bboxes"][idx]
#         ori_shape = batch["ori_shape"][si]
#         imgsz = batch["img"].shape[2:]
#         ratio_pad = batch["ratio_pad"][si]
#         if len(cls):
#             bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
#             ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
#         return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}
class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolo11n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """
    # 继承自BaseValidator的目标检测验证器类
    # 专门用于YOLO目标检测模型的验证

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        # 初始化检测模型的必要变量和设置
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        # 初始化各种统计变量
        self.nt_per_class = None     # 每个类别的目标数量
        self.nt_per_image = None     # 每个图像的目标数量
        self.is_coco = False         # 是否为COCO数据集
        self.is_lvis = False         # 是否为LVIS数据集
        self.class_map = None        # 类别映射
        
        # 设置任务为目标检测
        self.args.task = "detect"
        
        # 初始化检测指标
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        
        # 创建IoU向量，用于计算mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU向量，用于0.5到0.95的mAP计算
        self.niou = self.iouv.numel()  # IoU向量的元素数量
        
        # 用于自动标注的标签列表
        self.lb = []
        
        # 保存混合标签的警告
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' 将追加真实标签到预测结果以进行自动标注。\n"
                "WARNING ⚠️ 'save_hybrid=True' 将导致不正确的mAP计算。\n"
            )

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        # 预处理图像批次
        
        # 将图像移动到指定设备
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        
        # 图像归一化处理：半精度或全精度，并缩放到[0, 1]
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        
        # 将批次索引、类别和边界框移动到指定设备
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        # 如果启用混合保存，处理标签
        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            
            # 缩放边界框到图像尺寸
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            
            # 为每个批次创建标签
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        # 初始化YOLO的评估指标
        
        # 获取验证集路径
        val = self.data.get(self.args.split, "")
        
        # 判断是否为COCO数据集
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )
        
        # 判断是否为LVIS数据集
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco
        
        # 设置类别映射
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        
        # 根据数据集类型决定是否保存JSON结果
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training
        
        # 设置模型相关属性
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        
        # 配置指标
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        
        # 初始化混淆矩阵
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        
        # 初始化统计变量
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        # 返回格式化的指标摘要字符串
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        # 对预测输出应用非极大值抑制
        return ops.non_max_suppression(
            preds,                   # 原始预测结果
            self.args.conf,          # 置信度阈值
            self.args.iou,           # IoU阈值
            labels=self.lb,          # 标签
            nc=self.nc,              # 类别数量
            multi_label=True,        # 是否允许多标签
            agnostic=self.args.single_cls or self.args.agnostic_nms,  # 是否使用类别无关的NMS
            max_det=self.args.max_det,  # 最大检测数
            end2end=self.end2end,    # 是否为端到端模型
            rotated=self.args.task == "obb",  # 是否处理旋转边界框
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        # 为验证准备图像和标注批次
        
        # 获取特定批次的索引
        idx = batch["batch_idx"] == si
        
        # 提取类别和边界框
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        
        # 获取原始图像尺寸和处理后的图像尺寸
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        
        # 如果存在类别标签
        if len(cls):
            # 将边界框从XYWH转换为XYXY格式
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            
            # 将边界框缩放到原始图像尺寸
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
        
        # 返回处理后的批次信息
        return {
            "cls": cls, 
            "bbox": bbox, 
            "ori_shape": ori_shape, 
            "imgsz": imgsz, 
            "ratio_pad": ratio_pad
        }


    # def _prepare_pred(self, pred, pbatch):
    #     """Prepares a batch of images and annotations for validation."""
    #     predn = pred.clone()
    #     ops.scale_boxes(
    #         pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
    #     )  # native-space pred
    #     return predn

    # def update_metrics(self, preds, batch):
    #     """Metrics."""
    #     for si, pred in enumerate(preds):
    #         self.seen += 1
    #         npr = len(pred)
    #         stat = dict(
    #             conf=torch.zeros(0, device=self.device),
    #             pred_cls=torch.zeros(0, device=self.device),
    #             tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
    #         )
    #         pbatch = self._prepare_batch(si, batch)
    #         cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    #         nl = len(cls)
    #         stat["target_cls"] = cls
    #         stat["target_img"] = cls.unique()
    #         if npr == 0:
    #             if nl:
    #                 for k in self.stats.keys():
    #                     self.stats[k].append(stat[k])
    #                 if self.args.plots:
    #                     self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
    #             continue

    #         # Predictions
    #         if self.args.single_cls:
    #             pred[:, 5] = 0
    #         predn = self._prepare_pred(pred, pbatch)
    #         stat["conf"] = predn[:, 4]
    #         stat["pred_cls"] = predn[:, 5]

    #         # Evaluate
    #         if nl:
    #             stat["tp"] = self._process_batch(predn, bbox, cls)
    #         if self.args.plots:
    #             self.confusion_matrix.process_batch(predn, bbox, cls)
    #         for k in self.stats.keys():
    #             self.stats[k].append(stat[k])

    #         # Save
    #         if self.args.save_json:
    #             self.pred_to_json(predn, batch["im_file"][si])
    #         if self.args.save_txt:
    #             self.save_one_txt(
    #                 predn,
    #                 self.args.save_conf,
    #                 pbatch["ori_shape"],
    #                 self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
    #             )

    # def finalize_metrics(self, *args, **kwargs):
    #     """Set final values for metrics speed and confusion matrix."""
    #     self.metrics.speed = self.speed
    #     self.metrics.confusion_matrix = self.confusion_matrix

    # def get_stats(self):
    #     """Returns metrics statistics and results dictionary."""
    #     stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
    #     self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
    #     self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
    #     stats.pop("target_img", None)
    #     if len(stats) and stats["tp"].any():
    #         self.metrics.process(**stats)
    #     return self.metrics.results_dict

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        # 为验证准备预测结果
        
        # 克隆预测结果，避免修改原始数据
        predn = pred.clone()
        
        # 将预测框从处理后的图像尺寸缩放到原始图像尺寸
        ops.scale_boxes(
            pbatch["imgsz"],        # 处理后的图像尺寸
            predn[:, :4],           # 预测框坐标
            pbatch["ori_shape"],    # 原始图像尺寸
            ratio_pad=pbatch["ratio_pad"]  # 尺寸调整参数
        )
        
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        # 更新评估指标
        
        # 遍历每个预测结果
        for si, pred in enumerate(preds):
            # 累计已处理图像数量
            self.seen += 1
            
            # 获取预测框数量
            npr = len(pred)
            
            # 初始化统计字典
            stat = dict(
                # 置信度张量，初始为空
                conf=torch.zeros(0, device=self.device),
                # 预测类别张量，初始为空
                pred_cls=torch.zeros(0, device=self.device),
                # 真正例（True Positive）张量，初始为False
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            
            # 准备当前批次的标注信息
            pbatch = self._prepare_batch(si, batch)
            
            # 提取类别和边界框
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            
            # 获取真实标签数量
            nl = len(cls)
            
            # 记录目标类别和图像
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            
            # 如果没有预测框
            if npr == 0:
                # 如果存在真实标签
                if nl:
                    # 更新统计信息
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    
                    # 如果需要绘图，处理混淆矩阵
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 处理预测
            # 如果是单类别任务，强制将预测类别设为0
            if self.args.single_cls:
                pred[:, 5] = 0
            
            # 准备预测结果（缩放到原始图像尺寸）
            predn = self._prepare_pred(pred, pbatch)
            
            # 记录置信度和预测类别
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # 评估
            # 如果存在真实标签，计算真正例
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            
            # 如果需要绘图，处理混淆矩阵
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            
            # 更新统计信息
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # 保存结果
            # 如果需要保存JSON格式
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            
            # 如果需要保存文本格式
            if self.args.save_txt:
                self.save_one_txt(
                    predn,                   # 预测结果
                    self.args.save_conf,     # 是否保存置信度
                    pbatch["ori_shape"],     # 原始图像尺寸
                    # 保存路径：labels目录下以图像文件名命名的txt文件
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        # 设置最终的指标速度和混淆矩阵
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        # 返回指标统计和结果字典
        
        # 将统计信息转换为NumPy数组
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
        
        # 计算每个类别的目标数量
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        
        # 计算每个图像的目标数量
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        
        # 移除目标图像统计信息
        stats.pop("target_img", None)
        
        # 如果存在统计信息且存在真正例
        if len(stats) and stats["tp"].any():
            # 处理指标
            self.metrics.process(**stats)
        
        # 返回结果字典
        return self.metrics.results_dict


    # def print_results(self):
    #     """Prints training/validation set metrics per class."""
    #     pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
    #     LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
    #     if self.nt_per_class.sum() == 0:
    #         LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

    #     # Print results per class
    #     if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
    #         for i, c in enumerate(self.metrics.ap_class_index):
    #             LOGGER.info(
    #                 pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
    #             )

    #     if self.args.plots:
    #         for normalize in True, False:
    #             self.confusion_matrix.plot(
    #                 save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
    #             )

    # def _process_batch(self, detections, gt_bboxes, gt_cls):
    #     """
    #     Return correct prediction matrix.

    #     Args:
    #         detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
    #             (x1, y1, x2, y2, conf, class).
    #         gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
    #             bounding box is of the format: (x1, y1, x2, y2).
    #         gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

    #     Returns:
    #         (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

    #     Note:
    #         The function does not return any value directly usable for metrics calculation. Instead, it provides an
    #         intermediate representation used for evaluating predictions against ground truth.
    #     """
    #     iou = box_iou(gt_bboxes, detections[:, :4])
    #     return self.match_predictions(detections[:, 5], gt_cls, iou)

    # def build_dataset(self, img_path, mode="val", batch=None):
    #     """
    #     Build YOLO Dataset.

    #     Args:
    #         img_path (str): Path to the folder containing images.
    #         mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
    #         batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
    #     """
    #     return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    # def get_dataloader(self, dataset_path, batch_size):
    #     """Construct and return dataloader."""
    #     dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
    #     return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    # def plot_val_samples(self, batch, ni):
    #     """Plot validation image samples."""
    #     plot_images(
    #         batch["img"],
    #         batch["batch_idx"],
    #         batch["cls"].squeeze(-1),
    #         batch["bboxes"],
    #         paths=batch["im_file"],
    #         fname=self.save_dir / f"val_batch{ni}_labels.jpg",
    #         names=self.names,
    #         on_plot=self.on_plot,
    #     )

def print_results(self):
    """Prints training/validation set metrics per class."""
    # 打印训练/验证集的类别指标
    
    # 定义打印格式：类别名称、图像数、实例数、性能指标
    pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
    
    # 记录整体指标：所有类别、处理图像数、总实例数、平均性能指标
    LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
    
    # 如果没有标签，记录警告
    if self.nt_per_class.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

    # 逐类别打印结果
    # 条件：详细模式开启、非训练阶段、多类别、存在统计信息
    if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
        for i, c in enumerate(self.metrics.ap_class_index):
            # 打印每个类别的详细指标
            LOGGER.info(
                pf % (
                    self.names[c],          # 类别名称
                    self.nt_per_image[c],   # 每个图像的目标数
                    self.nt_per_class[c],   # 每个类别的总目标数
                    *self.metrics.class_result(i)  # 类别性能指标
                )
            )

    # 绘制混淆矩阵
    if self.args.plots:
        # 绘制两种类型的混淆矩阵：归一化和非归一化
        for normalize in True, False:
            self.confusion_matrix.plot(
                save_dir=self.save_dir,     # 保存目录
                names=self.names.values(),  # 类别名称
                normalize=normalize,        # 是否归一化
                on_plot=self.on_plot        # 绘图回调函数
            )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        # 计算预测框和真实框之间的IoU
        iou = box_iou(gt_bboxes, detections[:, :4])
        
        # 匹配预测和真实标签
        return self.match_predictions(
            detections[:, 5],  # 预测类别
            gt_cls,            # 真实类别
            iou                # IoU矩阵
        )

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or [val](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/models/yolo/detect/val.py:608:4-648:20) mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # 构建YOLO数据集
        return build_yolo_dataset(
            self.args,         # 参数
            img_path,          # 图像路径
            batch,             # 批次大小
            self.data,         # 数据配置
            mode=mode,         # 模式（训练/验证）
            stride=self.stride # 步长
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        # 构建并返回数据加载器
        
        # 构建数据集
        dataset = self.build_dataset(
            dataset_path, 
            batch=batch_size, 
            mode="val"
        )
        
        # 构建数据加载器
        return build_dataloader(
            dataset,               # 数据集
            batch_size,            # 批次大小
            self.args.workers,     # 工作进程数
            shuffle=False,         # 不打乱顺序
            rank=-1                # 分布式训练设置
        )

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        # 绘制验证图像样本
        plot_images(
            batch["img"],              # 图像
            batch["batch_idx"],        # 批次索引
            batch["cls"].squeeze(-1),  # 类别标签
            batch["bboxes"],           # 边界框
            paths=batch["im_file"],    # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",  # 保存文件名
            names=self.names,          # 类别名称
            on_plot=self.on_plot       # 绘图回调函数
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        # 绘制预测的边界框并保存结果
        plot_images(
            batch["img"],                   # 输入图像
            *output_to_target(               # 将预测结果转换为目标格式
                preds, 
                max_det=self.args.max_det    # 最大检测数限制
            ),
            paths=batch["im_file"],          # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",  # 保存文件名
            names=self.names,                # 类别名称
            on_plot=self.on_plot             # 绘图回调函数
        )
    
    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        # 将YOLO检测结果保存为txt文件，使用归一化坐标
        
        # 导入Results类
        from ultralytics.engine.results import Results
    
        # 创建Results对象并保存为文本
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # 创建空白图像
            path=None,                       # 路径为空
            names=self.names,                # 类别名称
            boxes=predn[:, :6],              # 预测框（前6列）
        ).save_txt(
            file,                            # 保存文件
            save_conf=save_conf              # 是否保存置信度
        )
    
    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        # 将YOLO预测结果序列化为COCO JSON格式
        
        # 获取文件名stem
        stem = Path(filename).stem
        
        # 尝试将stem转换为图像ID
        image_id = int(stem) if stem.isnumeric() else stem
        
        # 将边界框从XYXY转换为XYWH格式
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        
        # 调整边界框中心点（从中心转换到左上角）
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        
        # 遍历预测结果和边界框
        for p, b in zip(predn.tolist(), box.tolist()):
            # 添加JSON字典
            self.jdict.append(
                {
                    "image_id": image_id,                          # 图像ID
                    "category_id": self.class_map[int(p[5])],      # 类别ID（使用映射）
                    "bbox": [round(x, 3) for x in b],              # 边界框（保留3位小数）
                    "score": round(p[4], 5),                       # 置信度（保留5位小数）
                }
            )
    
    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        # 评估JSON格式的YOLO输出并返回性能统计
        
        # 检查是否需要保存JSON且为支持的数据集
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            # 预测结果JSON路径
            pred_json = self.save_dir / "predictions.json"
            
            # 注释JSON路径（根据数据集类型）
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )
            
            # 选择评估工具包
            pkg = "pycocotools" if self.is_coco else "lvis"
            
            # 记录评估信息
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            
            try:
                # 检查文件是否存在
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                
                # 检查依赖
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                
                # COCO数据集评估
                if self.is_coco:
                    from pycocotools.coco import COCO
                    from pycocotools.cocoeval import COCOeval
    
                    # 初始化注释和预测API
                    anno = COCO(str(anno_json))
                    pred = anno.loadRes(str(pred_json))
                    val = COCOeval(anno, pred, "bbox")
                
                # LVIS数据集评估
                else:
                    from lvis import LVIS, LVISEval
    
                    # 初始化注释和预测API
                    anno = LVIS(str(anno_json))
                    pred = anno._load_json(str(pred_json))
                    val = LVISEval(anno, pred, "bbox")
                
                # 设置要评估的图像
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                
                # 执行评估流程
                val.evaluate()
                val.accumulate()
                val.summarize()
                
                # 对于LVIS，显式调用结果打印
                if self.is_lvis:
                    val.print_results()
                
                # 更新mAP指标
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            
            # 处理可能的异常
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
            
        return stats

