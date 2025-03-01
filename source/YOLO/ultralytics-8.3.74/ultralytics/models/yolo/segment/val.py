# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.
    一个扩展 DetectionValidator 类的类，用于基于分割模型进行验证。

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml")
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.
        初始化 SegmentationValidator，并将任务设置为 'segment'，指标设置为 SegmentMetrics。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None  # 初始化绘图掩码
        self.process = None  # 初始化处理方法
        self.args.task = "segment"  # 设置任务为分割
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)  # 初始化指标

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device.
        通过将掩码转换为浮点数并发送到设备来预处理批次。"""
        batch = super().preprocess(batch)  # 调用父类的预处理方法
        batch["masks"] = batch["masks"].to(self.device).float()  # 将掩码转换为浮点数并移动到设备
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag.
        初始化指标并根据 save_json 标志选择掩码处理函数。"""
        super().init_metrics(model)  # 调用父类的初始化指标方法
        self.plot_masks = []  # 初始化绘图掩码列表
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")  # 检查 pycocotools 依赖
        # 根据 save_json 标志选择更精确或更快速的处理函数
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])  # 初始化统计字典

    def get_desc(self):
        """Return a formatted description of evaluation metrics.
        返回格式化的评估指标描述。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Post-processes YOLO predictions and returns output detections with proto.
        后处理 YOLO 预测，并返回带有原型的输出检测。"""
        p = super().postprocess(preds[0])  # 调用父类的后处理方法
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # 如果第二个输出长度为 3，则取最后一个
        return p, proto  # 返回处理后的预测和原型

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by processing images and targets.
        通过处理图像和目标为训练或推理准备批次。"""
        prepared_batch = super()._prepare_batch(si, batch)  # 调用父类的准备批次方法
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si  # 根据重叠掩码标志选择索引
        prepared_batch["masks"] = batch["masks"][midx]  # 准备掩码
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """Prepares a batch for training or inference by processing images and targets.
        通过处理图像和目标为训练或推理准备批次。"""
        predn = super()._prepare_pred(pred, pbatch)  # 调用父类的准备预测方法
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])  # 处理预测掩码
        return predn, pred_masks  # 返回处理后的预测和掩码

    def update_metrics(self, preds, batch):
        """Metrics. 更新指标。"""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):  # 遍历预测和原型
            self.seen += 1  # 增加已见样本计数
            npr = len(pred)  # 预测数量
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )  # 初始化统计字典
            pbatch = self._prepare_batch(si, batch)  # 准备批次
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")  # 提取类别和边界框
            nl = len(cls)  # 类别数量
            stat["target_cls"] = cls  # 记录目标类别
            stat["target_img"] = cls.unique()  # 记录目标图像
            if npr == 0:  # 如果没有预测
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])  # 更新统计
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)  # 处理混淆矩阵
                continue

            # Masks
            gt_masks = pbatch.pop("masks")  # 提取真实掩码
            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0  # 如果是单类，则将预测类别设置为 0
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)  # 准备预测
            stat["conf"] = predn[:, 4]  # 记录置信度
            stat["pred_cls"] = predn[:, 5]  # 记录预测类别

            # Evaluate
            if nl:  # 如果有目标类别
                stat["tp"] = self._process_batch(predn, bbox, cls)  # 处理预测和真实边界框
                stat["tp_m"] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                )  # 处理预测和真实掩码
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)  # 处理混淆矩阵

            for k in self.stats.keys():
                self.stats[k].append(stat[k])  # 更新统计

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)  # 将预测掩码转换为张量
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # 过滤前 15 个以绘图

            # Save
            if self.args.save_json:  # 如果需要保存为 JSON
                self.pred_to_json(
                    predn,
                    batch["im_file"][si],
                    ops.scale_image(
                        pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        pbatch["ori_shape"],
                        ratio_pad=batch["ratio_pad"][si],
                    ),
                )  # 保存预测为 JSON
            if self.args.save_txt:  # 如果需要保存为 TXT
                self.save_one_txt(
                    predn,
                    pred_masks,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )  # 保存预测为 TXT

    def finalize_metrics(self, *args, **kwargs):
        """Sets speed and confusion matrix for evaluation metrics.
        设置评估指标的速度和混淆矩阵。"""
        self.metrics.speed = self.speed  # 设置速度
        self.metrics.confusion_matrix = self.confusion_matrix  # 设置混淆矩阵

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.
        根据边界框和可选掩码计算批次的正确预测矩阵。

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detected bounding boxes and
                associated confidence scores and class indices. Each row is of the format [x1, y1, x2, y2, conf, class].
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground truth bounding box coordinates.
                Each row is of the format [x1, y1, x2, y2].
            gt_cls (torch.Tensor): Tensor of shape (M,) representing ground truth class indices.
            pred_masks (torch.Tensor | None): Tensor representing predicted masks, if available. The shape should
                match the ground truth masks.
            gt_masks (torch.Tensor | None): Tensor of shape (M, H, W) representing ground truth masks, if available.
            overlap (bool): Flag indicating if overlapping masks should be considered.
            masks (bool): Flag indicating if the batch contains mask data.

        Returns:
            (torch.Tensor): A correct prediction matrix of shape (N, 10), where 10 represents different IoU levels.

        Note:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.

        Example:
            ```python
            detections = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            gt_cls = torch.tensor([1, 0])
            correct_preds = validator._process_batch(detections, gt_bboxes, gt_cls)
            ```
        """
        if masks:  # 如果处理掩码
            if overlap:  # 如果考虑重叠
                nl = len(gt_cls)  # 真实类别数量
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1  # 创建索引
                gt_masks = gt_masks.repeat(nl, 1, 1)  # 重复真实掩码以匹配数量
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)  # 根据索引设置掩码
            if gt_masks.shape[1:] != pred_masks.shape[1:]:  # 如果形状不匹配
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]  # 调整形状
                gt_masks = gt_masks.gt_(0.5)  # 二值化掩码
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))  # 计算 IoU
        else:  # 处理边界框
            iou = box_iou(gt_bboxes, detections[:, :4])  # 计算 IoU

        return self.match_predictions(detections[:, 5], gt_cls, iou)  # 返回匹配的预测

    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels.
        绘制带有边界框标签的验证样本。"""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes.
        绘制带有掩码和边界框的批次预测。"""
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=15),  # 不设置为 self.args.max_det 以提高绘图速度
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,  # 如果有掩码则绘制
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 预测
        self.plot_masks.clear()  # 清空绘图掩码

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.
        将 YOLO 检测结果以特定格式保存到 TXT 文件中，坐标为归一化坐标。"""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # 创建零数组
            path=None,
            names=self.names,
            boxes=predn[:, :6],  # 提取边界框
            masks=pred_masks,  # 提取掩码
        ).save_txt(file, save_conf=save_conf)  # 保存为 TXT 文件

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result.
        保存一个 JSON 结果。

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict.
            将预测掩码编码为 RLE 并将结果附加到 jdict。"""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]  # 编码掩码
            rle["counts"] = rle["counts"].decode("utf-8")  # 解码计数
            return rle

        stem = Path(filename).stem  # 获取文件名的主干部分
        image_id = int(stem) if stem.isnumeric() else stem  # 如果是数字则转换为整数
        box = ops.xyxy2xywh(predn[:, :4])  # 转换为 xywh 格式
        box[:, :2] -= box[:, 2:] / 2  # 将 xy 中心转换为左上角
        pred_masks = np.transpose(pred_masks, (2, 0, 1))  # 转置掩码
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)  # 使用线程池编码掩码
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):  # 遍历预测和边界框
            self.jdict.append(  # 将结果添加到 jdict
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],  # 获取类别 ID
                    "bbox": [round(x, 3) for x in b],  # 四舍五入边界框
                    "score": round(p[4], 5),  # 四舍五入置信度
                    "segmentation": rles[i],  # 添加分割信息
                }
            )

    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics.
        返回 COCO 风格的目标检测评估指标。"""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # 注释文件路径
            pred_json = self.save_dir / "predictions.json"  # 预测文件路径
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")  # 日志信息
            try:  # 尝试评估
                check_requirements("pycocotools>=2.0.6")  # 检查 pycocotools 依赖
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"  # 检查文件是否存在
                anno = COCO(str(anno_json))  # 初始化注释 API
                pred = anno.loadRes(str(pred_json))  # 初始化预测 API（必须传递字符串，而不是 Path）
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # 设置评估图像
                    eval.evaluate()  # 评估
                    eval.accumulate()  # 累积结果
                    eval.summarize()  # 总结结果
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # 更新 mAP50-95 和 mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")  # 记录警告
        return stats  # 返回统计信息
