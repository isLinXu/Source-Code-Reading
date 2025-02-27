# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class PoseValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a pose model.
    一个继承自DetectionValidator类的验证器，专门用于基于姿态估计模型的验证。

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseValidator

        args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
        validator = PoseValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'PoseValidator' object with custom parameters and assigned attributes."""
        # 使用自定义参数和分配的属性初始化PoseValidator对象
        
        # 调用父类初始化方法
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        # 初始化关键点相关属性
        self.sigma = None  # 关键点匹配的权重系数
        self.kpt_shape = None  # 关键点形状
        
        # 强制设置任务为"pose"（姿态估计）
        self.args.task = "pose"
        
        # 创建姿态估计专用的度量指标对象
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        
        # 检查是否使用Apple MPS设备
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            # 对于姿态估计模型，警告使用MPS可能存在已知问题
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
        # 预处理批次数据，将关键点数据转换为浮点型并移动到指定设备
        
        # 调用父类预处理方法
        batch = super().preprocess(batch)
        
        # 将关键点数据转换为浮点型并移动到指定设备
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        
        return batch

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        # 返回评估指标的描述字符串
        return ("%22s" + "%11s" * 10) % (
            "Class",        # 类别
            "Images",       # 图像数
            "Instances",    # 实例数
            "Box(P",        # 边界框精确率
            "R",            # 边界框召回率
            "mAP50",        # 边界框50%IoU平均精度
            "mAP50-95)",    # 边界框0-95%IoU平均精度
            "Pose(P",       # 姿态精确率
            "R",            # 姿态召回率
            "mAP50",        # 姿态50%IoU平均精度
            "mAP50-95)",    # 姿态0-95%IoU平均精度
        )

    def init_metrics(self, model):
        """Initiate pose estimation metrics for YOLO model."""
        # 为YOLO模型初始化姿态估计指标
        
        # 调用父类初始化指标方法
        super().init_metrics(model)
        
        # 获取关键点形状
        self.kpt_shape = self.data["kpt_shape"]
        
        # 判断是否为标准姿态估计（COCO数据集的17个关键点）
        is_pose = self.kpt_shape == [17, 3]
        
        # 获取关键点数量
        nkpt = self.kpt_shape[0]
        
        # 设置关键点匹配的权重系数
        # - 对于标准姿态估计，使用预定义的OKS_SIGMA
        # - 对于其他情况，使用均匀分布的权重
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        
        # 初始化统计指标字典
        self.stats = dict(
            tp_p=[], # 关键点匹配的真正例
            tp=[],   # 边界框匹配的真正例
            conf=[], # 置信度
            pred_cls=[], # 预测类别
            target_cls=[], # 目标类别
            target_img=[] # 目标图像
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch for processing by converting keypoints to float and moving to device."""
        # 准备批次处理，转换关键点为浮点型并移动到设备
        
        # 调用父类批次准备方法
        pbatch = super()._prepare_batch(si, batch)
        
        # 提取指定批次索引的关键点
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        
        # 获取图像尺寸
        h, w = pbatch["imgsz"]
        
        # 克隆关键点数据
        kpts = kpts.clone()
        
        # 将关键点坐标缩放到图像尺寸
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        
        # 将关键点坐标缩放到原始图像尺寸
        kpts = ops.scale_coords(
            pbatch["imgsz"], 
            kpts, 
            pbatch["ori_shape"], 
            ratio_pad=pbatch["ratio_pad"]
        )
        
        # 将处理后的关键点添加到批次数据中
        pbatch["kpts"] = kpts
        
        return pbatch

    def _prepare_pred(self, pred, pbatch):
        """Prepares and scales keypoints in a batch for pose processing."""
        # 为姿态处理准备和缩放批次中的关键点
        
        # 调用父类预测准备方法
        predn = super()._prepare_pred(pred, pbatch)
        
        # 获取关键点数量
        nk = pbatch["kpts"].shape[1]
        
        # 重塑关键点预测
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        
        # 将关键点坐标缩放到原始图像尺寸
        ops.scale_coords(
            pbatch["imgsz"], 
            pred_kpts, 
            pbatch["ori_shape"], 
            ratio_pad=pbatch["ratio_pad"]
        )
        
        return predn, pred_kpts


    def update_metrics(self, preds, batch):
        """Metrics."""
        # 更新评估指标
        
        # 遍历每个预测结果
        for si, pred in enumerate(preds):
            # 增加已处理图像数
            self.seen += 1
            
            # 获取预测数量
            npr = len(pred)
            
            # 初始化统计字典
            stat = dict(
                conf=torch.zeros(0, device=self.device),  # 置信度
                pred_cls=torch.zeros(0, device=self.device),  # 预测类别
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # 边界框真正例
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # 关键点真正例
            )
            
            # 准备批次数据
            pbatch = self._prepare_batch(si, batch)
            
            # 提取类别和边界框
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            
            # 获取目标数量
            nl = len(cls)
            
            # 记录目标类别和图像
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            
            # 处理无预测结果的情况
            if npr == 0:
                if nl:
                    # 记录统计信息
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    
                    # 处理混淆矩阵
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 处理单类别情况
            if self.args.single_cls:
                pred[:, 5] = 0
            
            # 准备预测结果和关键点
            predn, pred_kpts = self._prepare_pred(pred, pbatch)
            
            # 记录置信度和预测类别
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # 评估预测结果
            if nl:
                # 评估边界框匹配
                stat["tp"] = self._process_batch(predn, bbox, cls)
                
                # 评估关键点匹配
                stat["tp_p"] = self._process_batch(predn, bbox, cls, pred_kpts, pbatch["kpts"])
            
            # 处理混淆矩阵
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            # 更新统计信息
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # 保存结果
            if self.args.save_json:
                # 将预测转换为JSON格式
                self.pred_to_json(predn, batch["im_file"][si])
            
            if self.args.save_txt:
                # 保存预测为文本文件
                self.save_one_txt(
                    predn,
                    pred_kpts,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_kpts=None, gt_kpts=None):
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth.
        通过计算检测结果和真实标注之间的交并比（IoU）返回正确的预测矩阵。

        Args:
            detections (torch.Tensor): Tensor with shape (N, 6) representing detection boxes and scores, where each
                detection is of the format (x1, y1, x2, y2, conf, class).
                形状为(N, 6)的张量，表示检测框和置信度，每个检测的格式为(x1, y1, x2, y2, 置信度, 类别)。
            gt_bboxes (torch.Tensor): Tensor with shape (M, 4) representing ground truth bounding boxes, where each
                box is of the format (x1, y1, x2, y2).
                形状为(M, 4)的张量，表示真实边界框，每个框的格式为(x1, y1, x2, y2)。
            gt_cls (torch.Tensor): Tensor with shape (M,) representing ground truth class indices.
                形状为(M,)的张量，表示真实类别索引。
            pred_kpts (torch.Tensor | None): Optional tensor with shape (N, 51) representing predicted keypoints, where
                51 corresponds to 17 keypoints each having 3 values.
                可选的形状为(N, 51)的张量，表示预测关键点，其中51对应17个关键点，每个关键点有3个值。
            gt_kpts (torch.Tensor | None): Optional tensor with shape (N, 51) representing ground truth keypoints.
                可选的形状为(N, 51)的张量，表示真实关键点。

        Returns:
            (torch.Tensor): A tensor with shape (N, 10) representing the correct prediction matrix for 10 IoU levels,
                where N is the number of detections.
            形状为(N, 10)的张量，表示10个IoU级别的正确预测矩阵，其中N是检测数量。

        Example:
            ```python
            detections = torch.rand(100, 6)  # 100 predictions: (x1, y1, x2, y2, conf, class)
            gt_bboxes = torch.rand(50, 4)  # 50 ground truth boxes: (x1, y1, x2, y2)
            gt_cls = torch.randint(0, 2, (50,))  # 50 ground truth class indices
            pred_kpts = torch.rand(100, 51)  # 100 predicted keypoints
            gt_kpts = torch.rand(50, 51)  # 50 ground truth keypoints
            correct_preds = _process_batch(detections, gt_bboxes, gt_cls, pred_kpts, gt_kpts)
            ```

        Note:
            `0.53` scale factor used in area computation is referenced from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
            面积计算中使用的`0.53`比例因子来自于指定的GitHub仓库链接。
        """
        # 处理关键点匹配的情况
        if pred_kpts is not None and gt_kpts is not None:
            # 使用`0.53`比例因子计算边界框面积
            # 参考: https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            
            # 计算关键点IoU
            # - gt_kpts: 真实关键点
            # - pred_kpts: 预测关键点
            # - sigma: 关键点匹配权重
            # - area: 边界框面积
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
        else:  # 处理边界框匹配的情况
            # 计算边界框IoU
            iou = box_iou(gt_bboxes, detections[:, :4])

        # 匹配预测结果
        # - detections[:, 5]: 预测类别
        # - gt_cls: 真实类别
        # - iou: 交并比
        return self.match_predictions(detections[:, 5], gt_cls, iou)


    def plot_val_samples(self, batch, ni):
        """
        Plots and saves validation set samples with predicted bounding boxes and keypoints.
        绘制并保存带有预测边界框和关键点的验证集样本。
        """
        plot_images(
            batch["img"],                 # 输入图像
            batch["batch_idx"],           # 批次索引
            batch["cls"].squeeze(-1),     # 类别标签（去除额外维度）
            batch["bboxes"],              # 边界框
            kpts=batch["keypoints"],      # 关键点
            paths=batch["im_file"],       # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",  # 保存文件名
            names=self.names,             # 类别名称
            on_plot=self.on_plot,         # 绘图回调函数
        )

    def plot_predictions(self, batch, preds, ni):
        """
        Plots predictions for YOLO model.
        绘制YOLO模型的预测结果。
        """
        # 提取并重塑预测的关键点
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        
        plot_images(
            batch["img"],                 # 输入图像
            *output_to_target(preds, max_det=self.args.max_det),  # 预测目标（使用最大检测数限制）
            kpts=pred_kpts,               # 预测关键点
            paths=batch["im_file"],       # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",  # 保存文件名
            names=self.names,             # 类别名称
            on_plot=self.on_plot,         # 绘图回调函数
        )

    def save_one_txt(self, predn, pred_kpts, save_conf, shape, file):
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.
        以特定格式将YOLO检测结果保存为txt文件，使用归一化坐标。
        """
        # 导入Results类
        from ultralytics.engine.results import Results

        # 创建Results对象并保存
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # 创建空白图像
            path=None,                    # 路径为空
            names=self.names,             # 类别名称
            boxes=predn[:, :6],           # 边界框信息
            keypoints=pred_kpts,          # 关键点
        ).save_txt(file, save_conf=save_conf)  # 保存为文本文件

    def pred_to_json(self, predn, filename):
        """
        Converts YOLO predictions to COCO JSON format.
        将YOLO预测结果转换为COCO JSON格式。
        """
        # 获取文件名的stem（不包含扩展名的文件名）
        stem = Path(filename).stem
        
        # 尝试将stem转换为图像ID，如果不是数字则保持原值
        image_id = int(stem) if stem.isnumeric() else stem
        
        # 将边界框坐标转换为XYWH格式
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        
        # 调整边界框坐标：从中心点转换为左上角
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        
        # 遍历预测结果和边界框
        for p, b in zip(predn.tolist(), box.tolist()):
            # 添加JSON字典到结果列表
            self.jdict.append(
                {
                    "image_id": image_id,                          # 图像ID
                    "category_id": self.class_map[int(p[5])],      # 类别ID（使用映射）
                    "bbox": [round(x, 3) for x in b],              # 边界框（保留3位小数）
                    "keypoints": p[6:],                            # 关键点
                    "score": round(p[4], 5),                       # 置信度（保留5位小数）
                }
            )

    def eval_json(self, stats):
        """
        Evaluates object detection model using COCO JSON format.
        使用COCO JSON格式评估目标检测模型。
        """
        # 检查是否需要保存JSON且为COCO数据集
        if self.args.save_json and self.is_coco and len(self.jdict):
            # 设置标注和预测JSON文件路径
            anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # 标注文件
            pred_json = self.save_dir / "predictions.json"  # 预测文件
            
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            
            try:
                # 检查并导入pycocotools
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                # 验证文件存在
                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                
                # 初始化COCO标注和预测API
                anno = COCO(str(anno_json))  # 初始化标注API
                pred = anno.loadRes(str(pred_json))  # 初始化预测API
                
                # 评估边界框和关键点
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "keypoints")]):
                    if self.is_coco:
                        # 设置要评估的图像ID
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                    
                    # 执行评估流程
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    
                    # 更新统计指标
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[:2]
            
            except Exception as e:
                # 处理pycocotools运行异常
                LOGGER.warning(f"pycocotools unable to run: {e}")
        
        return stats