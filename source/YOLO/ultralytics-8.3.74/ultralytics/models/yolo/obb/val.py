# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images


# class OBBValidator(DetectionValidator):
#     """
#     A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

#     Example:
#         ```python
#         from ultralytics.models.yolo.obb import OBBValidator

#         args = dict(model="yolo11n-obb.pt", data="dota8.yaml")
#         validator = OBBValidator(args=args)
#         validator(model=args["model"])
#         ```
#     """

#     def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
#         """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
#         super().__init__(dataloader, save_dir, pbar, args, _callbacks)
#         self.args.task = "obb"
#         self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

#     def init_metrics(self, model):
#         """Initialize evaluation metrics for YOLO."""
#         super().init_metrics(model)
#         val = self.data.get(self.args.split, "")  # validation path
#         self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO

#     def _process_batch(self, detections, gt_bboxes, gt_cls):
#         """
#         Perform computation of the correct prediction matrix for a batch of detections and ground truth bounding boxes.

#         Args:
#             detections (torch.Tensor): A tensor of shape (N, 7) representing the detected bounding boxes and associated
#                 data. Each detection is represented as (x1, y1, x2, y2, conf, class, angle).
#             gt_bboxes (torch.Tensor): A tensor of shape (M, 5) representing the ground truth bounding boxes. Each box is
#                 represented as (x1, y1, x2, y2, angle).
#             gt_cls (torch.Tensor): A tensor of shape (M,) representing class labels for the ground truth bounding boxes.

#         Returns:
#             (torch.Tensor): The correct prediction matrix with shape (N, 10), which includes 10 IoU (Intersection over
#                 Union) levels for each detection, indicating the accuracy of predictions compared to the ground truth.

#         Example:
#             ```python
#             detections = torch.rand(100, 7)  # 100 sample detections
#             gt_bboxes = torch.rand(50, 5)  # 50 sample ground truth boxes
#             gt_cls = torch.randint(0, 5, (50,))  # 50 ground truth class labels
#             correct_matrix = OBBValidator._process_batch(detections, gt_bboxes, gt_cls)
#             ```

#         Note:
#             This method relies on `batch_probiou` to calculate IoU between detections and ground truth bounding boxes.
#         """
#         iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
#         return self.match_predictions(detections[:, 5], gt_cls, iou)

#     def _prepare_batch(self, si, batch):
#         """Prepares and returns a batch for OBB validation."""
#         idx = batch["batch_idx"] == si
#         cls = batch["cls"][idx].squeeze(-1)
#         bbox = batch["bboxes"][idx]
#         ori_shape = batch["ori_shape"][si]
#         imgsz = batch["img"].shape[2:]
#         ratio_pad = batch["ratio_pad"][si]
#         if len(cls):
#             bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
#             ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
#         return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

#     def _prepare_pred(self, pred, pbatch):
#         """Prepares and returns a batch for OBB validation with scaled and padded bounding boxes."""
#         predn = pred.clone()
#         ops.scale_boxes(
#             pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
#         )  # native-space pred
#         return predn

#     def plot_predictions(self, batch, preds, ni):
#         """Plots predicted bounding boxes on input images and saves the result."""
#         plot_images(
#             batch["img"],
#             *output_to_rotated_target(preds, max_det=self.args.max_det),
#             paths=batch["im_file"],
#             fname=self.save_dir / f"val_batch{ni}_pred.jpg",
#             names=self.names,
#             on_plot=self.on_plot,
#         )  # pred

class OBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.
    一个继承自DetectionValidator类的验证器，专门用于基于带方向边界框（OBB）模型的验证。

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model="yolo11n-obb.pt", data="dota8.yaml")
        validator = OBBValidator(args=args)
        validator(model=args["model"])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        # 初始化OBBValidator，设置任务为'obb'，并使用OBBMetrics指标
        
        # 调用父类初始化方法
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        # 强制设置任务为"obb"
        self.args.task = "obb"
        
        # 初始化OBB特定的指标
        self.metrics = OBBMetrics(
            save_dir=self.save_dir,  # 保存目录
            plot=True,               # 启用绘图
            on_plot=self.on_plot     # 绘图回调函数
        )

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        # 初始化YOLO评估指标
        
        # 调用父类的指标初始化方法
        super().init_metrics(model)
        
        # 获取验证数据路径
        val = self.data.get(self.args.split, "")
        
        # 检查是否为DOTA数据集
        self.is_dota = isinstance(val, str) and "DOTA" in val

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Perform computation of the correct prediction matrix for a batch of detections and ground truth bounding boxes.
        为一批检测和真实边界框执行正确预测矩阵的计算。

        Args:
            detections (torch.Tensor): 形状为(N, 7)的张量，表示检测到的边界框和相关数据。
                每个检测表示为 (x1, y1, x2, y2, conf, class, angle)
            gt_bboxes (torch.Tensor): 形状为(M, 5)的张量，表示真实边界框。
                每个框表示为 (x1, y1, x2, y2, angle)
            gt_cls (torch.Tensor): 形状为(M,)的张量，表示真实边界框的类别标签。

        Returns:
            (torch.Tensor): 形状为(N, 10)的正确预测矩阵，包含10个IoU（交并比）级别，
                表示预测与真实值的准确性。
        """
        # 使用batch_probiou计算检测框和真实框的IoU
        # 将检测框的前4列（坐标）和最后一列（角度）拼接
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        
        # 匹配预测结果
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for OBB validation."""
        # 为OBB验证准备批次数据
        
        # 获取特定批次的索引
        idx = batch["batch_idx"] == si
        
        # 提取类别标签
        cls = batch["cls"][idx].squeeze(-1)
        
        # 提取边界框
        bbox = batch["bboxes"][idx]
        
        # 获取原始图像形状
        ori_shape = batch["ori_shape"][si]
        
        # 获取图像大小
        imgsz = batch["img"].shape[2:]
        
        # 获取缩放和填充比率
        ratio_pad = batch["ratio_pad"][si]
        
        # 如果存在类别标签
        if len(cls):
            # 根据图像大小缩放边界框坐标
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])
            
            # 将边界框缩放到原始图像空间
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)
        
        # 返回准备好的批次数据
        return {
            "cls": cls,             # 类别标签
            "bbox": bbox,            # 边界框
            "ori_shape": ori_shape,  # 原始图像形状
            "imgsz": imgsz,          # 图像大小
            "ratio_pad": ratio_pad   # 缩放和填充比率
        }

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for OBB validation with scaled and padded bounding boxes."""
        # 为OBB验证准备预测批次，包括缩放和填充边界框
        
        # 克隆预测结果
        predn = pred.clone()
        
        # 将预测框缩放到原始图像空间
        ops.scale_boxes(
            pbatch["imgsz"],         # 图像大小
            predn[:, :4],            # 预测框坐标
            pbatch["ori_shape"],     # 原始图像形状
            ratio_pad=pbatch["ratio_pad"],  # 缩放和填充比率
            xywh=True                # 使用XYWH格式
        )
        
        # 返回缩放后的预测结果
        return predn

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        # 在输入图像上绘制预测的边界框并保存结果
        plot_images(
            batch["img"],                   # 输入图像
            *output_to_rotated_target(      # 将输出转换为旋转目标
                preds, 
                max_det=self.args.max_det   # 最大检测数
            ),
            paths=batch["im_file"],         # 图像文件路径
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",  # 保存文件名
            names=self.names,               # 类别名称
            on_plot=self.on_plot            # 绘图回调函数
        )

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        # 将YOLO预测结果序列化为COCO JSON格式
        
        # 获取文件名的stem（不包含扩展名的文件名）
        stem = Path(filename).stem
        
        # 尝试将stem转换为图像ID，如果不是数字则保持原值
        image_id = int(stem) if stem.isnumeric() else stem
        
        # 组合边界框坐标和旋转角度
        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        
        # 将旋转边界框转换为多边形顶点坐标
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        
        # 遍历每个预测结果
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            # 添加JSON字典到结果列表
            self.jdict.append(
                {
                    "image_id": image_id,                          # 图像ID
                    "category_id": self.class_map[int(predn[i, 5].item())],  # 类别ID（使用映射）
                    "score": round(predn[i, 4].item(), 5),         # 置信度（保留5位小数）
                    "rbox": [round(x, 3) for x in r],              # 旋转边界框（保留3位小数）
                    "poly": [round(x, 3) for x in b],              # 多边形顶点坐标（保留3位小数）
                }
            )
    
    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        # 将YOLO检测结果保存为txt文件，使用归一化坐标
        
        # 导入必要的库
        import numpy as np
        from ultralytics.engine.results import Results
        
        # 组合边界框坐标和旋转角度
        rboxes = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        
        # 组合边界框信息：xywh、旋转角度、置信度、类别
        obb = torch.cat([rboxes, predn[:, 4:6]], dim=-1)
        
        # 创建Results对象并保存为文本
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # 创建空白图像
            path=None,                       # 路径为空
            names=self.names,                # 类别名称
            obb=obb,                         # 带方向的边界框信息
        ).save_txt(file, save_conf=save_conf)  # 保存为文本文件
    
    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        # 评估JSON格式的YOLO输出并返回性能统计
        
        # 检查是否需要保存JSON且为DOTA数据集
        if self.args.save_json and self.is_dota and len(self.jdict):
            # 导入必要的库
            import json
            import re
            from collections import defaultdict
            
            # 预测结果JSON路径
            pred_json = self.save_dir / "predictions.json"
            
            # 预测结果文本保存路径
            pred_txt = self.save_dir / "predictions_txt"
            pred_txt.mkdir(parents=True, exist_ok=True)
            
            # 加载JSON数据
            data = json.load(open(pred_json))
            
            # 保存分割结果
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                # 提取图像ID
                image_id = d["image_id"]
                
                # 提取置信度
                score = d["score"]
                
                # 获取类别名称（替换空格为连字符）
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                
                # 获取多边形顶点坐标
                p = d["poly"]
                
                # 保存到类别特定的文本文件
                with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            
            # 保存合并结果
            pred_merged_txt = self.save_dir / "predictions_merged_txt"
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            
            # 初始化合并结果字典
            merged_results = defaultdict(list)
            
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            
            # 处理和合并预测结果
            for d in data:
                # 提取图像ID的基本部分
                image_id = d["image_id"].split("__")[0]
                
                # 提取图像分割坐标
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                
                # 提取边界框、置信度和类别信息
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                
                # 调整边界框坐标
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                
                # 按图像ID合并结果
                merged_results[image_id].append(bbox)
            
            # 处理合并后的结果
            for image_id, bbox in merged_results.items():
                # 转换为张量
                bbox = torch.tensor(bbox)
                
                # 计算类别偏移
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh
                
                # 提取置信度
                scores = bbox[:, 5]
                
                # 克隆边界框
                b = bbox[:, :5].clone()
                b[:, :2] += c
                
                # 使用旋转NMS
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]
                
                # 转换为多边形顶点坐标
                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                
                # 保存最终结果
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    # 获取类别名称
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    
                    # 处理多边形坐标
                    p = [round(i, 3) for i in x[:-2]]
                    
                    # 处理置信度
                    score = round(x[-2], 3)
                    
                    # 保存到合并结果文件
                    with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            
            return stats
