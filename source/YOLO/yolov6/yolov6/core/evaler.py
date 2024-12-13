#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
import json
import torch
import yaml
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolov6.data.data_load import create_dataloader
from yolov6.utils.events import LOGGER, NCOLS
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.general import download_ckpt
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.utils.torch_utils import time_sync, get_model_info


class Evaler:
    """YOLOv6评估器类，用于模型评估和性能测试"""
    def __init__(self,
                 data,              # 数据配置
                 batch_size=32,     # 批次大小
                 img_size=640,      # 输入图像尺寸
                 conf_thres=0.03,   # 置信度阈值
                 iou_thres=0.65,    # IoU阈值
                 device='',         # 运行设备
                 half=True,         # 是否使用半精度(FP16)推理
                 save_dir='',       # 结果保存目录
                 shrink_size=640,   # 图像缩放尺寸
                 infer_on_rect=False,  # 是否在矩形图像上进行推理
                 verbose=False,     # 是否输出详细信息
                 do_coco_metric=True,  # 是否计算COCO指标
                 do_pr_metric=False,   # 是否计算PR曲线指标
                 plot_curve=True,      # 是否绘制评估曲线
                 plot_confusion_matrix=False,  # 是否绘制混淆矩阵
                 specific_shape=False,  # 是否使用指定形状
                 height=640,           # 指定高度
                 width=640            # 指定宽度
                 ):
        # 确保至少启用一种评估指标 | Assert at least one metric is enabled
        assert do_pr_metric or do_coco_metric, 'ERROR: at least set one val metric'
        
        # 初始化评估器的各项参数
        self.data = data                    # 数据集配置
        self.batch_size = batch_size        # 批处理大小
        self.img_size = img_size            # 图像大小
        self.conf_thres = conf_thres        # 检测置信度阈值
        self.iou_thres = iou_thres          # NMS的IoU阈值
        self.device = device                # 运行设备(CPU/GPU)
        self.half = half                    # 是否使用FP16推理
        self.save_dir = save_dir            # 结果保存路径
        self.shrink_size = shrink_size      # 图像缩放尺寸
        self.infer_on_rect = infer_on_rect  # 是否使用矩形推理
        self.verbose = verbose              # 是否输出详细日志
        self.do_coco_metric = do_coco_metric  # 是否计算COCO评估指标
        self.do_pr_metric = do_pr_metric    # 是否计算PR曲线指标
        self.plot_curve = plot_curve        # 是否绘制评估曲线
        self.plot_confusion_matrix = plot_confusion_matrix  # 是否绘制混淆矩阵
        self.specific_shape = specific_shape  # 是否使用指定输入形状
        self.height = height                # 指定输入高度
        self.width = width                  # 指定输入宽度

    def init_model(self, model, weights, task):
        """初始化模型
        Args:
            model: 模型对象
            weights: 权重文件路径
            task: 任务类型
        Returns:
            初始化后的模型
        """
        # 非训练模式下需要加载预训练权重
        if task != 'train':
            # 检查权重文件是否存在，不存在则下载
            if not os.path.exists(weights):
                download_ckpt(weights)
            # 加载检查点权重到指定设备
            model = load_checkpoint(weights, map_location=self.device)
            # 获取模型的最大步长
            self.stride = int(model.stride.max())
            
            # switch to deploy | 切换到部署模式
            from yolov6.layers.common import RepVGGBlock
            # 遍历模型的所有层
            for layer in model.modules():
                # RepVGG块切换到部署模式
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                # 处理上采样层的PyTorch兼容性问题
                elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                    layer.recompute_scale_factor = None  # torch 1.11.0 compatibility | PyTorch 1.11.0版本兼容性处理
            
            # 输出模型切换到部署模式的日志
            LOGGER.info("Switch model to deploy modality.")
            # 输出模型信息摘要
            LOGGER.info("Model Summary: {}".format(get_model_info(model, self.img_size)))
        
        # GPU设备上进行模型预热
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(model.parameters())))
        
        # 根据配置设置模型精度（半精度或全精度）
        model.half() if self.half else model.float()
        return model

    def init_data(self, dataloader, task):
        '''Initialize dataloader. | 初始化数据加载器
        Returns a dataloader for task val or speed. | 返回用于验证或速度测试的数据加载器
        Args:
            dataloader: 数据加载器对象
            task: 任务类型（'train', 'val', 'test'）
        Returns:
            配置好的数据加载器对象
        '''
        # 检查是否为COCO数据集，默认为False
        self.is_coco = self.data.get("is_coco", False)
        
        # 根据数据集类型设置类别ID映射
        # 如果是COCO数据集，将80类映射到91类（COCO官方类别ID）
        # 如果不是COCO数据集，创建一个0-999的类别列表
        self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        
        # 非训练模式下的特殊配置
        if task != 'train':
            # 评估相关的超参数设置
            eval_hyp = {
                "shrink_size": self.shrink_size,  # 图像缩放尺寸，用于控制推理时的图像大小
            }
            
            # 获取是否使用矩形推理的配置
            rect = self.infer_on_rect
            
            # 设置填充参数：
            # 如果使用矩形推理，填充比例为0.5
            # 如果不使用矩形推理，不进行填充
            pad = 0.5 if rect else 0.0
            
            # 创建并配置数据加载器
            dataloader = create_dataloader(
                # 根据task类型选择数据集路径，如果task不是train/val/test之一，则默认使用val
                self.data[task if task in ('train', 'val', 'test') else 'val'],
                self.img_size,          # 图像尺寸
                self.batch_size,        # 批次大小
                self.stride,            # 模型步长
                hyp=eval_hyp,           # 评估超参数
                check_labels=True,      # 检查标签有效性
                pad=pad,                # 填充比例
                rect=rect,              # 是否使用矩形推理
                data_dict=self.data,    # 数据集配置字典
                task=task,              # 任务类型
                specific_shape=self.specific_shape,  # 是否使用指定形状
                height=self.height,     # 指定图像高度
                width=self.width        # 指定图像宽度
            )[0]  # create_dataloader返回一个元组，取第一个元素作为数据加载器
        
        return dataloader

    def predict_model(self, model, dataloader, task):
        '''Model prediction | 模型预测
        Predicts the whole dataset and gets the prediced results and inference time. | 预测整个数据集并获取预测结果和推理时间
        Args:
            model: 模型对象
            dataloader: 数据加载器
            task: 任务类型
        Returns:
            预测结果、可视化输出和对应的图像路径
        '''
        # 初始化速度测试结果张量，包含4个指标：总推理数量、预处理时间、推理时间、后处理时间
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []  # 存储预测结果
        # 创建进度条
        pbar = tqdm(dataloader, desc=f"Inferencing model in {task} datasets.", ncols=NCOLS)

        # whether to compute metric and plot PR curve and P、R、F1 curve under iou50 match rule
        # 是否计算评估指标并绘制PR曲线和P、R、F1曲线（基于IOU=0.5的匹配规则）
        if self.do_pr_metric:
            stats, ap = [], []  # 统计数据和平均精度列表
            seen = 0  # 已处理图像计数
            iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95 | 用于计算mAP@0.5:0.95的IOU阈值向量
            niou = iouv.numel()  # IOU阈值的数量
            # 是否需要绘制混淆矩阵
            if self.plot_confusion_matrix:
                from yolov6.utils.metrics import ConfusionMatrix
                confusion_matrix = ConfusionMatrix(nc=model.nc)  # 创建混淆矩阵对象

        # 遍历数据集
        for i, (imgs, targets, paths, shapes) in enumerate(pbar):
            # pre-process | 预处理阶段
            t1 = time_sync()  # 记录开始时间
            imgs = imgs.to(self.device, non_blocking=True)  # 将图像数据移至指定设备
            imgs = imgs.half() if self.half else imgs.float()  # 根据配置转换数据类型（半精度或全精度）
            imgs /= 255  # 归一化处理
            self.speed_result[1] += time_sync() - t1  # pre-process time | 累加预处理时间

            # Inference | 推理阶段
            t2 = time_sync()  # 记录推理开始时间
            outputs, _ = model(imgs)  # 执行模型推理
            self.speed_result[2] += time_sync() - t2  # inference time | 累加推理时间

            # post-process | 后处理阶段
            t3 = time_sync()  # 记录后处理开始时间
            # 执行非极大值抑制
            outputs = non_max_suppression(outputs, self.conf_thres, self.iou_thres, multi_label=True)
            self.speed_result[3] += time_sync() - t3  # post-process time | 累加后处理时间
            self.speed_result[0] += len(outputs)  # 累加处理的检测框数量

            # 如果需要计算PR指标，创建输出结果的深拷贝
            if self.do_pr_metric:
                import copy
                eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])

            # save result | 保存结果
            pred_results.extend(self.convert_to_coco_format(outputs, imgs, paths, shapes, self.ids))

            # for tensorboard visualization, maximum images to show: 8
            # 用于Tensorboard可视化，最多显示8张图像
            if i == 0:
                vis_num = min(len(imgs), 8)
                vis_outputs = outputs[:vis_num]
                vis_paths = paths[:vis_num]

            if not self.do_pr_metric:
                continue

            # Statistics per image | 对每张图像进行统计
            # This code is based on | 这段代码基于
            # https://github.com/ultralytics/yolov5/blob/master/val.py
            for si, pred in enumerate(eval_outputs):
                # 获取当前图像的标签
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)  # 标签数量
                tcls = labels[:, 0].tolist() if nl else []  # target class | 目标类别列表
                seen += 1  # 已处理图像数+1

                # 如果没有预测结果
                if len(pred) == 0:
                    if nl:  # 有标签但没有预测结果
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions | 处理预测结果
                predn = pred.clone()  # 克隆预测结果
                # 将预测坐标转换到原始图像空间
                self.scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

                # Assign all predictions as incorrect | 初始化所有预测为错误
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                
                # 如果有标签，进行评估
                if nl:
                    from yolov6.utils.nms import xywh2xyxy

                    # target boxes | 处理目标框
                    tbox = xywh2xyxy(labels[:, 1:5])  # 将标签框从xywh格式转换为xyxy格式
                    # 将标签框坐标缩放到原始图像尺寸
                    tbox[:, [0, 2]] *= imgs[si].shape[1:][1]
                    tbox[:, [1, 3]] *= imgs[si].shape[1:][0]

                    # 将标签框坐标转换到原始图像空间
                    self.scale_coords(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])

                    # 将类别信息和框坐标组合
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels | 原始空间标签

                    from yolov6.utils.metrics import process_batch
                    # 计算预测结果的正确性
                    correct = process_batch(predn, labelsn, iouv)
                    # 如果需要，更新混淆矩阵
                    if self.plot_confusion_matrix:
                        confusion_matrix.process_batch(predn, labelsn)

                # Append statistics (correct, conf, pcls, tcls) | 添加统计信息（正确性、置信度、预测类别、真实类别）
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # 如果需要计算PR指标
        if self.do_pr_metric:
            # Compute statistics | 计算统计指标
            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 转换为numpy数组
            if len(stats) and stats[0].any():
                from yolov6.utils.metrics import ap_per_class
                # 计算每个类别的AP值
                p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.plot_curve, save_dir=self.save_dir, names=model.names)
                # 找到最佳F1分数对应的阈值索引
                AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() -1
                LOGGER.info(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx/1000.0}.")
                
                # 计算各项指标
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95 | AP@0.5和AP@0.5:0.95
                mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=model.nc)  # 每个类别的目标数量

                # Print results | 打印结果
                # 打印表头
                s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
                LOGGER.info(s)
                # 打印总体结果
                pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format | 打印格式
                LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))

                # 保存PR指标结果
                self.pr_metric_result = (map50, map)

                # Print results per class | 打印每个类别的结果
                if self.verbose and model.nc > 1:
                    for i, c in enumerate(ap_class):
                        LOGGER.info(pf % (model.names[c], seen, nt[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                           f1[i, AP50_F1_max_idx], ap50[i], ap[i]))

                # 如果需要，绘制混淆矩阵
                if self.plot_confusion_matrix:
                    confusion_matrix.plot(save_dir=self.save_dir, names=list(model.names))
            else:
                # 计算指标失败的情况
                LOGGER.info("Calculate metric failed, might check dataset.")
                self.pr_metric_result = (0.0, 0.0)

        return pred_results, vis_outputs, vis_paths


    def eval_model(self, pred_results, model, dataloader, task):
        '''Evaluate models | 评估模型
        For task speed, this function only evaluates the speed of model and outputs inference time. | 对于速度测试任务，仅评估模型速度并输出推理时间
        For task val, this function evaluates the speed and mAP by pycocotools, and returns | 对于验证任务，使用pycocotools评估速度和mAP
        inference time and mAP value. | 返回推理时间和mAP值
        Args:
            pred_results: 预测结果列表
            model: 模型对象
            dataloader: 数据加载器
            task: 任务类型（'speed'或'val'）
        Returns:
            tuple: (map50, map) mAP@0.5和mAP@0.5:0.95的值
        '''
        # 首先评估模型速度
        LOGGER.info(f'\nEvaluating speed.')
        self.eval_speed(task)

        # 如果不需要COCO指标但需要PR指标，直接返回PR指标结果
        if not self.do_coco_metric and self.do_pr_metric:
            return self.pr_metric_result
        
        # 使用pycocotools评估mAP
        LOGGER.info(f'\nEvaluating mAP by pycocotools.')
        if task != 'speed' and len(pred_results):
            # 获取标注文件路径
            if 'anno_path' in self.data:
                anno_json = self.data['anno_path']
            else:
                # 在数据集初始化时生成COCO格式的标签
                # 如果是训练任务，使用验证集配置
                task = 'val' if task == 'train' else task
                if not isinstance(self.data[task], list):
                    self.data[task] = [self.data[task]]
                # 构建标注文件路径
                dataset_root = os.path.dirname(os.path.dirname(self.data[task][0]))
                base_name = os.path.basename(self.data[task][0])
                anno_json = os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json')
            
            # 保存预测结果为JSON文件
            pred_json = os.path.join(self.save_dir, "predictions.json")
            LOGGER.info(f'Saving {pred_json}...')
            with open(pred_json, 'w') as f:
                json.dump(pred_results, f)

            # 初始化COCO评估器
            anno = COCO(anno_json)  # 加载标注文件
            pred = anno.loadRes(pred_json)  # 加载预测结果
            cocoEval = COCOeval(anno, pred, 'bbox')  # 创建评估器对象
            
            # 如果是COCO数据集，设置图像ID
            if self.is_coco:
                imgIds = [int(os.path.basename(x).split(".")[0])
                            for x in dataloader.dataset.img_paths]
                cocoEval.params.imgIds = imgIds
            
            # 执行评估
            cocoEval.evaluate()  # 运行评估
            cocoEval.accumulate()  # 累积评估结果

            # 如果需要详细输出，打印每个类别的AP值
            if self.verbose:
                import copy
                # 统计验证集图像和标注数量
                val_dataset_img_count = cocoEval.cocoGt.imgToAnns.__len__()
                val_dataset_anns_count = 0
                # 初始化每个类别的统计字典
                label_count_dict = {"images":set(), "anns":0}
                label_count_dicts = [copy.deepcopy(label_count_dict) for _ in range(model.nc)]
                
                # 统计每个类别的图像和标注数量
                for _, ann_i in cocoEval.cocoGt.anns.items():
                    if ann_i["ignore"]:
                        continue
                    val_dataset_anns_count += 1
                    # 获取类别索引
                    nc_i = self.coco80_to_coco91_class().index(ann_i['category_id']) if self.is_coco else ann_i['category_id']
                    label_count_dicts[nc_i]["images"].add(ann_i["image_id"])
                    label_count_dicts[nc_i]["anns"] += 1

                # 打印评估结果表头
                s = ('%-16s' + '%12s' * 7) % ('Class', 'Labeled_images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
                LOGGER.info(s)
                
                # 计算总体指标
                # 获取所有精度值：[IOU阈值, 召回率阈值, 类别, 面积范围, 最大检测数]
                coco_p = cocoEval.eval['precision']
                coco_p_all = coco_p[:, :, :, 0, 2]  # 选择所有类别在area=all，maxDets=100的精度
                map = np.mean(coco_p_all[coco_p_all>-1])  # 计算mAP@0.5:0.95

                # 计算IOU=0.5时的指标
                coco_p_iou50 = coco_p[0, :, :, 0, 2]  # 选择IOU=0.5的精度
                map50 = np.mean(coco_p_iou50[coco_p_iou50>-1])  # 计算mAP@0.5
                # 计算每个召回率阈值下的平均精度
                mp = np.array([np.mean(coco_p_iou50[ii][coco_p_iou50[ii]>-1]) for ii in range(coco_p_iou50.shape[0])])
                # 生成召回率序列
                mr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
                # 计算F1分数
                mf1 = 2 * mp * mr / (mp + mr + 1e-16)
                i = mf1.argmax()  # 找到最大F1分数的索引

                # 打印总体结果
                pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format | 打印格式
                LOGGER.info(pf % ('all', val_dataset_img_count, val_dataset_anns_count, mp[i], mr[i], mf1[i], map50, map))

                # 计算每个类别的最佳F1分数及对应的精度和召回率
                for nc_i in range(model.nc):
                    # 获取当前类别的精度
                    coco_p_c = coco_p[:, :, nc_i, 0, 2]
                    map = np.mean(coco_p_c[coco_p_c>-1])  # 计算当前类别的mAP@0.5:0.95

                    # 计算IOU=0.5时的指标
                    coco_p_c_iou50 = coco_p[0, :, nc_i, 0, 2]
                    map50 = np.mean(coco_p_c_iou50[coco_p_c_iou50>-1])  # 计算当前类别的mAP@0.5
                    p = coco_p_c_iou50  # 精度
                    r = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)  # 召回率
                    f1 = 2 * p * r / (p + r + 1e-16)  # F1分数
                    i = f1.argmax()  # 最大F1分数的索引
                    # 打印当前类别的结果
                    LOGGER.info(pf % (model.names[nc_i], len(label_count_dicts[nc_i]["images"]), label_count_dicts[nc_i]["anns"], p[i], r[i], f1[i], map50, map))
            
            # 输出评估摘要
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # 获取mAP@0.5:0.95和mAP@0.5的值
            
            # 将模型转换回float类型用于训练
            model.float()  # for training | 用于训练
            
            # 如果不是训练任务，输出结果保存路径
            if task != 'train':
                LOGGER.info(f"Results saved to {self.save_dir}")
            return (map50, map)
        return (0.0, 0.0)

    def eval_speed(self, task):
        '''Evaluate model inference speed. | 评估模型推理速度'''
        if task != 'train':
            # 获取样本数量和时间统计结果
            n_samples = self.speed_result[0].item()
            # 计算预处理、推理和NMS的平均时间（毫秒）
            pre_time, inf_time, nms_time = 1000 * self.speed_result[1:].cpu().numpy() / n_samples
            # 打印每个阶段的平均时间
            for n, v in zip(["pre-process", "inference", "NMS"],[pre_time, inf_time, nms_time]):
                LOGGER.info("Average {} time: {:.2f} ms".format(n, v))

    def box_convert(self, x):
        '''Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right. | 
        将形状为[n, 4]的边界框从[x1, y1, x2, y2]格式转换为[x, y, w, h]格式，其中x1y1为左上角坐标，x2y2为右下角坐标'''
        # 根据输入类型选择克隆或复制
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        # 计算中心点x坐标
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center | x中心点
        # 计算中心点y坐标
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center | y中心点
        # 计算宽度
        y[:, 2] = x[:, 2] - x[:, 0]  # width | 宽度
        # 计算高度
        y[:, 3] = x[:, 3] - x[:, 1]  # height | 高度
        return y

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        '''Rescale coords (xyxy) from img1_shape to img0_shape. | 将坐标从img1尺寸缩放到img0尺寸'''
        # 获取缩放比例和填充值
        gain = ratio_pad[0]
        pad = ratio_pad[1]

        # 处理x坐标：减去填充并除以缩放比例
        coords[:, [0, 2]] -= pad[0]  # x padding | x方向填充
        coords[:, [0, 2]] /= gain[1]  # raw x gain | x方向缩放
        # 处理y坐标：减去填充并除以缩放比例
        coords[:, [1, 3]] -= pad[1]  # y padding | y方向填充
        coords[:, [1, 3]] /= gain[0]  # y gain | y方向缩放

        # 将坐标限制在原始图像范围内
        if isinstance(coords, torch.Tensor):  # faster individually | 对于张量，单独处理更快
            coords[:, 0].clamp_(0, img0_shape[1])  # x1 | x1坐标限制
            coords[:, 1].clamp_(0, img0_shape[0])  # y1 | y1坐标限制
            coords[:, 2].clamp_(0, img0_shape[1])  # x2 | x2坐标限制
            coords[:, 3].clamp_(0, img0_shape[0])  # y2 | y2坐标限制
        else:  # np.array (faster grouped) | numpy数组，分组处理更快
            coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2 | x1,x2坐标限制
            coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2 | y1,y2坐标限制
        return coords

    def convert_to_coco_format(self, outputs, imgs, paths, shapes, ids):
        '''将模型输出转换为COCO评估格式'''
        pred_results = []
        for i, pred in enumerate(outputs):
            # 跳过空预测结果
            if len(pred) == 0:
                continue
            # 获取图像路径和原始形状
            path, shape = Path(paths[i]), shapes[i][0]
            # 将预测框坐标缩放到原始图像尺寸
            self.scale_coords(imgs[i].shape[1:], pred[:, :4], shape, shapes[i][1])
            # 获取图像ID
            image_id = int(path.stem) if self.is_coco else path.stem
            # 将边界框转换为COCO格式 [x,y,w,h]
            bboxes = self.box_convert(pred[:, 0:4])
            # 调整边界框坐标：从中心点坐标转换为左上角坐标
            bboxes[:, :2] -= bboxes[:, 2:] / 2
            # 获取类别和置信度
            cls = pred[:, 5]
            scores = pred[:, 4]
            # 遍历每个预测框，构建COCO格式的预测结果
            for ind in range(pred.shape[0]):
                category_id = ids[int(cls[ind])]
                bbox = [round(x, 3) for x in bboxes[ind].tolist()]
                score = round(scores[ind].item(), 5)
                pred_data = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score
                }
                pred_results.append(pred_data)
        return pred_results

    @staticmethod
    def check_task(task):
        '''检查任务类型是否有效'''
        if task not in ['train', 'val', 'test', 'speed']:
            raise Exception("task argument error: only support 'train' / 'val' / 'test' / 'speed' task.")

    @staticmethod
    def check_thres(conf_thres, iou_thres, task):
        '''Check whether confidence and iou threshold are best for task val/speed | 检查置信度和IOU阈值是否最适合验证/速度测试任务'''
        if task != 'train':
            if task == 'val' or task == 'test':
                # 验证任务的最佳阈值建议
                if conf_thres > 0.03:
                    LOGGER.warning(f'The best conf_thresh when evaluate the model is less than 0.03, while you set it to: {conf_thres}')
                if iou_thres != 0.65:
                    LOGGER.warning(f'The best iou_thresh when evaluate the model is 0.65, while you set it to: {iou_thres}')
            # 速度测试任务的最佳阈值建议
            if task == 'speed' and conf_thres < 0.4:
                LOGGER.warning(f'The best conf_thresh when test the speed of the model is larger than 0.4, while you set it to: {conf_thres}')

    @staticmethod
    def reload_device(device, model, task):
        '''重新配置设备设置'''
        # device = 'cpu' or '0' or '0,1,2,3'
        if task == 'train':
            # 训练任务使用模型当前的设备
            device = next(model.parameters()).device
        else:
            # 根据指定设备配置环境
            if device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            elif device:
                os.environ['CUDA_VISIBLE_DEVICES'] = device
                assert torch.cuda.is_available()
            # 确定最终使用的设备
            cuda = device != 'cpu' and torch.cuda.is_available()
            device = torch.device('cuda:0' if cuda else 'cpu')
        return device

    @staticmethod
    def reload_dataset(data, task='val'):
        '''重新加载数据集配置'''
        # 读取数据集配置文件
        with open(data, errors='ignore') as yaml_file:
            data = yaml.safe_load(yaml_file)
        # 确定任务类型
        task = 'test' if task == 'test' else 'val'
        # 获取数据集路径
        path = data.get(task, 'val')
        if not isinstance(path, list):
            path = [path]
        # 检查数据集路径是否存在
        for p in path:
            if not os.path.exists(p):
                raise Exception(f'Dataset path {p} not found.')
        return data

    @staticmethod
    def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper) | 将80类索引（val2014）转换为91类索引（论文）
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        # COCO数据集类别映射表，用于将80类映射到91类索引
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

    def eval_trt(self, engine, stride=32):
        '''使用TensorRT引擎评估模型性能'''
        self.stride = stride
        def init_engine(engine):
            '''初始化TensorRT引擎'''
            import tensorrt as trt
            from collections import namedtuple,OrderedDict
            # 定义绑定数据结构
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            # 创建TensorRT日志记录器
            logger = trt.Logger(trt.Logger.ERROR)
            # 初始化NVIDIA推理插件
            trt.init_libnvinfer_plugins(logger, namespace="")
            # 加载和反序列化TensorRT引擎
            with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            # 创建绑定字典
            bindings = OrderedDict()
            # 遍历所有绑定并设置内存
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            # 创建绑定地址字典
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            # 创建执行上下文
            context = model.create_execution_context()
            return context, bindings, binding_addrs, model.get_binding_shape(0)[0]

        def init_data(dataloader, task):
            '''初始化数据加载器和相关配置'''
            # 检查是否为COCO数据集
            self.is_coco = self.data.get("is_coco", False)
            # 设置类别ID映射
            self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
            pad = 0.0
            # 创建数据加载器
            dataloader = create_dataloader(self.data[task if task in ('train', 'val', 'test') else 'val'],
                                           self.img_size, self.batch_size, self.stride, check_labels=True, pad=pad, rect=False,
                                           data_dict=self.data, task=task)[0]
            return dataloader

        def convert_to_coco_format_trt(nums, boxes, scores, classes, paths, shapes, ids):
            '''将TensorRT输出转换为COCO评估格式'''
            pred_results = []
            # 遍历每个批次的预测结果
            for i, (num, detbox, detscore, detcls) in enumerate(zip(nums, boxes, scores, classes)):
                # 获取有效检测框数量
                n = int(num[0])
                if n == 0:
                    continue
                # 获取图像路径和原始形状
                path, shape = Path(paths[i]), shapes[i][0]
                # 计算缩放和填充参数
                gain = shapes[i][1][0][0]
                pad = torch.tensor(shapes[i][1][1]*2).to(self.device)
                # 处理检测框坐标
                detbox = detbox[:n, :]
                detbox -= pad  # 减去填充
                detbox /= gain  # 缩放还原
                # 限制坐标范围在原始图像尺寸内
                detbox[:, 0].clamp_(0, shape[1])  # x1
                detbox[:, 1].clamp_(0, shape[0])  # y1
                detbox[:, 2].clamp_(0, shape[1])  # x2
                detbox[:, 3].clamp_(0, shape[0])  # y2
                # 转换为[x,y,w,h]格式
                detbox[:,2:] = detbox[:,2:] - detbox[:,:2]
                # 获取分数和类别
                detscore = detscore[:n]
                detcls = detcls[:n]

                # 获取图像ID
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem

                # 构建COCO格式的预测结果
                for ind in range(n):
                    category_id = ids[int(detcls[ind])]
                    bbox = [round(x, 3) for x in detbox[ind].tolist()]
                    score = round(detscore[ind].item(), 5)
                    pred_data = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": score
                    }
                    pred_results.append(pred_data)
            return pred_results

        # 初始化TensorRT引擎和相关配置
        context, bindings, binding_addrs, trt_batch_size = init_engine(engine)
        # 检查批次大小是否合法
        assert trt_batch_size >= self.batch_size, f'The batch size you set is {self.batch_size}, it must <= tensorrt binding batch size {trt_batch_size}.'
        
        # 创建临时输入数据进行预热
        tmp = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        # 预热模型10次
        for _ in range(10):
            binding_addrs['images'] = int(tmp.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
        
        # 初始化数据加载器
        dataloader = init_data(None,'val')
        # 初始化速度测试结果数组
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        # 创建进度条
        pbar = tqdm(dataloader, desc="Inferencing model in validation dataset.", ncols=NCOLS)
        
        # 遍历数据集进行推理
        for imgs, targets, paths, shapes in pbar:
            nb_img = imgs.shape[0]
            if nb_img != self.batch_size:
                # 如果批次大小不足，进行填充
                zeros = torch.zeros(self.batch_size - nb_img, 3, *imgs.shape[2:])
                imgs = torch.cat([imgs, zeros],0)
            
            # 记录预处理开始时间
            t1 = time_sync()
            # 将图像数据转移到设备并预处理
            imgs = imgs.to(self.device, non_blocking=True)
            imgs = imgs.float()  # 转换为浮点型
            imgs /= 255  # 归一化

            # 记录预处理时间
            self.speed_result[1] += time_sync() - t1  # pre-process time | 预处理时间

            # 推理阶段
            t2 = time_sync()
            # 设置输入数据地址并执行推理
            binding_addrs['images'] = int(imgs.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
            # 获取有效的检测结果
            nums = bindings['num_dets'].data[:nb_img]
            boxes = bindings['det_boxes'].data[:nb_img]
            scores = bindings['det_scores'].data[:nb_img]
            classes = bindings['det_classes'].data[:nb_img]
            # 记录推理时间
            self.speed_result[2] += time_sync() - t2  # inference time | 推理时间

            # NMS时间（TensorRT已集成NMS，此处为0）
            self.speed_result[3] += 0
            # 转换结果格式并累加批次大小
            pred_results.extend(convert_to_coco_format_trt(nums, boxes, scores, classes, paths, shapes, self.ids))
            self.speed_result[0] += self.batch_size
        return dataloader, pred_results