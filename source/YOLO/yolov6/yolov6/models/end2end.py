import torch
import torch.nn as nn
import random


class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    # ONNX-Runtime 非极大值抑制（NMS）操作

    @staticmethod
    def forward(ctx,
                boxes,  # 预测框的坐标
                scores,  # 预测框的分数
                max_output_boxes_per_class=torch.tensor([100]),  # 每个类别的最大输出框数，默认为100
                iou_threshold=torch.tensor([0.45]),  # IOU阈值，默认为0.45
                score_threshold=torch.tensor([0.25])):  # 分数阈值，默认为0.25
        device = boxes.device  # 获取boxes的设备信息（CPU或GPU）
        batch = scores.shape[0]  # 获取批次大小
        num_det = random.randint(0, 100)  # 随机生成一个检测框的数量（0到100之间）
        
        # 随机选择batch中的一些索引
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)  # 随机选择batch中的索引并排序
        idxs = torch.arange(100, 100 + num_det).to(device)  # 生成从100到100+num_det的索引
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)  # 创建一个全为0的张量，大小为num_det
        
        # 将batches、zeros和idxs拼接在一起，形成选中的索引
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)  # 转换为int64类型
        
        return selected_indices  # 返回选中的索引

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        # 定义图的符号表示
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

class TRT8_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    # TensorRT 非极大值抑制（NMS）操作

    @staticmethod
    def forward(
        ctx,
        boxes,  # 预测框的坐标
        scores,  # 预测框的分数
        background_class=-1,  # 背景类的索引，默认为-1
        box_coding=1,  # 盒子编码方式，默认为1
        iou_threshold=0.45,  # IOU阈值，默认为0.45
        max_output_boxes=100,  # 最大输出框数，默认为100
        plugin_version="1",  # 插件版本，默认为"1"
        score_activation=0,  # 分数激活函数，默认为0
        score_threshold=0.25,  # 分数阈值，默认为0.25
    ):
        # 获取批次大小、框的数量和类别数量
        batch_size, num_boxes, num_classes = scores.shape
        
        # 随机生成每个批次的检测框数量
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        
        # 初始化检测框、检测分数和检测类别
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)  # 随机生成检测框坐标
        det_scores = torch.randn(batch_size, max_output_boxes)  # 随机生成检测分数
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)  # 随机生成检测类别
        
        return num_det, det_boxes, det_scores, det_classes  # 返回检测数量、检测框、检测分数和检测类别

    @staticmethod
    def symbolic(g,
                 boxes,  # 预测框的坐标
                 scores,  # 预测框的分数
                 background_class=-1,  # 背景类的索引，默认为-1
                 box_coding=1,  # 盒子编码方式，默认为1
                 iou_threshold=0.45,  # IOU阈值，默认为0.45
                 max_output_boxes=100,  # 最大输出框数，默认为100
                 plugin_version="1",  # 插件版本，默认为"1"
                 score_activation=0,  # 分数激活函数，默认为0
                 score_threshold=0.25):  # 分数阈值，默认为0.25
        # 定义图的符号表示
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,  # 背景类索引
                   box_coding_i=box_coding,  # 盒子编码方式
                   iou_threshold_f=iou_threshold,  # IOU阈值
                   max_output_boxes_i=max_output_boxes,  # 最大输出框数
                   plugin_version_s=plugin_version,  # 插件版本
                   score_activation_i=score_activation,  # 分数激活函数
                   score_threshold_f=score_threshold,  # 分数阈值
                   outputs=4)  # 输出数量为4
        
        nums, boxes, scores, classes = out  # 解包输出
        return nums, boxes, scores, classes  # 返回检测数量、检测框、检测分数和检测类别

class TRT7_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        plugin_version="1",
        shareLocation=1,
        backgroundLabelId=-1,
        numClasses=80,
        topK=1000,
        keepTopK=100,
        scoreThreshold=0.25,
        iouThreshold=0.45,
        isNormalized=0,
        clipBoxes=0,
        scoreBits=16,
        caffeSemantics=1,
    ):
        batch_size, num_boxes, numClasses = scores.shape
        num_det = torch.randint(0, keepTopK, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, keepTopK, 4)
        det_scores = torch.randn(batch_size, keepTopK)
        det_classes = torch.randint(0, numClasses, (batch_size, keepTopK)).float()
        return num_det, det_boxes, det_scores, det_classes
    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 plugin_version='1',
                 shareLocation=1,
                 backgroundLabelId=-1,
                 numClasses=80,
                 topK=1000,
                 keepTopK=100,
                 scoreThreshold=0.25,
                 iouThreshold=0.45,
                 isNormalized=0,
                 clipBoxes=0,
                 scoreBits=16,
                 caffeSemantics=1,
                 ):
        out = g.op("TRT::BatchedNMSDynamic_TRT", # BatchedNMS_TRT BatchedNMSDynamic_TRT
                   boxes,
                   scores,
                   shareLocation_i=shareLocation,
                   plugin_version_s=plugin_version,
                   backgroundLabelId_i=backgroundLabelId,
                   numClasses_i=numClasses,
                   topK_i=topK,
                   keepTopK_i=keepTopK,
                   scoreThreshold_f=scoreThreshold,
                   iouThreshold_f=iouThreshold,
                   isNormalized_i=isNormalized,
                   clipBoxes_i=clipBoxes,
                   scoreBits_i=scoreBits,
                   caffeSemantics_i=caffeSemantics,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    # 带有ONNX-Runtime非极大值抑制（NMS）操作的ONNX模块

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")  # 设置设备，默认为CPU
        self.max_obj = torch.tensor([max_obj]).to(device)  # 最大对象数量，转换为张量并移动到指定设备
        self.iou_threshold = torch.tensor([iou_thres]).to(device)  # IOU阈值，转换为张量并移动到指定设备
        self.score_threshold = torch.tensor([score_thres]).to(device)  # 分数阈值，转换为张量并移动到指定设备
        
        # 转换矩阵，用于坐标变换
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [-0.5, 0, 0.5, 0], 
                                             [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)  # 转换矩阵，转换为float32类型并移动到指定设备

    def forward(self, x):
        # 前向传播方法
        batch, anchors, _ = x.shape  # 获取输入张量的批次大小、锚框数量和特征维度
        box = x[:, :, :4]  # 提取前四个维度作为边界框坐标
        conf = x[:, :, 4:5]  # 提取第五个维度作为置信度
        score = x[:, :, 5:]  # 提取第六个维度及之后的分数
        score *= conf  # 将分数与置信度相乘，得到最终得分

        nms_box = box @ self.convert_matrix  # 使用转换矩阵对边界框进行转换
        nms_score = score.transpose(1, 2).contiguous()  # 转置得分张量以便于后续处理

        # 调用ORT_NMS的apply方法进行非极大值抑制
        selected_indices = ORT_NMS.apply(nms_box, nms_score, self.max_obj, self.iou_threshold, self.score_threshold)
        
        # 解包选中的索引
        batch_inds, cls_inds, box_inds = selected_indices.unbind(1)  
        selected_score = nms_score[batch_inds, cls_inds, box_inds].unsqueeze(1)  # 获取选中框的得分
        selected_box = nms_box[batch_inds, box_inds, ...]  # 获取选中框的坐标

        # 将选中的框和得分拼接在一起
        dets = torch.cat([selected_box, selected_score], dim=1)

        batched_dets = dets.unsqueeze(0).repeat(batch, 1, 1)  # 扩展维度并重复以匹配批次大小
        batch_template = torch.arange(0, batch, dtype=batch_inds.dtype, device=batch_inds.device)  # 创建批次模板
        batched_dets = batched_dets.where((batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1), batched_dets.new_zeros(1))  # 根据批次索引选择框

        batched_labels = cls_inds.unsqueeze(0).repeat(batch, 1)  # 扩展类别索引以匹配批次大小
        batched_labels = batched_labels.where((batch_inds == batch_template.unsqueeze(1)), batched_labels.new_ones(1) * -1)  # 根据批次索引选择类别

        N = batched_dets.shape[0]  # 获取检测框的数量

        # 为检测框和标签添加额外的零填充
        batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)  # 在最后添加零填充
        batched_labels = torch.cat((batched_labels, -batched_labels.new_ones((N, 1))), 1)  # 在最后添加负一填充

        # 对得分进行排序以获取前k个检测框
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)

        topk_batch_inds = torch.arange(batch, dtype=topk_inds.dtype, device=topk_inds.device).view(-1, 1)  # 创建批次索引
        batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]  # 根据索引选择检测框
        det_classes = batched_labels[topk_batch_inds, topk_inds, ...]  # 根据索引选择检测类别
        det_boxes, det_scores = batched_dets.split((4, 1), -1)  # 拆分检测框和得分
        det_scores = det_scores.squeeze(-1)  # 去除最后一个维度
        num_det = (det_scores > 0).sum(1, keepdim=True)  # 计算每个批次的检测数量
        
        return num_det, det_boxes, det_scores, det_classes  # 返回检测数量、检测框、检测分数和检测类别

class ONNX_TRT7(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    # 带有TensorRT非极大值抑制（NMS）操作的ONNX模块

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')  # 设置设备，默认为CPU
        self.shareLocation = 1  # 是否共享位置
        self.backgroundLabelId = -1  # 背景标签的索引，默认为-1
        self.numClasses = 80  # 类别数量，默认为80
        self.topK = 1000  # 最大检测框数，默认为1000
        self.keepTopK = max_obj  # 保留的最大对象数量
        self.scoreThreshold = score_thres  # 分数阈值
        self.iouThreshold = iou_thres  # IOU阈值
        self.isNormalized = 0  # 是否归一化，默认为0
        self.clipBoxes = 0  # 是否裁剪框，默认为0
        self.scoreBits = 16  # 分数位数，默认为16
        self.caffeSemantics = 1  # Caffe语义，默认为1
        self.plugin_version = '1'  # 插件版本，默认为'1'
        
        # 转换矩阵，用于坐标变换
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [-0.5, 0, 0.5, 0], 
                                             [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)  # 转换矩阵，转换为float32类型并移动到指定设备

    def forward(self, x):
        # 前向传播方法
        box = x[:, :, :4]  # 提取前四个维度作为边界框坐标
        conf = x[:, :, 4:5]  # 提取第五个维度作为置信度
        score = x[:, :, 5:]  # 提取第六个维度及之后的分数
        score *= conf  # 将分数与置信度相乘，得到最终得分

        box @= self.convert_matrix  # 使用转换矩阵对边界框进行变换
        box = box.unsqueeze(2)  # 扩展边界框的维度以便于后续处理
        self.numClasses = int(score.shape[2])  # 更新类别数量

        # 调用TRT7_NMS的apply方法进行非极大值抑制
        num_det, det_boxes, det_scores, det_classes = TRT7_NMS.apply(box, score, self.plugin_version,
                                                                     self.shareLocation,
                                                                     self.backgroundLabelId,
                                                                     self.numClasses,
                                                                     self.topK,
                                                                     self.keepTopK,
                                                                     self.scoreThreshold,
                                                                     self.iouThreshold,
                                                                     self.isNormalized,
                                                                     self.clipBoxes,
                                                                     self.scoreBits,
                                                                     self.caffeSemantics,
                                                                     )
        return num_det, det_boxes, det_scores, det_classes.int()  # 返回检测数量、检测框、检测分数和检测类别（转换为整数）


class ONNX_TRT8(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    # 带有TensorRT非极大值抑制（NMS）操作的ONNX模块

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')  # 设置设备，默认为CPU
        self.background_class = -1  # 背景类的索引，默认为-1
        self.box_coding = 1  # 盒子编码方式，默认为1
        self.iou_threshold = iou_thres  # IOU阈值
        self.max_obj = max_obj  # 最大对象数量
        self.plugin_version = '1'  # 插件版本，默认为'1'
        self.score_activation = 0  # 分数激活函数，默认为0
        self.score_threshold = score_thres  # 分数阈值

    def forward(self, x):
        # 前向传播方法
        box = x[:, :, :4]  # 提取前四个维度作为边界框坐标
        conf = x[:, :, 4:5]  # 提取第五个维度作为置信度
        score = x[:, :, 5:]  # 提取第六个维度及之后的分数
        score *= conf  # 将分数与置信度相乘，得到最终得分
        
        # 调用TRT8_NMS的apply方法进行非极大值抑制
        num_det, det_boxes, det_scores, det_classes = TRT8_NMS.apply(box, score, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes  # 返回检测数量、检测框、检测分数和检测类别


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    # 导出ONNX或TensorRT模型并进行NMS操作

    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None, ort=False, trt_version=8, with_preprocess=False):
        super().__init__()
        device = device if device else torch.device('cpu')  # 设置设备，默认为CPU
        self.with_preprocess = with_preprocess  # 是否进行预处理
        self.model = model.to(device)  # 将模型移动到指定设备
        TRT = ONNX_TRT8 if trt_version >= 8 else ONNX_TRT7  # 根据TensorRT版本选择相应的NMS模块
        self.patch_model = ONNX_ORT if ort else TRT  # 根据是否使用ONNX-Runtime选择模型
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, device)  # 实例化NMS模块
        self.end2end.eval()  # 设置为评估模式

    def forward(self, x):
        # 前向传播方法
        if self.with_preprocess:
            x = x[:, [2, 1, 0], ...]  # 预处理：调整通道顺序
            x = x * (1 / 255)  # 归一化到[0, 1]范围
        x = self.model(x)  # 通过主模型进行前向传播
        if isinstance(x, list):
            x = x[0]  # 如果输出是列表，取第一个元素
        else:
            x = x  # 否则保持不变
        x = self.end2end(x)  # 通过NMS模块进行前向传播
        return x  # 返回最终输出
