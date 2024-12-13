# Model validation metrics
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from . import general

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves. 计算每个类别的平均精确度，给定召回率和精确度曲线。
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10). 
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    # 参数
        tp：真阳性（numpy数组，nx1或nx10）  # 真阳性（numpy数组，nx1或nx10）
        conf：物体置信度值，范围从0到1（numpy数组）  # 物体置信度值，范围从0到1（numpy数组）
        pred_cls：预测的物体类别（numpy数组）  # 预测的物体类别（numpy数组）
        target_cls：真实的物体类别（numpy数组）  # 真实的物体类别（numpy数组）
        plot：是否绘制精确度-召回率曲线，mAP@0.5  # 是否绘制精确度-召回率曲线，mAP@0.5
        save_dir：绘图保存目录  # 绘图保存目录
    # 返回
        平均精确度  # 平均精确度
    """

    # Sort by objectness | 根据置信度排序
    i = np.argsort(-conf)  # 根据置信度排序，返回排序后的索引
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]  # 根据排序的索引重新排列真阳性、置信度和预测类别

    # Find unique classes | 找到唯一的类别
    unique_classes = np.unique(target_cls)  # 找到唯一的类别
    nc = unique_classes.shape[0]  # number of classes, number of detections # 类别数量，检测数量

    # Create Precision-Recall curve and compute AP for each class | 创建精确度-召回率曲线并计算每个类别的平均精确度
    px, py = np.linspace(0, 1, 1000), []  # for plotting 为绘图创建精确度-召回率曲线
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000)) # 初始化平均精确度、精确度和召回率
    for ci, c in enumerate(unique_classes): # 遍历每个类别
        i = pred_cls == c # 找到预测类别等于当前类别的索引
        n_l = (target_cls == c).sum()  # number of labels 真实标签数量
        n_p = i.sum()  # number of predictions # 预测数量

        if n_p == 0 or n_l == 0:  # 如果没有预测或标签
            continue  # 跳过当前类别
        else:
            # Accumulate FPs and TPs 累积假阳性和真阳性
            fpc = (1 - tp[i]).cumsum(0)  # 计算假阳性累积
            tpc = tp[i].cumsum(0)  # 计算真阳性累积

            # Recall 召回率
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision 精确度
            precision = tpc / (tpc + fpc)  # precision curve 召回率曲线
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score # 负x，xp因为xp递减

            # AP from recall-precision curve 从召回率-精确度曲线计算平均精确度
            for j in range(tp.shape[1]): # 遍历每个tp
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j]) # 计算平均精确度
                if plot and j == 0: # 如果需要绘图且为第一个tp
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5 # 在mAP@0.5处的精确度

    # Compute F1 (harmonic mean of precision and recall) 计算F1（精确度和召回率的调和平均数）
    f1 = 2 * p * r / (p + r + 1e-16) # 计算F1分数（精确度和召回率的调和平均数）
    if plot: # 如果需要绘图
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)  # 绘制精确度-召回率曲线
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')  # 绘制F1曲线
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')  # 绘制精确度曲线
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')  # 绘制召回率曲线

    # i = f1.mean(0).argmax()  # 最大F1索引  # 最大F1索引
    # return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')  # 返回精确度、召回率、平均精确度、F1分数和唯一类别
    return p, r, ap, f1, unique_classes.astype('int32')  # 返回精确度、召回率、平均精确度、F1分数和唯一类别


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves 计算平均精确度，给定召回率和精确度曲线。
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    # 参数
        recall：召回率曲线（列表）  # 召回率曲线（列表）
        precision：精确度曲线（列表）  # 精确度曲线（列表）
    # 返回
        平均精确度、精确度曲线、召回率曲线  # 平均精确度、精确度曲线、召回率曲线
    """

    # Append sentinel values to beginning and end 在开头和结尾添加哨兵值
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))  # 在开头和结尾添加哨兵值
    mpre = np.concatenate(([1.], precision, [0.]))  # 在开头和结尾添加哨兵值


    # Compute the precision envelope 计算精确度包络线
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp' # 方法：'连续'，'插值'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO) 如果使用插值方法
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate 积分计算
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes x轴（召回率）变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve 曲线下的面积计算

    return ap, mpre, mrec # 返回平均精确度、精确度曲线和召回率曲线

# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve 精确度-召回率曲线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True) # 创建绘图对象
    py = np.stack(py, axis=1) # 将py堆叠成二维数组

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes 如果类别数量小于21，显示每个类别的图例
        for i, y in enumerate(py.T): # 遍历每个类别的精确度-召回率曲线
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision) # 绘制召回率-精确度曲线
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)  绘制召回率-精确度曲线

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean()) # # 绘制所有类别的平均精确度曲线
    ax.set_xlabel('Recall') # 设置x轴标签
    ax.set_ylabel('Precision') # 设置y轴标签
    ax.set_xlim(0, 1)  # 设置x轴范围
    ax.set_ylim(0, 1)  # 设置y轴范围
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # 添加图例
    fig.savefig(Path(save_dir), dpi=250)  # 保存绘图



def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve 指标-置信度曲线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes # 如果类别数量小于21，显示每个类别的图例
        for i, y in enumerate(py): # 遍历每个类别的指标曲线
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric) # 绘制指标-置信度曲线
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric) # 绘制指标-置信度曲线

    y = py.mean(0) # 计算所有类别的平均指标
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)  # 设置x轴标签
    ax.set_ylabel(ylabel)  # 设置y轴标签
    ax.set_xlim(0, 1)  # 设置x轴范围
    ax.set_ylim(0, 1)  # 设置y轴范围
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left") # 添加图例
    fig.savefig(Path(save_dir), dpi=250) # 保存绘图

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    返回正确预测矩阵。两个集合的框都在（x1，y1，x2，y2）格式。
    参数：
        detections（Array[N，6]），x1，y1，x2，y2，conf，class  # 检测框，包含坐标、置信度和类别
        labels（Array[M，5]），class，x1，y1，x2，y2  # 真实框，包含类别和坐标
    返回：
        correct（Array[N，10]），对于10个IoU级别  # 正确预测的矩阵
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)  # 初始化正确预测矩阵
    iou = general.box_iou(labels[:, 1:], detections[:, :4])  # 计算IoU
    correct_class = labels[:, 0:1] == detections[:, 5]  # 检查类别是否匹配
    for i in range(len(iouv)): # 遍历每个IoU阈值
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match # IoU > 阈值且类别匹配
        if x[0].shape[0]: # 如果有匹配的框
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按照IoU降序排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 去重
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 去重
            correct[matches[:, 1].astype(int), i] = True                           # 标记正确预测
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device) # 返回正确预测矩阵

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes # 类别数量
        self.conf = conf # 置信度阈值
        self.iou_thres = iou_thres # IoU阈值

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes. 返回交并比（Jaccard指数）框。
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        两个集合的框都应该是（x1，y1，x2，y2）格式。
        参数：
            detections（Array[N，6]），x1，y1，x2，y2，conf，class  # 检测框，包含坐标、置信度和类别
            labels（Array[M，5]），class，x1，y1，x2，y2  # 真实框，包含类别和坐标
        返回：
            无，更新混淆矩阵  # 更新混淆矩阵
        """
        detections = detections[detections[:, 4] > self.conf]  # 过滤低置信度的检测框
        gt_classes = labels[:, 0].int()  # 获取真实类别
        detection_classes = detections[:, 5].int()  # 获取检测类别
        iou = general.box_iou(labels[:, 1:], detections[:, :4])  # 计算IoU


        # x = torch.where(iou > self.iou_thres)
        # if x[0].shape[0]:
        #     matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        #     if x[0].shape[0] > 1:
        #         matches = matches[matches[:, 2].argsort()[::-1]]
        #         matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        #         matches = matches[matches[:, 2].argsort()[::-1]]
        #         matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        # else:
        #     matches = np.zeros((0, 3))

        # n = matches.shape[0] > 0
        # m0, m1, _ = matches.transpose().astype(int)
        x = torch.where(iou > self.iou_thres)  # 找到IoU大于阈值的匹配
        if x[0].shape[0]:  # 如果有匹配
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label，detect，iou]
            if x[0].shape[0] > 1:  # 如果有多个匹配
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按照IoU降序排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 去重
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按照IoU降序排序
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 去重
        else:
            matches = np.zeros((0, 3))  # 如果没有匹配，初始化为空数组

        n = matches.shape[0] > 0  # 检查是否有匹配
        m0, m1, _ = matches.transpose().astype(int)  # 转置匹配数组并转换为整数
        
        for i, gc in enumerate(gt_classes): # 遍历真实类别
            j = m0 == i # 找到与真实类别匹配的检测
            if n and sum(j) == 1: # 如果有匹配且只有一个
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct # 正确
            else:
                self.matrix[self.nc, gc] += 1  # background FP # 背景FP

        if n: # 如果有匹配
            for i, dc in enumerate(detection_classes): # 遍历检测类别
                if not any(m1 == i): # 如果没有与真实类别匹配
                    self.matrix[dc, self.nc] += 1  # background FN 背景FN

    def matrix(self):
        return self.matrix  # 返回混淆矩阵

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives # 真阳性
        fp = self.matrix.sum(1) - tp  # false positives # 假阳性
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections) # 假阴性（未检测到的目标）
        return tp[:-1], fp[:-1]  # remove background class # 删除背景类别

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn # 导入seaborn库，用于绘制热图

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns # 归一化列
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00) # 不注释（将显示为0.00）

            fig = plt.figure(figsize=(12, 9), tight_layout=True)  # 创建绘图对象
            nc, nn = self.nc, len(names)  # number of classes, names # 类别数量，名称
            sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size # 标签大小
            labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels # 应用名称到刻度标签
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered # 忽略空矩阵RuntimeWarning：遇到所有NaN切片
                sn.heatmap(array,
                           annot=nc < 30,
                           annot_kws={
                               "size": 8},
                           cmap='Blues',
                           fmt='.2f',
                           square=True,
                           vmin=0.0,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1)) # 保存混淆矩阵图
            fig.axes[0].set_xlabel('True') # 设置x轴标签
            fig.axes[0].set_ylabel('Predicted') # 设置y轴标签
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close() # 关闭图形
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i]))) # 打印混淆矩阵每一行
