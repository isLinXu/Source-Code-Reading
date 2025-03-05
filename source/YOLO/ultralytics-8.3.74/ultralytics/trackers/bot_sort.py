# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 的 AGPL-3.0 许可证

from collections import deque  # 从 collections 导入 deque 类

import numpy as np  # 导入 NumPy 库

from .basetrack import TrackState  # 从 basetrack 模块导入 TrackState 类
from .byte_tracker import BYTETracker, STrack  # 从 byte_tracker 模块导入 BYTETracker 和 STrack 类
from .utils import matching  # 从 utils 模块导入 matching
from .utils.gmc import GMC  # 从 utils.gmc 模块导入 GMC 类
from .utils.kalman_filter import KalmanFilterXYWH  # 从 utils.kalman_filter 模块导入 KalmanFilterXYWH 类


class BOTrack(STrack):  # 定义 BOTrack 类，继承自 STrack
    """
    An extended version of the STrack class for YOLOv8, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.  # 所有 BOTrack 实例共享的 Kalman 滤波器
        smooth_feat (np.ndarray): Smoothed feature vector.  # 平滑后的特征向量
        curr_feat (np.ndarray): Current feature vector.  # 当前特征向量
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.  # 用于存储特征向量的双端队列，最大长度由 `feat_history` 定义
        alpha (float): Smoothing factor for the exponential moving average of features.  # 特征的指数移动平均平滑因子
        mean (np.ndarray): The mean state of the Kalman filter.  # Kalman 滤波器的均值状态
        covariance (np.ndarray): The covariance matrix of the Kalman filter.  # Kalman 滤波器的协方差矩阵

    Methods:
        update_features(feat): Update features vector and smooth it using exponential moving average.  # 更新特征向量并使用指数移动平均进行平滑
        predict(): Predicts the mean and covariance using Kalman filter.  # 使用 Kalman 滤波器预测均值和协方差
        re_activate(new_track, frame_id, new_id): Reactivates a track with updated features and optionally new ID.  # 使用更新的特征重新激活跟踪，并可选择分配新 ID
        update(new_track, frame_id): Update the YOLOv8 instance with new track and frame ID.  # 使用新跟踪和帧 ID 更新 YOLOv8 实例
        tlwh: Property that gets the current position in tlwh format [(top left x, top left y, width, height)](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/trackers/basetrack.py:107:4-109:58).  # 获取当前位于 tlwh 格式的属性 [(左上角 x, 左上角 y, 宽度, 高度)](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/trackers/basetrack.py:107:4-109:58)
        multi_predict(stracks): Predicts the mean and covariance of multiple object tracks using shared Kalman filter.  # 使用共享的 Kalman 滤波器预测多个对象跟踪的均值和协方差
        convert_coords(tlwh): Converts tlwh bounding box coordinates to xywh format.  # 将 tlwh 边界框坐标转换为 xywh 格式
        tlwh_to_xywh(tlwh): Convert bounding box to xywh format [(center x, center y, width, height)](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/trackers/basetrack.py:107:4-109:58).  # 将边界框转换为 xywh 格式 [(中心 x, 中心 y, 宽度, 高度)](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/trackers/basetrack.py:107:4-109:58)

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(tlwh=[100, 50, 80, 40], score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(tlwh=[110, 60, 80, 40], score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()  # 定义一个共享的 Kalman 滤波器实例

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):  # 初始化 BOTrack 对象，接受边界框、分数、类别、特征和特征历史长度
        """
        Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            tlwh (np.ndarray): Bounding box coordinates in tlwh format (top left x, top left y, width, height).  # tlwh 格式的边界框坐标 (左上角 x, 左上角 y, 宽度, 高度)
            score (float): Confidence score of the detection.  # 检测的置信度分数
            cls (int): Class ID of the detected object.  # 检测对象的类别 ID
            feat (np.ndarray | None): Feature vector associated with the detection.  # 与检测相关的特征向量
            feat_history (int): Maximum length of the feature history deque.  # 特征历史双端队列的最大长度

        Examples:
            Initialize a BOTrack object with bounding box, score, class ID, and feature vector
            >>> tlwh = np.array([100, 50, 80, 120])
            >>> score = 0.9
            >>> cls = 1
            >>> feat = np.random.rand(128)
            >>> bo_track = BOTrack(tlwh, score, cls, feat)  # 初始化 BOTrack 对象
        """
        super().__init__(tlwh, score, cls)  # 调用父类构造函数

        self.smooth_feat = None  # 平滑特征初始化为 None
        self.curr_feat = None  # 当前特征初始化为 None
        if feat is not None:  # 如果提供了特征
            self.update_features(feat)  # 更新特征
        self.features = deque([], maxlen=feat_history)  # 初始化特征双端队列，最大长度为 feat_history
        self.alpha = 0.9  # 设置平滑因子

    def update_features(self, feat):  # 更新特征方法
        """Update the feature vector and apply exponential moving average smoothing."""  # 更新特征向量并应用指数移动平均平滑
        feat /= np.linalg.norm(feat)  # 归一化特征向量
        self.curr_feat = feat  # 更新当前特征
        if self.smooth_feat is None:  # 如果平滑特征为 None
            self.smooth_feat = feat  # 设置平滑特征为当前特征
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat  # 使用指数移动平均更新平滑特征
        self.features.append(feat)  # 将当前特征添加到特征队列
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  # 归一化平滑特征

    def predict(self):  # 预测方法
        """Predicts the object's future state using the Kalman filter to update its mean and covariance."""  # 使用 Kalman 滤波器预测对象的未来状态，更新其均值和协方差
        mean_state = self.mean.copy()  # 复制均值状态
        if self.state != TrackState.Tracked:  # 如果状态不是被跟踪
            mean_state[6] = 0  # 将第 7 个元素设置为 0
            mean_state[7] = 0  # 将第 8 个元素设置为 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)  # 使用 Kalman 滤波器进行预测

    def re_activate(self, new_track, frame_id, new_id=False):  # 重新激活方法
        """Reactivates a track with updated features and optionally assigns a new ID."""  # 使用更新的特征重新激活跟踪，并可选择分配新 ID
        if new_track.curr_feat is not None:  # 如果新跟踪的当前特征不为 None
            self.update_features(new_track.curr_feat)  # 更新特征
        super().re_activate(new_track, frame_id, new_id)  # 调用父类的重新激活方法

    def update(self, new_track, frame_id):  # 更新方法
        """Updates the YOLOv8 instance with new track information and the current frame ID."""  # 使用新跟踪信息和当前帧 ID 更新 YOLOv8 实例
        if new_track.curr_feat is not None:  # 如果新跟踪的当前特征不为 None
            self.update_features(new_track.curr_feat)  # 更新特征
        super().update(new_track, frame_id)  # 调用父类的更新方法

    @property
    def tlwh(self):  # tlwh 属性
        """Returns the current bounding box position in [(top left x, top left y, width, height)](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/trackers/basetrack.py:107:4-109:58) format."""  # 返回当前边界框位置，格式为 [(左上角 x, 左上角 y, 宽度, 高度)](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/trackers/basetrack.py:107:4-109:58)
        if self.mean is None:  # 如果均值为 None
            return self._tlwh.copy()  # 返回当前 tlwh 的副本
        ret = self.mean[:4].copy()  # 获取均值的前 4 个元素的副本
        ret[:2] -= ret[2:] / 2  # 将左上角坐标调整为中心坐标
        return ret  # 返回调整后的坐标

    @staticmethod
    def multi_predict(stracks):  # 静态方法，进行多对象预测
        """Predicts the mean and covariance for multiple object tracks using a shared Kalman filter."""  # 使用共享的 Kalman 滤波器预测多个对象跟踪的均值和协方差
        if len(stracks) <= 0:  # 如果跟踪列表为空
            return  # 返回
        multi_mean = np.asarray([st.mean.copy() for st in stracks])  # 复制所有跟踪的均值
        multi_covariance = np.asarray([st.covariance for st in stracks])  # 复制所有跟踪的协方差
        for i, st in enumerate(stracks):  # 遍历每个跟踪
            if st.state != TrackState.Tracked:  # 如果状态不是被跟踪
                multi_mean[i][6] = 0  # 将第 7 个元素设置为 0
                multi_mean[i][7] = 0  # 将第 8 个元素设置为 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)  # 使用共享的 Kalman 滤波器进行预测
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):  # 遍历每个均值和协方差
            stracks[i].mean = mean  # 更新跟踪的均值
            stracks[i].covariance = cov  # 更新跟踪的协方差

    def convert_coords(self, tlwh):  # 坐标转换方法
        """Converts tlwh bounding box coordinates to xywh format."""  # 将 tlwh 边界框坐标转换为 xywh 格式
        return self.tlwh_to_xywh(tlwh)  # 调用 tlwh_to_xywh 方法进行转换

    @staticmethod
    def tlwh_to_xywh(tlwh):  # 静态方法，进行坐标转换
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""  # 将边界框从 tlwh (左上角-宽度-高度) 转换为 xywh (中心-x-中心-y-宽度-高度) 格式
        ret = np.asarray(tlwh).copy()  # 复制 tlwh 数组
        ret[:2] += ret[2:] / 2  # 将左上角坐标调整为中心坐标
        return ret  # 返回调整后的坐标


class BOTSORT(BYTETracker):  # 定义 BOTSORT 类，继承自 BYTETracker
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.  # 跟踪和检测之间空间接近度 (IoU) 的阈值
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.  # 跟踪和检测之间外观相似度 (ReID 嵌入) 的阈值
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.  # 处理 ReID 嵌入的对象，如果未启用 ReID，则设置为 None
        gmc (GMC): An instance of the GMC algorithm for data association.  # 数据关联的 GMC 算法实例
        args (Any): Parsed command-line arguments containing tracking parameters.  # 解析的命令行参数，包含跟踪参数

    Methods:
        get_kalmanfilter(): Returns an instance of KalmanFilterXYWH for object tracking.  # 返回 KalmanFilterXYWH 的实例，用于对象跟踪
        init_track(dets, scores, cls, img): Initialize track with detections, scores, and classes.  # 使用检测、分数和类别初始化跟踪
        get_dists(tracks, detections): Get distances between tracks and detections using IoU and (optionally) ReID.  # 使用 IoU 和 (可选) ReID 获取跟踪和检测之间的距离
        multi_predict(tracks): Predict and track multiple objects with YOLOv8 model.  # 使用 YOLOv8 模型预测和跟踪多个对象

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(dets, scores, cls, img)
        >>> bot_sort.multi_predict(tracks)

    Note:
        The class is designed to work with the YOLOv8 object detection model and supports ReID only if enabled via args.  # 该类设计用于与 YOLOv8 对象检测模型配合使用，仅在通过 args 启用时支持 ReID
    """

    def __init__(self, args, frame_rate=30):  # 初始化 BOTSORT 类，接受参数和帧率
        """
        Initialize YOLOv8 object with ReID module and GMC algorithm.

        Args:
            args (object): Parsed command-line arguments containing tracking parameters.  # 解析的命令行参数，包含跟踪参数
            frame_rate (int): Frame rate of the video being processed.  # 正在处理的视频的帧率

        Examples:
            Initialize BOTSORT with command-line arguments and a specified frame rate:
            >>> args = parse_args()
            >>> bot_sort = BOTSORT(args, frame_rate=30)  # 使用命令行参数和指定帧率初始化 BOTSORT
        """
        super().__init__(args, frame_rate)  # 调用父类构造函数
        # ReID module
        self.proximity_thresh = args.proximity_thresh  # 设置空间接近度阈值
        self.appearance_thresh = args.appearance_thresh  # 设置外观相似度阈值

        if args.with_reid:  # 如果启用了 ReID
            # Haven't supported BoT-SORT(reid) yet  # 还未支持 BoT-SORT (reid)
            self.encoder = None  # 设置编码器为 None
        self.gmc = GMC(method=args.gmc_method)  # 创建 GMC 实例，使用指定的方法

    def get_kalmanfilter(self):  # 获取 Kalman 滤波器的方法
        """Returns an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""  # 返回 KalmanFilterXYWH 的实例，用于预测和更新跟踪过程中的对象状态
        return KalmanFilterXYWH()  # 返回 KalmanFilterXYWH 实例

    def init_track(self, dets, scores, cls, img=None):  # 初始化跟踪的方法
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""  # 使用检测边界框、分数、类别标签和可选的 ReID 特征初始化对象跟踪
        if len(dets) == 0:  # 如果检测列表为空
            return []  # 返回空列表
        if self.args.with_reid and self.encoder is not None:  # 如果启用了 ReID 且编码器不为 None
            features_keep = self.encoder.inference(img, dets)  # 使用编码器进行推理，获取特征
            return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]  # 返回包含特征的 BOTrack 实例
        else:
            return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]  # 返回不包含特征的 BOTrack 实例

    def get_dists(self, tracks, detections):  # 获取距离的方法
        """Calculates distances between tracks and detections using IoU and optionally ReID embeddings."""  # 使用 IoU 和可选的 ReID 嵌入计算跟踪和检测之间的距离
        dists = matching.iou_distance(tracks, detections)  # 计算 IoU 距离
        dists_mask = dists > self.proximity_thresh  # 创建距离掩码，筛选出小于阈值的距离

        if self.args.fuse_score:  # 如果启用了分数融合
            dists = matching.fuse_score(dists, detections)  # 融合分数

        if self.args.with_reid and self.encoder is not None:  # 如果启用了 ReID 且编码器不为 None
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0  # 计算嵌入距离并除以 2
            emb_dists[emb_dists > self.appearance_thresh] = 1.0  # 将大于外观阈值的嵌入距离设置为 1.0
            emb_dists[dists_mask] = 1.0  # 将掩码对应的嵌入距离设置为 1.0
            dists = np.minimum(dists, emb_dists)  # 取 IoU 距离和嵌入距离的最小值
        return dists  # 返回计算后的距离

    def multi_predict(self, tracks):  # 多对象预测的方法
        """Predicts the mean and covariance for multiple object tracks using a shared Kalman filter."""  # 使用共享的 Kalman 滤波器预测多个对象跟踪的均值和协方差
        BOTrack.multi_predict(tracks)  # 调用 BOTrack 的多预测方法

    def reset(self):  # 重置方法
        """Resets the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""  # 将 BOTSORT 跟踪器重置为初始状态，清除所有跟踪对象和内部状态
        super().reset()  # 调用父类的重置方法
        self.gmc.reset_params()  # 重置 GMC 的参数