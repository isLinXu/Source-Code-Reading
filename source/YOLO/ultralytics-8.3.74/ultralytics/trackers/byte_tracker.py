# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


import numpy as np  # 导入 NumPy 库

from ..utils import LOGGER  # 从 utils 模块导入 LOGGER
from ..utils.ops import xywh2ltwh  # 从 utils.ops 模块导入 xywh2ltwh 函数
from .basetrack import BaseTrack, TrackState  # 从 basetrack 模块导入 BaseTrack 和 TrackState 类
from .utils import matching  # 从 utils 模块导入 matching
from .utils.kalman_filter import KalmanFilterXYAH  # 从 utils.kalman_filter 模块导入 KalmanFilterXYAH 类


class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.  # 使用 Kalman 滤波器进行状态估计的单对象跟踪表示。

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.  # 该类负责存储有关单个跟踪段的所有信息，并根据 Kalman 滤波器执行状态更新和预测。

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter that is used across all STrack instances for prediction.  # 共享的 Kalman 滤波器，用于所有 STrack 实例的预测。
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.  # 私有属性，用于存储边界框的左上角坐标和宽度、高度。
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.  # 用于此特定对象跟踪的 Kalman 滤波器实例。
        mean (np.ndarray): Mean state estimate vector.  # 均值状态估计向量。
        covariance (np.ndarray): Covariance of state estimate.  # 状态估计的协方差。
        is_activated (bool): Boolean flag indicating if the track has been activated.  # 布尔标志，指示跟踪是否已激活。
        score (float): Confidence score of the track.  # 跟踪的置信度分数。
        tracklet_len (int): Length of the tracklet.  # 跟踪段的长度。
        cls (Any): Class label for the object.  # 对象的类别标签。
        idx (int): Index or identifier for the object.  # 对象的索引或标识符。
        frame_id (int): Current frame ID.  # 当前帧 ID。
        start_frame (int): Frame where the object was first detected.  # 对象首次被检测到的帧。

    Methods:
        predict(): Predict the next state of the object using Kalman filter.  # 使用 Kalman 滤波器预测对象的下一个状态。
        multi_predict(stracks): Predict the next states for multiple tracks.  # 预测多个跟踪的下一个状态。
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.  # 使用单应性矩阵更新多个跟踪状态。
        activate(kalman_filter, frame_id): Activate a new tracklet.  # 激活一个新的跟踪段。
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.  # 重新激活之前丢失的跟踪段。
        update(new_track, frame_id): Update the state of a matched track.  # 更新匹配跟踪的状态。
        convert_coords(tlwh): Convert bounding box to x-y-aspect-height format.  # 将边界框转换为 x-y-宽度-高度格式。
        tlwh_to_xyah(tlwh): Convert tlwh bounding box to xyah format.  # 将 tlwh 边界框转换为 xyah 格式。

    Examples:
        Initialize and activate a new track  # 初始化并激活一个新跟踪
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")  # 创建 STrack 实例
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)  # 激活跟踪段
    """

    shared_kalman = KalmanFilterXYAH()  # 定义一个共享的 Kalman 滤波器实例

    def __init__(self, xywh, score, cls):
        """
        Initialize a new STrack instance.  # 初始化一个新的 STrack 实例。

        Args:
            xywh (List[float]): Bounding box coordinates and dimensions in the format (x, y, w, h, [a], idx), where
                (x, y) is the center, (w, h) are width and height, [a] is optional aspect ratio, and idx is the id.  # 边界框坐标和尺寸，格式为 (x, y, w, h, [a], idx)，其中 (x, y) 是中心，(w, h) 是宽度和高度，[a] 是可选的宽高比，idx 是 ID。
            score (float): Confidence score of the detection.  # 检测的置信度分数。
            cls (Any): Class label for the detected object.  # 检测对象的类别标签。

        Examples:
            >>> xywh = [100.0, 150.0, 50.0, 75.0, 1]  # 边界框坐标
            >>> score = 0.9  # 置信度分数
            >>> cls = "person"  # 类别标签
            >>> track = STrack(xywh, score, cls)  # 创建 STrack 实例
        """
        super().__init__()  # 调用父类的构造函数
        # xywh+idx or xywha+idx  # xywh+idx 或 xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"  # 确保 xywh 的长度为 5 或 6
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)  # 将 xywh 转换为 tlwh 格式并存储
        self.kalman_filter = None  # 初始化 Kalman 滤波器为 None
        self.mean, self.covariance = None, None  # 初始化均值和协方差为 None
        self.is_activated = False  # 激活状态初始化为 False

        self.score = score  # 设置置信度分数
        self.tracklet_len = 0  # 跟踪段长度初始化为 0
        self.cls = cls  # 设置类别标签
        self.idx = xywh[-1]  # 设置对象的索引
        self.angle = xywh[4] if len(xywh) == 6 else None  # 如果有角度，则存储角度

    def predict(self):
        """Predicts the next state (mean and covariance) of the object using the Kalman filter.  # 使用 Kalman 滤波器预测对象的下一个状态（均值和协方差）。"""
        mean_state = self.mean.copy()  # 复制当前均值状态
        if self.state != TrackState.Tracked:  # 如果状态不是被跟踪
            mean_state[7] = 0  # 将第 7 个元素设置为 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)  # 使用 Kalman 滤波器进行预测

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances.  # 使用 Kalman 滤波器对提供的 STrack 实例列表执行多对象预测跟踪。"""
        if len(stracks) <= 0:  # 如果没有跟踪实例
            return  # 返回
        multi_mean = np.asarray([st.mean.copy() for st in stracks])  # 复制所有跟踪的均值
        multi_covariance = np.asarray([st.covariance for st in stracks])  # 复制所有跟踪的协方差
        for i, st in enumerate(stracks):  # 遍历每个跟踪
            if st.state != TrackState.Tracked:  # 如果状态不是被跟踪
                multi_mean[i][7] = 0  # 将第 7 个元素设置为 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)  # 使用共享的 Kalman 滤波器进行预测
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):  # 遍历每个均值和协方差
            stracks[i].mean = mean  # 更新跟踪的均值
            stracks[i].covariance = cov  # 更新跟踪的协方差

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks.  # 使用单应性矩阵更新多个跟踪的状态位置和协方差。"""
        if len(stracks) > 0:  # 如果有跟踪实例
            multi_mean = np.asarray([st.mean.copy() for st in stracks])  # 复制所有跟踪的均值
            multi_covariance = np.asarray([st.covariance for st in stracks])  # 复制所有跟踪的协方差

            R = H[:2, :2]  # 提取单应性矩阵的前 2 行 2 列
            R8x8 = np.kron(np.eye(4, dtype=float), R)  # 创建 8x8 的扩展矩阵
            t = H[:2, 2]  # 提取单应性矩阵的平移部分

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):  # 遍历每个均值和协方差
                mean = R8x8.dot(mean)  # 使用扩展矩阵更新均值
                mean[:2] += t  # 将平移添加到均值
                cov = R8x8.dot(cov).dot(R8x8.transpose())  # 更新协方差

                stracks[i].mean = mean  # 更新跟踪的均值
                stracks[i].covariance = cov  # 更新跟踪的协方差

    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance.  # 使用提供的 Kalman 滤波器激活一个新的跟踪段，并初始化其状态和协方差。"""
        self.kalman_filter = kalman_filter  # 设置 Kalman 滤波器
        self.track_id = self.next_id()  # 获取下一个跟踪 ID
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))  # 初始化均值和协方差

        self.tracklet_len = 0  # 跟踪段长度初始化为 0
        self.state = TrackState.Tracked  # 设置状态为被跟踪
        if frame_id == 1:  # 如果是第一帧
            self.is_activated = True  # 激活状态设置为 True
        self.frame_id = frame_id  # 设置当前帧 ID
        self.start_frame = frame_id  # 设置开始帧 ID

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track using new detection data and updates its state and attributes.  # 使用新的检测数据重新激活之前丢失的跟踪，并更新其状态和属性。"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)  # 使用新的 tlwh 坐标更新均值和协方差
        )
        self.tracklet_len = 0  # 跟踪段长度初始化为 0
        self.state = TrackState.Tracked  # 设置状态为被跟踪
        self.is_activated = True  # 激活状态设置为 True
        self.frame_id = frame_id  # 设置当前帧 ID
        if new_id:  # 如果需要新的 ID
            self.track_id = self.next_id()  # 获取下一个跟踪 ID
        self.score = new_track.score  # 更新置信度分数
        self.cls = new_track.cls  # 更新类别标签
        self.angle = new_track.angle  # 更新角度
        self.idx = new_track.idx  # 更新索引

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.  # 更新匹配跟踪的状态。

        Args:
            new_track (STrack): The new track containing updated information.  # 包含更新信息的新跟踪。
            frame_id (int): The ID of the current frame.  # 当前帧的 ID。

        Examples:
            Update the state of a track with new detection information  # 使用新的检测信息更新跟踪状态
            >>> track = STrack([100, 200, 50, 80, 0.9, 1])  # 创建 STrack 实例
            >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])  # 创建新的 STrack 实例
            >>> track.update(new_track, 2)  # 更新跟踪
        """
        self.frame_id = frame_id  # 设置当前帧 ID
        self.tracklet_len += 1  # 跟踪段长度加 1

        new_tlwh = new_track.tlwh  # 获取新的 tlwh 坐标
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)  # 更新均值和协方差
        )
        self.state = TrackState.Tracked  # 设置状态为被跟踪
        self.is_activated = True  # 激活状态设置为 True

        self.score = new_track.score  # 更新置信度分数
        self.cls = new_track.cls  # 更新类别标签
        self.angle = new_track.angle  # 更新角度
        self.idx = new_track.idx  # 更新索引

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent.  # 将边界框的左上角-宽度-高度格式转换为 x-y-宽度-高度 格式。"""
        return self.tlwh_to_xyah(tlwh)  # 调用 tlwh_to_xyah 方法进行转换

    @property
    def tlwh(self):
        """Returns the bounding box in top-left-width-height format from the current state estimate.  # 从当前状态估计中返回边界框的左上角-宽度-高度格式。"""
        if self.mean is None:  # 如果均值为 None
            return self._tlwh.copy()  # 返回当前 tlwh 的副本
        ret = self.mean[:4].copy()  # 获取均值的前 4 个元素的副本
        ret[2] *= ret[3]  # 将宽度乘以高度
        ret[:2] -= ret[2:] / 2  # 将左上角坐标调整为中心坐标
        return ret  # 返回调整后的坐标

    @property
    def xyxy(self):
        """Converts bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format.  # 将边界框从 (左上角 x, 左上角 y, 宽度, 高度) 格式转换为 (最小 x, 最小 y, 最大 x, 最大 y) 格式。"""
        ret = self.tlwh.copy()  # 获取 tlwh 的副本
        ret[2:] += ret[:2]  # 将左上角坐标加到宽度和高度上
        return ret  # 返回转换后的坐标

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format.  # 将边界框从 tlwh 格式转换为中心-x-中心-y-宽度-高度 (xyah) 格式。"""
        ret = np.asarray(tlwh).copy()  # 复制 tlwh 数组
        ret[:2] += ret[2:] / 2  # 将左上角坐标调整为中心坐标
        ret[2] /= ret[3]  # 计算宽高比
        return ret  # 返回转换后的坐标

    @property
    def xywh(self):
        """Returns the current position of the bounding box in (center x, center y, width, height) format.  # 返回边界框的当前坐标，格式为 (中心 x, 中心 y, 宽度, 高度)。"""
        ret = np.asarray(self.tlwh).copy()  # 获取 tlwh 的副本
        ret[:2] += ret[2:] / 2  # 将左上角坐标加到宽度和高度上
        return ret  # 返回转换后的坐标

    @property
    def xywha(self):
        """Returns position in (center x, center y, width, height, angle) format, warning if angle is missing.  # 返回 (中心 x, 中心 y, 宽度, 高度, 角度) 格式的位置，如果缺少角度则发出警告。"""
        if self.angle is None:  # 如果角度为 None
            LOGGER.warning("WARNING ⚠️ `angle` attr not found, returning [xywh](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/trackers/bot_sort.py:138:4-143:46) instead.")  # 发出警告，返回 xywh
            return self.xywh  # 返回 xywh
        return np.concatenate([self.xywh, self.angle[None]])  # 将 xywh 和角度连接返回

    @property
    def result(self):
        """Returns the current tracking results in the appropriate bounding box format.  # 以适当的边界框格式返回当前跟踪结果。"""
        coords = self.xyxy if self.angle is None else self.xywha  # 根据是否有角度选择坐标格式
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]  # 返回结果列表

    def __repr__(self):
        """Returns a string representation of the STrack object including start frame, end frame, and track ID.  # 返回 STrack 对象的字符串表示，包括开始帧、结束帧和跟踪 ID。"""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"  # 格式化返回字符串


class BYTETracker:
    """
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.  # BYTETracker：基于 YOLOv8 的对象检测和跟踪算法。

    Responsible for initializing, updating, and managing the tracks for detected objects in a video sequence.  # 负责初始化、更新和管理视频序列中检测到的对象的跟踪。
    It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for predicting  # 它维护跟踪、丢失和移除的状态，并利用 Kalman 滤波器预测
    the new object locations, and performs data association.  # 新对象的位置，并执行数据关联。

    Attributes:
        tracked_stracks (List[STrack]): List of successfully activated tracks.  # 成功激活的跟踪列表。
        lost_stracks (List[STrack]): List of lost tracks.  # 丢失的跟踪列表。
        removed_stracks (List[STrack]): List of removed tracks.  # 移除的跟踪列表。
        frame_id (int): The current frame ID.  # 当前帧 ID。
        args (Namespace): Command-line arguments.  # 命令行参数。
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.  # 跟踪被视为“丢失”的最大帧数。
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.  # Kalman 滤波器对象。

    Methods:
        update(results, img=None): Updates object tracker with new detections.  # 使用新的检测更新对象跟踪器。
        get_kalmanfilter(): Returns a Kalman filter object for tracking bounding boxes.  # 返回用于跟踪边界框的 Kalman 滤波器对象。
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.  # 使用检测初始化对象跟踪。
        get_dists(tracks, detections): Calculates the distance between tracks and detections.  # 计算跟踪和检测之间的距离。
        multi_predict(tracks): Predicts the location of tracks.  # 预测跟踪的位置。
        reset_id(): Resets the ID counter of STrack.  # 重置 STrack 的 ID 计数器。
        joint_stracks(tlista, tlistb): Combines two lists of stracks.  # 合并两个 stracks 列表。
        sub_stracks(tlista, tlistb): Filters out the stracks present in the second list from the first list.  # 从第一个列表中过滤出第二个列表中存在的 stracks。
        remove_duplicate_stracks(stracksa, stracksb): Removes duplicate stracks based on IoU.  # 根据 IoU 移除重复的 stracks。

    Examples:
        Initialize BYTETracker and update with detection results  # 初始化 BYTETracker 并使用检测结果更新
        >>> tracker = BYTETracker(args, frame_rate=30)  # 创建 BYTETracker 实例
        >>> results = yolo_model.detect(image)  # 使用 YOLO 模型进行检测
        >>> tracked_objects = tracker.update(results)  # 更新跟踪对象
    """

    def __init__(self, args, frame_rate=30):
        """
        Initialize a BYTETracker instance for object tracking.  # 初始化 BYTETracker 实例以进行对象跟踪。

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.  # 包含跟踪参数的命令行参数。
            frame_rate (int): Frame rate of the video sequence.  # 视频序列的帧率。

        Examples:
            Initialize BYTETracker with command-line arguments and a frame rate of 30  # 使用命令行参数和 30 帧率初始化 BYTETracker
            >>> args = Namespace(track_buffer=30)  # 创建命令行参数实例
            >>> tracker = BYTETracker(args, frame_rate=30)  # 创建 BYTETracker 实例
        """
        self.tracked_stracks = []  # type: list[STrack]  # 成功激活的跟踪列表
        self.lost_stracks = []  # type: list[STrack]  # 丢失的跟踪列表
        self.removed_stracks = []  # type: list[STrack]  # 移除的跟踪列表

        self.frame_id = 0  # 当前帧 ID 初始化为 0
        self.args = args  # 设置命令行参数
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)  # 计算最大丢失时间
        self.kalman_filter = self.get_kalmanfilter()  # 获取 Kalman 滤波器
        self.reset_id()  # 重置 ID

    def update(self, results, img=None):
        """Updates the tracker with new detections and returns the current list of tracked objects.  # 使用新的检测更新跟踪器，并返回当前跟踪对象的列表。"""
        self.frame_id += 1  # 当前帧 ID 加 1
        activated_stracks = []  # 激活的跟踪列表
        refind_stracks = []  # 重新找到的跟踪列表
        lost_stracks = []  # 丢失的跟踪列表
        removed_stracks = []  # 移除的跟踪列表

        scores = results.conf  # 获取检测的置信度分数
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh  # 获取边界框
        # Add index  # 添加索引
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)  # 将索引添加到边界框
        cls = results.cls  # 获取类别标签

        remain_inds = scores >= self.args.track_high_thresh  # 获取高于阈值的索引
        inds_low = scores > self.args.track_low_thresh  # 获取低于阈值的索引
        inds_high = scores < self.args.track_high_thresh  # 获取高于阈值的索引

        inds_second = inds_low & inds_high  # 获取第二次检测的索引
        dets_second = bboxes[inds_second]  # 获取第二次检测的边界框
        dets = bboxes[remain_inds]  # 获取高置信度的边界框
        scores_keep = scores[remain_inds]  # 获取高置信度的分数
        scores_second = scores[inds_second]  # 获取第二次检测的分数
        cls_keep = cls[remain_inds]  # 获取高置信度的类别标签
        cls_second = cls[inds_second]  # 获取第二次检测的类别标签

        detections = self.init_track(dets, scores_keep, cls_keep, img)  # 初始化跟踪
        # Add newly detected tracklets to tracked_stracks  # 将新检测的跟踪段添加到 tracked_stracks
        unconfirmed = []  # 未确认的跟踪列表
        tracked_stracks = []  # type: list[STrack]  # 成功激活的跟踪列表
        for track in self.tracked_stracks:  # 遍历已跟踪的跟踪
            if not track.is_activated:  # 如果未激活
                unconfirmed.append(track)  # 添加到未确认列表
            else:
                tracked_stracks.append(track)  # 添加到成功激活列表
        # Step 2: First association, with high score detection boxes  # 第 2 步：第一次关联，使用高分检测框
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)  # 合并成功激活和丢失的跟踪
        # Predict the current location with KF  # 使用 Kalman 滤波器预测当前位置信息
        self.multi_predict(strack_pool)  # 对合并的跟踪进行多预测
        if hasattr(self, "gmc") and img is not None:  # 如果有 GMC 且图像不为 None
            warp = self.gmc.apply(img, dets)  # 应用 GMC 进行图像变换
            STrack.multi_gmc(strack_pool, warp)  # 更新合并的跟踪状态
            STrack.multi_gmc(unconfirmed, warp)  # 更新未确认的跟踪状态

        dists = self.get_dists(strack_pool, detections)  # 计算合并跟踪和检测之间的距离
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)  # 进行线性分配，获取匹配结果

        for itracked, idet in matches:  # 遍历每个匹配
            track = strack_pool[itracked]  # 获取跟踪对象
            det = detections[idet]  # 获取检测对象
            if track.state == TrackState.Tracked:  # 如果状态是被跟踪
                track.update(det, self.frame_id)  # 更新跟踪状态
                activated_stracks.append(track)  # 添加到激活列表
            else:
                track.re_activate(det, self.frame_id, new_id=False)  # 重新激活跟踪
                refind_stracks.append(track)  # 添加到重新找到的列表
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections  # 第 3 步：第二次关联，使用低分检测框将未跟踪的对象关联到低分检测
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)  # 初始化第二次检测的跟踪
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]  # 获取被跟踪的跟踪对象
        # TODO  # 待办事项
        dists = matching.iou_distance(r_tracked_stracks, detections_second)  # 计算 IoU 距离
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)  # 进行线性分配，获取匹配结果
        for itracked, idet in matches:  # 遍历每个匹配
            track = r_tracked_stracks[itracked]  # 获取跟踪对象
            det = detections_second[idet]  # 获取检测对象
            if track.state == TrackState.Tracked:  # 如果状态是被跟踪
                track.update(det, self.frame_id)  # 更新跟踪状态
                activated_stracks.append(track)  # 添加到激活列表
            else:
                track.re_activate(det, self.frame_id, new_id=False)  # 重新激活跟踪
                refind_stracks.append(track)  # 添加到重新找到的列表

        for it in u_track:  # 遍历未确认的跟踪
            track = r_tracked_stracks[it]  # 获取跟踪对象
            if track.state != TrackState.Lost:  # 如果状态不是丢失
                track.mark_lost()  # 标记为丢失
                lost_stracks.append(track)  # 添加到丢失列表
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame  # 处理未确认的跟踪，通常是仅有一个开始帧的跟踪
        detections = [detections[i] for i in u_detection]  # 获取未确认的检测
        dists = self.get_dists(unconfirmed, detections)  # 计算未确认的跟踪和检测之间的距离
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)  # 进行线性分配，获取匹配结果
        for itracked, idet in matches:  # 遍历每个匹配
            unconfirmed[itracked].update(detections[idet], self.frame_id)  # 更新未确认的跟踪
            activated_stracks.append(unconfirmed[itracked])  # 添加到激活列表
        for it in u_unconfirmed:  # 遍历未确认的未匹配
            track = unconfirmed[it]  # 获取未确认的跟踪对象
            track.mark_removed()  # 标记为移除
            removed_stracks.append(track)  # 添加到移除列表
        # Step 4: Init new stracks  # 第 4 步：初始化新的跟踪
        for inew in u_detection:  # 遍历未确认的检测
            track = detections[inew]  # 获取检测对象
            if track.score < self.args.new_track_thresh:  # 如果分数低于阈值
                continue  # 跳过
            track.activate(self.kalman_filter, self.frame_id)  # 激活新的跟踪
            activated_stracks.append(track)  # 添加到激活列表
        # Step 5: Update state  # 第 5 步：更新状态
        for track in self.lost_stracks:  # 遍历丢失的跟踪
            if self.frame_id - track.end_frame > self.max_time_lost:  # 如果丢失时间超过最大丢失时间
                track.mark_removed()  # 标记为移除
                removed_stracks.append(track)  # 添加到移除列表

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]  # 过滤出被跟踪的对象
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)  # 合并激活的跟踪
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)  # 合并重新找到的跟踪
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)  # 从丢失的跟踪中去除已被跟踪的对象
        self.lost_stracks.extend(lost_stracks)  # 添加丢失的跟踪
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)  # 从丢失的跟踪中去除已移除的对象
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)  # 移除重复的跟踪
        self.removed_stracks.extend(removed_stracks)  # 添加移除的跟踪
        if len(self.removed_stracks) > 1000:  # 如果移除的跟踪数量超过 1000
            self.removed_stracks = self.removed_stracks[-999:]  # 将移除的跟踪限制为最多 999 个

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)  # 返回激活的跟踪结果

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH.  # 返回用于跟踪边界框的 Kalman 滤波器对象。"""
        return KalmanFilterXYAH()  # 返回 Kalman 滤波器对象

    def init_track(self, dets, scores, cls, img=None):
        """Initializes object tracking with given detections, scores, and class labels using the STrack algorithm.  # 使用 STrack 算法初始化对象跟踪，给定检测、分数和类别标签。"""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # 如果有检测，返回 STrack 实例列表

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IoU and optionally fuses scores.  # 使用 IoU 计算跟踪和检测之间的距离，并可选择融合分数。"""
        dists = matching.iou_distance(tracks, detections)  # 计算 IoU 距离
        if self.args.fuse_score:  # 如果启用分数融合
            dists = matching.fuse_score(dists, detections)  # 融合分数
        return dists  # 返回距离

    def multi_predict(self, tracks):
        """Predict the next states for multiple tracks using Kalman filter.  # 使用 Kalman 滤波器预测多个跟踪的下一个状态。"""
        STrack.multi_predict(tracks)  # 调用 STrack 的多预测方法

    @staticmethod
    def reset_id():
        """Resets the ID counter for STrack instances to ensure unique track IDs across tracking sessions.  # 重置 STrack 实例的 ID 计数器，以确保跟踪会话中唯一的跟踪 ID。"""
        STrack.reset_id()  # 调用 STrack 的重置 ID 方法

    def reset(self):
        """Resets the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter.  # 通过清除所有跟踪、丢失和移除的跟踪并重新初始化 Kalman 滤波器来重置跟踪器。"""
        self.tracked_stracks = []  # type: list[STrack]  # 成功激活的跟踪列表
        self.lost_stracks = []  # type: list[STrack]  # 丢失的跟踪列表
        self.removed_stracks = []  # type: list[STrack]  # 移除的跟踪列表
        self.frame_id = 0  # 当前帧 ID 初始化为 0
        self.kalman_filter = self.get_kalmanfilter()  # 获取 Kalman 滤波器
        self.reset_id()  # 重置 ID

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combines two lists of STrack objects into a single list, ensuring no duplicates based on track IDs.  # 将两个 STrack 对象列表合并为一个列表，确保基于跟踪 ID 没有重复。"""
        exists = {}  # 存储已存在的跟踪 ID
        res = []  # 合并后的结果列表
        for t in tlista:  # 遍历第一个列表
            exists[t.track_id] = 1  # 将跟踪 ID 标记为存在
            res.append(t)  # 添加到结果列表
        for t in tlistb:  # 遍历第二个列表
            tid = t.track_id  # 获取当前跟踪 ID
            if not exists.get(tid, 0):  # 如果该 ID 不在已存在的列表中
                exists[tid] = 1  # 将 ID 标记为存在
                res.append(t)  # 添加到结果列表
        return res  # 返回合并后的结果列表

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Filters out the stracks present in the second list from the first list.  # 从第一个列表中过滤出第二个列表中存在的 stracks。"""
        track_ids_b = {t.track_id for t in tlistb}  # 获取第二个列表中的所有跟踪 ID
        return [t for t in tlista if t.track_id not in track_ids_b]  # 返回第一个列表中不在第二个列表中的跟踪

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Removes duplicate stracks from two lists based on Intersection over Union (IoU) distance.  # 根据交并比 (IoU) 距离从两个列表中移除重复的 stracks。"""
        pdist = matching.iou_distance(stracksa, stracksb)  # 计算两个列表之间的 IoU 距离
        pairs = np.where(pdist < 0.15)  # 获取 IoU 距离小于 0.15 的索引对
        dupa, dupb = [], []  # 存储重复的索引
        for p, q in zip(*pairs):  # 遍历所有重复的索引对
            timep = stracksa[p].frame_id - stracksa[p].start_frame  # 计算第一个跟踪的时间
            timeq = stracksb[q].frame_id - stracksb[q].start_frame  # 计算第二个跟踪的时间
            if timep > timeq:  # 如果第一个跟踪的时间较长
                dupb.append(q)  # 添加到第二个跟踪的重复列表
            else:
                dupa.append(p)  # 添加到第一个跟踪的重复列表
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]  # 过滤掉第一个列表中的重复跟踪
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]  # 过滤掉第二个列表中的重复跟踪
        return resa, resb  # 返回过滤后的两个列表