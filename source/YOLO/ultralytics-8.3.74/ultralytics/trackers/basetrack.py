# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 的 AGPL-3.0 许可证
"""Module defines the base classes and structures for object tracking in YOLO."""  # 模块定义了 YOLO 中对象跟踪的基本类和结构

from collections import OrderedDict  # 从 collections 导入 OrderedDict 类

import numpy as np  # 导入 NumPy 库


class TrackState:  # 定义 TrackState 类
    """
    Enumeration class representing the possible states of an object being tracked.

    Attributes:
        New (int): State when the object is newly detected.  # 状态为新检测到的对象
        Tracked (int): State when the object is successfully tracked in subsequent frames.  # 状态为在后续帧中成功跟踪的对象
        Lost (int): State when the object is no longer tracked.  # 状态为对象不再被跟踪
        Removed (int): State when the object is removed from tracking.  # 状态为对象从跟踪中移除

    Examples:
        >>> state = TrackState.New
        >>> if state == TrackState.New:
        >>>     print("Object is newly detected.")  # 对象是新检测到的
    """

    New = 0  # 新状态
    Tracked = 1  # 被跟踪状态
    Lost = 2  # 丢失状态
    Removed = 3  # 被移除状态


class BaseTrack:  # 定义 BaseTrack 类
    """
    Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count (int): Class-level counter for unique track IDs.  # 类级别的唯一跟踪 ID 计数器
        track_id (int): Unique identifier for the track.  # 跟踪的唯一标识符
        is_activated (bool): Flag indicating whether the track is currently active.  # 标志，指示跟踪是否当前处于活动状态
        state (TrackState): Current state of the track.  # 跟踪的当前状态
        history (OrderedDict): Ordered history of the track's states.  # 跟踪状态的有序历史记录
        features (List): List of features extracted from the object for tracking.  # 从对象中提取的用于跟踪的特征列表
        curr_feature (Any): The current feature of the object being tracked.  # 当前被跟踪对象的特征
        score (float): The confidence score of the tracking.  # 跟踪的置信度分数
        start_frame (int): The frame number where tracking started.  # 跟踪开始的帧编号
        frame_id (int): The most recent frame ID processed by the track.  # 跟踪处理的最新帧 ID
        time_since_update (int): Frames passed since the last update.  # 自上次更新以来经过的帧数
        location (tuple): The location of the object in the context of multi-camera tracking.  # 在多摄像头跟踪中的对象位置

    Methods:
        end_frame: Returns the ID of the last frame where the object was tracked.  # 返回对象被跟踪的最后一帧的 ID
        next_id: Increments and returns the next global track ID.  # 增加并返回下一个全局跟踪 ID
        activate: Abstract method to activate the track.  # 抽象方法，用于激活跟踪
        predict: Abstract method to predict the next state of the track.  # 抽象方法，用于预测跟踪的下一个状态
        update: Abstract method to update the track with new data.  # 抽象方法，用于使用新数据更新跟踪
        mark_lost: Marks the track as lost.  # 将跟踪标记为丢失
        mark_removed: Marks the track as removed.  # 将跟踪标记为已移除
        reset_id: Resets the global track ID counter.  # 重置全局跟踪 ID 计数器

    Examples:
        Initialize a new track and mark it as lost:
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # Output: 2 (TrackState.Lost)  # 输出: 2 (TrackState.Lost)
    """

    _count = 0  # 类级别的唯一跟踪 ID 计数器

    def __init__(self):  # 初始化方法
        """
        Initializes a new track with a unique ID and foundational tracking attributes.

        Examples:
            Initialize a new track
            >>> track = BaseTrack()
            >>> print(track.track_id)  # Output: 0  # 输出: 0
        """
        self.track_id = 0  # 跟踪 ID 初始化为 0
        self.is_activated = False  # 跟踪状态初始化为未激活
        self.state = TrackState.New  # 跟踪状态初始化为新状态
        self.history = OrderedDict()  # 初始化状态历史为有序字典
        self.features = []  # 初始化特征列表为空
        self.curr_feature = None  # 当前特征初始化为 None
        self.score = 0  # 置信度分数初始化为 0
        self.start_frame = 0  # 跟踪开始帧初始化为 0
        self.frame_id = 0  # 最新帧 ID 初始化为 0
        self.time_since_update = 0  # 自上次更新以来的帧数初始化为 0
        self.location = (np.inf, np.inf)  # 位置初始化为无穷大

    @property
    def end_frame(self):  # 结束帧属性
        """Returns the ID of the most recent frame where the object was tracked."""  # 返回对象被跟踪的最新帧的 ID
        return self.frame_id  # 返回当前帧 ID

    @staticmethod
    def next_id():  # 静态方法，获取下一个唯一跟踪 ID
        """Increment and return the next unique global track ID for object tracking."""  # 增加并返回下一个唯一全局跟踪 ID
        BaseTrack._count += 1  # 增加计数器
        return BaseTrack._count  # 返回当前计数器值

    def activate(self, *args):  # 激活跟踪方法
        """Activates the track with provided arguments, initializing necessary attributes for tracking."""  # 使用提供的参数激活跟踪，初始化必要的跟踪属性
        raise NotImplementedError  # 抛出未实现错误

    def predict(self):  # 预测方法
        """Predicts the next state of the track based on the current state and tracking model."""  # 根据当前状态和跟踪模型预测跟踪的下一个状态
        raise NotImplementedError  # 抛出未实现错误

    def update(self, *args, **kwargs):  # 更新方法
        """Updates the track with new observations and data, modifying its state and attributes accordingly."""  # 使用新观察和数据更新跟踪，修改其状态和属性
        raise NotImplementedError  # 抛出未实现错误

    def mark_lost(self):  # 标记为丢失的方法
        """Marks the track as lost by updating its state to TrackState.Lost."""  # 通过将状态更新为 TrackState.Lost 将跟踪标记为丢失
        self.state = TrackState.Lost  # 更新状态为丢失

    def mark_removed(self):  # 标记为移除的方法
        """Marks the track as removed by setting its state to TrackState.Removed."""  # 通过将状态设置为 TrackState.Removed 将跟踪标记为移除
        self.state = TrackState.Removed  # 更新状态为已移除

    @staticmethod
    def reset_id():  # 重置 ID 的静态方法
        """Reset the global track ID counter to its initial value."""  # 将全局跟踪 ID 计数器重置为初始值
        BaseTrack._count = 0  # 将计数器重置为 0