# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from time import time  # 从time模块导入time函数，用于获取当前时间

import numpy as np  # 导入NumPy库，用于数组和数值计算

from ultralytics.solutions.solutions import BaseSolution  # 从ultralytics.solutions模块导入BaseSolution类
from ultralytics.utils.plotting import Annotator, colors  # 从ultralytics.utils.plotting导入Annotator和colors

class SpeedEstimator(BaseSolution):
    """
    A class to estimate the speed of objects in a real-time video stream based on their tracks.
    一个类，用于根据对象的轨迹估计实时视频流中对象的速度。

    This class extends the BaseSolution class and provides functionality for estimating object speeds using
    tracking data in video streams.
    该类扩展了BaseSolution类，提供使用视频流中的跟踪数据估计对象速度的功能。

    Attributes:
        spd (Dict[int, float]): Dictionary storing speed data for tracked objects.
        spd (Dict[int, float]): 存储被跟踪对象速度数据的字典。
        trkd_ids (List[int]): List of tracked object IDs that have already been speed-estimated.
        trkd_ids (List[int]): 存储已进行速度估计的被跟踪对象ID的列表。
        trk_pt (Dict[int, float]): Dictionary storing previous timestamps for tracked objects.
        trk_pt (Dict[int, float]): 存储被跟踪对象的先前时间戳的字典。
        trk_pp (Dict[int, Tuple[float, float]]): Dictionary storing previous positions for tracked objects.
        trk_pp (Dict[int, Tuple[float, float]]): 存储被跟踪对象的先前位置的字典。
        annotator (Annotator): Annotator object for drawing on images.
        annotator (Annotator): 用于在图像上绘制的Annotator对象。
        region (List[Tuple[int, int]]): List of points defining the speed estimation region.
        region (List[Tuple[int, int]]): 定义速度估计区域的点的列表。
        track_line (List[Tuple[float, float]]): List of points representing the object's track.
        track_line (List[Tuple[float, float]]): 表示对象轨迹的点的列表。
        r_s (LineString): LineString object representing the speed estimation region.
        r_s (LineString): 表示速度估计区域的LineString对象。

    Methods:
        initialize_region: Initializes the speed estimation region.
        initialize_region: 初始化速度估计区域。
        estimate_speed: Estimates the speed of objects based on tracking data.
        estimate_speed: 基于跟踪数据估计对象的速度。
        store_tracking_history: Stores the tracking history for an object.
        store_tracking_history: 存储对象的跟踪历史。
        extract_tracks: Extracts tracks from the current frame.
        extract_tracks: 从当前帧中提取轨迹。
        display_output: Displays the output with annotations.
        display_output: 显示带注释的输出。

    Examples:
        >>> estimator = SpeedEstimator()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = estimator.estimate_speed(frame)
        >>> cv2.imshow("Speed Estimation", processed_frame)
    """

    def __init__(self, **kwargs):
        """Initializes the SpeedEstimator object with speed estimation parameters and data structures."""
        super().__init__(**kwargs)  # 调用父类的初始化方法

        self.initialize_region()  # 初始化速度区域

        self.spd = {}  # 设置速度数据字典
        self.trkd_ids = []  # 存储已进行速度估计的ID的列表
        self.trk_pt = {}  # 存储跟踪对象的先前时间
        self.trk_pp = {}  # 存储跟踪对象的先前位置

    def estimate_speed(self, im0):
        """
        Estimates the speed of objects based on tracking data.

        Args:
            im0 (np.ndarray): Input image for processing. Shape is typically (H, W, C) for RGB images.
            im0 (np.ndarray): 输入图像，用于处理。形状通常为(H, W, C)表示RGB图像。

        Returns:
            (np.ndarray): Processed image with speed estimations and annotations.
            (np.ndarray): 处理后的图像，包含速度估计和注释。

        Examples:
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> processed_image = estimator.estimate_speed(image)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化注释器
        self.extract_tracks(im0)  # 提取轨迹

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # 绘制速度估计区域

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):  # 遍历边界框、轨迹ID和类别
            self.store_tracking_history(track_id, box)  # 存储轨迹历史

            # Check if track_id is already in self.trk_pp or trk_pt initialize if not
            if track_id not in self.trk_pt:  # 检查轨迹ID是否已在时间戳字典中
                self.trk_pt[track_id] = 0  # 初始化时间戳
            if track_id not in self.trk_pp:  # 检查轨迹ID是否已在位置字典中
                self.trk_pp[track_id] = self.track_line[-1]  # 初始化位置为当前轨迹的最后一个点

            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]  # 设置速度标签
            self.annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # 绘制边界框和速度标签

            # Draw tracks of objects 绘制对象的轨迹
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            # Calculate object speed and direction based on region intersection
            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):  # 检查轨迹线是否与速度区域相交
                direction = "known"  # 方向为已知
            else:
                direction = "unknown"  # 方向为未知

            # Perform speed calculation and tracking updates if direction is valid
            if direction == "known" and track_id not in self.trkd_ids:  # 如果方向已知且未记录速度
                self.trkd_ids.append(track_id)  # 添加轨迹ID到已记录列表
                time_difference = time() - self.trk_pt[track_id]  # 计算时间差
                if time_difference > 0:  # 如果时间差大于0
                    self.spd[track_id] = np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1]) / time_difference  # 计算速度

            self.trk_pt[track_id] = time()  # 更新当前时间戳
            self.trk_pp[track_id] = self.track_line[-1]  # 更新当前点

        self.display_output(im0)  # 使用基类函数显示输出

        return im0  # 返回处理后的图像以供进一步使用