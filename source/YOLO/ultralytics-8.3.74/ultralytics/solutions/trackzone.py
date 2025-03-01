# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class TrackZone(BaseSolution):
    """
    A class to manage region-based object tracking in a video stream.

    This class extends the BaseSolution class and provides functionality for tracking objects within a specific region
    defined by a polygonal area. Objects outside the region are excluded from tracking. It supports dynamic initialization
    of the region, allowing either a default region or a user-specified polygon.

    Attributes:
        region (ndarray): The polygonal region for tracking, represented as a convex hull.

    Methods:
        trackzone: Processes each frame of the video, applying region-based tracking.

    Examples:
        >>> tracker = TrackZone()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = tracker.trackzone(frame)
        >>> cv2.imshow("Tracked Frame", processed_frame)
    """

    def __init__(self, **kwargs):
        """Initializes the TrackZone class for tracking objects within a defined region in video streams."""
        super().__init__(**kwargs)
        default_region = [(150, 150), (1130, 150), (1130, 570), (150, 570)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))

    def trackzone(self, im0):
        """
        Processes the input frame to track objects within a defined region.

        This method initializes the annotator, creates a mask for the specified region, extracts tracks
        only from the masked area, and updates tracking information. Objects outside the region are ignored.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed.

        Returns:
            (numpy.ndarray): The processed image with tracking id and bounding boxes annotations.

        Examples:
            >>> tracker = TrackZone()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> tracker.trackzone(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        # Create a mask for the region and extract tracks from the masked image
        masked_frame = cv2.bitwise_and(im0, im0, mask=cv2.fillPoly(np.zeros_like(im0[:, :, 0]), [self.region], 255))
        self.extract_tracks(masked_frame)

        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)

        # Iterate over boxes, track ids, classes indexes list and draw bounding boxes
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=f"{self.names[cls]}:{track_id}", color=colors(track_id, True))

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2  # 导入OpenCV库以处理图像
import numpy as np  # 导入NumPy库，用于数组和数值计算

from ultralytics.solutions.solutions import BaseSolution  # 从ultralytics.solutions模块导入BaseSolution类
from ultralytics.utils.plotting import Annotator, colors  # 从ultralytics.utils.plotting导入Annotator和colors

class TrackZone(BaseSolution):
    """
    A class to manage region-based object tracking in a video stream.
    一个类，用于管理视频流中的基于区域的对象跟踪。

    This class extends the BaseSolution class and provides functionality for tracking objects within a specific region
    defined by a polygonal area. Objects outside the region are excluded from tracking. It supports dynamic initialization
    of the region, allowing either a default region or a user-specified polygon.
    该类扩展了BaseSolution类，提供在多边形区域内跟踪对象的功能。区域外的对象将被排除在跟踪之外。它支持区域的动态初始化，允许使用默认区域或用户指定的多边形。

    Attributes:
        region (ndarray): The polygonal region for tracking, represented as a convex hull.
        region (ndarray): 用于跟踪的多边形区域，表示为凸包。

    Methods:
        trackzone: Processes each frame of the video, applying region-based tracking.
        trackzone: 处理视频的每一帧，应用基于区域的跟踪。

    Examples:
        >>> tracker = TrackZone()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = tracker.trackzone(frame)
        >>> cv2.imshow("Tracked Frame", processed_frame)
    """

    def __init__(self, **kwargs):
        """Initializes the TrackZone class for tracking objects within a defined region in video streams."""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        default_region = [(150, 150), (1130, 150), (1130, 570), (150, 570)]  # 定义默认区域
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))  # 将区域转换为凸包

    def trackzone(self, im0):
        """
        Processes the input frame to track objects within a defined region.

        This method initializes the annotator, creates a mask for the specified region, extracts tracks
        only from the masked area, and updates tracking information. Objects outside the region are ignored.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed. 输入图像或帧，用于处理。

        Returns:
            (numpy.ndarray): The processed image with tracking id and bounding boxes annotations.
            (numpy.ndarray): 处理后的图像，包含跟踪ID和边界框注释。

        Examples:
            >>> tracker = TrackZone()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> tracker.trackzone(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化注释器
        # Create a mask for the region and extract tracks from the masked image
        masked_frame = cv2.bitwise_and(im0, im0, mask=cv2.fillPoly(np.zeros_like(im0[:, :, 0]), [self.region], 255))  # 创建区域掩码并提取轨迹
        self.extract_tracks(masked_frame)  # 从掩码图像中提取轨迹

        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)  # 绘制区域边界

        # Iterate over boxes, track ids, classes indexes list and draw bounding boxes
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):  # 遍历边界框、轨迹ID和类别
            self.annotator.box_label(box, label=f"{self.names[cls]}:{track_id}", color=colors(track_id, True))  # 绘制边界框和标签

        self.display_output(im0)  # 使用基类函数显示输出

        return im0  # 返回处理后的图像以供进一步使用