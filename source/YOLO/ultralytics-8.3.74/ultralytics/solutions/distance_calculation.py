# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math

import cv2

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class DistanceCalculation(BaseSolution):
    """
    A class to calculate distance between two objects in a real-time video stream based on their tracks.
    
    这个类用于根据对象在实时视频流中的轨迹计算两个对象之间的距离。
    
    This class extends BaseSolution to provide functionality for selecting objects and calculating the distance
    between them in a video stream using YOLO object detection and tracking.
    
    此类扩展了 BaseSolution，以提供在视频流中选择对象并计算它们之间距离的功能，使用 YOLO 目标检测和跟踪。
    
    Attributes:
        left_mouse_count (int): Counter for left mouse button clicks.
        left_mouse_count (int): 左键单击计数器。
        selected_boxes (Dict[int, List[float]]): Dictionary to store selected bounding boxes and their track IDs.
        selected_boxes (Dict[int, List[float]]): 字典，用于存储选定的边界框及其轨迹 ID。
        annotator (Annotator): An instance of the Annotator class for drawing on the image.
        annotator (Annotator): Annotator 类的实例，用于在图像上绘制。
        boxes (List[List[float]]): List of bounding boxes for detected objects.
        boxes (List[List[float]]): 检测到的对象的边界框列表。
        track_ids (List[int]): List of track IDs for detected objects.
        track_ids (List[int]): 检测到的对象的轨迹 ID 列表。
        clss (List[int]): List of class indices for detected objects.
        clss (List[int]): 检测到的对象的类别索引列表。
        names (List[str]): List of class names that the model can detect.
        names (List[str]): 模型可以检测的类别名称列表。
        centroids (List[List[int]]): List to store centroids of selected bounding boxes.
        centroids (List[List[int]]): 存储选定边界框的质心的列表。
    
    Methods:
        mouse_event_for_distance: Handles mouse events for selecting objects in the video stream.
        mouse_event_for_distance: 处理鼠标事件以选择视频流中的对象。
        calculate: Processes video frames and calculates the distance between selected objects.
        calculate: 处理视频帧并计算所选对象之间的距离。
    
    Examples:
        >>> distance_calc = DistanceCalculation()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = distance_calc.calculate(frame)
        >>> cv2.imshow("Distance Calculation", processed_frame)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs):
        """Initializes the DistanceCalculation class for measuring object distances in video streams."""
        """初始化 DistanceCalculation 类以测量视频流中对象的距离。"""
        super().__init__(**kwargs)

        # Mouse event information
        # 鼠标事件信息
        self.left_mouse_count = 0
        self.selected_boxes = {}

        self.centroids = []  # Initialize empty list to store centroids
        self.centroids = []  # 初始化空列表以存储质心

    def mouse_event_for_distance(self, event, x, y, flags, param):
        """
        Handles mouse events to select regions in a real-time video stream for distance calculation.
        
        处理鼠标事件以选择实时视频流中的区域以进行距离计算。
        
        Args:
            event (int): Type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN).
            event (int): 鼠标事件的类型（例如，cv2.EVENT_MOUSEMOVE，cv2.EVENT_LBUTTONDOWN）。
            x (int): X-coordinate of the mouse pointer.
            x (int): 鼠标指针的 X 坐标。
            y (int): Y-coordinate of the mouse pointer.
            y (int): 鼠标指针的 Y 坐标。
            flags (int): Flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY, cv2.EVENT_FLAG_SHIFTKEY).
            flags (int): 与事件相关的标志（例如，cv2.EVENT_FLAG_CTRLKEY，cv2.EVENT_FLAG_SHIFTKEY）。
            param (Dict): Additional parameters passed to the function.
            param (Dict): 传递给函数的附加参数。
        
        Examples:
            >>> # Assuming 'dc' is an instance of DistanceCalculation
            >>> cv2.setMouseCallback("window_name", dc.mouse_event_for_distance)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.track_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = box
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def calculate(self, im0):
        """
        Processes a video frame and calculates the distance between two selected bounding boxes.
        
        处理视频帧并计算两个选定边界框之间的距离。
        
        This method extracts tracks from the input frame, annotates bounding boxes, and calculates the distance
        between two user-selected objects if they have been chosen.
        
        此方法从输入帧中提取轨迹，注释边界框，并计算两个用户选择的对象之间的距离（如果已选择）。
        
        Args:
            im0 (numpy.ndarray): The input image frame to process.
            im0 (numpy.ndarray): 要处理的输入图像帧。
        
        Returns:
            (numpy.ndarray): The processed image frame with annotations and distance calculations.
            (numpy.ndarray): 带有注释和距离计算的处理图像帧。
        
        Examples:
            >>> import numpy as np
            >>> from ultralytics.solutions import DistanceCalculation
            >>> dc = DistanceCalculation()
            >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> processed_frame = dc.calculate(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes, track ids and classes index
        # 遍历边界框、轨迹 ID 和类别索引
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes.keys():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            # Store user selected boxes in centroids list
            # 将用户选择的框存储在质心列表中
            self.centroids.extend(
                [[int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)] for box in self.selected_boxes.values()]
            )
            # Calculate pixels distance
            # 计算像素距离
            pixels_distance = math.sqrt(
                (self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2
            )
            self.annotator.plot_distance_and_line(pixels_distance, self.centroids)

        self.centroids = []

        self.display_output(im0)  # display output with base class function
        cv2.setMouseCallback("Ultralytics Solutions", self.mouse_event_for_distance)

        return im0  # return output image for more usage
