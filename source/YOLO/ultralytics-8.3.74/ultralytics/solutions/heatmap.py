# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np

from ultralytics.solutions.object_counter import ObjectCounter
from ultralytics.utils.plotting import Annotator


class Heatmap(ObjectCounter):
    """
    A class to draw heatmaps in real-time video streams based on object tracks.
    
    这个类用于根据对象轨迹在实时视频流中绘制热图。
    
    This class extends the ObjectCounter class to generate and visualize heatmaps of object movements in video
    streams. It uses tracked object positions to create a cumulative heatmap effect over time.
    
    此类扩展了 ObjectCounter 类，以生成和可视化视频流中对象运动的热图。它使用跟踪的对象位置来创建随时间变化的累积热图效果。
    
    Attributes:
        initialized (bool): Flag indicating whether the heatmap has been initialized.
        initialized (bool): 标志，指示热图是否已初始化。
        colormap (int): OpenCV colormap used for heatmap visualization.
        colormap (int): 用于热图可视化的 OpenCV 颜色映射。
        heatmap (np.ndarray): Array storing the cumulative heatmap data.
        heatmap (np.ndarray): 存储累积热图数据的数组。
        annotator (Annotator): Object for drawing annotations on the image.
        annotator (Annotator): 用于在图像上绘制注释的对象。
    
    Methods:
        heatmap_effect: Calculates and updates the heatmap effect for a given bounding box.
        heatmap_effect: 计算并更新给定边界框的热图效果。
        generate_heatmap: Generates and applies the heatmap effect to each frame.
        generate_heatmap: 为每帧生成并应用热图效果。
    
    Examples:
        >>> from ultralytics.solutions import Heatmap
        >>> heatmap = Heatmap(model="yolo11n.pt", colormap=cv2.COLORMAP_JET)
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = heatmap.generate_heatmap(frame)
    """

    def __init__(self, **kwargs):
        """Initializes the Heatmap class for real-time video stream heatmap generation based on object tracks."""
        """初始化 Heatmap 类，以根据对象轨迹生成实时视频流热图。"""
        super().__init__(**kwargs)

        self.initialized = False  # bool variable for heatmap initialization
        self.initialized = False  # 热图初始化的布尔变量
        if self.region is not None:  # check if user provided the region coordinates
            if self.region is not None:  # 检查用户是否提供了区域坐标
                self.initialize_region()

        # store colormap
        # 存储颜色映射
        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]
        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]  # 使用配置中的颜色映射
        self.heatmap = None

    def heatmap_effect(self, box):
        """
        Efficiently calculates heatmap area and effect location for applying colormap.
        
        高效计算热图区域和效果位置，以应用颜色映射。
        
        Args:
            box (List[float]): Bounding box coordinates [x0, y0, x1, y1].
            box (List[float]): 边界框坐标 [x0, y0, x1, y1]。
        
        Examples:
            >>> heatmap = Heatmap()
            >>> box = [100, 100, 200, 200]
            >>> heatmap.heatmap_effect(box)
        """
        x0, y0, x1, y1 = map(int, box)
        x0, y0, x1, y1 = map(int, box)  # 将边界框坐标转换为整数
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2  # 计算半径的平方

        # Create a meshgrid with region of interest (ROI) for vectorized distance calculations
        # 创建一个网格，包含感兴趣区域（ROI），用于向量化距离计算
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        # Calculate squared distances from the center
        # 计算与中心的平方距离
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        # Create a mask of points within the radius
        # 创建半径内点的掩码
        within_radius = dist_squared <= radius_squared

        # Update only the values within the bounding box in a single vectorized operation
        # 在单个向量化操作中仅更新边界框内的值
        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def generate_heatmap(self, im0):
        """
        Generate heatmap for each frame using Ultralytics.
        
        使用 Ultralytics 为每帧生成热图。
        
        Args:
            im0 (np.ndarray): Input image array for processing.
            im0 (np.ndarray): 用于处理的输入图像数组。
        
        Returns:
            (np.ndarray): Processed image with heatmap overlay and object counts (if region is specified).
            (np.ndarray): 带有热图叠加和对象计数的处理图像（如果指定了区域）。
        
        Examples:
            >>> heatmap = Heatmap()
            >>> im0 = cv2.imread("image.jpg")
            >>> result = heatmap.generate_heatmap(im0)
        """
        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99
        if not self.initialized:  # 如果尚未初始化
            self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99  # 创建热图的零数组
        self.initialized = True  # Initialize heatmap only once
        self.initialized = True  # 仅初始化热图一次

        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化注释器
        self.extract_tracks(im0)  # Extract tracks
        self.extract_tracks(im0)  # 提取轨迹

        # Iterate over bounding boxes, track ids and classes index
        # 遍历边界框、轨迹 ID 和类别索引
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box and counting region
            # 绘制边界框和计数区域
            self.heatmap_effect(box)

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)
                self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)  # 绘制区域
                self.store_tracking_history(track_id, box)  # Store track history
                self.store_tracking_history(track_id, box)  # 存储轨迹历史
                self.store_classwise_counts(cls)  # store classwise counts in dict
                self.store_classwise_counts(cls)  # 在字典中存储类别计数
                current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)  # 计算当前质心
                # Store tracking previous position and perform object counting
                # 存储跟踪的先前位置并执行对象计数
                prev_position = None
                prev_position = None  # 初始化先前位置
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                if len(self.track_history[track_id]) > 1:  # 如果轨迹历史长度大于 1
                    prev_position = self.track_history[track_id][-2]  # 获取先前位置
                self.count_objects(current_centroid, track_id, prev_position, cls)  # Perform object counting
                self.count_objects(current_centroid, track_id, prev_position, cls)  # 执行对象计数

        if self.region is not None:
            self.display_counts(im0)  # Display the counts on the frame
        if self.region is not None:  # 如果指定了区域
            self.display_counts(im0)  # 在帧上显示计数

        # Normalize, apply colormap to heatmap and combine with original image
        # 归一化，应用颜色映射到热图并与原始图像结合
        if self.track_data.id is not None:
            im0 = cv2.addWeighted(
                im0,
                0.5,
                cv2.applyColorMap(
                    cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.colormap
                ),
                0.5,
                0,
            )

        self.display_output(im0)  # display output with base class function
        self.display_output(im0)  # 使用基类函数显示输出
        return im0  # return output image for more usage
        return im0  # 返回输出图像以供进一步使用
