# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution  # 从ultralytics的solutions模块导入BaseSolution类
from ultralytics.utils.plotting import Annotator, colors  # 从ultralytics的utils.plotting模块导入Annotator和colors

class QueueManager(BaseSolution):
    """
    Manages queue counting in real-time video streams based on object tracks.
    管理基于对象轨迹的实时视频流中的队列计数。

    This class extends BaseSolution to provide functionality for tracking and counting objects within a specified
    region in video frames.
    该类扩展了BaseSolution，提供了在视频帧中跟踪和计数指定区域内对象的功能。

    Attributes:
        counts (int): The current count of objects in the queue. 队列中对象的当前计数。
        rect_color (Tuple[int, int, int]): RGB color tuple for drawing the queue region rectangle. 绘制队列区域矩形的RGB颜色元组。
        region_length (int): The number of points defining the queue region. 定义队列区域的点数。
        annotator (Annotator): An instance of the Annotator class for drawing on frames. 用于在帧上绘图的Annotator类实例。
        track_line (List[Tuple[int, int]]): List of track line coordinates. 轨迹线坐标的列表。
        track_history (Dict[int, List[Tuple[int, int]]]): Dictionary storing tracking history for each object. 存储每个对象跟踪历史的字典。

    Methods:
        initialize_region: Initializes the queue region. 初始化队列区域。
        process_queue: Processes a single frame for queue management. 处理单个帧以进行队列管理。
        extract_tracks: Extracts object tracks from the current frame. 从当前帧中提取对象轨迹。
        store_tracking_history: Stores the tracking history for an object. 存储对象的跟踪历史。
        display_output: Displays the processed output. 显示处理后的输出。

    Examples:
        >>> cap = cv2.VideoCapture("Path/to/video/file.mp4")
        >>> queue_manager = QueueManager(region=[100, 100, 200, 200, 300, 300])
        >>> while cap.isOpened():
        >>>     success, im0 = cap.read()
        >>>     if not success:
        >>>         break
        >>>     out = queue.process_queue(im0)
    """

    def __init__(self, **kwargs):
        """Initializes the QueueManager with parameters for tracking and counting objects in a video stream."""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.initialize_region()  # 初始化队列区域
        self.counts = 0  # Queue counts Information 队列计数信息
        self.rect_color = (255, 255, 255)  # Rectangle color 矩形颜色
        self.region_length = len(self.region)  # Store region length for further usage 存储区域长度以备后用

    def process_queue(self, im0):
        """
        Processes the queue management for a single frame of video.

        Args:
            im0 (numpy.ndarray): Input image for processing, typically a frame from a video stream. 输入图像以供处理，通常是视频流中的一帧。

        Returns:
            (numpy.ndarray): Processed image with annotations, bounding boxes, and queue counts. 处理后的图像，包含注释、边界框和队列计数。

        This method performs the following steps:
        1. Resets the queue count for the current frame. 重置当前帧的队列计数。
        2. Initializes an Annotator object for drawing on the image. 初始化Annotator对象以在图像上绘图。
        3. Extracts tracks from the image. 从图像中提取轨迹。
        4. Draws the counting region on the image. 在图像上绘制计数区域。
        5. For each detected object:
           - Draws bounding boxes and labels. 绘制边界框和标签。
           - Stores tracking history. 存储跟踪历史。
           - Draws centroids and tracks. 绘制质心和轨迹。
           - Checks if the object is inside the counting region and updates the count. 检查对象是否在计数区域内并更新计数。
        6. Displays the queue count on the image. 在图像上显示队列计数。
        7. Displays the processed output. 显示处理后的输出。

        Examples:
            >>> queue_manager = QueueManager()
            >>> frame = cv2.imread("frame.jpg")
            >>> processed_frame = queue_manager.process_queue(frame)
        """
        self.counts = 0  # Reset counts every frame 每帧重置计数
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator 初始化注释器
        self.extract_tracks(im0)  # Extract tracks 从图像中提取轨迹

        self.annotator.draw_region(  # 绘制区域
            reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):  # 遍历每个边界框、轨迹ID和类别
            # Draw bounding box and counting region 绘制边界框和计数区域
            self.annotator.box_label(box, label=self.names[cls], color=colors(track_id, True))  # 在边界框上绘制标签
            self.store_tracking_history(track_id, box)  # 存储轨迹历史

            # Draw tracks of objects 绘制对象的轨迹
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            # Cache frequently accessed attributes 缓存频繁访问的属性
            track_history = self.track_history.get(track_id, [])  # 获取轨迹历史

            # store previous position of track and check if the object is inside the counting region
            prev_position = None  # 初始化前一个位置为None
            if len(track_history) > 1:  # 如果轨迹历史长度大于1
                prev_position = track_history[-2]  # 获取前一个位置
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):  # 检查对象是否在计数区域内
                self.counts += 1  # 更新计数

        # Display queue counts 显示队列计数
        self.annotator.queue_counts_display(  # 在图像上显示队列计数
            f"Queue Counts : {str(self.counts)}",
            points=self.region,
            region_color=self.rect_color,
            txt_color=(104, 31, 17),
        )
        self.display_output(im0)  # display output with base class function 使用基类函数显示输出

        return im0  # return output image for more usage 返回处理后的图像以供进一步使用