# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution  # 从ultralytics的solutions模块导入BaseSolution类
from ultralytics.utils.plotting import Annotator, colors  # 从ultralytics的utils.plotting模块导入Annotator和colors

class ObjectCounter(BaseSolution):  # 定义ObjectCounter类，继承自BaseSolution
    """
    A class to manage the counting of objects in a real-time video stream based on their tracks.
    一个类，用于管理基于轨迹在实时视频流中计数对象。

    This class extends the BaseSolution class and provides functionality for counting objects moving in and out of a
    specified region in a video stream. It supports both polygonal and linear regions for counting.
    该类扩展了BaseSolution类，并提供了在视频流中计数进出指定区域的对象的功能。它支持多边形和线性区域的计数。

    Attributes:
        in_count (int): Counter for objects moving inward.  向内移动的对象计数器。
        out_count (int): Counter for objects moving outward. 向外移动的对象计数器。
        counted_ids (List[int]): List of IDs of objects that have been counted. 已计数对象的ID列表。
        classwise_counts (Dict[str, Dict[str, int]]): Dictionary for counts, categorized by object class. 按对象类别分类的计数字典。
        region_initialized (bool): Flag indicating whether the counting region has been initialized. 指示计数区域是否已初始化的标志。
        show_in (bool): Flag to control display of inward count. 控制向内计数显示的标志。
        show_out (bool): Flag to control display of outward count. 控制向外计数显示的标志。

    Methods:
        count_objects: Counts objects within a polygonal or linear region. 在多边形或线性区域内计数对象。
        store_classwise_counts: Initializes class-wise counts if not already present. 如果尚未存在，则初始化类别计数。
        display_counts: Displays object counts on the frame. 在帧上显示对象计数。
        count: Processes input data (frames or object tracks) and updates counts. 处理输入数据（帧或对象轨迹）并更新计数。

    Examples:
        >>> counter = ObjectCounter()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = counter.count(frame)
        >>> print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}")
    """

    def __init__(self, **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
        # 初始化ObjectCounter类，用于实时视频流中的对象计数
        super().__init__(**kwargs)  # 调用父类的初始化方法

        self.in_count = 0  # Counter for objects moving inward  向内移动的对象计数器
        self.out_count = 0  # Counter for objects moving outward 向外移动的对象计数器
        self.counted_ids = []  # List of IDs of objects that have been counted 已计数对象的ID列表
        self.classwise_counts = {}  # Dictionary for counts, categorized by object class 按对象类别分类的计数字典
        self.region_initialized = False  # Bool variable for region initialization 区域初始化的布尔变量

        self.show_in = self.CFG["show_in"]  # 从配置中获取是否显示向内计数的标志
        self.show_out = self.CFG["show_out"]  # 从配置中获取是否显示向外计数的标志

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """
        Counts objects within a polygonal or linear region based on their tracks.

        Args:
            current_centroid (Tuple[float, float]): Current centroid values in the current frame. 当前帧中的质心值。
            track_id (int): Unique identifier for the tracked object. 被跟踪对象的唯一标识符。
            prev_position (Tuple[float, float]): Last frame position coordinates (x, y) of the track. 轨迹的最后一帧位置坐标（x，y）。
            cls (int): Class index for classwise count updates. 类别计数更新的类别索引。

        Examples:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id = 1
            >>> prev_position = (120, 220)
            >>> cls = 0
            >>> counter.count_objects(current_centroid, track_id, prev_position, cls)
        """
        if prev_position is None or track_id in self.counted_ids:  # 如果前一个位置为空或track_id已经被计数
            return  # 直接返回，不进行计数

        if len(self.region) == 2:  # Linear region (defined as a line segment) 线性区域（定义为线段）
            line = self.LineString(self.region)  # Check if the line intersects the trajectory of the object 检查线是否与对象的轨迹相交
            if line.intersects(self.LineString([prev_position, current_centroid])):  # 如果线与当前质心和前一个位置之间的线相交
                # Determine orientation of the region (vertical or horizontal) 确定区域的方向（垂直或水平）
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # Vertical region: Compare x-coordinates to determine direction 垂直区域：比较x坐标以确定方向
                    if current_centroid[0] > prev_position[0]:  # Moving right 向右移动
                        self.in_count += 1  # 向内计数加一
                        self.classwise_counts[self.names[cls]]["IN"] += 1  # 类别向内计数加一
                    else:  # Moving left 向左移动
                        self.out_count += 1  # 向外计数加一
                        self.classwise_counts[self.names[cls]]["OUT"] += 1  # 类别向外计数加一
                # Horizontal region: Compare y-coordinates to determine direction 水平区域：比较y坐标以确定方向
                elif current_centroid[1] > prev_position[1]:  # Moving downward 向下移动
                    self.in_count += 1  # 向内计数加一
                    self.classwise_counts[self.names[cls]]["IN"] += 1  # 类别向内计数加一
                else:  # Moving upward 向上移动
                    self.out_count += 1  # 向外计数加一
                    self.classwise_counts[self.names[cls]]["OUT"] += 1  # 类别向外计数加一
                self.counted_ids.append(track_id)  # 将track_id添加到已计数ID列表中

        elif len(self.region) > 2:  # Polygonal region 多边形区域
            polygon = self.Polygon(self.region)  # 创建多边形对象
            if polygon.contains(self.Point(current_centroid)):  # 如果多边形包含当前质心
                # Determine motion direction for vertical or horizontal polygons 确定垂直或水平多边形的运动方向
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)  # 计算区域宽度
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)  # 计算区域高度

                if (
                    region_width < region_height
                    and current_centroid[0] > prev_position[0]
                    or region_width >= region_height
                    and current_centroid[1] > prev_position[1]
                ):  # Moving right 向右移动
                    self.in_count += 1  # 向内计数加一
                    self.classwise_counts[self.names[cls]]["IN"] += 1  # 类别向内计数加一
                else:  # Moving left 向左移动
                    self.out_count += 1  # 向外计数加一
                    self.classwise_counts[self.names[cls]]["OUT"] += 1  # 类别向外计数加一
                self.counted_ids.append(track_id)  # 将track_id添加到已计数ID列表中

    def store_classwise_counts(self, cls):
        """
        Initialize class-wise counts for a specific object class if not already present.

        Args:
            cls (int): Class index for classwise count updates. 类别计数更新的类别索引。

        This method ensures that the 'classwise_counts' dictionary contains an entry for the specified class,
        initializing 'IN' and 'OUT' counts to zero if the class is not already present.
        此方法确保'classwise_counts'字典包含指定类别的条目，如果类别尚未存在，则将'IN'和'OUT'计数初始化为零。

        Examples:
            >>> counter = ObjectCounter()
            >>> counter.store_classwise_counts(0)  # Initialize counts for class index 0
            >>> print(counter.classwise_counts)
            {'person': {'IN': 0, 'OUT': 0}}
        """
        if self.names[cls] not in self.classwise_counts:  # 如果类别名称不在计数字典中
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}  # 初始化该类别的计数

    def display_counts(self, im0):
        """
        Displays object counts on the input image or frame.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on. 输入图像或帧以显示计数。

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0  # 仅当有计数时才显示
        }

        if labels_dict:  # 如果有标签字典
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)  # 在图像上显示计数

    def count(self, im0):
        """
        Processes input data (frames or object tracks) and updates object counts.

        This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates
        object counts, and displays the results on the input image.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed. 需要处理的输入图像或帧。

        Returns:
            (numpy.ndarray): The processed image with annotations and count information. 处理后的图像，包含注释和计数信息。

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = counter.count(frame)
        """
        if not self.region_initialized:  # 如果区域尚未初始化
            self.initialize_region()  # 初始化区域
            self.region_initialized = True  # 设置区域已初始化标志

        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator 初始化注释器
        self.extract_tracks(im0)  # 提取轨迹

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # Draw region 绘制区域

        # Iterate over bounding boxes, track ids and classes index 遍历边界框、轨迹ID和类别索引
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box and counting region 绘制边界框和计数区域
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))  # 在边界框上显示标签
            self.store_tracking_history(track_id, box)  # 存储轨迹历史
            self.store_classwise_counts(cls)  # 存储类别计数

            # Draw tracks of objects 绘制对象的轨迹
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(cls), True), track_thickness=self.line_width
            )
            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)  # 计算当前质心
            # store previous position of track for object counting 存储对象计数的轨迹前一个位置
            prev_position = None  # 初始化前一个位置为None
            if len(self.track_history[track_id]) > 1:  # 如果轨迹历史长度大于1
                prev_position = self.track_history[track_id][-2]  # 获取前一个位置
            self.count_objects(current_centroid, track_id, prev_position, cls)  # 进行对象计数

        self.display_counts(im0)  # 在帧上显示计数
        self.display_output(im0)  # 使用基类函数显示输出

        return im0  # 返回处理后的图像以供进一步使用