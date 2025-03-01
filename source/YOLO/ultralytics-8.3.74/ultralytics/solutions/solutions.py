# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict  # 从collections模块导入defaultdict，用于创建默认字典

import cv2  # 导入OpenCV库以处理图像

from ultralytics import YOLO  # 从ultralytics模块导入YOLO类
from ultralytics.utils import ASSETS_URL, DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER  # 导入ultralytics.utils中的常量和日志记录器
from ultralytics.utils.checks import check_imshow, check_requirements  # 导入检查函数

class BaseSolution:
    """
    A base class for managing Ultralytics Solutions.
    Ultralytics解决方案的基类。

    This class provides core functionality for various Ultralytics Solutions, including model loading, object tracking,
    and region initialization.
    该类提供各种Ultralytics解决方案的核心功能，包括模型加载、对象跟踪和区域初始化。

    Attributes:
        LineString (shapely.geometry.LineString): Class for creating line string geometries.
        LineString (shapely.geometry.LineString): 用于创建线字符串几何图形的类。
        Polygon (shapely.geometry.Polygon): Class for creating polygon geometries.
        Polygon (shapely.geometry.Polygon): 用于创建多边形几何图形的类。
        Point (shapely.geometry.Point): Class for creating point geometries.
        Point (shapely.geometry.Point): 用于创建点几何图形的类。
        CFG (Dict): Configuration dictionary loaded from a YAML file and updated with kwargs.
        CFG (Dict): 从YAML文件加载的配置字典，并用kwargs更新。
        region (List[Tuple[int, int]]): List of coordinate tuples defining a region of interest.
        region (List[Tuple[int, int]]): 定义感兴趣区域的坐标元组列表。
        line_width (int): Width of lines used in visualizations.
        line_width (int): 可视化中使用的线条宽度。
        model (ultralytics.YOLO): Loaded YOLO model instance.
        model (ultralytics.YOLO): 加载的YOLO模型实例。
        names (Dict[int, str]): Dictionary mapping class indices to class names.
        names (Dict[int, str]): 将类索引映射到类名称的字典。
        env_check (bool): Flag indicating whether the environment supports image display.
        env_check (bool): 标志，指示环境是否支持图像显示。
        track_history (collections.defaultdict): Dictionary to store tracking history for each object.
        track_history (collections.defaultdict): 用于存储每个对象跟踪历史的字典。

    Methods:
        extract_tracks: Apply object tracking and extract tracks from an input image.
        extract_tracks: 应用对象跟踪并从输入图像中提取轨迹。
        store_tracking_history: Store object tracking history for a given track ID and bounding box.
        store_tracking_history: 存储给定轨迹ID和边界框的对象跟踪历史。
        initialize_region: Initialize the counting region and line segment based on configuration.
        initialize_region: 根据配置初始化计数区域和线段。
        display_output: Display the results of processing, including showing frames or saving results.
        display_output: 显示处理结果，包括显示帧或保存结果。

    Examples:
        >>> solution = BaseSolution(model="yolo11n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> solution.initialize_region()
        >>> image = cv2.imread("image.jpg")
        >>> solution.extract_tracks(image)
        >>> solution.display_output(image)
    """

    def __init__(self, IS_CLI=False, **kwargs):
        """
        Initializes the `BaseSolution` class with configuration settings and the YOLO model for Ultralytics solutions.

        Initializes the `BaseSolution`类，配置设置和Ultralytics解决方案的YOLO模型。

        IS_CLI (optional): Enables CLI mode if set.
        IS_CLI（可选）：如果设置，则启用CLI模式。
        """
        check_requirements("shapely>=2.0.0")  # 检查shapely库的版本要求
        from shapely.geometry import LineString, Point, Polygon  # 从shapely.geometry导入LineString、Point和Polygon
        from shapely.prepared import prep  # 从shapely.prepared导入prep

        self.LineString = LineString  # 将LineString类赋值给self.LineString
        self.Polygon = Polygon  # 将Polygon类赋值给self.Polygon
        self.Point = Point  # 将Point类赋值给self.Point
        self.prep = prep  # 将prep函数赋值给self.prep
        self.annotator = None  # 初始化注释器为None
        self.tracks = None  # 初始化轨迹为None
        self.track_data = None  # 初始化跟踪数据为None
        self.boxes = []  # 初始化边界框列表
        self.clss = []  # 初始化类别列表
        self.track_ids = []  # 初始化轨迹ID列表
        self.track_line = None  # 初始化轨迹线为None
        self.r_s = None  # 初始化区域线段为None

        # Load config and update with args
        DEFAULT_SOL_DICT.update(kwargs)  # 更新默认解决方案字典
        DEFAULT_CFG_DICT.update(kwargs)  # 更新默认配置字典
        self.CFG = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}  # 合并字典
        LOGGER.info(f"Ultralytics Solutions: ✅ {DEFAULT_SOL_DICT}")  # 记录Ultralytics解决方案的信息

        self.region = self.CFG["region"]  # 存储区域数据以供其他类使用
        self.line_width = (
            self.CFG["line_width"] if self.CFG["line_width"] is not None else 2
        )  # 存储线宽以供使用

        # Load Model and store classes names
        if self.CFG["model"] is None:  # 如果模型未指定
            self.CFG["model"] = "yolo11n.pt"  # 设置默认模型
        self.model = YOLO(self.CFG["model"])  # 加载YOLO模型
        self.names = self.model.names  # 存储类名称

        self.track_add_args = {  # Tracker additional arguments for advance configuration
            k: self.CFG[k] for k in ["verbose", "iou", "conf", "device", "max_det", "half", "tracker"]
        }  # 存储跟踪器的附加参数以进行高级配置

        if IS_CLI and self.CFG["source"] is None:  # 如果在CLI模式下且未提供源
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"  # 设置默认源
            LOGGER.warning(f"⚠️ WARNING: source not provided. using default source {ASSETS_URL}/{d_s}")  # 记录警告信息
            from ultralytics.utils.downloads import safe_download  # 导入安全下载函数

            safe_download(f"{ASSETS_URL}/{d_s}")  # 从Ultralytics资源下载源
            self.CFG["source"] = d_s  # 设置默认源

        # Initialize environment and region setup
        self.env_check = check_imshow(warn=True)  # 检查环境是否支持图像显示
        self.track_history = defaultdict(list)  # 初始化跟踪历史字典

    def extract_tracks(self, im0):
        """
        Applies object tracking and extracts tracks from an input image or frame.

        Args:
            im0 (ndarray): The input image or frame. 输入图像或帧。

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)  # 应用对象跟踪并提取轨迹

        # Extract tracks for OBB or object detection
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes  # 提取轨迹数据

        if self.track_data and self.track_data.id is not None:  # 如果有轨迹数据且ID不为None
            self.boxes = self.track_data.xyxy.cpu()  # 获取边界框
            self.clss = self.track_data.cls.cpu().tolist()  # 获取类别
            self.track_ids = self.track_data.id.int().cpu().tolist()  # 获取轨迹ID
        else:
            LOGGER.warning("WARNING ⚠️ no tracks found!")  # 记录未找到轨迹的警告
            self.boxes, self.clss, self.track_ids = [], [], []  # 清空边界框、类别和轨迹ID

    def store_tracking_history(self, track_id, box):
        """
        Stores the tracking history of an object.

        This method updates the tracking history for a given object by appending the center point of its
        bounding box to the track line. It maintains a maximum of 30 points in the tracking history.

        Args:
            track_id (int): The unique identifier for the tracked object. 被跟踪对象的唯一标识符。
            box (List[float]): The bounding box coordinates of the object in the format [x1, y1, x2, y2]. 对象的边界框坐标，格式为[x1, y1, x2, y2]。

        Examples:
            >>> solution = BaseSolution()
            >>> solution.store_tracking_history(1, [100, 200, 300, 400])
        """
        # Store tracking history
        self.track_line = self.track_history[track_id]  # 获取指定轨迹ID的跟踪线
        self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))  # 将边界框中心点添加到跟踪线
        if len(self.track_line) > 30:  # 如果跟踪线超过30个点
            self.track_line.pop(0)  # 删除最旧的点

    def initialize_region(self):
        """Initialize the counting region and line segment based on configuration settings."""
        if self.region is None:  # 如果区域未定义
            self.region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # 设置默认区域
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )  # 根据区域的点数选择使用Polygon或LineString

    def display_output(self, im0):
        """
        Display the results of the processing, which could involve showing frames, printing counts, or saving results.

        This method is responsible for visualizing the output of the object detection and tracking process. It displays
        the processed frame with annotations, and allows for user interaction to close the display.

        Args:
            im0 (numpy.ndarray): The input image or frame that has been processed and annotated. 输入图像或帧，已处理和注释。

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.display_output(frame)

        Notes:
            - This method will only display output if the 'show' configuration is set to True and the environment
              supports image display.
            - The display can be closed by pressing the 'q' key.
        """
        if self.CFG.get("show") and self.env_check:  # 如果配置中设置了显示并且环境支持
            cv2.imshow("Ultralytics Solutions", im0)  # 显示图像
            if cv2.waitKey(1) & 0xFF == ord("q"):  # 如果按下'q'键
                return  # 退出显示