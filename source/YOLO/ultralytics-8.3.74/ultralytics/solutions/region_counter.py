# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution  # 从ultralytics的solutions模块导入BaseSolution类
from ultralytics.utils import LOGGER  # 从ultralytics的utils模块导入LOGGER，用于日志记录
from ultralytics.utils.plotting import Annotator, colors  # 从ultralytics的utils.plotting模块导入Annotator和colors

class RegionCounter(BaseSolution):
    """
    A class designed for real-time counting of objects within user-defined regions in a video stream.
    一个类，旨在实时计数视频流中用户定义区域内的对象。

    This class inherits from `BaseSolution` and offers functionalities to define polygonal regions in a video
    frame, track objects, and count those objects that pass through each defined region. This makes it useful
    for applications that require counting in specified areas, such as monitoring zones or segmented sections.
    该类继承自`BaseSolution`，提供在视频帧中定义多边形区域、跟踪对象和计数通过每个定义区域的对象的功能。这使其在需要在指定区域计数的应用中非常有用，例如监控区域或分段部分。

    Attributes:
        region_template (dict): A template for creating new counting regions with default attributes including
                                the name, polygon coordinates, and display colors.
        region_template (dict): 创建新的计数区域的模板，包含默认属性，如名称、多边形坐标和显示颜色。
        counting_regions (list): A list storing all defined regions, where each entry is based on `region_template`
                                 and includes specific region settings like name, coordinates, and color.
        counting_regions (list): 存储所有定义区域的列表，每个条目基于`region_template`，包括特定区域设置，如名称、坐标和颜色。

    Methods:
        add_region: Adds a new counting region with specified attributes, such as the region's name, polygon points,
                    region color, and text color.
        add_region: 添加一个新的计数区域，具有指定的属性，如区域名称、多边形点、区域颜色和文本颜色。
        count: Processes video frames to count objects in each region, drawing regions and displaying counts
               on the frame. Handles object detection, region definition, and containment checks.
        count: 处理视频帧以计数每个区域中的对象，绘制区域并在帧上显示计数。处理对象检测、区域定义和包含检查。
    """

    def __init__(self, **kwargs):
        """Initializes the RegionCounter class for real-time counting in different regions of the video streams."""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.region_template = {  # 定义区域模板
            "name": "Default Region",  # 默认区域名称
            "polygon": None,  # 多边形坐标
            "counts": 0,  # 计数初始化为0
            "dragging": False,  # 拖动状态
            "region_color": (255, 255, 255),  # 区域颜色
            "text_color": (0, 0, 0),  # 文本颜色
        }
        self.counting_regions = []  # 初始化计数区域列表

    def add_region(self, name, polygon_points, region_color, text_color):
        """
        Adds a new region to the counting list based on the provided template with specific attributes.

        Args:
            name (str): Name assigned to the new region. 新区域的名称。
            polygon_points (list[tuple]): List of (x, y) coordinates defining the region's polygon. 定义区域多边形的(x, y)坐标列表。
            region_color (tuple): BGR color for region visualization. 区域可视化的BGR颜色。
            text_color (tuple): BGR color for the text within the region. 区域内文本的BGR颜色。
        """
        region = self.region_template.copy()  # 复制区域模板
        region.update(  # 更新区域属性
            {
                "name": name,  # 设置区域名称
                "polygon": self.Polygon(polygon_points),  # 创建多边形对象
                "region_color": region_color,  # 设置区域颜色
                "text_color": text_color,  # 设置文本颜色
            }
        )
        self.counting_regions.append(region)  # 将新区域添加到计数区域列表中

    def count(self, im0):
        """
        Processes the input frame to detect and count objects within each defined region.

        Args:
            im0 (numpy.ndarray): Input image frame where objects and regions are annotated. 输入图像帧，其中注释了对象和区域。

        Returns:
           im0 (numpy.ndarray): Processed image frame with annotated counting information. 处理后的图像帧，包含注释的计数信息。
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化注释器
        self.extract_tracks(im0)  # 从图像中提取轨迹

        # Region initialization and conversion 区域初始化和转换
        if self.region is None:  # 如果区域未定义
            self.initialize_region()  # 初始化区域
            regions = {"Region#01": self.region}  # 创建包含默认区域的字典
        else:
            regions = self.region if isinstance(self.region, dict) else {"Region#01": self.region}  # 根据区域类型定义区域

        # Draw regions and process counts for each defined area 绘制区域并处理每个定义区域的计数
        for idx, (region_name, reg_pts) in enumerate(regions.items(), start=1):  # 遍历每个区域
            if not isinstance(reg_pts, list) or not all(isinstance(pt, tuple) for pt in reg_pts):  # 检查区域点的有效性
                LOGGER.warning(f"Invalid region points for {region_name}: {reg_pts}")  # 记录警告日志
                continue  # 跳过无效条目
            color = colors(idx, True)  # 获取区域颜色
            self.annotator.draw_region(reg_pts=reg_pts, color=color, thickness=self.line_width * 2)  # 绘制区域
            self.add_region(region_name, reg_pts, color, self.annotator.get_txt_color())  # 添加区域到计数列表

        # Prepare regions for containment check 准备区域以进行包含检查
        for region in self.counting_regions:  # 遍历所有计数区域
            region["prepared_polygon"] = self.prep(region["polygon"])  # 准备多边形用于检查

        # Process bounding boxes and count objects within each region 处理边界框并计数每个区域内的对象
        for box, cls in zip(self.boxes, self.clss):  # 遍历每个边界框和类别
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))  # 在边界框上绘制标签
            bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)  # 计算边界框的中心坐标

            for region in self.counting_regions:  # 遍历每个计数区域
                if region["prepared_polygon"].contains(self.Point(bbox_center)):  # 检查中心点是否在区域内
                    region["counts"] += 1  # 更新计数

        # Display counts in each region 显示每个区域的计数
        for region in self.counting_regions:  # 遍历每个计数区域
            self.annotator.text_label(  # 在区域内显示计数
                region["polygon"].bounds,  # 获取区域边界
                label=str(region["counts"]),  # 显示计数
                color=region["region_color"],  # 区域颜色
                txt_color=region["text_color"],  # 文本颜色
            )
            region["counts"] = 0  # 重置计数以备下一帧使用

        self.display_output(im0)  # 显示输出
        return im0  # 返回处理后的图像