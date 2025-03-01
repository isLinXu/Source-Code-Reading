# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json  # 导入json模块，用于处理JSON数据

import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数组和矩阵操作

from ultralytics.solutions.solutions import BaseSolution  # 从ultralytics的solutions模块导入BaseSolution类
from ultralytics.utils import LOGGER  # 从ultralytics的utils模块导入LOGGER，用于日志记录
from ultralytics.utils.checks import check_requirements  # 从ultralytics的utils.checks模块导入check_requirements，用于检查依赖
from ultralytics.utils.plotting import Annotator  # 从ultralytics的utils.plotting模块导入Annotator，用于绘图

class ParkingPtsSelection:
    """
    A class for selecting and managing parking zone points on images using a Tkinter-based UI.
    一个类，用于使用基于Tkinter的UI选择和管理图像上的停车区域点。

    This class provides functionality to upload an image, select points to define parking zones, and save the
    selected points to a JSON file. It uses Tkinter for the graphical user interface.
    该类提供了上传图像、选择点以定义停车区域并将所选点保存到JSON文件的功能。它使用Tkinter作为图形用户界面。

    Attributes:
        tk (module): The Tkinter module for GUI operations. Tkinter模块，用于GUI操作。
        filedialog (module): Tkinter's filedialog module for file selection operations. Tkinter的filedialog模块，用于文件选择操作。
        messagebox (module): Tkinter's messagebox module for displaying message boxes. Tkinter的messagebox模块，用于显示消息框。
        master (tk.Tk): The main Tkinter window. 主Tkinter窗口。
        canvas (tk.Canvas): The canvas widget for displaying the image and drawing bounding boxes. 用于显示图像和绘制边界框的画布小部件。
        image (PIL.Image.Image): The uploaded image. 上传的图像。
        canvas_image (ImageTk.PhotoImage): The image displayed on the canvas. 显示在画布上的图像。
        rg_data (List[List[Tuple[int, int]]]): List of bounding boxes, each defined by 4 points. 边界框列表，每个边界框由4个点定义。
        current_box (List[Tuple[int, int]]): Temporary storage for the points of the current bounding box. 当前边界框点的临时存储。
        imgw (int): Original width of the uploaded image. 上传图像的原始宽度。
        imgh (int): Original height of the uploaded image. 上传图像的原始高度。
        canvas_max_width (int): Maximum width of the canvas. 画布的最大宽度。
        canvas_max_height (int): Maximum height of the canvas. 画布的最大高度。

    Methods:
        initialize_properties: Initializes the necessary properties. 初始化必要的属性。
        upload_image: Uploads an image, resizes it to fit the canvas, and displays it. 上传图像，将其调整大小以适应画布并显示。
        on_canvas_click: Handles mouse clicks to add points for bounding boxes. 处理鼠标点击以添加边界框的点。
        draw_box: Draws a bounding box on the canvas. 在画布上绘制边界框。
        remove_last_bounding_box: Removes the last bounding box and redraws the canvas. 移除最后一个边界框并重绘画布。
        redraw_canvas: Redraws the canvas with the image and all bounding boxes. 使用图像和所有边界框重绘画布。
        save_to_json: Saves the bounding boxes to a JSON file. 将边界框保存到JSON文件中。

    Examples:
        >>> parking_selector = ParkingPtsSelection()
        >>> # Use the GUI to upload an image, select parking zones, and save the data
    """

    def __init__(self):
        """Initializes the ParkingPtsSelection class, setting up UI and properties for parking zone point selection."""
        check_requirements("tkinter")  # 检查是否安装了tkinter模块
        import tkinter as tk  # 导入tkinter模块
        from tkinter import filedialog, messagebox  # 从tkinter导入filedialog和messagebox模块

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox  # 初始化tkinter相关模块
        self.master = self.tk.Tk()  # Reference to the main application window or parent widget 主应用程序窗口或父小部件的引用
        self.master.title("Ultralytics Parking Zones Points Selector")  # 设置窗口标题
        self.master.resizable(False, False)  # 禁止调整窗口大小

        self.canvas = self.tk.Canvas(self.master, bg="white")  # Canvas widget for displaying images or graphics 用于显示图像或图形的画布小部件
        self.canvas.pack(side=self.tk.BOTTOM)  # 将画布放置在窗口底部

        self.image = None  # Variable to store the loaded image 用于存储加载图像的变量
        self.canvas_image = None  # Reference to the image displayed on the canvas 显示在画布上的图像的引用
        self.canvas_max_width = None  # Maximum allowed width for the canvas 画布的最大允许宽度
        self.canvas_max_height = None  # Maximum allowed height for the canvas 画布的最大允许高度
        self.rg_data = None  # Data related to region or annotation management 与区域或注释管理相关的数据
        self.current_box = None  # Stores the currently selected or active bounding box 存储当前选择或活动的边界框
        self.imgh = None  # Height of the current image 当前图像的高度
        self.imgw = None  # Width of the current image 当前图像的宽度

        # Button frame with buttons 按钮框，包含按钮
        button_frame = self.tk.Frame(self.master)  # 创建按钮框
        button_frame.pack(side=self.tk.TOP)  # 将按钮框放置在窗口顶部

        for text, cmd in [  # 遍历按钮文本和命令
            ("Upload Image", self.upload_image),  # 上传图像按钮
            ("Remove Last BBox", self.remove_last_bounding_box),  # 移除最后一个边界框按钮
            ("Save", self.save_to_json),  # 保存按钮
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)  # 创建按钮并放置在按钮框中

        self.initialize_properties()  # 初始化属性
        self.master.mainloop()  # 启动主事件循环

    def initialize_properties(self):
        """Initialize properties for image, canvas, bounding boxes, and dimensions."""
        self.image = self.canvas_image = None  # 初始化图像和画布图像为None
        self.rg_data, self.current_box = [], []  # 初始化区域数据和当前边界框为空列表
        self.imgw = self.imgh = 0  # 初始化图像宽度和高度为0
        self.canvas_max_width, self.canvas_max_height = 1280, 720  # 设置画布的最大宽度和高度

    def upload_image(self):
        """Uploads and displays an image on the canvas, resizing it to fit within specified dimensions."""
        from PIL import Image, ImageTk  # scope because ImageTk requires tkinter package 作用域内导入Image和ImageTk模块，因为ImageTk需要tkinter包

        self.image = Image.open(self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")]))  # 上传图像文件
        if not self.image:  # 如果没有加载图像
            return  # 直接返回

        self.imgw, self.imgh = self.image.size  # 获取图像的宽度和高度
        aspect_ratio = self.imgw / self.imgh  # 计算宽高比
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )  # 计算画布宽度
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )  # 计算画布高度

        self.canvas.config(width=canvas_width, height=canvas_height)  # 配置画布的宽度和高度
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height)))  # 将图像调整大小并转换为PhotoImage
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)  # 在画布上创建图像
        self.canvas.bind("<Button-1>", self.on_canvas_click)  # 绑定鼠标左键点击事件

        self.rg_data.clear(), self.current_box.clear()  # 清空区域数据和当前边界框

    def on_canvas_click(self, event):
        """Handles mouse clicks to add points for bounding boxes on the canvas."""
        self.current_box.append((event.x, event.y))  # 将点击坐标添加到当前边界框
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")  # 在画布上绘制红色圆点
        if len(self.current_box) == 4:  # 如果当前边界框的点数为4
            self.rg_data.append(self.current_box.copy())  # 将当前边界框复制并添加到区域数据
            self.draw_box(self.current_box)  # 绘制边界框
            self.current_box.clear()  # 清空当前边界框

    def draw_box(self, box):
        """Draws a bounding box on the canvas using the provided coordinates."""
        for i in range(4):  # 遍历边界框的4个点
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)  # 在画布上绘制边界框的线段

    def remove_last_bounding_box(self):
        """Removes the last bounding box from the list and redraws the canvas."""
        if not self.rg_data:  # 如果没有边界框
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")  # 显示警告消息
            return  # 直接返回
        self.rg_data.pop()  # 移除最后一个边界框
        self.redraw_canvas()  # 重绘画布

    def redraw_canvas(self):
        """Redraws the canvas with the image and all bounding boxes."""
        self.canvas.delete("all")  # 清空画布
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)  # 重新创建图像
        for box in self.rg_data:  # 遍历所有边界框
            self.draw_box(box)  # 绘制每个边界框

    def save_to_json(self):
        """Saves the selected parking zone points to a JSON file with scaled coordinates."""
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()  # 计算缩放比例
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]  # 生成JSON数据

        from io import StringIO  # Function level import, as it's only required to store coordinates, not every frame 函数级导入，因为它只用于存储坐标，而不是每个帧

        write_buffer = StringIO()  # 创建StringIO对象用于存储数据
        json.dump(data, write_buffer, indent=4)  # 将数据写入StringIO对象
        with open("bounding_boxes.json", "w", encoding="utf-8") as f:  # 打开文件以写入JSON数据
            f.write(write_buffer.getvalue())  # 写入数据
        self.messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")  # 显示成功消息

class ParkingManagement(BaseSolution):
    """
    Manages parking occupancy and availability using YOLO model for real-time monitoring and visualization.
    使用YOLO模型管理停车场的占用情况和可用性，以实现实时监控和可视化。

    This class extends BaseSolution to provide functionality for parking lot management, including detection of
    occupied spaces, visualization of parking regions, and display of occupancy statistics.
    该类扩展了BaseSolution，提供停车场管理的功能，包括占用空间的检测、停车区域的可视化和占用统计的显示。

    Attributes:
        json_file (str): Path to the JSON file containing parking region details. 包含停车区域详细信息的JSON文件路径。
        json (List[Dict]): Loaded JSON data containing parking region information. 加载的JSON数据，包含停车区域信息。
        pr_info (Dict[str, int]): Dictionary storing parking information (Occupancy and Available spaces). 存储停车信息（占用和可用空间）的字典。
        arc (Tuple[int, int, int]): RGB color tuple for available region visualization. 可用区域可视化的RGB颜色元组。
        occ (Tuple[int, int, int]): RGB color tuple for occupied region visualization. 占用区域可视化的RGB颜色元组。
        dc (Tuple[int, int, int]): RGB color tuple for centroid visualization of detected objects. 检测到的对象质心可视化的RGB颜色元组。

    Methods:
        process_data: Processes model data for parking lot management and visualization. 处理停车场管理和可视化的模型数据。

    Examples:
        >>> from ultralytics.solutions import ParkingManagement
        >>> parking_manager = ParkingManagement(model="yolo11n.pt", json_file="parking_regions.json")
        >>> print(f"Occupied spaces: {parking_manager.pr_info['Occupancy']}")
        >>> print(f"Available spaces: {parking_manager.pr_info['Available']}")
    """

    def __init__(self, **kwargs):
        """Initializes the parking management system with a YOLO model and visualization settings."""
        super().__init__(**kwargs)  # 调用父类的初始化方法

        self.json_file = self.CFG["json_file"]  # Load JSON data 加载JSON数据
        if self.json_file is None:  # 如果json_file为空
            LOGGER.warning("❌ json_file argument missing. Parking region details required.")  # 记录警告日志
            raise ValueError("❌ Json file path can not be empty")  # 引发错误

        with open(self.json_file) as f:  # 打开JSON文件
            self.json = json.load(f)  # 加载JSON数据

        self.pr_info = {"Occupancy": 0, "Available": 0}  # dictionary for parking information 停车信息的字典

        self.arc = (0, 0, 255)  # available region color 可用区域颜色
        self.occ = (0, 255, 0)  # occupied region color 占用区域颜色
        self.dc = (255, 0, 189)  # centroid color for each box 每个框的质心颜色

    def process_data(self, im0):
        """
        Processes the model data for parking lot management.

        This function analyzes the input image, extracts tracks, and determines the occupancy status of parking
        regions defined in the JSON file. It annotates the image with occupied and available parking spots,
        and updates the parking information.

        Args:
            im0 (np.ndarray): The input inference image. 输入推理图像。

        Examples:
            >>> parking_manager = ParkingManagement(json_file="parking_regions.json")
            >>> image = cv2.imread("parking_lot.jpg")
            >>> parking_manager.process_data(image)
        """
        self.extract_tracks(im0)  # extract tracks from im0 从im0中提取轨迹
        es, fs = len(self.json), 0  # empty slots, filled slots 空槽位，已填槽位
        annotator = Annotator(im0, self.line_width)  # init annotator 初始化注释器

        for region in self.json:  # 遍历JSON中的每个区域
            # Convert points to a NumPy array with the correct dtype and reshape properly 将点转换为具有正确数据类型的NumPy数组并正确重塑
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))  # 创建点数组
            rg_occupied = False  # occupied region initialization 占用区域初始化
            for box, cls in zip(self.boxes, self.clss):  # 遍历每个边界框和类别
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)  # 计算边界框的中心坐标
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)  # 计算点到多边形的距离
                if dist >= 0:  # 如果点在多边形内
                    # cv2.circle(im0, (xc, yc), radius=self.line_width * 4, color=self.dc, thickness=-1)
                    annotator.display_objects_labels(  # 在图像上显示对象标签
                        im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                    )
                    rg_occupied = True  # 标记区域为占用
                    break  # 退出循环
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)  # 更新空槽位和已填槽位
            # Plotting regions 绘制区域
            cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=2)  # 绘制多边形

        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, es  # 更新停车信息

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)  # 显示停车信息
        self.display_output(im0)  # display output with base class function 使用基类函数显示输出
        return im0  # return output image for more usage 返回处理后的图像以供进一步使用