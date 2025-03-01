# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io  # 导入io模块，用于处理字节流
from typing import Any  # 从typing模块导入Any，用于类型注解

import cv2  # 导入OpenCV库以处理图像

from ultralytics import YOLO  # 从ultralytics模块导入YOLO类
from ultralytics.utils import LOGGER  # 从ultralytics.utils导入LOGGER，用于日志记录
from ultralytics.utils.checks import check_requirements  # 从ultralytics.utils.checks导入check_requirements函数
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS  # 从ultralytics.utils.downloads导入GITHUB_ASSETS_STEMS

class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference using
    Streamlit and Ultralytics YOLO models. It provides the functionalities such as loading models, configuring settings,
    uploading video files, and performing real-time inference.
    一个类，用于使用Streamlit和Ultralytics YOLO模型执行对象检测、图像分类、图像分割和姿态估计推断。它提供了加载模型、配置设置、上传视频文件和执行实时推断等功能。

    Attributes:
        st (module): Streamlit module for UI creation.
        st (module): 用于UI创建的Streamlit模块。
        temp_dict (dict): Temporary dictionary to store the model path.
        temp_dict (dict): 存储模型路径的临时字典。
        model_path (str): Path to the loaded model.
        model_path (str): 加载模型的路径。
        model (YOLO): The YOLO model instance.
        model (YOLO): YOLO模型实例。
        source (str): Selected video source.
        source (str): 选择的视频源。
        enable_trk (str): Enable tracking option.
        enable_trk (str): 启用跟踪选项。
        conf (float): Confidence threshold.
        conf (float): 置信度阈值。
        iou (float): IoU threshold for non-max suppression.
        iou (float): 非最大抑制的IoU阈值。
        vid_file_name (str): Name of the uploaded video file.
        vid_file_name (str): 上传视频文件的名称。
        selected_ind (list): List of selected class indices.
        selected_ind (list): 选定类别索引的列表。

    Methods:
        web_ui: Sets up the Streamlit web interface with custom HTML elements.
        web_ui: 使用自定义HTML元素设置Streamlit网页界面。
        sidebar: Configures the Streamlit sidebar for model and inference settings.
        sidebar: 配置Streamlit侧边栏以进行模型和推断设置。
        source_upload: Handles video file uploads through the Streamlit interface.
        source_upload: 通过Streamlit界面处理视频文件上传。
        configure: Configures the model and loads selected classes for inference.
        configure: 配置模型并加载选定的类以进行推断。
        inference: Performs real-time object detection inference.
        inference: 执行实时对象检测推断。

    Examples:
        >>> inf = solutions.Inference(model="path/to/model.pt")  # Model is not necessary argument.
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the Inference class, checking Streamlit requirements and setting up the model path.

        Args:
            **kwargs (Any): Additional keyword arguments for model configuration.
            **kwargs (Any): 模型配置的附加关键字参数。
        """
        check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
        import streamlit as st  # 导入Streamlit模块

        self.st = st  # Reference to the Streamlit class instance
        self.source = None  # Placeholder for video or webcam source details
        self.enable_trk = False  # Flag to toggle object tracking
        self.conf = 0.25  # Confidence threshold for detection
        self.iou = 0.45  # Intersection-over-Union (IoU) threshold for non-maximum suppression
        self.org_frame = None  # Container for the original frame to be displayed
        self.ann_frame = None  # Container for the annotated frame to be displayed
        self.vid_file_name = None  # Holds the name of the video file
        self.selected_ind = []  # List of selected classes for detection or tracking
        self.model = None  # Container for the loaded model instance

        self.temp_dict = {"model": None, **kwargs}  # 临时字典，包含模型路径和其他参数
        self.model_path = None  # Store model file name with path
        if self.temp_dict["model"] is not None:  # 如果提供了模型路径
            self.model_path = self.temp_dict["model"]  # 设置模型路径

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")  # 记录Ultralytics解决方案的信息

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""
        # Streamlit应用程序的主标题

        # Subtitle of streamlit application
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""
        # Streamlit应用程序的副标题

        # Set html page configuration and append custom HTML
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")  # 设置网页配置
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)  # 添加自定义HTML
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)  # 添加主标题
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)  # 添加副标题

    def sidebar(self):
        """Configures the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:  # Add Ultralytics LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"  # Ultralytics LOGO链接
            self.st.image(logo, width=250)  # 显示LOGO

        self.st.sidebar.title("User Configuration")  # 添加用户配置标题
        self.source = self.st.sidebar.selectbox(  # 添加源选择下拉框
            "Video",
            ("webcam", "video"),
        )  # 视频源选择
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))  # 启用对象跟踪的单选框
        self.conf = float(  # 置信度阈值滑块
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )  # 置信度阈值滑块
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # IoU阈值滑块

        col1, col2 = self.st.columns(2)  # 创建两个列
        self.org_frame = col1.empty()  # 原始帧的占位符
        self.ann_frame = col2.empty()  # 注释帧的占位符

    def source_upload(self):
        """Handles video file uploads through the Streamlit interface."""
        self.vid_file_name = ""  # 初始化视频文件名称
        if self.source == "video":  # 如果选择的视频源为视频文件
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])  # 视频文件上传组件
            if vid_file is not None:  # 如果选择了文件
                g = io.BytesIO(vid_file.read())  # 创建BytesIO对象
                with open("ultralytics.mp4", "wb") as out:  # 以字节方式打开临时文件
                    out.write(g.read())  # 将字节写入文件
                self.vid_file_name = "ultralytics.mp4"  # 设置视频文件名称
        elif self.source == "webcam":  # 如果选择的源为网络摄像头
            self.vid_file_name = 0  # 设置视频文件名称为0，表示使用摄像头

    def configure(self):
        """Configures the model and loads selected classes for inference."""
        # Add dropdown menu for model selection
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]  # 获取可用模型列表
        if self.model_path:  # 如果用户提供了自定义模型
            available_models.insert(0, self.model_path.split(".pt")[0])  # 将自定义模型添加到可用模型列表
        selected_model = self.st.sidebar.selectbox("Model", available_models)  # 模型选择下拉框

        with self.st.spinner("Model is downloading..."):  # 显示加载模型的进度条
            self.model = YOLO(f"{selected_model.lower()}.pt")  # 加载YOLO模型
            class_names = list(self.model.names.values())  # 将类名称字典转换为列表
        self.st.success("Model loaded successfully!")  # 显示成功加载模型的消息

        # Multiselect box with class names and get indices of selected classes
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])  # 类别多选框
        self.selected_ind = [class_names.index(option) for option in selected_classes]  # 获取所选类的索引

        if not isinstance(self.selected_ind, list):  # 确保selected_ind是列表
            self.selected_ind = list(self.selected_ind)  # 转换为列表

    def inference(self):
        """Performs real-time object detection inference."""
        self.web_ui()  # 初始化网页界面
        self.sidebar()  # 创建侧边栏
        self.source_upload()  # 上传视频源
        self.configure()  # 配置应用

        if self.st.sidebar.button("Start"):  # 如果点击开始按钮
            stop_button = self.st.button("Stop")  # 停止推断的按钮
            cap = cv2.VideoCapture(self.vid_file_name)  # 捕获视频
            if not cap.isOpened():  # 如果无法打开视频源
                self.st.error("Could not open webcam.")  # 显示错误信息
            while cap.isOpened():  # 当视频源打开时
                success, frame = cap.read()  # 读取帧
                if not success:  # 如果读取失败
                    self.st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")  # 显示警告信息
                    break

                # Store model predictions
                if self.enable_trk == "Yes":  # 如果启用跟踪
                    results = self.model.track(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                    )  # 执行跟踪
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)  # 执行检测
                annotated_frame = results[0].plot()  # 在帧上添加注释

                if stop_button:  # 如果点击停止按钮
                    cap.release()  # 释放视频捕获
                    self.st.stop()  # 停止Streamlit应用

                self.org_frame.image(frame, channels="BGR")  # 显示原始帧
                self.ann_frame.image(annotated_frame, channels="BGR")  # 显示处理后的帧

            cap.release()  # 释放视频捕获
        cv2.destroyAllWindows()  # 销毁所有OpenCV窗口


if __name__ == "__main__":  # 如果该文件是主程序
    import sys  # 导入sys模块以访问命令行参数

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)  # 获取命令行参数的数量
    model = sys.argv[1] if args > 1 else None  # 将第一个参数赋值为模型名称
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()  # 创建Inference类的实例并执行推断