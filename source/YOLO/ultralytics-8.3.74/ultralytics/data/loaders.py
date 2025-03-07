# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import imread


@dataclass
class SourceTypes:
    """
    Class to represent various types of input sources for predictions.
    类，用于表示用于预测的各种输入源类型。

    This class uses dataclass to define boolean flags for different types of input sources that can be used for
    making predictions with YOLO models.
    该类使用数据类定义布尔标志，表示可以用于YOLO模型进行预测的不同类型的输入源。

    Attributes:
        stream (bool): Flag indicating if the input source is a video stream.
        stream (bool): 标志，指示输入源是否为视频流。
        screenshot (bool): Flag indicating if the input source is a screenshot.
        screenshot (bool): 标志，指示输入源是否为截图。
        from_img (bool): Flag indicating if the input source is an image file.
        from_img (bool): 标志，指示输入源是否为图像文件。

    Examples:
        >>> source_types = SourceTypes(stream=True, screenshot=False, from_img=False)
        >>> print(source_types.stream)
        True
        >>> print(source_types.from_img)
        False
    """

    stream: bool = False  # 输入源是否为视频流，默认为False
    screenshot: bool = False  # 输入源是否为截图，默认为False
    from_img: bool = False  # 输入源是否为图像文件，默认为False
    tensor: bool = False  # 输入源是否为张量，默认为False


class LoadStreams:
    """
    Stream Loader for various types of video streams.
    各种类型视频流的加载器。

    Supports RTSP, RTMP, HTTP, and TCP streams. This class handles the loading and processing of multiple video
    streams simultaneously, making it suitable for real-time video analysis tasks.
    支持RTSP、RTMP、HTTP和TCP流。该类处理多个视频流的加载和处理，适合实时视频分析任务。

    Attributes:
        sources (List[str]): The source input paths or URLs for the video streams.
        sources (List[str]): 视频流的输入路径或URL。
        vid_stride (int): Video frame-rate stride.
        vid_stride (int): 视频帧率步幅。
        buffer (bool): Whether to buffer input streams.
        buffer (bool): 是否缓冲输入流。
        running (bool): Flag to indicate if the streaming thread is running.
        running (bool): 指示流线程是否正在运行的标志。
        mode (str): Set to 'stream' indicating real-time capture.
        mode (str): 设置为'stream'，指示实时捕获。
        imgs (List[List[np.ndarray]]): List of image frames for each stream.
        imgs (List[List[np.ndarray]]): 每个流的图像帧列表。
        fps (List[float]): List of FPS for each stream.
        fps (List[float]): 每个流的FPS列表。
        frames (List[int]): List of total frames for each stream.
        frames (List[int]): 每个流的总帧数列表。
        threads (List[Thread]): List of threads for each stream.
        threads (List[Thread]): 每个流的线程列表。
        shape (List[Tuple[int, int, int]]): List of shapes for each stream.
        shape (List[Tuple[int, int, int]]): 每个流的形状列表。
        caps (List[cv2.VideoCapture]): List of cv2.VideoCapture objects for each stream.
        caps (List[cv2.VideoCapture]): 每个流的cv2.VideoCapture对象列表。
        bs (int): Batch size for processing.
        bs (int): 处理的批量大小。

    Methods:
        update: Read stream frames in daemon thread.
        update: 在守护线程中读取流帧。
        close: Close stream loader and release resources.
        close: 关闭流加载器并释放资源。
        __iter__: Returns an iterator object for the class.
        __iter__: 返回类的迭代器对象。
        __next__: Returns source paths, transformed, and original images for processing.
        __next__: 返回源路径、转换后的图像和原始图像以进行处理。
        __len__: Return the length of the sources object.
        __len__: 返回sources对象的长度。

    Examples:
        >>> stream_loader = LoadStreams("rtsp://example.com/stream1.mp4")
        >>> for sources, imgs, _ in stream_loader:
        ...     # Process the images
        ...     pass
        >>> stream_loader.close()

    Notes:
        - The class uses threading to efficiently load frames from multiple streams simultaneously.
        - 该类使用线程高效地同时从多个流加载帧。
        - It automatically handles YouTube links, converting them to the best available stream URL.
        - 它自动处理YouTube链接，将其转换为最佳可用流URL。
        - The class implements a buffer system to manage frame storage and retrieval.
        - 该类实现了一个缓冲系统来管理帧存储和检索。
    """

    def __init__(self, sources="file.streams", vid_stride=1, buffer=False):
        """Initialize stream loader for multiple video sources, supporting various stream types."""
        # 初始化多个视频源的流加载器，支持各种流类型。
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        # torch.backends.cudnn.benchmark = True  # 在固定大小推理时更快
        self.buffer = buffer  # buffer input streams
        # self.buffer = buffer  # 缓冲输入流
        self.running = True  # running flag for Thread
        # self.running = True  # 线程的运行标志
        self.mode = "stream"  # 设置为'stream'
        self.vid_stride = vid_stride  # video frame-rate stride
        # self.vid_stride = vid_stride  # 视频帧率步幅

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        # 如果sources是文件，则读取其内容并分割为列表，否则将sources作为单个元素列表
        n = len(sources)  # 获取源的数量
        self.bs = n  # 批量大小等于源的数量
        self.fps = [0] * n  # frames per second
        # self.fps = [0] * n  # 每秒帧数
        self.frames = [0] * n  # 每个流的帧数
        self.threads = [None] * n  # 每个流的线程列表
        self.caps = [None] * n  # video capture objects
        # self.caps = [None] * n  # 视频捕获对象
        self.imgs = [[] for _ in range(n)]  # images
        # self.imgs = [[] for _ in range(n)]  # 图像
        self.shape = [[] for _ in range(n)]  # image shapes
        # self.shape = [[] for _ in range(n)]  # 图像形状
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        # self.sources = [ops.clean_str(x) for x in sources]  # 清理源名称以供后用
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            # 启动线程以从视频流读取帧
            st = f"{i + 1}/{n}: {s}... "  # 生成当前源的状态信息
            if urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:  # if source is YouTube video
                # 如果源是YouTube视频
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Jsn8D3aC840' or 'https://youtu.be/Jsn8D3aC840'
                s = get_best_youtube_url(s)  # 获取最佳YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            # 如果s是数字，则评估它，否则保持不变
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' webcam not supported in Colab and Kaggle notebooks. "
                    "'source=0' 的网络摄像头在Colab和Kaggle笔记本中不受支持。"
                    "Try running 'source=0' in a local environment."
                    "尝试在本地环境中运行'source=0'。"
                )
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            # 存储视频捕获对象
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}Failed to open {s}")
                # 如果未成功打开，抛出连接错误
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            # 获取FPS，警告：可能返回0或nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback
            # 获取帧数，若为0则设置为无限流
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
            # 30 FPS fallback
            # 获取FPS，若为无穷大则设置为30

            success, im = self.caps[i].read()  # guarantee first frame
            # 确保读取第一帧
            if not success or im is None:
                raise ConnectionError(f"{st}Failed to read images from {s}")
                # 如果读取失败，抛出连接错误
            self.imgs[i].append(im)  # 添加第一帧到图像列表
            self.shape[i] = im.shape  # 存储图像形状
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            # 启动线程以更新图像
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            # 记录成功信息
            self.threads[i].start()  # 启动线程
        LOGGER.info("")  # newline

    def update(self, i, cap, stream):
        """Read stream frames in daemon thread and update image buffer."""
        # 在守护线程中读取流帧并更新图像缓冲区
        n, f = 0, self.frames[i]  # frame number, frame array
        # 帧编号，帧数组
        while self.running and cap.isOpened() and n < (f - 1):
            # 当线程运行且视频流打开且帧数小于总帧数时
            if len(self.imgs[i]) < 30:  # keep a <=30-image buffer
                # 保持图像缓冲区不超过30张
                n += 1  # 增加帧计数
                cap.grab()  # .read() = .grab() followed by .retrieve()
                # 抓取下一帧
                if n % self.vid_stride == 0:  # 根据帧率步幅读取帧
                    success, im = cap.retrieve()  # 读取当前帧
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)  # 创建空图像
                        LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                        # 记录警告信息
                        cap.open(stream)  # re-open stream if signal was lost
                        # 如果信号丢失，则重新打开流
                    if self.buffer:
                        self.imgs[i].append(im)  # 如果启用缓冲，则添加图像
                    else:
                        self.imgs[i] = [im]  # 否则仅保留当前图像
            else:
                time.sleep(0.01)  # wait until the buffer is empty
                # 等待直到缓冲区为空

    def close(self):
        """Terminates stream loader, stops threads, and releases video capture resources."""
        # 终止流加载器，停止线程并释放视频捕获资源
        self.running = False  # stop flag for Thread
        # 停止线程的标志
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
                # 如果线程仍在运行，则等待其结束，设置超时
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            # 遍历存储的VideoCapture对象
            try:
                cap.release()  # release video capture
                # 释放视频捕获
            except Exception as e:
                LOGGER.warning(f"WARNING ⚠️ Could not release VideoCapture object: {e}")
                # 记录警告信息
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        # 遍历YOLO图像流并重新打开无响应的流
        self.count = -1  # 初始化计数
        return self  # 返回自身作为迭代器

    def __next__(self):
        """Returns the next batch of frames from multiple video streams for processing."""
        # 返回多个视频流的下一批帧以进行处理
        self.count += 1  # 增加计数

        images = []  # 初始化图像列表
        for i, x in enumerate(self.imgs):  # 遍历每个流的图像
            # Wait until a frame is available in each buffer
            # 等待每个缓冲区中有帧可用
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord("q"):  # q to quit
                    # 如果线程不再存活或按下'q'键，则关闭
                    self.close()  # 关闭流
                    raise StopIteration  # 停止迭代
                time.sleep(1 / min(self.fps))  # 等待直到帧可用
                x = self.imgs[i]  # 更新当前流的图像
                if not x:
                    LOGGER.warning(f"WARNING ⚠️ Waiting for stream {i}")
                    # 记录警告信息

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))  # 从缓冲区获取并移除第一帧

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                # 获取最后一帧，如果没有则创建空图像
                x.clear()  # 清空缓冲区

        return self.sources, images, [""] * self.bs  # 返回源、图像和空字符串列表

    def __len__(self):
        """Return the number of video streams in the LoadStreams object."""
        # 返回LoadStreams对象中的视频流数量
        return self.bs  # 1E12帧 = 32个流在30FPS下持续30年

class LoadScreenshots:
    """
    Ultralytics screenshot dataloader for capturing and processing screen images.
    Ultralytics截图数据加载器，用于捕获和处理屏幕图像。

    This class manages the loading of screenshot images for processing with YOLO. It is suitable for use with
    `yolo predict source=screen`.
    该类管理截图图像的加载，以便与YOLO一起处理。适用于使用`yolo predict source=screen`。

    Attributes:
        source (str): The source input indicating which screen to capture.
        source (str): 输入源，指示要捕获哪个屏幕。
        screen (int): The screen number to capture.
        screen (int): 要捕获的屏幕编号。
        left (int): The left coordinate for screen capture area.
        left (int): 屏幕捕获区域的左坐标。
        top (int): The top coordinate for screen capture area.
        top (int): 屏幕捕获区域的顶部坐标。
        width (int): The width of the screen capture area.
        width (int): 屏幕捕获区域的宽度。
        height (int): The height of the screen capture area.
        height (int): 屏幕捕获区域的高度。
        mode (str): Set to 'stream' indicating real-time capture.
        mode (str): 设置为'stream'，指示实时捕获。
        frame (int): Counter for captured frames.
        frame (int): 捕获帧的计数器。
        sct (mss.mss): Screen capture object from `mss` library.
        sct (mss.mss): 来自`mss`库的屏幕捕获对象。
        bs (int): Batch size, set to 1.
        bs (int): 批量大小，设置为1。
        fps (int): Frames per second, set to 30.
        fps (int): 每秒帧数，设置为30。
        monitor (Dict[str, int]): Monitor configuration details.
        monitor (Dict[str, int]): 显示器配置细节。

    Methods:
        __iter__: Returns an iterator object.
        __iter__: 返回一个迭代器对象。
        __next__: Captures the next screenshot and returns it.
        __next__: 捕获下一个截图并返回。

    Examples:
        >>> loader = LoadScreenshots("0 100 100 640 480")  # screen 0, top-left (100,100), 640x480
        >>> for source, im, im0s, vid_cap, s in loader:
        ...     print(f"Captured frame: {im.shape}")
    """

    def __init__(self, source):
        """Initialize screenshot capture with specified screen and region parameters."""
        # 初始化带有指定屏幕和区域参数的截图捕获。
        check_requirements("mss")  # 检查是否安装了mss库
        import mss  # noqa  # 导入mss库

        source, *params = source.split()  # 将输入源分割为屏幕编号和参数
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        # 默认设置为全屏0
        if len(params) == 1:
            self.screen = int(params[0])  # 如果只有一个参数，则设置屏幕编号
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)  # 如果有四个参数，则分别设置左、上、宽、高
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)  # 如果有五个参数，则分别设置屏幕编号和区域参数
        self.mode = "stream"  # 设置模式为'stream'
        self.frame = 0  # 初始化帧计数器
        self.sct = mss.mss()  # 创建屏幕捕获对象
        self.bs = 1  # 批量大小设置为1
        self.fps = 30  # 每秒帧数设置为30

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]  # 获取指定屏幕的监视器信息
        self.top = monitor["top"] if top is None else (monitor["top"] + top)  # 设置顶部坐标
        self.left = monitor["left"] if left is None else (monitor["left"] + left)  # 设置左侧坐标
        self.width = width or monitor["width"]  # 设置宽度
        self.height = height or monitor["height"]  # 设置高度
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}  # 存储监视器的配置

    def __iter__(self):
        """Yields the next screenshot image from the specified screen or region for processing."""
        # 从指定屏幕或区域生成下一个截图图像以供处理。
        return self  # 返回自身作为迭代器

    def __next__(self):
        """Captures and returns the next screenshot as a numpy array using the mss library."""
        # 使用mss库捕获并返回下一个截图作为numpy数组。
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]  # BGRA to BGR  # 将BGRA转换为BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "  # 生成屏幕信息字符串

        self.frame += 1  # 增加帧计数
        return [str(self.screen)], [im0], [s]  # 返回屏幕编号、图像和信息字符串


class LoadImagesAndVideos:
    """
    A class for loading and processing images and videos for YOLO object detection.
    用于加载和处理YOLO目标检测的图像和视频的类。

    This class manages the loading and pre-processing of image and video data from various sources, including
    single image files, video files, and lists of image and video paths.
    该类管理来自各种来源的图像和视频数据的加载和预处理，包括单个图像文件、视频文件以及图像和视频路径的列表。

    Attributes:
        files (List[str]): List of image and video file paths.
        files (List[str]): 图像和视频文件路径的列表。
        nf (int): Total number of files (images and videos).
        nf (int): 文件总数（图像和视频）。
        video_flag (List[bool]): Flags indicating whether a file is a video (True) or an image (False).
        video_flag (List[bool]): 标志，指示文件是否为视频（True）或图像（False）。
        mode (str): Current mode, 'image' or 'video'.
        mode (str): 当前模式，'image'或'video'。
        vid_stride (int): Stride for video frame-rate.
        vid_stride (int): 视频帧率的步幅。
        bs (int): Batch size.
        bs (int): 批量大小。
        cap (cv2.VideoCapture): Video capture object for OpenCV.
        cap (cv2.VideoCapture): OpenCV的视频捕获对象。
        frame (int): Frame counter for video.
        frame (int): 视频的帧计数器。
        frames (int): Total number of frames in the video.
        frames (int): 视频中的总帧数。
        count (int): Counter for iteration, initialized at 0 during __iter__().
        count (int): 迭代计数器，在__iter__()期间初始化为0。
        ni (int): Number of images.
        ni (int): 图像数量。

    Methods:
        __init__: Initialize the LoadImagesAndVideos object.
        __init__: 初始化LoadImagesAndVideos对象。
        __iter__: Returns an iterator object for VideoStream or ImageFolder.
        __iter__: 返回VideoStream或ImageFolder的迭代器对象。
        __next__: Returns the next batch of images or video frames along with their paths and metadata.
        __next__: 返回下一批图像或视频帧及其路径和元数据。
        _new_video: Creates a new video capture object for the given path.
        _new_video: 为给定路径创建新的视频捕获对象。
        __len__: Returns the number of batches in the object.
        __len__: 返回对象中的批次数量。

    Examples:
        >>> loader = LoadImagesAndVideos("path/to/data", batch=32, vid_stride=1)
        >>> for paths, imgs, info in loader:
        ...     # Process batch of images or video frames
        ...     pass

    Notes:
        - Supports various image formats including HEIC.
        - 支持多种图像格式，包括HEIC。
        - Handles both local files and directories.
        - 处理本地文件和目录。
        - Can read from a text file containing paths to images and videos.
        - 可以从包含图像和视频路径的文本文件中读取。
    """

    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize dataloader for images and videos, supporting various input formats."""
        # 初始化图像和视频的数据加载器，支持各种输入格式。
        parent = None  # 初始化父目录为None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            # 如果路径是字符串且为.txt文件，则读取文件内容
            parent = Path(path).parent  # 获取父目录
            path = Path(path).read_text().splitlines()  # 读取源列表
        files = []  # 初始化文件列表
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            # 遍历路径，确保路径是列表或元组
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            # 获取绝对路径
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
                # 如果路径中包含通配符，则使用glob扩展文件
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # dir
                # 如果路径是目录，则扩展目录下的所有文件
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
                # 如果路径是文件，则添加到文件列表
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
                # 如果路径是相对路径且在父目录中存在，则添加到文件列表
            else:
                raise FileNotFoundError(f"{p} does not exist")  # 如果路径不存在，则抛出文件未找到错误

        # Define files as images or videos
        images, videos = [], []  # 初始化图像和视频列表
        for f in files:
            suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
            # 获取文件扩展名并转换为小写
            if suffix in IMG_FORMATS:
                images.append(f)  # 如果是图像格式，则添加到图像列表
            elif suffix in VID_FORMATS:
                videos.append(f)  # 如果是视频格式，则添加到视频列表
        ni, nv = len(images), len(videos)  # 获取图像和视频的数量

        self.files = images + videos  # 合并图像和视频列表
        self.nf = ni + nv  # number of files
        self.ni = ni  # number of images
        self.video_flag = [False] * ni + [True] * nv  # 创建标志列表，指示文件类型
        self.mode = "video" if ni == 0 else "image"  # default to video if no images
        # 如果没有图像，则默认设置为视频
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch  # 批量大小设置
        if any(videos):
            self._new_video(videos[0])  # new video
            # 如果有视频，则创建新视频
        else:
            self.cap = None  # 如果没有视频，则设置捕获对象为None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")
            # 如果没有找到图像或视频，则抛出文件未找到错误

    def __iter__(self):
        """Iterates through image/video files, yielding source paths, images, and metadata."""
        # 遍历图像/视频文件，生成源路径、图像和元数据。
        self.count = 0  # 初始化计数器
        return self  # 返回自身作为迭代器

    def __next__(self):
        """Returns the next batch of images or video frames with their paths and metadata."""
        # 返回下一批图像或视频帧及其路径和元数据。
        paths, imgs, info = [], [], []  # 初始化路径、图像和信息列表
        while len(imgs) < self.bs:  # 当图像数量小于批量大小时
            if self.count >= self.nf:  # end of file list
                # 如果到达文件列表末尾
                if imgs:
                    return paths, imgs, info  # return last partial batch
                    # 返回最后一部分批次
                else:
                    raise StopIteration  # 如果没有图像，则停止迭代

            path = self.files[self.count]  # 获取当前文件路径
            if self.video_flag[self.count]:  # 如果当前文件是视频
                self.mode = "video"  # 设置模式为视频
                if not self.cap or not self.cap.isOpened():  # 检查视频捕获对象
                    self._new_video(path)  # 创建新的视频捕获对象

                success = False  # 初始化成功标志
                for _ in range(self.vid_stride):  # 根据帧率步幅抓取帧
                    success = self.cap.grab()  # 抓取下一帧
                    if not success:
                        break  # end of video or failure
                        # 如果抓取失败，则跳出循环

                if success:  # 如果抓取成功
                    success, im0 = self.cap.retrieve()  # 从视频捕获对象中获取图像
                    if success:  # 如果成功获取图像
                        self.frame += 1  # 增加帧计数
                        paths.append(path)  # 添加路径到列表
                        imgs.append(im0)  # 添加图像到列表
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        # 添加信息字符串
                        if self.frame == self.frames:  # end of video
                            self.count += 1  # 增加计数器
                            self.cap.release()  # 释放视频捕获对象
                else:
                    # Move to the next file if the current video ended or failed to open
                    self.count += 1  # 增加计数器
                    if self.cap:
                        self.cap.release()  # 释放视频捕获对象
                    if self.count < self.nf:  # 如果还有文件
                        self._new_video(self.files[self.count])  # 创建新视频
            else:
                # Handle image files (including HEIC)
                self.mode = "image"  # 设置模式为图像
                if path.split(".")[-1].lower() == "heic":  # 如果是HEIC格式
                    # Load HEIC image using Pillow with pillow-heif
                    check_requirements("pillow-heif")  # 检查是否安装了pillow-heif库

                    from pillow_heif import register_heif_opener  # 从pillow_heif导入注册器

                    register_heif_opener()  # Register HEIF opener with Pillow
                    # 使用Pillow注册HEIF打开器
                    with Image.open(path) as img:  # 打开HEIC图像
                        im0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # convert image to BGR nparray
                        # 将图像转换为BGR格式的numpy数组
                else:
                    im0 = imread(path)  # BGR  # 读取图像
                if im0 is None:  # 如果图像读取失败
                    LOGGER.warning(f"WARNING ⚠️ Image Read Error {path}")  # 记录警告信息
                else:
                    paths.append(path)  # 添加路径到列表
                    imgs.append(im0)  # 添加图像到列表
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")  # 添加信息字符串
                self.count += 1  # move to the next file  # 移动到下一个文件
                if self.count >= self.ni:  # end of image list
                    break  # 如果到达图像列表末尾，则跳出循环

        return paths, imgs, info  # 返回路径、图像和信息

    def _new_video(self, path):
        """Creates a new video capture object for the given path and initializes video-related attributes."""
        # 为给定路径创建新的视频捕获对象并初始化与视频相关的属性。
        self.frame = 0  # 初始化帧计数
        self.cap = cv2.VideoCapture(path)  # 创建视频捕获对象
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # 获取视频的FPS
        if not self.cap.isOpened():  # 检查视频捕获对象是否成功打开
            raise FileNotFoundError(f"Failed to open video {path}")  # 如果未成功打开，则抛出错误
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)  # 获取视频的总帧数

    def __len__(self):
        """Returns the number of files (images and videos) in the dataset."""
        # 返回数据集中图像和视频的数量。
        return math.ceil(self.nf / self.bs)  # number of batches

class LoadPilAndNumpy:
    """
    Load images from PIL and Numpy arrays for batch processing.
    从PIL和Numpy数组加载图像以进行批处理。

    This class manages loading and pre-processing of image data from both PIL and Numpy formats. It performs basic
    validation and format conversion to ensure that the images are in the required format for downstream processing.
    该类管理从PIL和Numpy格式加载和预处理图像数据。它执行基本的验证和格式转换，以确保图像符合后续处理的要求。

    Attributes:
        paths (List[str]): List of image paths or autogenerated filenames.
        paths (List[str]): 图像路径或自动生成的文件名列表。
        im0 (List[np.ndarray]): List of images stored as Numpy arrays.
        im0 (List[np.ndarray]): 存储为Numpy数组的图像列表。
        mode (str): Type of data being processed, set to 'image'.
        mode (str): 正在处理的数据类型，设置为'image'。
        bs (int): Batch size, equivalent to the length of `im0`.
        bs (int): 批量大小，相当于`im0`的长度。

    Methods:
        _single_check: Validate and format a single image to a Numpy array.
        _single_check: 验证并格式化单个图像为Numpy数组。

    Examples:
        >>> from PIL import Image
        >>> import numpy as np
        >>> pil_img = Image.new("RGB", (100, 100))
        >>> np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> loader = LoadPilAndNumpy([pil_img, np_img])
        >>> paths, images, _ = next(iter(loader))
        >>> print(f"Loaded {len(images)} images")
        Loaded 2 images
    """

    def __init__(self, im0):
        """Initializes a loader for PIL and Numpy images, converting inputs to a standardized format."""
        # 初始化PIL和Numpy图像的加载器，将输入转换为标准化格式。
        if not isinstance(im0, list):
            im0 = [im0]  # 如果输入不是列表，则将其转换为列表。
        # use `image{i}.jpg` when Image.filename returns an empty path.
        self.paths = [getattr(im, "filename", "") or f"image{i}.jpg" for i, im in enumerate(im0)]
        # 获取每个图像的文件名，如果没有则使用自动生成的名称。
        self.im0 = [self._single_check(im) for im in im0]  # 验证并格式化每个图像。
        self.mode = "image"  # 设置模式为'image'
        self.bs = len(self.im0)  # 批量大小等于图像数量。

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array, ensuring RGB order and contiguous memory."""
        # 验证并格式化图像为Numpy数组，确保RGB顺序和连续内存。
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        # 确保输入是PIL图像或Numpy数组。
        if isinstance(im, Image.Image):
            if im.mode != "RGB":
                im = im.convert("RGB")  # 如果不是RGB模式，则转换为RGB模式。
            im = np.asarray(im)[:, :, ::-1]  # 将图像转换为Numpy数组并调整通道顺序。
            im = np.ascontiguousarray(im)  # 确保数组是连续的。
        return im  # 返回格式化后的图像。

    def __len__(self):
        """Returns the length of the 'im0' attribute, representing the number of loaded images."""
        # 返回'im0'属性的长度，表示加载的图像数量。
        return len(self.im0)  # 返回图像数量。

    def __next__(self):
        """Returns the next batch of images, paths, and metadata for processing."""
        # 返回下一批图像、路径和元数据以供处理。
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration  # 如果已经循环一次，则停止迭代。
        self.count += 1  # 增加计数。
        return self.paths, self.im0, [""] * self.bs  # 返回路径、图像和空字符串列表。

    def __iter__(self):
        """Iterates through PIL/numpy images, yielding paths, raw images, and metadata for processing."""
        # 遍历PIL/Numpy图像，生成路径、原始图像和元数据以供处理。
        self.count = 0  # 初始化计数器。
        return self  # 返回自身作为迭代器。


class LoadTensor:
    """
    A class for loading and processing tensor data for object detection tasks.
    用于加载和处理目标检测任务的张量数据的类。

    This class handles the loading and pre-processing of image data from PyTorch tensors, preparing them for
    further processing in object detection pipelines.
    该类处理从PyTorch张量加载和预处理图像数据，为目标检测管道的进一步处理做好准备。

    Attributes:
        im0 (torch.Tensor): The input tensor containing the image(s) with shape (B, C, H, W).
        im0 (torch.Tensor): 输入张量，包含形状为(B, C, H, W)的图像。
        bs (int): Batch size, inferred from the shape of `im0`.
        bs (int): 批量大小，从`im0`的形状推断得出。
        mode (str): Current processing mode, set to 'image'.
        mode (str): 当前处理模式，设置为'image'。
        paths (List[str]): List of image paths or auto-generated filenames.
        paths (List[str]): 图像路径或自动生成的文件名列表。

    Methods:
        _single_check: Validates and formats an input tensor.
        _single_check: 验证并格式化输入张量。

    Examples:
        >>> import torch
        >>> tensor = torch.rand(1, 3, 640, 640)
        >>> loader = LoadTensor(tensor)
        >>> paths, images, info = next(iter(loader))
        >>> print(f"Processed {len(images)} images")
    """

    def __init__(self, im0) -> None:
        """Initialize LoadTensor object for processing torch.Tensor image data."""
        # 初始化LoadTensor对象以处理torch.Tensor图像数据。
        self.im0 = self._single_check(im0)  # 验证并格式化输入张量。
        self.bs = self.im0.shape[0]  # 批量大小从张量的形状推断。
        self.mode = "image"  # 设置模式为'image'。
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]
        # 获取每个图像的文件名，如果没有则使用自动生成的名称。

    @staticmethod
    def _single_check(im, stride=32):
        """Validates and formats a single image tensor, ensuring correct shape and normalization."""
        # 验证并格式化单个图像张量，确保形状正确和归一化。
        s = (
            f"WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        )  # 构造警告信息。
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)  # 如果形状不正确，则抛出错误。
            LOGGER.warning(s)  # 记录警告信息。
            im = im.unsqueeze(0)  # 如果是3D张量，则增加一个维度。
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)  # 如果形状不符合步幅要求，则抛出错误。
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:  # torch.float32 eps is 1.2e-07
            LOGGER.warning(
                f"WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. "
                f"Dividing input by 255."
            )  # 如果最大值大于1，则记录警告信息并归一化。
            im = im.float() / 255.0  # 归一化张量。

        return im  # 返回格式化后的张量。

    def __iter__(self):
        """Yields an iterator object for iterating through tensor image data."""
        # 生成一个迭代器对象，用于遍历张量图像数据。
        self.count = 0  # 初始化计数器。
        return self  # 返回自身作为迭代器。

    def __next__(self):
        """Yields the next batch of tensor images and metadata for processing."""
        # 生成下一批张量图像及其元数据以供处理。
        if self.count == 1:  # 只循环一次，因为这是批推理
            raise StopIteration  # 如果已经循环一次，则停止迭代。
        self.count += 1  # 增加计数。
        return self.paths, self.im0, [""] * self.bs  # 返回路径、图像和空字符串列表。

    def __len__(self):
        """Returns the batch size of the tensor input."""
        # 返回张量输入的批量大小。
        return self.bs  # 返回批量大小。


def autocast_list(source):
    """Merges a list of sources into a list of numpy arrays or PIL images for Ultralytics prediction."""
    # 将源列表合并为用于Ultralytics预测的Numpy数组或PIL图像列表。
    files = []  # 初始化文件列表。
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            # 如果是文件名或URI
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im))
            # 打开图像，如果是HTTP链接则请求图像。
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np Image
            # 如果是PIL图像或Numpy数组
            files.append(im)  # 添加到文件列表。
        else:
            raise TypeError(
                f"type {type(im).__name__} is not a supported Ultralytics prediction source type. \n"
                f"See https://docs.ultralytics.com/modes/predict for supported source types."
            )  # 如果类型不支持，则抛出错误。

    return files  # 返回合并后的文件列表。


def get_best_youtube_url(url, method="pytube"):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.
    从给定的YouTube视频中检索最佳质量的MP4视频流的URL。

    Args:
        url (str): The URL of the YouTube video.
        url (str): YouTube视频的URL。
        method (str): The method to use for extracting video info. Options are "pytube", "pafy", and "yt-dlp".
            Defaults to "pytube".
        method (str): 用于提取视频信息的方法。选项为"pytube"、"pafy"和"yt-dlp"。默认为"pytube"。

    Returns:
        (str | None): The URL of the best quality MP4 video stream, or None if no suitable stream is found.
        (str | None): 最佳质量MP4视频流的URL，如果未找到合适的流，则返回None。

    Examples:
        >>> url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        >>> best_url = get_best_youtube_url(url)
        >>> print(best_url)
        https://rr4---sn-q4flrnek.googlevideo.com/videoplayback?expire=...

    Notes:
        - Requires additional libraries based on the chosen method: pytubefix, pafy, or yt-dlp.
        - 根据选择的方法，需要额外的库：pytubefix、pafy或yt-dlp。
        - The function prioritizes streams with at least 1080p resolution when available.
        - 该函数优先选择分辨率至少为1080p的流（如果可用）。
        - For the "yt-dlp" method, it looks for formats with video codec, no audio, and *.mp4 extension.
        - 对于"yt-dlp"方法，它查找具有视频编解码器、无音频和*.mp4扩展名的格式。
    """
    if method == "pytube":
        # Switched from pytube to pytubefix to resolve https://github.com/pytube/pytube/issues/1954
        check_requirements("pytubefix>=6.5.2")  # 检查是否安装了pytubefix库。
        from pytubefix import YouTube  # 导入YouTube类。

        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)  # 获取mp4格式的视频流。
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)  # sort streams by resolution
        # 按分辨率对视频流进行排序。
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:  # check if resolution is at least 1080p
                return stream.url  # 返回最佳质量的流URL。

    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))  # 检查是否安装了pafy和youtube_dl库。
        import pafy  # noqa  # 导入pafy库。

        return pafy.new(url).getbestvideo(preftype="mp4").url  # 返回最佳视频流的URL。

    elif method == "yt-dlp":
        check_requirements("yt-dlp")  # 检查是否安装了yt-dlp库。
        import yt_dlp  # 导入yt_dlp库。

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:  # 创建yt-dlp对象。
            info_dict = ydl.extract_info(url, download=False)  # 提取视频信息。
        for f in reversed(info_dict.get("formats", [])):  # reversed because best is usually last
            # 反向遍历格式列表，因为最佳格式通常在最后。
            # Find a format with video codec, no audio, *.mp4 extension at least 1920x1080 size
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080  # 检查格式大小。
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")  # 返回符合条件的流的URL。


# Define constants
LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots)  # 定义加载器常量