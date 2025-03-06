# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, TryExcept, ops, plt_settings, threaded
from ultralytics.utils.checks import check_font, check_version, is_ascii
from ultralytics.utils.files import increment_path


class Colors:
    """
    Ultralytics color palette https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors.
    Ultralytics 颜色调色板 https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors。

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.
    该类提供了处理 Ultralytics 颜色调色板的方法，包括将十六进制颜色代码转换为 RGB 值。

    Attributes:
        palette (list of tuple): List of RGB color values.
        palette（元组列表）：RGB 颜色值的列表。
        n (int): The number of colors in the palette.
        n（整数）：调色板中的颜色数量。
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
        pose_palette（np.ndarray）：具有 dtype np.uint8 的特定颜色调色板数组。

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"
    !!! 注意 "Ultralytics 品牌颜色"

        For Ultralytics brand colors see [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand). Please use the official Ultralytics colors for all marketing materials.
        有关 Ultralytics 品牌颜色，请参见 [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand)。请在所有营销材料中使用官方的 Ultralytics 颜色。
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        # 初始化颜色为 hex = matplotlib.colors.TABLEAU_COLORS.values()。
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        # 将十六进制颜色转换为 RGB 并存储在调色板中
        self.n = len(self.palette)
        # 计算调色板中的颜色数量
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )
        # 定义特定的姿态颜色调色板，类型为 np.uint8

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        # 将十六进制颜色代码转换为 RGB 值。
        c = self.palette[int(i) % self.n]
        # 获取调色板中对应索引的颜色
        return (c[2], c[1], c[0]) if bgr else c
        # 如果 bgr 为 True，则返回 BGR 格式的颜色，否则返回 RGB 格式的颜色

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        # 将十六进制颜色代码转换为 RGB 值（即默认的 PIL 顺序）。
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
        # 将十六进制字符串转换为 RGB 元组


colors = Colors()  # create instance for 'from utils.plots import colors'
# 创建 Colors 类的实例，用于 'from utils.plots import colors'


class Annotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.
    Ultralytics 注释器，用于训练/验证马赛克和 JPG 及预测注释。

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        im（Image.Image 或 numpy 数组）：要注释的图像。
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        pil（布尔值）：是否使用 PIL 或 cv2 绘制注释。
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        font（ImageFont.truetype 或 ImageFont.load_default）：用于文本注释的字体。
        lw (float): Line width for drawing.
        lw（浮点数）：绘制的线宽。
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        skeleton（List[List[int]]）：关键点的骨架结构。
        limb_color (List[int]): Color palette for limbs.
        limb_color（List[int]）：肢体的颜色调色板。
        kpt_color (List[int]): Color palette for keypoints.
        kpt_color（List[int]）：关键点的颜色调色板。
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        # 使用图像和线宽初始化 Annotator 类，并提供关键点和肢体的颜色调色板。
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        # 检查示例是否包含非 ASCII 字符（例如，亚洲、阿拉伯、斯拉夫字符）
        input_is_pil = isinstance(im, Image.Image)
        # 检查输入图像是否为 PIL 图像
        self.pil = pil or non_ascii or input_is_pil
        # 根据条件设置是否使用 PIL
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        # 设置线宽，默认根据图像大小计算
        if self.pil:  # use PIL
            # 如果使用 PIL
            self.im = im if input_is_pil else Image.fromarray(im)
            # 将输入图像转换为 PIL 图像
            self.draw = ImageDraw.Draw(self.im)
            # 创建用于绘制的 ImageDraw 对象
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                # 检查并加载合适的字体
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                # 设置字体大小，默认根据图像大小计算
                self.font = ImageFont.truetype(str(font), size)
                # 使用指定字体和大小创建字体对象
            except Exception:
                self.font = ImageFont.load_default()
                # 如果加载字体失败，则使用默认字体
            # Deprecation fix for w, h = getsize(string) -> _, _, w, h = getbox(string)
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
                # 修复获取文本大小的方法
        else:  # use cv2
            # 如果使用 cv2
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            # 确保图像是连续的
            self.im = im if im.flags.writeable else im.copy()
            # 将输入图像赋值给实例变量
            self.tf = max(self.lw - 1, 1)  # font thickness
            # 设置字体厚度
            self.sf = self.lw / 3  # font scale
            # 设置字体缩放比例
        # Pose
        # 姿态
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]
        # 定义关键点的骨架结构

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        # 设置肢体的颜色调色板
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        # 设置关键点的颜色调色板
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        # 定义深色调色板
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }
        # 定义浅色调色板

    def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
        """
        Assign text color based on background color.
        根据背景颜色分配文本颜色。

        Args:
            color (tuple, optional): The background color of the rectangle for text (B, G, R).
            color（元组，可选）：文本矩形的背景颜色 (B, G, R)。
            txt_color (tuple, optional): The color of the text (R, G, B).
            txt_color（元组，可选）：文本的颜色 (R, G, B)。

        Returns:
            txt_color (tuple): Text color for label
            txt_color（元组）：标签的文本颜色
        """
        if color in self.dark_colors:
            return 104, 31, 17
            # 如果背景颜色在深色调色板中，则返回深色文本颜色
        elif color in self.light_colors:
            return 255, 255, 255
            # 如果背景颜色在浅色调色板中，则返回浅色文本颜色
        else:
            return txt_color
            # 否则返回默认文本颜色

    def circle_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=2):
        """
        Draws a label with a background circle centered within a given bounding box.
        在给定的边界框内绘制带有背景圆圈的标签。

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            box（元组）：边界框坐标 (x1, y1, x2, y2)。
            label (str): The text label to be displayed.
            label（字符串）：要显示的文本标签。
            color (tuple, optional): The background color of the rectangle (B, G, R).
            color（元组，可选）：矩形的背景颜色 (B, G, R)。
            txt_color (tuple, optional): The color of the text (R, G, B).
            txt_color（元组，可选）：文本的颜色 (R, G, B)。
            margin (int, optional): The margin between the text and the rectangle border.
            margin（整数，可选）：文本与矩形边框之间的边距。
        """
        # If label have more than 3 characters, skip other characters, due to circle size
        # 如果标签超过 3 个字符，则跳过其他字符，以适应圆圈大小
        if len(label) > 3:
            print(
                f"Length of label is {len(label)}, initial 3 label characters will be considered for circle annotation!"
            )
            # 打印标签长度信息
            label = label[:3]
            # 仅保留前 3 个字符

        # Calculate the center of the box
        # 计算边界框的中心
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # 获取文本大小
        text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]
        # 计算适应文本及边距所需的半径
        required_radius = int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin
        # 使用所需半径绘制圆圈
        cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
        # 计算文本位置
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # 绘制文本
        cv2.putText(
            self.im,
            str(label),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def text_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=5):
        """
        Draws a label with a background rectangle centered within a given bounding box.
        在给定的边界框内绘制带有背景矩形的标签。
    
        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            box（元组）：边界框坐标 (x1, y1, x2, y2)。
            label (str): The text label to be displayed.
            label（字符串）：要显示的文本标签。
            color (tuple, optional): The background color of the rectangle (B, G, R).
            color（元组，可选）：矩形的背景颜色 (B, G, R)。
            txt_color (tuple, optional): The color of the text (R, G, B).
            txt_color（元组，可选）：文本的颜色 (R, G, B)。
            margin (int, optional): The margin between the text and the rectangle border.
            margin（整数，可选）：文本与矩形边框之间的边距。
        """
        # Calculate the center of the bounding box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # 计算边界框的中心坐标
    
        # Get the size of the text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        # 获取文本的大小
    
        # Calculate the top-left corner of the text (to center it)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # 计算文本的左上角坐标，以便居中显示
    
        # Calculate the coordinates of the background rectangle
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # 计算背景矩形的坐标
    
        # Draw the background rectangle
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # 绘制背景矩形
    
        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.1,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )
        # 在矩形上绘制文本
    
    
    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """
        Draws a bounding box to image with label.
        在图像上绘制带标签的边界框。
    
        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            box（元组）：边界框坐标 (x1, y1, x2, y2)。
            label (str): The text label to be displayed.
            label（字符串）：要显示的文本标签。
            color (tuple, optional): The background color of the rectangle (B, G, R).
            color（元组，可选）：矩形的背景颜色 (B, G, R)。
            txt_color (tuple, optional): The color of the text (R, G, B).
            txt_color（元组，可选）：文本的颜色 (R, G, B)。
            rotated (bool, optional): Variable used to check if task is OBB
            rotated（布尔值，可选）：用于检查任务是否为 OBB 的变量。
        """
        txt_color = self.get_txt_color(color, txt_color)
        # 根据背景颜色获取文本颜色
        if isinstance(box, torch.Tensor):
            box = box.tolist()
            # 如果 box 是张量，则转换为列表
        if self.pil or not is_ascii(label):
            # 如果使用 PIL 或标签不是 ASCII 字符
            if rotated:
                p1 = box[0]
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)  # PIL requires tuple box
                # 绘制旋转的多边形边界框
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)  # box
                # 绘制矩形边界框
            if label:
                w, h = self.font.getsize(label)  # text width, height
                # 获取文本的宽度和高度
                outside = p1[1] >= h  # label fits outside box
                # 检查标签是否适合框外
                if p1[0] > self.im.size[0] - w:  # size is (w, h), check if label extend beyond right side of image
                    p1 = self.im.size[0] - w, p1[1]
                    # 如果标签超出图像右侧，则调整位置
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # 绘制文本背景矩形
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
                # 在图像上绘制文本
        else:  # cv2
            # 如果使用 cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)  # cv2 requires nparray box
                # 绘制旋转的多边形边界框
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
                # 绘制矩形边界框
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                # 获取文本的宽度和高度
                h += 3  # add pixels to pad text
                # 为文本添加额外的像素以增加填充
                outside = p1[1] >= h  # label fits outside box
                # 检查标签是否适合框外
                if p1[0] > self.im.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                    p1 = self.im.shape[1] - w, p1[1]
                    # 如果标签超出图像右侧，则调整位置
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                # 绘制填充的文本背景矩形
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )
                # 在图像上绘制文本
    
        def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
            """
            Plot masks on image.
            在图像上绘制掩码。
    
            Args:
                masks (tensor): Predicted masks on cuda, shape: [n, h, w]
                masks（张量）：在 cuda 上的预测掩码，形状为 [n, h, w]
                colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
                colors（列表）：预测掩码的颜色，形状为 [[r, g, b] * n]
                im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
                im_gpu（张量）：图像在 cuda 上，形状为 [3, h, w]，范围为 [0, 1]
                alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
                alpha（浮点数）：掩码透明度：0.0 完全透明，1.0 不透明
                retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
                retina_masks（布尔值）：是否使用高分辨率掩码。默认为 False。
            """
            if self.pil:
                # Convert to numpy first
                self.im = np.asarray(self.im).copy()
                # 首先转换为 numpy 数组
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
                # 如果没有掩码，则将图像设置为 GPU 图像
            if im_gpu.device != masks.device:
                im_gpu = im_gpu.to(masks.device)
                # 如果图像设备与掩码设备不同，则将图像移动到掩码设备
            colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
            colors = colors[:, None, None]  # shape(n,1,1,3)
            masks = masks.unsqueeze(3)  # shape(n,h,w,1)
            masks_color = masks * (colors * alpha)  # shape(n,h,w,3)
    
            inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
            mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)
    
            im_gpu = im_gpu.flip(dims=[0])  # flip channel
            im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
            im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
            im_mask = im_gpu * 255
            im_mask_np = im_mask.byte().cpu().numpy()
            self.im[:] = im_mask_np if retina_masks else ops.scale_image(im_mask_np, self.im.shape)
            # 将掩码应用到图像上
            if self.pil:
                # Convert im back to PIL and update draw
                self.fromarray(self.im)
    
        def kpts(self, kpts, shape=(640, 640), radius=None, kpt_line=True, conf_thres=0.25, kpt_color=None):
            """
            Plot keypoints on the image.
            在图像上绘制关键点。
    
            Args:
                kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
                kpts（torch.Tensor）：关键点，形状为 [17, 3]（x, y, 置信度）。
                shape (tuple, optional): Image shape (h, w). Defaults to (640, 640).
                shape（元组，可选）：图像形状 (h, w)。默认为 (640, 640)。
                radius (int, optional): Keypoint radius. Defaults to 5.
                radius（整数，可选）：关键点半径。默认为 5。
                kpt_line (bool, optional): Draw lines between keypoints. Defaults to True.
                kpt_line（布尔值，可选）：在关键点之间绘制线条。默认为 True。
                conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
                conf_thres（浮点数，可选）：置信度阈值。默认为 0.25。
                kpt_color (tuple, optional): Keypoint color (B, G, R). Defaults to None.
                kpt_color（元组，可选）：关键点颜色 (B, G, R)。默认为 None。
    
            Note:
                - `kpt_line=True` currently only supports human pose plotting.
                - Modifies self.im in-place.
                - If self.pil is True, converts image to numpy array and back to PIL.
            """
            radius = radius if radius is not None else self.lw
            # 如果未指定半径，则使用线宽
            if self.pil:
                # Convert to numpy first
                self.im = np.asarray(self.im).copy()
                # 首先转换为 numpy 数组
            nkpt, ndim = kpts.shape
            # 获取关键点的数量和维度
            is_pose = nkpt == 17 and ndim in {2, 3}
            # 检查是否为姿态关键点
            kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
            # 仅在姿态关键点时支持绘制线条
            for i, k in enumerate(kpts):
                color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
                # 获取关键点颜色
                x_coord, y_coord = k[0], k[1]
                # 获取关键点坐标
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    # 检查坐标是否在图像范围内
                    if len(k) == 3:
                        conf = k[2]
                        if conf < conf_thres:
                            continue
                        # 如果存在置信度并且低于阈值，则跳过
                    cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
                    # 绘制关键点圆圈
    
            if kpt_line:
                ndim = kpts.shape[-1]
                for i, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                    pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                    # 获取骨架连接的关键点坐标
                    if ndim == 3:
                        conf1 = kpts[(sk[0] - 1), 2]
                        conf2 = kpts[(sk[1] - 1), 2]
                        if conf1 < conf_thres or conf2 < conf_thres:
                            continue
                        # 检查置信度是否低于阈值
                    if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    # 检查坐标是否在图像范围内
                    if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    cv2.line(
                        self.im,
                        pos1,
                        pos2,
                        kpt_color or self.limb_color[i].tolist(),
                        thickness=int(np.ceil(self.lw / 2)),
                        lineType=cv2.LINE_AA,
                    )
                    # 绘制连接关键点的线条
            if self.pil:
                # Convert im back to PIL and update draw
                self.fromarray(self.im)
    
        def rectangle(self, xy, fill=None, outline=None, width=1):
            """Add rectangle to image (PIL-only)."""
            # 在图像上添加矩形（仅限 PIL）。
            self.draw.rectangle(xy, fill, outline, width)
    
        def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_style=False):
            """Adds text to an image using PIL or cv2."""
            # 使用 PIL 或 cv2 在图像上添加文本。
            if anchor == "bottom":  # start y from font bottom
                w, h = self.font.getsize(text)  # text width, height
                # 获取文本的宽度和高度
                xy[1] += 1 - h
                # 调整 y 坐标以从字体底部开始
            if self.pil:
                # 如果使用 PIL
                if box_style:
                    w, h = self.font.getsize(text)
                    self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                    # 使用 [txt_color](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/utils/plotting.py:288:4-311:44) 作为背景绘制矩形，并用白色绘制前景
                    txt_color = (255, 255, 255)
                if "\n" in text:
                    lines = text.split("\n")
                    _, h = self.font.getsize(text)
                    for line in lines:
                        self.draw.text(xy, line, fill=txt_color, font=self.font)
                        xy[1] += h
                else:
                    self.draw.text(xy, text, fill=txt_color, font=self.font)
            else:
                # 如果使用 cv2
                if box_style:
                    w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                    # 获取文本的宽度和高度
                    h += 3  # add pixels to pad text
                    # 为文本添加额外的像素以增加填充
                    outside = xy[1] >= h  # label fits outside box
                    # 检查标签是否适合框外
                    p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                    cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                    # 绘制填充的文本背景矩形
                    # 使用 [txt_color](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/utils/plotting.py:288:4-311:44) 作为背景绘制矩形，并用白色绘制前景
                    txt_color = (255, 255, 255)
                cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)
                # 在图像上绘制文本
    
        def fromarray(self, im):
            """Update self.im from a numpy array."""
            # 从 numpy 数组更新 self.im。
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            # 创建用于绘制的 ImageDraw 对象
    
        def result(self):
            """Return annotated image as array."""
            # 返回注释后的图像作为数组。
            return np.asarray(self.im)
    
        def show(self, title=None):
            """Show the annotated image."""
            # 显示注释后的图像。
            im = Image.fromarray(np.asarray(self.im)[..., ::-1])  # Convert numpy array to PIL Image with RGB to BGR
            # 将 numpy 数组转换为 PIL 图像，并将 RGB 转换为 BGR
            if IS_COLAB or IS_KAGGLE:  # can not use IS_JUPYTER as will run for all ipython environments
                # 如果在 Colab 或 Kaggle 中
                try:
                    display(im)  # noqa - display() function only available in ipython environments
                    # 显示图像
                except ImportError as e:
                    LOGGER.warning(f"Unable to display image in Jupyter notebooks: {e}")
                    # 如果无法在 Jupyter 笔记本中显示图像，则记录警告
            else:
                im.show(title=title)
                # 在默认图像查看器中显示图像
    
        def save(self, filename="image.jpg"):
            """Save the annotated image to 'filename'."""
            # 将注释后的图像保存到 'filename'。
            cv2.imwrite(filename, np.asarray(self.im))
    
        @staticmethod
        def get_bbox_dimension(bbox=None):
            """
            Calculate the area of a bounding box.
            计算边界框的面积。
    
            Args:
                bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
                bbox（元组）：边界框坐标，格式为 (x_min, y_min, x_max, y_max)。
    
            Returns:
                width (float): Width of the bounding box.
                width（浮点数）：边界框的宽度。
                height (float): Height of the bounding box.
                height（浮点数）：边界框的高度。
                area (float): Area enclosed by the bounding box.
                area（浮点数）：边界框所围成的面积。
            """
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            return width, height, width * height
            # 返回边界框的宽度、高度和面积
    
        def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
            """
            Draw region line.
            绘制区域线。
    
            Args:
                reg_pts (list): Region Points (for line 2 points, for region 4 points)
                reg_pts（列表）：区域点（对于线条为 2 个点，对于区域为 4 个点）
                color (tuple): Region Color value
                color（元组）：区域颜色值
                thickness (int): Region area thickness value
                thickness（整数）：区域厚度值
            """
            cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)
            # 在图像上绘制多边形区域
    
            # Draw small circles at the corner points
            for point in reg_pts:
                cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle
                # 在角点绘制小圆圈
    
        def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
            """
            Draw centroid point and track trails.
            绘制质心点和轨迹。
    
            Args:
                track (list): object tracking points for trails display
                track（列表）：用于显示轨迹的对象跟踪点
                color (tuple): tracks line color
                color（元组）：轨迹线颜色
                track_thickness (int): track line thickness value
                track_thickness（整数）：轨迹线厚度值
            """
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # 将轨迹点转换为适合绘制的格式
            cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
            # 绘制轨迹线
            cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)
            # 在最后一个轨迹点绘制圆圈
    
        def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0)):
            """
            Displays queue counts on an image centered at the points with customizable font size and colors.
            在图像上显示队列计数，居中于指定点，并可自定义字体大小和颜色。
    
            Args:
                label (str): Queue counts label.
                label（字符串）：队列计数标签。
                points (tuple): Region points for center point calculation to display text.
                points（元组）：用于计算文本显示中心点的区域点。
                region_color (tuple): RGB queue region color.
                region_color（元组）：RGB 队列区域颜色。
                txt_color (tuple): RGB text display color.
                txt_color（元组）：RGB 文本显示颜色。
            """
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            center_x = sum(x_values) // len(points)
            center_y = sum(y_values) // len(points)
            # 计算文本中心点坐标
    
            text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
            # 获取文本的宽度和高度
            text_width = text_size[0]
            text_height = text_size[1]
    
            rect_width = text_width + 20
            rect_height = text_height + 20
            rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
            rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
            cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)
            # 绘制文本背景矩形
    
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
    
            # Draw text
            cv2.putText(
                self.im,
                label,
                (text_x, text_y),
                0,
                fontScale=self.sf,
                color=txt_color,
                thickness=self.tf,
                lineType=cv2.LINE_AA,
            )
            # 在图像上绘制文本
    
        def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
            """
            Display the bounding boxes labels in parking management app.
            在停车管理应用中显示边界框标签。
    
            Args:
                im0 (ndarray): Inference image.
                im0（ndarray）：推理图像。
                text (str): Object/class name.
                text（字符串）：对象/类别名称。
                txt_color (tuple): Display color for text foreground.
                txt_color（元组）：文本前景显示颜色。
                bg_color (tuple): Display color for text background.
                bg_color（元组）：文本背景显示颜色。
                x_center (float): The x position center point for bounding box.
                x_center（浮点数）：边界框的 x 位置中心点。
                y_center (float): The y position center point for bounding box.
                y_center（浮点数）：边界框的 y 位置中心点。
                margin (int): The gap between text and rectangle for better display.
                margin（整数）：文本与矩形之间的间隙，以便更好地显示。
            """
            text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
            # 获取文本的宽度和高度
            text_x = x_center - text_size[0] // 2
            text_y = y_center + text_size[1] // 2
    
            rect_x1 = text_x - margin
            rect_y1 = text_y - text_size[1] - margin
            rect_x2 = text_x + text_size[0] + margin
            rect_y2 = text_y + margin
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            # 绘制文本背景矩形
            cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            # 在图像上绘制文本
            
def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
    """
    Display the bounding boxes labels in parking management app.
    在停车管理应用中显示边界框标签。

    Args:
        im0 (ndarray): Inference image.
        im0（ndarray）：推理图像。
        text (str): Object/class name.
        text（字符串）：对象/类别名称。
        txt_color (tuple): Display color for text foreground.
        txt_color（元组）：文本前景显示颜色。
        bg_color (tuple): Display color for text background.
        bg_color（元组）：文本背景显示颜色。
        x_center (float): The x position center point for bounding box.
        x_center（浮点数）：边界框的 x 位置中心点。
        y_center (float): The y position center point for bounding box.
        y_center（浮点数）：边界框的 y 位置中心点。
        margin (int): The gap between text and rectangle for better display.
        margin（整数）：文本与矩形之间的间隙，以便更好地显示。
    """
    text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # 获取文本的宽度和高度
    text_x = x_center - text_size[0] // 2  # 计算文本的 x 坐标，使其居中
    text_y = y_center + text_size[1] // 2  # 计算文本的 y 坐标，使其在中心点下方

    rect_x1 = text_x - margin  # 矩形左上角的 x 坐标
    rect_y1 = text_y - text_size[1] - margin  # 矩形左上角的 y 坐标
    rect_x2 = text_x + text_size[0] + margin  # 矩形右下角的 x 坐标
    rect_y2 = text_y + margin  # 矩形右下角的 y 坐标
    cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)  # 绘制文本背景矩形
    cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)  # 在图像上绘制文本

def display_analytics(self, im0, text, txt_color, bg_color, margin):
    """
    Display the overall statistics for parking lots.
    显示停车场的整体统计信息。

    Args:
        im0 (ndarray): Inference image.
        im0（ndarray）：推理图像。
        text (dict): Labels dictionary.
        text（字典）：标签字典。
        txt_color (tuple): Display color for text foreground.
        txt_color（元组）：文本前景显示颜色。
        bg_color (tuple): Display color for text background.
        bg_color（元组）：文本背景显示颜色。
        margin (int): Gap between text and rectangle for better display.
        margin（整数）：文本与矩形之间的间隙，以便更好地显示。
    """
    horizontal_gap = int(im0.shape[1] * 0.02)  # 计算水平间隙
    vertical_gap = int(im0.shape[0] * 0.01)  # 计算垂直间隙
    text_y_offset = 0  # 初始化文本的 y 偏移量
    for label, value in text.items():  # 遍历文本字典中的每个标签和对应的值
        txt = f"{label}: {value}"  # 格式化文本
        text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]  # 获取文本的大小
        if text_size[0] < 5 or text_size[1] < 5:  # 如果文本大小小于5，则设置为5
            text_size = (5, 5)
        text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap  # 计算文本的 x 坐标
        text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap  # 计算文本的 y 坐标
        rect_x1 = text_x - margin * 2  # 矩形的左上角 x 坐标
        rect_y1 = text_y - text_size[1] - margin * 2  # 矩形的左上角 y 坐标
        rect_x2 = text_x + text_size[0] + margin * 2  # 矩形的右下角 x 坐标
        rect_y2 = text_y + margin * 2  # 矩形的右下角 y 坐标
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)  # 绘制背景矩形
        cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)  # 在图像上绘制文本
        text_y_offset = rect_y2  # 更新 y 偏移量

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """
        Calculate the pose angle for object.
        计算物体的姿态角度。

        Args:
            a (float) : The value of pose point a
            a（浮动）: 姿态点 a 的值
            b (float): The value of pose point b
            b（浮动）: 姿态点 b 的值
            c (float): The value of pose point c
            c（浮动）: 姿态点 c 的值

        Returns:
            angle (degree): Degree value of angle between three points
            angle（度数）: 三个点之间的角度值
        """
        a, b, c = np.array(a), np.array(b), np.array(c)  # 将输入转换为 NumPy 数组
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])  # 计算弧度
        angle = np.abs(radians * 180.0 / np.pi)  # 将弧度转换为角度
        if angle > 180.0:  # 如果角度大于 180，则调整
            angle = 360 - angle
        return angle  # 返回角度

    def draw_specific_points(self, keypoints, indices=None, radius=2, conf_thres=0.25):
        """
        Draw specific keypoints for gym steps counting.
        绘制特定关键点以计数健身步骤。

        Args:
            keypoints (list): Keypoints data to be plotted.
            keypoints（列表）：要绘制的关键点数据。
            indices (list, optional): Keypoint indices to be plotted. Defaults to [2, 5, 7].
            indices（列表，可选）：要绘制的关键点索引。默认为 [2, 5, 7]。
            radius (int, optional): Keypoint radius. Defaults to 2.
            radius（整数，可选）：关键点半径。默认为 2。
            conf_thres (float, optional): Confidence threshold for keypoints. Defaults to 0.25.
            conf_thres（浮动，可选）：关键点的置信度阈值。默认为 0.25。

        Returns:
            (numpy.ndarray): Image with drawn keypoints.
            (numpy.ndarray): 绘制了关键点的图像。

        Note:
            Keypoint format: [x, y] or [x, y, confidence].
            关键点格式：[x, y] 或 [x, y, confidence]。
            Modifies self.im in-place.
            修改 self.im。
        """
        indices = indices or [2, 5, 7]  # 如果没有提供索引，则使用默认值
        points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thres]  # 根据置信度和索引筛选关键点

        # Draw lines between consecutive points
        for start, end in zip(points[:-1], points[1:]):  # 遍历相邻的关键点
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)  # 绘制连接线

        # Draw circles for keypoints
        for pt in points:  # 遍历关键点
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)  # 绘制关键点圆圈

        return self.im  # 返回绘制后的图像

    def plot_workout_information(self, display_text, position, color=(104, 31, 17), txt_color=(255, 255, 255)):
        """
        Draw text with a background on the image.
        在图像上绘制带背景的文本。
    
        Args:
            display_text (str): The text to be displayed.
            display_text（字符串）：要显示的文本。
            position (tuple): Coordinates (x, y) on the image where the text will be placed.
            position（元组）：图像上文本的位置坐标 (x, y)。
            color (tuple, optional): Text background color
            color（元组，可选）：文本背景颜色。
            txt_color (tuple, optional): Text foreground color
            txt_color（元组，可选）：文本前景颜色。
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, self.sf, self.tf)  # 获取文本的宽度和高度
    
        # Draw background rectangle
        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),  # 矩形的左上角坐标
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),  # 矩形的右下角坐标
            color,
            -1,
        )
        # Draw text
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)  # 在图像上绘制文本
    
        return text_height  # 返回文本高度
    
    def plot_angle_and_count_and_stage(
        self, angle_text, count_text, stage_text, center_kpt, color=(104, 31, 17), txt_color=(255, 255, 255)
    ):
        """
        Plot the pose angle, count value, and step stage.
        绘制姿态角度、计数值和步骤阶段。
    
        Args:
            angle_text (str): Angle value for workout monitoring
            angle_text（字符串）：用于健身监测的角度值。
            count_text (str): Counts value for workout monitoring
            count_text（字符串）：用于健身监测的计数值。
            stage_text (str): Stage decision for workout monitoring
            stage_text（字符串）：用于健身监测的阶段决策。
            center_kpt (list): Centroid pose index for workout monitoring
            center_kpt（列表）：用于健身监测的质心姿态索引。
            color (tuple, optional): Text background color
            color（元组，可选）：文本背景颜色。
            txt_color (tuple, optional): Text foreground color
            txt_color（元组，可选）：文本前景颜色。
        """
        # Format text
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"  # 格式化文本
    
        # Draw angle, count and stage text
        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color  # 绘制角度文本
        )
        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color  # 绘制计数文本
        )
        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color  # 绘制阶段文本
        )
    
    def seg_bbox(self, mask, mask_color=(255, 0, 255), label=None, txt_color=(255, 255, 255)):
        """
        Function for drawing segmented object in bounding box shape.
        绘制以边界框形状显示的分割对象。
    
        Args:
            mask (np.ndarray): A 2D array of shape (N, 2) containing the contour points of the segmented object.
            mask（np.ndarray）：形状为 (N, 2) 的二维数组，包含分割对象的轮廓点。
            mask_color (tuple): RGB color for the contour and label background.
            mask_color（元组）：轮廓和标签背景的RGB颜色。
            label (str, optional): Text label for the object. If None, no label is drawn.
            label（字符串，可选）：对象的文本标签。如果为None，则不绘制标签。
            txt_color (tuple): RGB color for the label text.
            txt_color（元组）：标签文本的RGB颜色。
        """
        if mask.size == 0:  # no masks to plot
            return  # 如果没有掩码，则返回
    
        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)  # 绘制轮廓
        if label:  # 如果提供了标签
            text_size, _ = cv2.getTextSize(label, 0, self.sf, self.tf)  # 获取标签文本的大小
            cv2.rectangle(
                self.im,
                (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),  # 矩形左上角坐标
                (int(mask[0][0]) + text_size[0] // 2 + 10, int(mask[0][1] + 10)),  # 矩形右下角坐标
                mask_color,
                -1,
            )
            cv2.putText(
                self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1])), 0, self.sf, txt_color, self.tf  # 绘制标签文本
            )
    
    def sweep_annotator(self, line_x=0, line_y=0, label=None, color=(221, 0, 186), txt_color=(255, 255, 255)):
        """
        Function for drawing a sweep annotation line and an optional label.
        绘制扫掠注释线和可选标签的函数。
    
        Args:
            line_x (int): The x-coordinate of the sweep line.
            line_x（整数）：扫掠线的x坐标。
            line_y (int): The y-coordinate limit of the sweep line.
            line_y（整数）：扫掠线的y坐标限制。
            label (str, optional): Text label to be drawn in center of sweep line. If None, no label is drawn.
            label（字符串，可选）：要绘制在扫掠线中心的文本标签。如果为None，则不绘制标签。
            color (tuple): RGB color for the line and label background.
            color（元组）：线和标签背景的RGB颜色。
            txt_color (tuple): RGB color for the label text.
            txt_color（元组）：标签文本的RGB颜色。
        """
        # Draw the sweep line
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)  # 绘制扫掠线
    
        # Draw label, if provided
        if label:  # 如果提供了标签
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)  # 获取标签文本的大小
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),  # 矩形左上角坐标
                (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),  # 矩形右下角坐标
                color,
                -1,
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),  # 绘制标签文本
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf,
            )
    
    def plot_distance_and_line(
        self, pixels_distance, centroids, line_color=(104, 31, 17), centroid_color=(255, 0, 255)
    ):
        """
        Plot the distance and line on frame.
        在帧上绘制距离和线条。
    
        Args:
            pixels_distance (float): Pixels distance between two bbox centroids.
            pixels_distance（浮点数）：两个边界框质心之间的像素距离。
            centroids (list): Bounding box centroids data.
            centroids（列表）：边界框质心数据。
            line_color (tuple, optional): Distance line color.
            line_color（元组，可选）：距离线的颜色。
            centroid_color (tuple, optional): Bounding box centroid color.
            centroid_color（元组，可选）：边界框质心的颜色。
        """
        # Get the text size
        text = f"Pixels Distance: {pixels_distance:.2f}"  # 格式化文本
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)  # 获取文本的宽度和高度
    
        # Define corners with 10-pixel margin and draw rectangle
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)  # 绘制背景矩形
    
        # Calculate the position for the text with a 10-pixel margin and draw text
        text_position = (25, 25 + text_height_m + 10)  # 计算文本位置
        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )  # 绘制文本
    
        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)  # 绘制连接两个质心的线
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)  # 绘制第一个质心
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)  # 绘制第二个质心
    
    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255)):
        """
        Function for pinpoint human-vision eye mapping and plotting.
        精确绘制人类视觉眼映射和绘图的函数。
    
        Args:
            box (list): Bounding box coordinates
            box（列表）：边界框坐标。
            center_point (tuple): center point for vision eye view
            center_point（元组）：视觉眼视图的中心点。
            color (tuple): object centroid and line color value
            color（元组）：对象质心和线条颜色值。
            pin_color (tuple): visioneye point color value
            pin_color（元组）：视觉眼点颜色值。
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)  # 计算边界框中心
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)  # 绘制视觉眼点
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)  # 绘制边界框中心
        cv2.line(self.im, center_point, center_bbox, color, self.tf)  # 绘制连接线


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.
    将图像裁剪保存为 {file}，裁剪大小乘以 {gain} 和 {pad} 像素。保存和/或返回裁剪图像。

    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.
    该函数接受一个边界框和一幅图像，然后根据边界框保存图像的裁剪部分。可选地，裁剪可以是正方形，并且该函数允许对边界框进行增益和填充调整。

    Args:
        xyxy (torch.Tensor or list): A tensor or list representing the bounding box in xyxy format.
        xyxy（torch.Tensor 或列表）：表示边界框的张量或列表，格式为 xyxy。
        im (numpy.ndarray): The input image.
        im（numpy.ndarray）：输入图像。
        file (Path, optional): The path where the cropped image will be saved. Defaults to 'im.jpg'.
        file（Path，可选）：裁剪图像将保存的路径。默认为 'im.jpg'。
        gain (float, optional): A multiplicative factor to increase the size of the bounding box. Defaults to 1.02.
        gain（浮点数，可选）：用于增加边界框大小的乘法因子。默认为 1.02。
        pad (int, optional): The number of pixels to add to the width and height of the bounding box. Defaults to 10.
        pad（整数，可选）：要添加到边界框宽度和高度的像素数。默认为 10。
        square (bool, optional): If True, the bounding box will be transformed into a square. Defaults to False.
        square（布尔值，可选）：如果为 True，则边界框将转换为正方形。默认为 False。
        BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB. Defaults to False.
        BGR（布尔值，可选）：如果为 True，则图像将以 BGR 格式保存，否则为 RGB。默认为 False。
        save (bool, optional): If True, the cropped image will be saved to disk. Defaults to True.
        save（布尔值，可选）：如果为 True，则裁剪的图像将保存到磁盘。默认为 True。

    Returns:
        (numpy.ndarray): The cropped image.
        (numpy.ndarray)：裁剪后的图像。

    Example:
        ```python
        from ultralytics.utils.plotting import save_one_box

        xyxy = [50, 50, 150, 150]
        im = cv2.imread("image.jpg")
        cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
        ```
    """
    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)  # 如果不是张量，可能是列表，将其转换为张量
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes 将 xyxy 格式的边界框转换为 xywh 格式
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square 尝试将矩形转换为正方形
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad 边界框宽高乘以增益并加上填充
    xyxy = ops.xywh2xyxy(b).long()  # 将 xywh 格式转换回 xyxy 格式并转换为长整型
    xyxy = ops.clip_boxes(xyxy, im.shape)  # 裁剪边界框以确保在图像范围内
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]  # 根据边界框裁剪图像
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory 创建目录
        f = str(increment_path(file).with_suffix(".jpg"))  # 生成文件路径并添加 .jpg 后缀
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB 保存为 RGB 格式
    return crop  # 返回裁剪后的图像

@threaded
def plot_images(
    images: Union[torch.Tensor, np.ndarray],
    batch_idx: Union[torch.Tensor, np.ndarray],
    cls: Union[torch.Tensor, np.ndarray],
    bboxes: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.float32),
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    masks: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.uint8),
    kpts: Union[torch.Tensor, np.ndarray] = np.zeros((0, 51), dtype=np.float32),
    paths: Optional[List[str]] = None,
    fname: str = "images.jpg",
    names: Optional[Dict[int, str]] = None,
    on_plot: Optional[Callable] = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Plot image grid with labels, bounding boxes, masks, and keypoints.
    绘制带标签、边界框、掩码和关键点的图像网格。

    Args:
        images: Batch of images to plot. Shape: (batch_size, channels, height, width).
        images：要绘制的图像批次。形状: (batch_size, channels, height, width)。
        batch_idx: Batch indices for each detection. Shape: (num_detections,).
        batch_idx：每个检测的批次索引。形状: (num_detections,)。
        cls: Class labels for each detection. Shape: (num_detections,).
        cls：每个检测的类别标签。形状: (num_detections,)。
        bboxes: Bounding boxes for each detection. Shape: (num_detections, 4) or (num_detections, 5) for rotated boxes.
        bboxes：每个检测的边界框。形状: (num_detections, 4) 或 (num_detections, 5) 用于旋转框。
        confs: Confidence scores for each detection. Shape: (num_detections,).
        confs：每个检测的置信度分数。形状: (num_detections,)。
        masks: Instance segmentation masks. Shape: (num_detections, height, width) or (1, height, width).
        masks：实例分割掩码。形状: (num_detections, height, width) 或 (1, height, width)。
        kpts: Keypoints for each detection. Shape: (num_detections, 51).
        kpts：每个检测的关键点。形状: (num_detections, 51)。
        paths: List of file paths for each image in the batch.
        paths：批次中每个图像的文件路径列表。
        fname: Output filename for the plotted image grid.
        fname：绘制的图像网格的输出文件名。
        names: Dictionary mapping class indices to class names.
        names：将类别索引映射到类别名称的字典。
        on_plot: Optional callback function to be called after saving the plot.
        on_plot：保存图像后调用的可选回调函数。
        max_size: Maximum size of the output image grid.
        max_size：输出图像网格的最大尺寸。
        max_subplots: Maximum number of subplots in the image grid.
        max_subplots：图像网格中的最大子图数量。
        save: Whether to save the plotted image grid to a file.
        save：是否将绘制的图像网格保存到文件。
        conf_thres: Confidence threshold for displaying detections.
        conf_thres：显示检测的置信度阈值。

    Returns:
        np.ndarray: Plotted image grid as a numpy array if save is False, None otherwise.
        np.ndarray：如果 save 为 False，则返回绘制的图像网格作为 numpy 数组，否则返回 None。
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()  # 将图像张量转换为 NumPy 数组
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()  # 将类别张量转换为 NumPy 数组
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()  # 将边界框张量转换为 NumPy 数组
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)  # 将掩码张量转换为 NumPy 数组并转换为整数类型
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()  # 将关键点张量转换为 NumPy 数组
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()  # 将批次索引张量转换为 NumPy 数组

    bs, _, h, w = images.shape  # batch size, _, height, width 获取批次大小、高度和宽度
    bs = min(bs, max_subplots)  # limit plot images 限制绘制的图像数量
    ns = np.ceil(bs**0.5)  # number of subplots (square) 计算子图数量（平方）
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional) 反归一化（可选）

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init 初始化图像网格
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin 计算块的原点
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)  # 将图像放入网格中

    # Resize (optional)
    scale = max_size / ns / max(h, w)  # 计算缩放因子
    if scale < 1:
        h = math.ceil(scale * h)  # 调整高度
        w = math.ceil(scale * w)  # 调整宽度
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))  # 调整图像网格大小

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size 计算字体大小
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)  # 创建注释器
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin 计算块的原点
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # 绘制边框
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # 绘制文件名
        if len(cls) > 0:
            idx = batch_idx == i  # 获取当前批次的索引
            classes = cls[idx].astype("int")  # 获取当前批次的类别
            labels = confs is None  # 检查是否有置信度

            if len(bboxes):
                boxes = bboxes[idx]  # 获取当前批次的边界框
                conf = confs[idx] if confs is not None else None  # 检查置信度是否存在
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1 如果经过归一化（容差 0.1）
                        boxes[..., [0, 2]] *= w  # scale to pixels 将边界框坐标缩放到像素
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales 如果图像缩放，绝对坐标需要缩放
                        boxes[..., :4] *= scale
                boxes[..., 0] += x  # 调整边界框的 x 坐标
                boxes[..., 1] += y  # 调整边界框的 y 坐标
                is_obb = boxes.shape[-1] == 5  # xywhr 检查是否为旋转边界框
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)  # 将边界框转换为 xyxy 格式
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]  # 获取类别
                    color = colors(c)  # 获取颜色
                    c = names.get(c, c) if names else c  # 获取类别名称
                    if labels or conf[j] > conf_thres:  # 检查是否需要绘制标签
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"  # 格式化标签
                        annotator.box_label(box, label, color=color, rotated=is_obb)  # 绘制边界框标签

            elif len(classes):
                for c in classes:
                    color = colors(c)  # 获取颜色
                    c = names.get(c, c) if names else c  # 获取类别名称
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)  # 绘制类别文本

            # Plot keypoints
            if len(kpts):
                kpts_ = kpts[idx].copy()  # 获取当前批次的关键点
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # if normalized with tolerance .01 如果经过归一化（容差 0.01）
                        kpts_[..., 0] *= w  # scale to pixels 将关键点坐标缩放到像素
                        kpts_[..., 1] *= h
                    elif scale < 1:  # absolute coords need scale if image scales 如果图像缩放，绝对坐标需要缩放
                        kpts_ *= scale
                kpts_[..., 0] += x  # 调整关键点的 x 坐标
                kpts_[..., 1] += y  # 调整关键点的 y 坐标
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:  # 检查是否需要绘制关键点
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)  # 绘制关键点

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]  # 获取当前批次的掩码
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()  # 计算当前批次的数量
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1  # 创建索引
                    image_masks = np.repeat(image_masks, nl, axis=0)  # 重复掩码
                    image_masks = np.where(image_masks == index, 1.0, 0.0)  # 根据索引创建掩码

                im = np.asarray(annotator.im).copy()  # 复制注释器图像
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:  # 检查是否需要绘制掩码
                        color = colors(classes[j])  # 获取颜色
                        mh, mw = image_masks[j].shape  # 获取掩码的高度和宽度
                        if mh != h or mw != w:  # 如果掩码的尺寸与图像不匹配
                            mask = image_masks[j].astype(np.uint8)  # 转换为无符号8位整数
                            mask = cv2.resize(mask, (w, h))  # 调整掩码大小
                            mask = mask.astype(bool)  # 转换为布尔类型
                        else:
                            mask = image_masks[j].astype(bool)  # 转换为布尔类型
                        try:
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )  # 将掩码区域的颜色混合
                        except Exception:
                            pass
                annotator.fromarray(im)  # 更新注释器图像
    if not save:
        return np.asarray(annotator.im)  # 如果不保存，则返回图像
    annotator.im.save(fname)  # 保存图像
    if on_plot:
        on_plot(fname)  # 调用回调函数



@plt_settings()
def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None):
    """
    Plot training results from a results CSV file. The function supports various types of data including segmentation,
    pose estimation, and classification. Plots are saved as 'results.png' in the directory where the CSV is located.
    从结果 CSV 文件中绘制训练结果。该函数支持多种类型的数据，包括分割、姿态估计和分类。图表将保存为 'results.png'，保存在 CSV 文件所在的目录中。

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to 'path/to/results.csv'.
        file（字符串，可选）：包含训练结果的 CSV 文件路径。默认为 'path/to/results.csv'。
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided. Defaults to ''.
        dir（字符串，可选）：如果未提供 'file'，则 CSV 文件所在的目录。默认为 ''。
        segment (bool, optional): Flag to indicate if the data is for segmentation. Defaults to False.
        segment（布尔值，可选）：指示数据是否用于分割的标志。默认为 False。
        pose (bool, optional): Flag to indicate if the data is for pose estimation. Defaults to False.
        pose（布尔值，可选）：指示数据是否用于姿态估计的标志。默认为 False。
        classify (bool, optional): Flag to indicate if the data is for classification. Defaults to False.
        classify（布尔值，可选）：指示数据是否用于分类的标志。默认为 False。
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.
            on_plot（可调用对象，可选）：绘图后执行的回调函数。接受文件名作为参数。默认为 None。

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results("path/to/results.csv", segment=True)
        ```
    """
    import pandas as pd  # scope for faster 'import ultralytics' 导入 pandas，以加快 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d  # 导入高斯滤波函数

    save_dir = Path(file).parent if file else Path(dir)  # 获取保存目录
    if classify:  # 如果是分类数据
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)  # 创建 2x2 的子图
        index = [2, 5, 3, 4]  # 指定要绘制的列索引
    elif segment:  # 如果是分割数据
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)  # 创建 2x8 的子图
        index = [2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 8, 9, 12, 13]  # 指定要绘制的列索引
    elif pose:  # 如果是姿态估计数据
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)  # 创建 2x9 的子图
        index = [2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 9, 10, 13, 14]  # 指定要绘制的列索引
    else:  # 默认情况
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)  # 创建 2x5 的子图
        index = [2, 3, 4, 5, 6, 9, 10, 11, 7, 8]  # 指定要绘制的列索引
    ax = ax.ravel()  # 将子图展平为一维数组
    files = list(save_dir.glob("results*.csv"))  # 获取所有以 results 开头的 CSV 文件
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."  # 检查是否找到文件
    for f in files:  # 遍历所有结果文件
        try:
            data = pd.read_csv(f)  # 读取 CSV 文件
            s = [x.strip() for x in data.columns]  # 获取列名并去除空格
            x = data.values[:, 0]  # 获取第一列数据
            for i, j in enumerate(index):  # 遍历要绘制的列索引
                y = data.values[:, j].astype("float")  # 获取当前列的数据并转换为浮点数
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results 绘制实际结果
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line 绘制平滑线
                ax[i].set_title(s[j], fontsize=12)  # 设置标题
                # if j in {8, 9, 10}:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.warning(f"WARNING: Plotting error for {f}: {e}")  # 记录绘图错误
    ax[1].legend()  # 添加图例
    fname = save_dir / "results.png"  # 保存结果图像的路径
    fig.savefig(fname, dpi=200)  # 保存图像
    plt.close()  # 关闭图像
    if on_plot:
        on_plot(fname)  # 调用回调函数


def plt_color_scatter(v, f, bins=20, cmap="viridis", alpha=0.8, edgecolors="none"):
    """
    Plots a scatter plot with points colored based on a 2D histogram.
    绘制散点图，点的颜色基于二维直方图。

    Args:
        v (array-like): Values for the x-axis.
        v（类数组）：x 轴的值。
        f (array-like): Values for the y-axis.
        f（类数组）：y 轴的值。
        bins (int, optional): Number of bins for the histogram. Defaults to 20.
        bins（整数，可选）：直方图的箱数。默认为 20。
        cmap (str, optional): Colormap for the scatter plot. Defaults to 'viridis'.
        cmap（字符串，可选）：散点图的颜色映射。默认为 'viridis'。
        alpha (float, optional): Alpha for the scatter plot. Defaults to 0.8.
        alpha（浮动，可选）：散点图的透明度。默认为 0.8。
        edgecolors (str, optional): Edge colors for the scatter plot. Defaults to 'none'.
        edgecolors（字符串，可选）：散点图的边缘颜色。默认为 'none'。

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    # Calculate 2D histogram and corresponding colors
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)  # 计算二维直方图及对应的颜色
    colors = [
        hist[
            min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1),  # 获取 x 轴的索引
            min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1),  # 获取 y 轴的索引
        ]
        for i in range(len(v))
    ]

    # Scatter plot
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)  # 绘制散点图


def plot_tune_results(csv_file="tune_results.csv"):
    """
    Plot the evolution results stored in a 'tune_results.csv' file. The function generates a scatter plot for each key
    in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on the plots.
    绘制存储在 'tune_results.csv' 文件中的演化结果。该函数为 CSV 中的每个键生成散点图，并根据适应度分数进行着色。最佳配置在图上突出显示。

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results. Defaults to 'tune_results.csv'.
        csv_file（字符串，可选）：包含调优结果的 CSV 文件路径。默认为 'tune_results.csv'。

    Examples:
        >>> plot_tune_results("path/to/tune_results.csv")
    """
    import pandas as pd  # scope for faster 'import ultralytics' 导入 pandas，以加快 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d  # 导入高斯滤波函数

    def _save_one_file(file):
        """Save one matplotlib plot to 'file'."""
        plt.savefig(file, dpi=200)  # 保存绘图
        plt.close()  # 关闭绘图
        LOGGER.info(f"Saved {file}")  # 记录保存信息

    # Scatter plots for each hyperparameter
    csv_file = Path(csv_file)  # 将 CSV 文件路径转换为 Path 对象
    data = pd.read_csv(csv_file)  # 读取 CSV 文件
    num_metrics_columns = 1  # 指标列数
    keys = [x.strip() for x in data.columns][num_metrics_columns:]  # 获取列名并去除空格
    x = data.values  # 获取数据值
    fitness = x[:, 0]  # fitness 获取适应度
    j = np.argmax(fitness)  # max fitness index 获取最大适应度的索引
    n = math.ceil(len(keys) ** 0.5)  # columns and rows in plot 计算绘图的列数和行数
    plt.figure(figsize=(10, 10), tight_layout=True)  # 创建绘图
    for i, k in enumerate(keys):  # 遍历每个超参数
        v = x[:, i + num_metrics_columns]  # 获取超参数的值
        mu = v[j]  # best single result 获取最佳单个结果
        plt.subplot(n, n, i + 1)  # 创建子图
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")  # 绘制散点图
        plt.plot(mu, fitness.max(), "k+", markersize=15)  # 绘制最佳结果
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # 设置标题
        plt.tick_params(axis="both", labelsize=8)  # 设置坐标轴标签大小
        if i % n != 0:
            plt.yticks([])  # 隐藏 y 轴标签
    _save_one_file(csv_file.with_name("tune_scatter_plots.png"))  # 保存散点图

    # Fitness vs iteration
    x = range(1, len(fitness) + 1)  # 迭代次数
    plt.figure(figsize=(10, 6), tight_layout=True)  # 创建绘图
    plt.plot(x, fitness, marker="o", linestyle="none", label="fitness")  # 绘制适应度
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ":", label="smoothed", linewidth=2)  # smoothing line 绘制平滑线
    plt.title("Fitness vs Iteration")  # 设置标题
    plt.xlabel("Iteration")  # 设置 x 轴标签
    plt.ylabel("Fitness")  # 设置 y 轴标签
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例
    _save_one_file(csv_file.with_name("tune_fitness.png"))  # 保存适应度图


def output_to_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []  # 初始化目标列表
    for i, o in enumerate(output):  # 遍历模型输出
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)  # 获取边界框、置信度和类别
        j = torch.full((conf.shape[0], 1), i)  # 创建批次 ID
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf), 1))  # 拼接目标
    targets = torch.cat(targets, 0).numpy()  # 拼接所有目标并转换为 numpy 数组
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]  # 返回目标格式


def output_to_rotated_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []  # 初始化目标列表
    for i, o in enumerate(output):  # 遍历模型输出
        box, conf, cls, angle = o[:max_det].cpu().split((4, 1, 1, 1), 1)  # 获取边界框、置信度、类别和角度
        j = torch.full((conf.shape[0], 1), i)  # 创建批次 ID
        targets.append(torch.cat((j, cls, box, angle, conf), 1))  # 拼接目标
    targets = torch.cat(targets, 0).numpy()  # 拼接所有目标并转换为 numpy 数组
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]  # 返回目标格式


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    Visualize feature maps of a given model module during inference.
    可视化给定模型模块在推理过程中的特征图。

    Args:
        x (torch.Tensor): Features to be visualized.
        x（torch.Tensor）：要可视化的特征。
        module_type (str): Module type.
        module_type（字符串）：模块类型。
        stage (int): Module stage within the model.
        stage（整数）：模型中的模块阶段。
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        n（整数，可选）：要绘制的最大特征图数量。默认为 32。
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
        save_dir（Path，可选）：保存结果的目录。默认为 Path('runs/detect/exp')。
    """
    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # all model heads
        if m in module_type:  # 如果模块类型是模型头之一
            return  # 不进行可视化
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # batch, channels, height, width 获取批次、通道、高度和宽度
        if height > 1 and width > 1:  # 确保高度和宽度大于1
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # 文件名

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # 选择批次索引 0，按通道分块
            n = min(n, channels)  # 绘图数量
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 行 x n/8 列
            ax = ax.ravel()  # 将子图展平为一维数组
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # 调整子图间距
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # 绘制特征图
                ax[i].axis("off")  # 关闭坐标轴

            LOGGER.info(f"Saving {f}... ({n}/{channels})")  # 记录保存信息
            plt.savefig(f, dpi=300, bbox_inches="tight")  # 保存图像
            plt.close()  # 关闭图像
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # 保存为 npy 格式