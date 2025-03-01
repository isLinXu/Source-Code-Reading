# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator


class AIGym(BaseSolution):
    """
    A class to manage gym steps of people in a real-time video stream based on their poses.
    一个类，用于根据人们的姿势在实时视频流中管理健身步骤。

    This class extends BaseSolution to monitor workouts using YOLO pose estimation models. It tracks and counts
    repetitions of exercises based on predefined angle thresholds for up and down positions.
    此类扩展了 BaseSolution，以使用 YOLO 姿态估计模型监控锻炼。它根据预定义的角度阈值跟踪和计数锻炼的重复次数。

    Attributes:
        count (List[int]): Repetition counts for each detected person.
        count (List[int]): 每个检测到的人的重复计数。
        angle (List[float]): Current angle of the tracked body part for each person.
        angle (List[float]): 每个人跟踪的身体部位的当前角度。
        stage (List[str]): Current exercise stage ('up', 'down', or '-') for each person.
        stage (List[str]): 每个人当前的锻炼阶段（'up'、'down' 或 '-'）。
        initial_stage (str | None): Initial stage of the exercise.
        initial_stage (str | None): 锻炼的初始阶段。
        up_angle (float): Angle threshold for considering the 'up' position of an exercise.
        up_angle (float): 用于考虑锻炼的“上”位置的角度阈值。
        down_angle (float): Angle threshold for considering the 'down' position of an exercise.
        down_angle (float): 用于考虑锻炼的“下”位置的角度阈值。
        kpts (List[int]): Indices of keypoints used for angle calculation.
        kpts (List[int]): 用于角度计算的关键点索引。
        annotator (Annotator): Object for drawing annotations on the image.
        annotator (Annotator): 用于在图像上绘制注释的对象。

    Methods:
        monitor: Processes a frame to detect poses, calculate angles, and count repetitions.
        monitor: 处理帧以检测姿势、计算角度和计数重复次数。

    Examples:
        >>> gym = AIGym(model="yolo11n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> processed_image = gym.monitor(image)
        >>> cv2.imshow("Processed Image", processed_image)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs):
        """Initializes AIGym for workout monitoring using pose estimation and predefined angles.
        初始化 AIGym 以使用姿态估计和预定义角度进行锻炼监控。"""
        # Check if the model name ends with '-pose'
        # 检查模型名称是否以 '-pose' 结尾
        if "model" in kwargs and "-pose" not in kwargs["model"]:
            kwargs["model"] = "yolo11n-pose.pt"  # 默认模型
        elif "model" not in kwargs:
            kwargs["model"] = "yolo11n-pose.pt"  # 默认模型

        super().__init__(**kwargs)  # 调用父类构造函数
        self.count = []  # List for counts, necessary where there are multiple objects in frame
        self.angle = []  # List for angle, necessary where there are multiple objects in frame
        self.stage = []  # List for stage, necessary where there are multiple objects in frame

        # Extract details from CFG single time for usage later
        # 从配置中提取详细信息以供后续使用
        self.initial_stage = None  # 初始阶段
        self.up_angle = float(self.CFG["up_angle"])  # Pose up predefined angle to consider up pose
        self.down_angle = float(self.CFG["down_angle"])  # Pose down predefined angle to consider down pose
        self.kpts = self.CFG["kpts"]  # 用户选择的锻炼关键点存储以供进一步使用

    def monitor(self, im0):
        """
        Monitors workouts using Ultralytics YOLO Pose Model.
        使用 Ultralytics YOLO 姿态模型监控锻炼。

        This function processes an input image to track and analyze human poses for workout monitoring. It uses
        the YOLO Pose model to detect keypoints, estimate angles, and count repetitions based on predefined
        angle thresholds.
        此函数处理输入图像以跟踪和分析人类姿势以进行锻炼监控。它使用 YOLO 姿态模型检测关键点，估计角度，并根据预定义的角度阈值计数重复次数。

        Args:
            im0 (ndarray): Input image for processing.
            im0 (ndarray): 输入图像以进行处理。

        Returns:
            (ndarray): Processed image with annotations for workout monitoring.
            (ndarray): 带有锻炼监控注释的处理图像。
        
        Examples:
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> processed_image = gym.monitor(image)
        """
        # Extract tracks
        # 提取跟踪信息
        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)[0]

        if tracks.boxes.id is not None:
            # Extract and check keypoints
            # 提取并检查关键点
            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)  # 新检测到的人数
                self.angle += [0] * new_human  # 初始化角度
                self.count += [0] * new_human  # 初始化计数
                self.stage += ["-"] * new_human  # 初始化阶段

            # Initialize annotator
            # 初始化注释器
            self.annotator = Annotator(im0, line_width=self.line_width)

            # Enumerate over keypoints
            # 遍历关键点
            for ind, k in enumerate(reversed(tracks.keypoints.data)):
                # Get keypoints and estimate the angle
                # 获取关键点并估计角度
                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]  # 获取关键点
                self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)  # 估计角度
                im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.line_width * 3)  # 绘制关键点

                # Determine stage and count logic based on angle thresholds
                # 根据角度阈值确定阶段和计数逻辑
                if self.angle[ind] < self.down_angle:
                    if self.stage[ind] == "up":
                        self.count[ind] += 1  # 增加计数
                    self.stage[ind] = "down"  # 设置阶段为下
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"  # 设置阶段为上

                # Display angle, count, and stage text
                # 显示角度、计数和阶段文本
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],  # 显示的角度文本
                    count_text=self.count[ind],  # 显示的计数文本
                    stage_text=self.stage[ind],  # 阶段位置文本
                    center_kpt=k[int(self.kpts[1])],  # 用于显示的中心关键点
                )

        self.display_output(im0)  # 显示输出图像（如果环境支持显示）
        return im0  # 返回图像以供写入或进一步使用