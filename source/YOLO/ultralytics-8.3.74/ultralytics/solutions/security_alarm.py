# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution  # 从ultralytics的solutions模块导入BaseSolution类
from ultralytics.utils import LOGGER  # 从ultralytics的utils模块导入LOGGER，用于日志记录
from ultralytics.utils.plotting import Annotator, colors  # 从ultralytics的utils.plotting模块导入Annotator和colors

class SecurityAlarm(BaseSolution):
    """
    A class to manage security alarm functionalities for real-time monitoring.
    一个类，用于管理实时监控的安全警报功能。

    This class extends the BaseSolution class and provides features to monitor
    objects in a frame, send email notifications when specific thresholds are
    exceeded for total detections, and annotate the output frame for visualization.
    该类扩展了BaseSolution类，提供监控图像中的对象、当总检测超过特定阈值时发送电子邮件通知，并对输出帧进行注释以进行可视化的功能。

    Attributes:
       email_sent (bool): Flag to track if an email has already been sent for the current event.
       email_sent (bool): 标志，跟踪当前事件是否已发送电子邮件。
       records (int): Threshold for the number of detected objects to trigger an alert.
       records (int): 触发警报的检测对象数量的阈值。

    Methods:
       authenticate: Sets up email server authentication for sending alerts.
       authenticate: 设置电子邮件服务器认证以发送警报。
       send_email: Sends an email notification with details and an image attachment.
       send_email: 发送带有详细信息和图像附件的电子邮件通知。
       monitor: Monitors the frame, processes detections, and triggers alerts if thresholds are crossed.
       monitor: 监控帧，处理检测，并在阈值被超越时触发警报。

    Examples:
        >>> security = SecurityAlarm()
        >>> security.authenticate("abc@gmail.com", "1111222233334444", "xyz@gmail.com")
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = security.monitor(frame)
    """

    def __init__(self, **kwargs):
        """Initializes the SecurityAlarm class with parameters for real-time object monitoring."""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.email_sent = False  # 初始化电子邮件发送标志为False
        self.records = self.CFG["records"]  # 从配置中获取触发警报的记录阈值
        self.server = None  # 初始化电子邮件服务器为None
        self.to_email = ""  # 收件人电子邮件地址
        self.from_email = ""  # 发件人电子邮件地址

    def authenticate(self, from_email, password, to_email):
        """
        Authenticates the email server for sending alert notifications.

        Args:
            from_email (str): Sender's email address. 发件人电子邮件地址。
            password (str): Password for the sender's email account. 发件人电子邮件账户的密码。
            to_email (str): Recipient's email address. 收件人电子邮件地址。

        This method initializes a secure connection with the SMTP server
        and logs in using the provided credentials.
        此方法初始化与SMTP服务器的安全连接，并使用提供的凭据登录。

        Examples:
            >>> alarm = SecurityAlarm()
            >>> alarm.authenticate("sender@example.com", "password123", "recipient@example.com")
        """
        import smtplib  # 导入smtplib模块以处理电子邮件发送

        self.server = smtplib.SMTP("smtp.gmail.com: 587")  # 连接到Gmail SMTP服务器
        self.server.starttls()  # 启动TLS加密
        self.server.login(from_email, password)  # 使用发件人电子邮件和密码登录
        self.to_email = to_email  # 设置收件人电子邮件
        self.from_email = from_email  # 设置发件人电子邮件

    def send_email(self, im0, records=5):
        """
        Sends an email notification with an image attachment indicating the number of objects detected.

        Args:
            im0 (numpy.ndarray): The input image or frame to be attached to the email. 输入图像或帧，将附加到电子邮件中。
            records (int): The number of detected objects to be included in the email message. 要包含在电子邮件消息中的检测对象数量。

        This method encodes the input image, composes the email message with
        details about the detection, and sends it to the specified recipient.
        此方法将输入图像编码，构建包含检测详细信息的电子邮件消息，并将其发送到指定的收件人。

        Examples:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> alarm.send_email(frame, records=10)
        """
        from email.mime.image import MIMEImage  # 导入MIMEImage用于图像附件
        from email.mime.multipart import MIMEMultipart  # 导入MIMEMultipart用于创建多部分邮件
        from email.mime.text import MIMEText  # 导入MIMEText用于文本邮件内容

        import cv2  # 导入OpenCV库以处理图像

        img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()  # 将图像编码为JPEG格式

        # Create the email 创建电子邮件
        message = MIMEMultipart()  # 创建多部分邮件对象
        message["From"] = self.from_email  # 设置发件人
        message["To"] = self.to_email  # 设置收件人
        message["Subject"] = "Security Alert"  # 设置邮件主题

        # Add the text message body 添加文本邮件正文
        message_body = f"Ultralytics ALERT!!! {records} objects have been detected!!"  # 邮件正文内容
        message.attach(MIMEText(message_body))  # 将文本内容附加到邮件

        # Attach the image 附加图像
        image_attachment = MIMEImage(img_bytes, name="ultralytics.jpg")  # 创建图像附件
        message.attach(image_attachment)  # 将图像附件附加到邮件

        # Send the email 发送电子邮件
        try:
            self.server.send_message(message)  # 发送邮件
            LOGGER.info("✅ Email sent successfully!")  # 记录成功发送邮件的日志
        except Exception as e:
            print(f"❌ Failed to send email: {e}")  # 打印发送邮件失败的错误信息

    def monitor(self, im0):
        """
        Monitors the frame, processes object detections, and triggers alerts if thresholds are exceeded.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed and annotated. 输入图像或帧，需处理和注释。

        This method processes the input frame, extracts detections, annotates the frame
        with bounding boxes, and sends an email notification if the number of detected objects
        surpasses the specified threshold and an alert has not already been sent.
        此方法处理输入帧，提取检测结果，在帧上添加边界框注释，并在检测到的对象数量超过指定阈值且未发送警报时发送电子邮件通知。

        Returns:
            (numpy.ndarray): The processed frame with annotations. 处理后的帧，包含注释。

        Examples:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = alarm.monitor(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化注释器
        self.extract_tracks(im0)  # 从图像中提取轨迹

        # Iterate over bounding boxes, track ids and classes index 遍历边界框、轨迹ID和类别索引
        for box, cls in zip(self.boxes, self.clss):  # 遍历每个边界框和类别
            # Draw bounding box 绘制边界框
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))  # 在边界框上绘制标签

        total_det = len(self.clss)  # 计算检测到的对象总数
        if total_det > self.records and not self.email_sent:  # 如果检测到的对象总数超过阈值且未发送邮件
            self.send_email(im0, total_det)  # 发送邮件
            self.email_sent = True  # 设置邮件已发送标志

        self.display_output(im0)  # display output with base class function 使用基类函数显示输出

        return im0  # return output image for more usage 返回处理后的图像以供进一步使用