# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import emojis  # 从ultralytics.utils导入emojis模块，用于处理表情符号


class HUBModelError(Exception):
    """
    Custom exception class for handling errors related to model fetching in Ultralytics YOLO.
    自定义异常类，用于处理与Ultralytics YOLO中模型获取相关的错误。

    This exception is raised when a requested model is not found or cannot be retrieved.
    当请求的模型未找到或无法检索时，将引发此异常。
    The message is also processed to include emojis for better user experience.
    消息也会被处理，以包含表情符号，以改善用户体验。

    Attributes:
        message (str): The error message displayed when the exception is raised.
        message (str): 当异常被引发时显示的错误消息。

    Note:
        The message is automatically processed through the 'emojis' function from the 'ultralytics.utils' package.
        注意：消息会通过'ultralytics.utils'包中的'emojis'函数自动处理。
    """

    def __init__(self, message="Model not found. Please check model URL and try again."):
        """Create an exception for when a model is not found.
        创建一个模型未找到时的异常。"""
        super().__init__(emojis(message))  # 调用父类的构造函数，并处理消息以包含表情符号