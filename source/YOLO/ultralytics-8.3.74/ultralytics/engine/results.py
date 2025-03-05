# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy  # 从copy模块导入deepcopy函数
from functools import lru_cache  # 从functools模块导入lru_cache装饰器
from pathlib import Path  # 从pathlib模块导入Path类

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库

from ultralytics.data.augment import LetterBox  # 从ultralytics.data.augment导入LetterBox类
from ultralytics.utils import LOGGER, SimpleClass, ops  # 从ultralytics.utils导入LOGGER、SimpleClass和ops
from ultralytics.utils.checks import check_requirements  # 从ultralytics.utils.checks导入check_requirements函数
from ultralytics.utils.plotting import Annotator, colors, save_one_box  # 从ultralytics.utils.plotting导入Annotator、colors和save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode  # 从ultralytics.utils.torch_utils导入smart_inference_mode函数


class BaseTensor(SimpleClass):
    """
    Base tensor class with additional methods for easy manipulation and device handling.  # 基础张量类，具有额外方法以方便操作和设备管理

    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.  # 数据属性，存储预测数据，如边界框、掩码或关键点
        orig_shape (Tuple[int, int]): Original shape of the image, typically in the format (height, width).  # 原始图像形状属性，通常格式为（高度，宽度）

    Methods:
        cpu: Return a copy of the tensor stored in CPU memory.  # 返回存储在CPU内存中的张量副本
        numpy: Returns a copy of the tensor as a numpy array.  # 返回张量的NumPy数组副本
        cuda: Moves the tensor to GPU memory, returning a new instance if necessary.  # 将张量移动到GPU内存，如果需要返回新实例
        to: Return a copy of the tensor with the specified device and dtype.  # 返回具有指定设备和数据类型的张量副本

    Examples:
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image.  # 用预测数据和原始图像形状初始化BaseTensor

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.  # 数据参数，存储预测数据，如边界框、掩码或关键点
            orig_shape (Tuple[int, int]): Original shape of the image in (height, width) format.  # orig_shape参数，原始图像形状，格式为（高度，宽度）

        Examples:
            >>> import torch
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"  # 确保数据是torch.Tensor或np.ndarray类型
        self.data = data  # 将数据赋值给self.data
        self.orig_shape = orig_shape  # 将原始形状赋值给self.orig_shape

    @property
    def shape(self):
        """
        Returns the shape of the underlying data tensor.  # 返回底层数据张量的形状

        Returns:
            (Tuple[int, ...]): The shape of the data tensor.  # 返回数据张量的形状元组

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape  # 返回数据张量的形状

    def cpu(self):
        """
        Returns a copy of the tensor stored in CPU memory.  # 返回存储在CPU内存中的张量副本

        Returns:
            (BaseTensor): A new BaseTensor object with the data tensor moved to CPU memory.  # 返回一个新的BaseTensor对象，其数据张量移动到CPU内存

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)  # 如果数据是NumPy数组，返回自身；否则返回一个新的BaseTensor对象

    def numpy(self):
        """
        Returns a copy of the tensor as a numpy array.  # 返回张量的NumPy数组副本

        Returns:
            (np.ndarray): A numpy array containing the same data as the original tensor.  # 返回一个包含与原始张量相同数据的NumPy数组

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_array = base_tensor.numpy()
            >>> print(type(numpy_array))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)  # 如果数据是NumPy数组，返回自身；否则返回一个新的BaseTensor对象

    def cuda(self):
        """
        Moves the tensor to GPU memory.  # 将张量移动到GPU内存

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to GPU memory if it's not already a
                numpy array, otherwise returns self.  # 如果数据不是NumPy数组，则返回一个新的BaseTensor实例，数据移动到GPU内存；否则返回自身

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)  # 返回一个新的BaseTensor对象，数据移动到GPU内存

    def to(self, *args, **kwargs):
        """
        Return a copy of the tensor with the specified device and dtype.  # 返回具有指定设备和数据类型的张量副本

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().  # 可变长度参数列表，将传递给torch.Tensor.to()方法
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().  # 任意关键字参数，将传递给torch.Tensor.to()方法

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to the specified device and/or dtype.  # 返回一个新的BaseTensor实例，数据移动到指定设备和/或数据类型

        Examples:
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)  # 返回一个新的BaseTensor对象，数据移动到指定设备和/或数据类型

    def __len__(self):  # override len(results)
        """
        Returns the length of the underlying data tensor.  # 返回底层数据张量的长度

        Returns:
            (int): The number of elements in the first dimension of the data tensor.  # 返回数据张量第一维的元素数量

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)  # 返回数据张量的长度

    def __getitem__(self, idx):
        """
        Returns a new BaseTensor instance containing the specified indexed elements of the data tensor.  # 返回一个新的BaseTensor实例，包含数据张量中指定索引的元素

        Args:
            idx (int | List[int] | torch.Tensor): Index or indices to select from the data tensor.  # 索引或索引列表，用于从数据张量中选择元素

        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed data.  # 返回一个新的BaseTensor实例，包含索引数据

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)  # 返回一个新的BaseTensor对象，包含指定索引的数据
class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.  # 用于存储和处理推理结果的类

    This class encapsulates the functionality for handling detection, segmentation, pose estimation,
    and classification results from YOLO models.  # 此类封装了处理YOLO模型的检测、分割、姿态估计和分类结果的功能

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.  # 原始图像，作为NumPy数组
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format.  # 原始图像形状，格式为（高度，宽度）
        boxes (Boxes | None): Object containing detection bounding boxes.  # 包含检测边界框的对象
        masks (Masks | None): Object containing detection masks.  # 包含检测掩码的对象
        probs (Probs | None): Object containing class probabilities for classification tasks.  # 包含分类任务的类概率的对象
        keypoints (Keypoints | None): Object containing detected keypoints for each object.  # 包含每个对象的检测关键点的对象
        obb (OBB | None): Object containing oriented bounding boxes.  # 包含定向边界框的对象
        speed (Dict[str, float | None]): Dictionary of preprocess, inference, and postprocess speeds.  # 预处理、推理和后处理速度的字典
        names (Dict[int, str]): Dictionary mapping class IDs to class names.  # 将类ID映射到类名称的字典
        path (str): Path to the image file.  # 图像文件的路径
        _keys (Tuple[str, ...]): Tuple of attribute names for internal use.  # 用于内部使用的属性名称元组

    Methods:
        update: Updates object attributes with new detection results.  # 用新的检测结果更新对象属性
        cpu: Returns a copy of the Results object with all tensors on CPU memory.  # 返回一个副本，所有张量在CPU内存中
        numpy: Returns a copy of the Results object with all tensors as numpy arrays.  # 返回一个副本，所有张量作为NumPy数组
        cuda: Returns a copy of the Results object with all tensors on GPU memory.  # 返回一个副本，所有张量在GPU内存中
        to: Returns a copy of the Results object with tensors on a specified device and dtype.  # 返回一个副本，张量在指定设备和数据类型上
        new: Returns a new Results object with the same image, path, and names.  # 返回一个新的Results对象，具有相同的图像、路径和名称
        plot: Plots detection results on an input image, returning an annotated image.  # 在输入图像上绘制检测结果，返回带注释的图像
        show: Shows annotated results on screen.  # 在屏幕上显示注释结果
        save: Saves annotated results to file.  # 将注释结果保存到文件
        verbose: Returns a log string for each task, detailing detections and classifications.  # 返回每个任务的日志字符串，详细说明检测和分类
        save_txt: Saves detection results to a text file.  # 将检测结果保存到文本文件
        save_crop: Saves cropped detection images.  # 保存裁剪的检测图像
        tojson: Converts detection results to JSON format.  # 将检测结果转换为JSON格式

    Examples:
        >>> results = model("path/to/image.jpg")  # 使用模型进行推理
        >>> for result in results:
        ...     print(result.boxes)  # 打印检测框
        ...     result.show()  # 显示带注释的图像
        ...     result.save(filename="result.jpg")  # 保存带注释的图像
    """

    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.  # 初始化Results类以存储和处理推理结果

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.  # 原始图像，作为NumPy数组
            path (str): The path to the image file.  # 图像文件的路径
            names (Dict): A dictionary of class names.  # 类名称的字典
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.  # 每个检测的边界框坐标的2D张量
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image.  # 检测掩码的3D张量，每个掩码是一个二进制图像
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification task.  # 每个类的概率的1D张量，用于分类任务
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection.  # 每个检测的关键点坐标的2D张量
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection.  # 每个检测的定向边界框坐标的2D张量
            speed (Dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).  # 包含预处理、推理和后处理速度的字典（毫秒/图像）

        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> result = results[0]  # 获取第一个结果
            >>> boxes = result.boxes  # 获取第一个结果的边界框
            >>> masks = result.masks  # 获取第一个结果的掩码

        Notes:
            For the default pose model, keypoint indices for human body pose estimation are:  # 对于默认姿态模型，人体姿态估计的关键点索引为：
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear  # 0: 鼻子，1: 左眼，2: 右眼，3: 左耳，4: 右耳
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow  # 5: 左肩，6: 右肩，7: 左肘，8: 右肘
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip  # 9: 左腕，10: 右腕，11: 左髋，12: 右髋
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle  # 13: 左膝，14: 右膝，15: 左踝，16: 右踝
        """
        self.orig_img = orig_img  # 将原始图像赋值给self.orig_img
        self.orig_shape = orig_img.shape[:2]  # 获取原始图像的形状并赋值给self.orig_shape
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # 如果有边界框，则创建Boxes对象
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # 如果有掩码，则创建Masks对象
        self.probs = Probs(probs) if probs is not None else None  # 如果有概率，则创建Probs对象
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None  # 如果有关键点，则创建Keypoints对象
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None  # 如果有定向边界框，则创建OBB对象
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}  # 设置速度属性
        self.names = names  # 将类名称赋值给self.names
        self.path = path  # 将路径赋值给self.path
        self.save_dir = None  # 初始化保存目录为None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"  # 初始化内部使用的属性名称元组

    def __getitem__(self, idx):
        """
        Return a Results object for a specific index of inference results.  # 返回特定索引的推理结果的Results对象

        Args:
            idx (int | slice): Index or slice to retrieve from the Results object.  # 索引或切片，用于从Results对象中检索

        Returns:
            (Results): A new Results object containing the specified subset of inference results.  # 返回一个新的Results对象，包含指定子集的推理结果

        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> single_result = results[0]  # 获取第一个结果
            >>> subset_results = results[1:4]  # 获取结果的切片
        """
        return self._apply("__getitem__", idx)  # 调用_apply方法，返回指定索引的结果

    def __len__(self):
        """
        Return the number of detections in the Results object.  # 返回Results对象中的检测数量

        Returns:
            (int): The number of detections, determined by the length of the first non-empty attribute
                (boxes, masks, probs, keypoints, or obb).  # 返回检测数量，取决于第一个非空属性的长度（边界框、掩码、概率、关键点或定向边界框）

        Examples:
            >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 4))  # 创建Results对象
            >>> len(results)  # 获取检测数量
            5
        """
        for k in self._keys:  # 遍历所有关键属性
            v = getattr(self, k)  # 获取属性值
            if v is not None:  # 如果属性值不为None
                return len(v)  # 返回属性值的长度

    def update(self, boxes=None, masks=None, probs=None, obb=None, keypoints=None):
        """
        Updates the Results object with new detection data.  # 用新的检测数据更新Results对象

        This method allows updating the boxes, masks, probabilities, and oriented bounding boxes (OBB) of the
        Results object. It ensures that boxes are clipped to the original image shape.  # 此方法允许更新Results对象的边界框、掩码、概率和定向边界框（OBB），并确保边界框裁剪到原始图像形状

        Args:
            boxes (torch.Tensor | None): A tensor of shape (N, 6) containing bounding box coordinates and
                confidence scores. The format is (x1, y1, x2, y2, conf, class).  # 边界框参数，包含边界框坐标和置信度的张量
            masks (torch.Tensor | None): A tensor of shape (N, H, W) containing segmentation masks.  # 掩码参数，包含分割掩码的张量
            probs (torch.Tensor | None): A tensor of shape (num_classes,) containing class probabilities.  # 概率参数，包含每个类概率的张量
            obb (torch.Tensor | None): A tensor of shape (N, 5) containing oriented bounding box coordinates.  # 定向边界框参数，包含定向边界框坐标的张量
            keypoints (torch.Tensor | None): A tensor of shape (N, 17, 3) containing keypoints.  # 关键点参数，包含关键点坐标的张量

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])  # 创建新的边界框
            >>> results[0].update(boxes=new_boxes)  # 更新第一个结果的边界框
        """
        if boxes is not None:  # 如果提供了边界框
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)  # 更新边界框并裁剪到原始形状
        if masks is not None:  # 如果提供了掩码
            self.masks = Masks(masks, self.orig_shape)  # 更新掩码
        if probs is not None:  # 如果提供了概率
            self.probs = probs  # 更新概率
        if obb is not None:  # 如果提供了定向边界框
            self.obb = OBB(obb, self.orig_shape)  # 更新定向边界框
        if keypoints is not None:  # 如果提供了关键点
            self.keypoints = Keypoints(keypoints, self.orig_shape)  # 更新关键点

    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes.  # 将函数应用于所有非空属性，并返回一个新的Results对象，具有修改后的属性

        This method is internally called by methods like .to(), .cuda(), .cpu(), etc.  # 此方法由.to()、.cuda()、.cpu()等方法内部调用

        Args:
            fn (str): The name of the function to apply.  # 要应用的函数名称
            *args (Any): Variable length argument list to pass to the function.  # 可变长度参数列表，将传递给函数
            **kwargs (Any): Arbitrary keyword arguments to pass to the function.  # 任意关键字参数，将传递给函数

        Returns:
            (Results): A new Results object with attributes modified by the applied function.  # 返回一个新的Results对象，属性由应用的函数修改

        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            ...     result_cuda = result.cuda()  # 将每个结果移动到GPU
            ...     result_cpu = result.cpu()  # 将每个结果移动到CPU
        """
        r = self.new()  # 创建一个新的Results对象
        for k in self._keys:  # 遍历所有关键属性
            v = getattr(self, k)  # 获取属性值
            if v is not None:  # 如果属性值不为None
                setattr(r, k, getattr(v, fn)(*args, **kwargs))  # 将应用函数的结果赋值给新对象的对应属性
        return r  # 返回新的Results对象

    def cpu(self):
        """
        Returns a copy of the Results object with all its tensors moved to CPU memory.  # 返回一个副本，所有张量在CPU内存中

        This method creates a new Results object with all tensor attributes (boxes, masks, probs, keypoints, obb)
        transferred to CPU memory. It's useful for moving data from GPU to CPU for further processing or saving.  # 此方法创建一个新的Results对象，所有张量属性（边界框、掩码、概率、关键点、定向边界框）转移到CPU内存中，适用于将数据从GPU移动到CPU以进行进一步处理或保存

        Returns:
            (Results): A new Results object with all tensor attributes on CPU memory.  # 返回一个新的Results对象，所有张量属性在CPU内存中

        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> cpu_result = results[0].cpu()  # 将第一个结果移动到CPU
            >>> print(cpu_result.boxes.device)  # 输出: cpu
        """
        return self._apply("cpu")  # 调用_apply方法，返回所有张量在CPU内存中的副本

    def numpy(self):
        """
        Converts all tensors in the Results object to numpy arrays.  # 将Results对象中的所有张量转换为NumPy数组

        Returns:
            (Results): A new Results object with all tensors converted to numpy arrays.  # 返回一个新的Results对象，所有张量转换为NumPy数组

        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> numpy_result = results[0].numpy()  # 获取第一个结果的NumPy数组
            >>> type(numpy_result.boxes.data)  # 输出: <class 'numpy.ndarray'>

        Notes:
            This method creates a new Results object, leaving the original unchanged. It's useful for
            interoperability with numpy-based libraries or when CPU-based operations are required.  # 此方法创建一个新的Results对象，原始对象保持不变，适用于与基于NumPy的库的互操作性或需要CPU操作时
        """
        return self._apply("numpy")  # 调用_apply方法，返回所有张量转换为NumPy数组的副本

    def cuda(self):
        """
        Moves all tensors in the Results object to GPU memory.  # 将Results对象中的所有张量移动到GPU内存

        Returns:
            (Results): A new Results object with all tensors moved to CUDA device.  # 返回一个新的Results对象，所有张量在CUDA设备上

        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> cuda_results = results[0].cuda()  # 将第一个结果移动到GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # 将每个结果移动到GPU
        """
        return self._apply("cuda")  # 调用_apply方法，返回所有张量在GPU内存中的副本

    def to(self, *args, **kwargs):
        """
        Moves all tensors in the Results object to the specified device and dtype.  # 将Results对象中的所有张量移动到指定设备和数据类型

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().  # 可变长度参数列表，将传递给torch.Tensor.to()方法
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().  # 任意关键字参数，将传递给torch.Tensor.to()方法

        Returns:
            (Results): A new Results object with all tensors moved to the specified device and dtype.  # 返回一个新的Results对象，所有张量在指定设备和数据类型上

        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> result_cuda = results[0].to("cuda")  # 将第一个结果移动到GPU
            >>> result_cpu = results[0].to("cpu")  # 将第一个结果移动到CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # 将第一个结果转换为半精度
        """
        return self._apply("to", *args, **kwargs)  # 调用_apply方法，返回所有张量在指定设备和数据类型上的副本

    def new(self):
        """
        Creates a new Results object with the same image, path, names, and speed attributes.  # 创建一个新的Results对象，具有相同的图像、路径、名称和速度属性

        Returns:
            (Results): A new Results object with copied attributes from the original instance.  # 返回一个新的Results对象，具有原始实例的属性副本

        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> new_result = results[0].new()  # 创建一个新的Results对象
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)  # 返回一个新的Results对象，具有相同的属性

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
    ):
        """
        Plots detection results on an input RGB image.  # 在输入RGB图像上绘制检测结果

        Args:
            conf (bool): Whether to plot detection confidence scores.  # 是否绘制检测置信度分数
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.  # 边界框的线宽，如果为None，则根据图像大小缩放
            font_size (float | None): Font size for text. If None, scaled to image size.  # 文本的字体大小，如果为None，则根据图像大小缩放
            font (str): Font to use for text.  # 用于文本的字体
            pil (bool): Whether to return the image as a PIL Image.  # 是否将图像作为PIL图像返回
            img (np.ndarray | None): Image to plot on. If None, uses original image.  # 要绘制的图像，如果为None，则使用原始图像
            im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.  # GPU上的标准化图像，以便更快地绘制掩码
            kpt_radius (int): Radius of drawn keypoints.  # 绘制关键点的半径
            kpt_line (bool): Whether to draw lines connecting keypoints.  # 是否绘制连接关键点的线
            labels (bool): Whether to plot labels of bounding boxes.  # 是否绘制边界框的标签
            boxes (bool): Whether to plot bounding boxes.  # 是否绘制边界框
            masks (bool): Whether to plot masks.  # 是否绘制掩码
            probs (bool): Whether to plot classification probabilities.  # 是否绘制分类概率
            show (bool): Whether to display the annotated image.  # 是否显示带注释的图像
            save (bool): Whether to save the annotated image.  # 是否保存带注释的图像
            filename (str | None): Filename to save image if save is True.  # 如果保存为True，则保存图像的文件名
            color_mode (bool): Specify the color mode, e.g., 'instance' or 'class'. Default to 'class'.  # 指定颜色模式，例如“实例”或“类”。默认为“类”。

        Returns:
            (np.ndarray): Annotated image as a numpy array.  # 返回带注释的图像，作为NumPy数组

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     im = result.plot()  # 绘制结果
            >>>     im.show()  # 显示图像
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."  # 确保颜色模式是“实例”或“类”
        if img is None and isinstance(self.orig_img, torch.Tensor):  # 如果未提供图像且原始图像是张量
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()  # 将张量转换为NumPy数组

        names = self.names  # 获取类名称
        is_obb = self.obb is not None  # 检查是否有定向边界框
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes  # 获取预测边界框
        pred_masks, show_masks = self.masks, masks  # 获取预测掩码
        pred_probs, show_probs = self.probs, probs  # 获取预测概率
        annotator = Annotator(  # 创建Annotator实例
            deepcopy(self.orig_img if img is None else img),  # 使用原始图像或提供的图像
            line_width,  # 线宽
            font_size,  # 字体大小
            font,  # 字体
            pil or (pred_probs is not None and show_probs),  # 如果是分类任务，默认设置pil为True
            example=names,  # 类名称示例
        )

        # Plot Segment results
        if pred_masks and show_masks:  # 如果有掩码并且需要显示掩码
            if im_gpu is None:  # 如果没有GPU图像
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())  # 调整图像大小以适应掩码
                im_gpu = (  # 创建标准化的GPU图像
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)  # 转换为张量
                    .permute(2, 0, 1)  # 调整维度
                    .flip(0)  # 翻转第一个维度
                    .contiguous()  # 确保张量是连续的
                    / 255  # 归一化
                )
            idx = (  # 获取索引
                pred_boxes.id
                if pred_boxes.id is not None and color_mode == "instance"  # 如果是实例模式
                else pred_boxes.cls  # 否则使用类索引
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))  # 反转范围
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)  # 绘制掩码

        # Plot Detect results
        if pred_boxes is not None and show_boxes:  # 如果有预测边界框并且需要显示边界框
            for i, d in enumerate(reversed(pred_boxes)):  # 遍历预测边界框
                c, d_conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())  # 获取类、置信度和ID
                name = ("" if id is None else f"id:{id} ") + names[c]  # 获取名称
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None  # 获取标签
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()  # 获取边界框
                annotator.box_label(  # 绘制边界框标签
                    box,
                    label,
                    color=colors(  # 设置颜色
                        c
                        if color_mode == "class"  # 如果是类模式
                        else id
                        if id is not None  # 如果ID不为None
                        else i
                        if color_mode == "instance"  # 如果是实例模式
                        else None,
                        True,
                    ),
                    rotated=is_obb,  # 是否为定向边界框
                )

        # Plot Classify results
        if pred_probs is not None and show_probs:  # 如果有预测概率并且需要显示概率
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)  # 获取前5个类的概率
            x = round(self.orig_shape[0] * 0.03)  # 计算文本的X坐标
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # 绘制文本

        # Plot Pose results
        if self.keypoints is not None:  # 如果有关键点
            for i, k in enumerate(reversed(self.keypoints.data)):  # 遍历关键点数据
                annotator.kpts(  # 绘制关键点
                    k,
                    self.orig_shape,
                    radius=kpt_radius,  # 关键点半径
                    kpt_line=kpt_line,  # 是否绘制关键点连线
                    kpt_color=colors(i, True) if color_mode == "instance" else None,  # 设置关键点颜色
                )

        # Show results
        if show:  # 如果需要显示结果
            annotator.show(self.path)  # 显示带注释的图像

        # Save results
        if save:  # 如果需要保存结果
            annotator.save(filename)  # 保存带注释的图像

        return annotator.im if pil else annotator.result()  # 返回图像或结果

    def show(self, *args, **kwargs):
        """
        Display the image with annotated inference results.  # 显示带注释的推理结果图像
    
        This method plots the detection results on the original image and displays it. It's a convenient way to
        visualize the model's predictions directly.  # 此方法在原始图像上绘制检测结果并显示。它是直接可视化模型预测的便捷方式。
    
        Args:
            *args (Any): Variable length argument list to be passed to the [plot()](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58) method.  # 可变长度参数列表，将传递给[plot()](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58)方法。
            **kwargs (Any): Arbitrary keyword arguments to be passed to the [plot()](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58) method.  # 任意关键字参数，将传递给[plot()](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58)方法。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> results[0].show()  # Display the first result  # 显示第一个结果
            >>> for result in results:
            >>>     result.show()  # Display all results  # 显示所有结果
        """
        self.plot(show=True, *args, **kwargs)  # 调用plot方法并显示结果
    
    def save(self, filename=None, *args, **kwargs):
        """
        Saves annotated inference results image to file.  # 将带注释的推理结果图像保存到文件。
    
        This method plots the detection results on the original image and saves the annotated image to a file. It
        utilizes the [plot](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58) method to generate the annotated image and then saves it to the specified filename.  # 此方法在原始图像上绘制检测结果，并将带注释的图像保存到文件。它利用[plot](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58)方法生成带注释的图像，然后将其保存到指定的文件名。
    
        Args:
            filename (str | Path | None): The filename to save the annotated image. If None, a default filename
                is generated based on the original image path.  # 要保存带注释图像的文件名。如果为None，则根据原始图像路径生成默认文件名。
            *args (Any): Variable length argument list to be passed to the [plot](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58) method.  # 可变长度参数列表，将传递给[plot](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58)方法。
            **kwargs (Any): Arbitrary keyword arguments to be passed to the [plot](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58) method.  # 任意关键字参数，将传递给[plot()](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:447:4-584:58)方法。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     result.save("annotated_image.jpg")  # 保存带注释的图像
            >>> # Or with custom plot arguments
            >>> for result in results:
            >>>     result.save("annotated_image.jpg", conf=False, line_width=2)  # 或使用自定义绘图参数
        """
        if not filename:  # 如果未提供文件名
            filename = f"results_{Path(self.path).name}"  # 生成默认文件名
        self.plot(save=True, filename=filename, *args, **kwargs)  # 调用plot方法保存图像
        return filename  # 返回保存的文件名
    
    def verbose(self):
        """
        Returns a log string for each task in the results, detailing detection and classification outcomes.  # 返回结果中每个任务的日志字符串，详细说明检测和分类结果。
    
        This method generates a human-readable string summarizing the detection and classification results. It includes
        the number of detections for each class and the top probabilities for classification tasks.  # 此方法生成一个人类可读的字符串，总结检测和分类结果。它包括每个类的检测数量和分类任务的最高概率。
    
        Returns:
            (str): A formatted string containing a summary of the results. For detection tasks, it includes the
                number of detections per class. For classification tasks, it includes the top 5 class probabilities.  # 返回一个格式化的字符串，包含结果的摘要。对于检测任务，它包括每个类的检测数量。对于分类任务，它包括前5个类的概率。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     print(result.verbose())  # 打印详细结果
            2 persons, 1 car, 3 traffic lights,  # 2个行人，1辆车，3个交通信号灯
            dog 0.92, cat 0.78, horse 0.64,  # 狗0.92，猫0.78，马0.64
        """
        log_string = ""  # 初始化日志字符串
        probs = self.probs  # 获取概率
        if len(self) == 0:  # 如果没有检测结果
            return log_string if probs is not None else f"{log_string}(no detections), "  # 返回日志字符串或无检测结果的提示
        if probs is not None:  # 如果有概率
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "  # 添加前5个类的概率
        if boxes := self.boxes:  # 如果存在边界框
            for c in boxes.cls.unique():  # 遍历每个类
                n = (boxes.cls == c).sum()  # 计算每个类的检测数量
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # 添加检测数量和类名
        return log_string  # 返回日志字符串
    
    def save_txt(self, txt_file, save_conf=False):
        """
        Save detection results to a text file.  # 将检测结果保存到文本文件。
    
        Args:
            txt_file (str | Path): Path to the output text file.  # 输出文本文件的路径。
            save_conf (bool): Whether to include confidence scores in the output.  # 是否在输出中包含置信度分数。
    
        Returns:
            (str): Path to the saved text file.  # 返回保存的文本文件路径。
    
        Examples:
            >>> from ultralytics import YOLO  # 从ultralytics导入YOLO
            >>> model = YOLO("yolo11n.pt")  # 加载模型
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     result.save_txt("output.txt")  # 保存结果到文本文件
    
        Notes:
            - The file will contain one line per detection or classification with the following structure:  # 文件将包含每个检测或分类的一行，结构如下：
              - For detections: `class confidence x_center y_center width height`  # 对于检测：`类 置信度 x中心 y中心 宽度 高度`
              - For classifications: `confidence class_name`  # 对于分类：`置信度 类名称`
              - For masks and keypoints, the specific formats will vary accordingly.  # 对于掩码和关键点，具体格式将相应变化。
            - The function will create the output directory if it does not exist.  # 如果输出目录不存在，函数将创建它。
            - If save_conf is False, the confidence scores will be excluded from the output.  # 如果save_conf为False，置信度分数将被排除在输出之外。
            - Existing contents of the file will not be overwritten; new results will be appended.  # 文件的现有内容不会被覆盖；新结果将被追加。
        """
        is_obb = self.obb is not None  # 检查是否存在定向边界框
        boxes = self.obb if is_obb else self.boxes  # 选择边界框
        masks = self.masks  # 获取掩码
        probs = self.probs  # 获取概率
        kpts = self.keypoints  # 获取关键点
        texts = []  # 初始化文本列表
        if probs is not None:  # 如果有概率
            # Classify  # 分类
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]  # 添加前5个类的置信度和名称
        elif boxes:  # 如果有边界框
            # Detect/segment/pose  # 检测/分割/姿态
            for j, d in enumerate(boxes):  # 遍历边界框
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())  # 获取类、置信度和ID
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))  # 创建输出行
                if masks:  # 如果有掩码
                    seg = masks[j].xyn[0].copy().reshape(-1)  # 复制并调整掩码形状
                    line = (c, *seg)  # 更新行
                if kpts is not None:  # 如果有关键点
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn  # 获取关键点
                    line += (*kpt.reshape(-1).tolist(),)  # 更新行
                line += (conf,) * save_conf + (() if id is None else (id,))  # 添加置信度和ID
                texts.append(("%g " * len(line)).rstrip() % line)  # 将行添加到文本列表
    
        if texts:  # 如果有文本
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
            with open(txt_file, "a") as f:  # 以追加模式打开文件
                f.writelines(text + "\n" for text in texts)  # 写入文本行
    
    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """
        Saves cropped detection images to specified directory.  # 将裁剪的检测图像保存到指定目录。
    
        This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
        subdirectory named after the object's class, with the filename based on the input file_name.  # 此方法将检测到的对象的裁剪图像保存到指定目录。每个裁剪图像保存在以对象类命名的子目录中，文件名基于输入的file_name。
    
        Args:
            save_dir (str | Path): Directory path where cropped images will be saved.  # 裁剪图像将保存的目录路径。
            file_name (str | Path): Base filename for the saved cropped images. Default is Path("im.jpg").  # 保存裁剪图像的基本文件名。默认是Path("im.jpg")。
    
        Notes:
            - This method does not support Classify or Oriented Bounding Box (OBB) tasks.  # 此方法不支持分类或定向边界框（OBB）任务。
            - Crops are saved as 'save_dir/class_name/file_name.jpg'.  # 裁剪图像保存为'save_dir/class_name/file_name.jpg'。
            - The method will create necessary subdirectories if they don't exist.  # 如果必要的子目录不存在，方法将创建它们。
            - Original image is copied before cropping to avoid modifying the original.  # 在裁剪之前复制原始图像，以避免修改原始图像。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     result.save_crop(save_dir="path/to/crops", file_name="detection")  # 保存裁剪图像
        """
        if self.probs is not None:  # 如果有概率
            LOGGER.warning("WARNING ⚠️ Classify task do not support [save_crop](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:721:4-755:13).")  # 警告：分类任务不支持[save_crop](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:721:4-755:13)。
            return  # 返回
        if self.obb is not None:  # 如果有定向边界框
            LOGGER.warning("WARNING ⚠️ OBB task do not support [save_crop](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:721:4-755:13).")  # 警告：OBB任务不支持[save_crop](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:721:4-755:13)。
            return  # 返回
        for d in self.boxes:  # 遍历边界框
            save_one_box(  # 保存单个边界框
                d.xyxy,  # 边界框坐标
                self.orig_img.copy(),  # 复制原始图像
                file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),  # 保存路径
                BGR=True,  # 使用BGR格式
            )
    
    def summary(self, normalize=False, decimals=5):
        """
        Converts inference results to a summarized dictionary with optional normalization for box coordinates.  # 将推理结果转换为总结字典，并可选择对边界框坐标进行归一化。
    
        This method creates a list of detection dictionaries, each containing information about a single
        detection or classification result. For classification tasks, it returns the top class and its
        confidence. For detection tasks, it includes class information, bounding box coordinates, and
        optionally mask segments and keypoints.  # 此方法创建一个检测字典列表，每个字典包含有关单个检测或分类结果的信息。对于分类任务，它返回最高类及其置信度。对于检测任务，它包括类信息、边界框坐标，以及可选的掩码段和关键点。
    
        Args:
            normalize (bool): Whether to normalize bounding box coordinates by image dimensions. Defaults to False.  # 是否根据图像尺寸归一化边界框坐标。默认为False。
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.  # 输出值四舍五入的小数位数。默认为5。
    
        Returns:
            (List[Dict]): A list of dictionaries, each containing summarized information for a single
                detection or classification result. The structure of each dictionary varies based on the
                task type (classification or detection) and available information (boxes, masks, keypoints).  # 返回一个字典列表，每个字典包含单个检测或分类结果的总结信息。每个字典的结构根据任务类型（分类或检测）和可用信息（边界框、掩码、关键点）而有所不同。
    
        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     summary = result.summary()  # 获取结果摘要
            >>>     print(summary)  # 打印摘要
        """
        # Create list of detection dictionaries  # 创建检测字典列表
        results = []  # 初始化结果列表
        if self.probs is not None:  # 如果有概率
            class_id = self.probs.top1  # 获取最高类ID
            results.append(  # 添加到结果列表
                {
                    "name": self.names[class_id],  # 类名称
                    "class": class_id,  # 类ID
                    "confidence": round(self.probs.top1conf.item(), decimals),  # 置信度
                }
            )
            return results  # 返回结果
    
        is_obb = self.obb is not None  # 检查是否存在定向边界框
        data = self.obb if is_obb else self.boxes  # 选择数据
        h, w = self.orig_shape if normalize else (1, 1)  # 获取图像高度和宽度
        for i, row in enumerate(data):  # 遍历数据
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)  # 获取类ID和置信度
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()  # 获取边界框
            xy = {}  # 初始化字典
            for j, b in enumerate(box):  # 遍历边界框坐标
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)  # 归一化x坐标
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)  # 归一化y坐标
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}  # 创建结果字典
            if data.is_track:  # 如果数据是跟踪
                result["track_id"] = int(row.id.item())  # 添加跟踪ID
            if self.masks:  # 如果有掩码
                result["segments"] = {  # 添加掩码信息
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),  # 归一化x坐标
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),  # 归一化y坐标
                }
            if self.keypoints is not None:  # 如果有关键点
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # 获取关键点数据
                result["keypoints"] = {  # 添加关键点信息
                    "x": (x / w).numpy().round(decimals).tolist(),  # 归一化x坐标
                    "y": (y / h).numpy().round(decimals).tolist(),  # 归一化y坐标
                    "visible": visible.numpy().round(decimals).tolist(),  # 可见性
                }
            results.append(result)  # 将结果添加到列表
    
        return results  # 返回结果列表
    
    def to_df(self, normalize=False, decimals=5):
        """
        Converts detection results to a Pandas Dataframe.  # 将检测结果转换为Pandas数据框。
    
        This method converts the detection results into Pandas Dataframe format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.  # 此方法将检测结果转换为Pandas数据框格式。它包括有关检测到的对象的信息，例如边界框、类名称、置信度分数，以及可选的分割掩码和关键点。
    
        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.  # 是否根据图像尺寸归一化边界框坐标。如果为True，坐标将作为0到1之间的浮点值返回。默认为False。
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.  # 输出值四舍五入的小数位数。默认为5。
    
        Returns:
            (DataFrame): A Pandas Dataframe containing all the information in results in an organized way.  # 返回一个包含所有结果信息的Pandas数据框，格式化良好。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     df_result = result.to_df()  # 获取数据框结果
            >>>     print(df_result)  # 打印数据框
        """
        import pandas as pd  # scope for faster 'import ultralytics'  # 为了更快的'import ultralytics'，在此作用域下导入pandas
    
        return pd.DataFrame(self.summary(normalize=normalize, decimals=decimals))  # 返回数据框
    
    def to_csv(self, normalize=False, decimals=5, *args, **kwargs):
        """
        Converts detection results to CSV format.  # 将检测结果转换为CSV格式。
    
        This method serializes the detection results into a CSV format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.  # 此方法将检测结果序列化为CSV格式。它包括有关检测到的对象的信息，例如边界框、类名称、置信度分数，以及可选的分割掩码和关键点。
    
        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.  # 是否根据图像尺寸归一化边界框坐标。如果为True，坐标将作为0到1之间的浮点值返回。默认为False。
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.  # 输出值四舍五入的小数位数。默认为5。
            *args (Any): Variable length argument list to be passed to pandas.DataFrame.to_csv().  # 可变长度参数列表，将传递给pandas.DataFrame.to_csv()。
            **kwargs (Any): Arbitrary keyword arguments to be passed to pandas.DataFrame.to_csv().  # 任意关键字参数，将传递给pandas.DataFrame.to_csv()。
    
        Returns:
            (str): CSV containing all the information in results in an organized way.  # 返回一个包含所有结果信息的CSV文件，格式化良好。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     csv_result = result.to_csv()  # 获取CSV结果
            >>>     print(csv_result)  # 打印CSV结果
        """
        return self.to_df(normalize=normalize, decimals=decimals).to_csv(*args, **kwargs)  # 返回CSV文件
    
    def to_xml(self, normalize=False, decimals=5, *args, **kwargs):
        """
        Converts detection results to XML format.  # 将检测结果转换为XML格式。
    
        This method serializes the detection results into an XML format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.  # 此方法将检测结果序列化为XML格式。它包括有关检测到的对象的信息，例如边界框、类名称、置信度分数，以及可选的分割掩码和关键点。
    
        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.  # 是否根据图像尺寸归一化边界框坐标。如果为True，坐标将作为0到1之间的浮点值返回。默认为False。
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.  # 输出值四舍五入的小数位数。默认为5。
            *args (Any): Variable length argument list to be passed to pandas.DataFrame.to_xml().  # 可变长度参数列表，将传递给pandas.DataFrame.to_xml()。
            **kwargs (Any): Arbitrary keyword arguments to be passed to pandas.DataFrame.to_xml().  # 任意关键字参数，将传递给pandas.DataFrame.to_xml()。
    
        Returns:
            (str): An XML string containing all the information in results in an organized way.  # 返回一个XML字符串，包含所有结果信息，格式化良好。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     xml_result = result.to_xml()  # 获取XML结果
            >>>     print(xml_result)  # 打印XML结果
        """
        check_requirements("lxml")  # 检查是否安装lxml库
        df = self.to_df(normalize=normalize, decimals=decimals)  # 获取数据框
        return '<?xml version="1.0" encoding="utf-8"?>\n<root></root>' if df.empty else df.to_xml(*args, **kwargs)  # 返回XML字符串
    
    def tojson(self, normalize=False, decimals=5):
        """Deprecated version of to_json()."""  # to_json()的弃用版本。
        LOGGER.warning("WARNING ⚠️ 'result.tojson()' is deprecated, replace with 'result.to_json()'.")  # 警告：'result.tojson()'已弃用，请替换为'result.to_json()'。
        return self.to_json(normalize, decimals)  # 调用to_json()
    
    def to_json(self, normalize=False, decimals=5):
        """
        Converts detection results to JSON format.  # 将检测结果转换为JSON格式。
    
        This method serializes the detection results into a JSON-compatible format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.  # 此方法将检测结果序列化为JSON兼容格式。它包括有关检测到的对象的信息，例如边界框、类名称、置信度分数，以及可选的分割掩码和关键点。
    
        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.  # 是否根据图像尺寸归一化边界框坐标。如果为True，坐标将作为0到1之间的浮点值返回。默认为False。
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.  # 输出值四舍五入的小数位数。默认为5。
    
        Returns:
            (str): A JSON string containing the serialized detection results.  # 返回一个JSON字符串，包含序列化的检测结果。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     json_result = result.to_json()  # 获取JSON结果
            >>>     print(json_result)  # 打印JSON结果
    
        Notes:
            - For classification tasks, the JSON will contain class probabilities instead of bounding boxes.  # 对于分类任务，JSON将包含类概率而不是边界框。
            - For object detection tasks, the JSON will include bounding box coordinates, class names, and
              confidence scores.  # 对于目标检测任务，JSON将包括边界框坐标、类名称和置信度分数。
            - If available, segmentation masks and keypoints will also be included in the JSON output.  # 如果可用，分割掩码和关键点也将包含在JSON输出中。
            - The method uses the [summary](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:757:4-821:22) method internally to generate the data structure before
              converting it to JSON.  # 此方法内部使用[summary](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/ultralytics-8.3.74/ultralytics/engine/results.py:757:4-821:22)方法生成数据结构，然后将其转换为JSON。
        """
        import json  # 导入json模块
    
        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)  # 返回JSON字符串
    
    def to_sql(self, table_name="results", normalize=False, decimals=5, db_path="results.db"):
        """
        Converts detection results to an SQL-compatible format.  # 将检测结果转换为SQL兼容格式。
    
        This method serializes the detection results into a format compatible with SQL databases.
        It includes information about detected objects such as bounding boxes, class names, confidence scores,
        and optionally segmentation masks, keypoints or oriented bounding boxes.  # 此方法将检测结果序列化为与SQL数据库兼容的格式。它包括有关检测到的对象的信息，例如边界框、类名称、置信度分数，以及可选的分割掩码、关键点或定向边界框。
    
        Args:
            table_name (str): Name of the SQL table where the data will be inserted. Defaults to "detection_results".  # SQL表的名称，数据将插入到该表中。默认为"detection_results"。
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.  # 是否根据图像尺寸归一化边界框坐标。如果为True，坐标将作为0到1之间的浮点值返回。默认为False。
            decimals (int): Number of decimal places to round the bounding boxes values to. Defaults to 5.  # 边界框值四舍五入的小数位数。默认为5。
            db_path (str): Path to the SQLite database file. Defaults to "results.db".  # SQLite数据库文件的路径。默认为"results.db"。
    
        Examples:
            >>> results = model("path/to/image.jpg")  # 使用模型进行推理
            >>> for result in results:
            >>>     result.to_sql()  # 保存结果到SQL
        """
        import json  # 导入json模块
        import sqlite3  # 导入sqlite3模块
    
        # Convert results to a list of dictionaries  # 将结果转换为字典列表
        data = self.summary(normalize=normalize, decimals=decimals)  # 获取结果摘要
        if len(data) == 0:  # 如果结果为空
            LOGGER.warning("⚠️ No results to save to SQL. Results dict is empty")  # 警告：没有结果可以保存到SQL。结果字典为空
            return  # 返回
    
        # Connect to the SQLite database  # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)  # 连接到数据库
        cursor = conn.cursor()  # 创建游标
    
        # Create table if it doesn't exist  # 如果表不存在，则创建表
        columns = (
            "id INTEGER PRIMARY KEY AUTOINCREMENT, class_name TEXT, confidence REAL, box TEXT, masks TEXT, kpts TEXT"
        )  # 定义表的列
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")  # 创建表
    
        # Insert data into the table  # 将数据插入表中
        for item in data:  # 遍历数据
            cursor.execute(
                f"INSERT INTO {table_name} (class_name, confidence, box, masks, kpts) VALUES (?, ?, ?, ?, ?)",  # 插入数据的SQL语句
                (
                    item.get("name"),  # 类名称
                    item.get("confidence"),  # 置信度
                    json.dumps(item.get("box", {})),  # 边界框
                    json.dumps(item.get("segments", {})),  # 掩码
                    json.dumps(item.get("keypoints", {})),  # 关键点
                ),
            )
    
        # Commit and close the connection  # 提交并关闭连接
        conn.commit()  # 提交更改
        conn.close()  # 关闭连接
    
        LOGGER.info(f"✅ Detection results successfully written to SQL table '{table_name}' in database '{db_path}'.")  # 日志信息：检测结果成功写入数据库'{db_path}'中的SQL表'{table_name}'。

class Boxes(BaseTensor):
    """
    A class for managing and manipulating detection boxes.  # 管理和操作检测框的类。

    This class provides functionality for handling detection boxes, including their coordinates, confidence scores,
    class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipulation
    and conversion between different coordinate systems.  # 此类提供处理检测框的功能，包括其坐标、置信度分数、类标签和可选的跟踪ID。它支持多种框格式，并提供便捷的操作和不同坐标系统之间的转换方法。

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor containing detection boxes and associated data.  # 包含检测框及其相关数据的原始张量。
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).  # 原始图像的尺寸（高度，宽度）。
        is_track (bool): Indicates whether tracking IDs are included in the box data.  # 指示框数据中是否包含跟踪ID。
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.  # 以[x1, y1, x2, y2]格式表示的框。
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.  # 每个框的置信度分数。
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.  # 每个框的类标签。
        id (torch.Tensor | numpy.ndarray): Tracking IDs for each box (if available).  # 每个框的跟踪ID（如果可用）。
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format.  # 以[x, y, 宽度, 高度]格式表示的框。
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.  # 相对于原始形状的归一化[x1, y1, x2, y2]框。
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.  # 相对于原始形状的归一化[x, y, 宽度, 高度]框。

    Methods:
        cpu(): Returns a copy of the object with all tensors on CPU memory.  # 返回一个副本，所有张量在CPU内存中。
        numpy(): Returns a copy of the object with all tensors as numpy arrays.  # 返回一个副本，所有张量作为NumPy数组。
        cuda(): Returns a copy of the object with all tensors on GPU memory.  # 返回一个副本，所有张量在GPU内存中。
        to(*args, **kwargs): Returns a copy of the object with tensors on specified device and dtype.  # 返回一个副本，张量在指定设备和数据类型上。

    Examples:
        >>> import torch  # 导入torch库
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])  # 创建检测框数据
        >>> orig_shape = (480, 640)  # height, width  # 原始图像尺寸（高度，宽度）
        >>> boxes = Boxes(boxes_data, orig_shape)  # 创建Boxes对象
        >>> print(boxes.xyxy)  # 打印xyxy格式的框
        >>> print(boxes.conf)  # 打印框的置信度
        >>> print(boxes.cls)  # 打印框的类标签
        >>> print(boxes.xywhn)  # 打印归一化的xywh格式框
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class with detection box data and the original image shape.  # 用检测框数据和原始图像形状初始化Boxes类。

        This class manages detection boxes, providing easy access and manipulation of box coordinates,
        confidence scores, class identifiers, and optional tracking IDs. It supports multiple formats
        for box coordinates, including both absolute and normalized forms.  # 此类管理检测框，提供对框坐标、置信度分数、类标识符和可选跟踪ID的便捷访问和操作。它支持多种框坐标格式，包括绝对和归一化形式。

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape
                (num_boxes, 6) or (num_boxes, 7). Columns should contain
                [x1, y1, x2, y2, confidence, class, (optional) track_id].  # 一个形状为(num_boxes, 6)或(num_boxes, 7)的张量或NumPy数组，列应包含[x1, y1, x2, y2, 置信度, 类, （可选）跟踪ID]。
            orig_shape (Tuple[int, int]): The original image shape as (height, width). Used for normalization.  # 原始图像形状（高度，宽度），用于归一化。

        Attributes:
            data (torch.Tensor): The raw tensor containing detection boxes and their associated data.  # 包含检测框及其相关数据的原始张量。
            orig_shape (Tuple[int, int]): The original image size, used for normalization.  # 原始图像大小，用于归一化。
            is_track (bool): Indicates whether tracking IDs are included in the box data.  # 指示框数据中是否包含跟踪ID。

        Examples:
            >>> import torch  # 导入torch库
            >>> boxes = torch.tensor([[100, 50, 150, 100, 0.9, 0]])  # 创建一个检测框
            >>> orig_shape = (480, 640)  # 原始图像形状
            >>> detection_boxes = Boxes(boxes, orig_shape)  # 创建Boxes对象
            >>> print(detection_boxes.xyxy)  # 打印xyxy格式的框
            tensor([[100.,  50., 150., 100.]])  # 输出结果示例
        """
        if boxes.ndim == 1:  # 如果输入的boxes是一维数组
            boxes = boxes[None, :]  # 将其转换为二维数组
        n = boxes.shape[-1]  # 获取最后一维的大小
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # 确保最后一维的大小为6或7
        super().__init__(boxes, orig_shape)  # 调用父类构造函数
        self.is_track = n == 7  # 判断是否包含跟踪ID
        self.orig_shape = orig_shape  # 保存原始图像形状

    @property
    def xyxy(self):
        """
        Returns bounding boxes in [x1, y1, x2, y2] format.  # 返回[x1, y1, x2, y2]格式的边界框。

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array of shape (n, 4) containing bounding box
                coordinates in [x1, y1, x2, y2] format, where n is the number of boxes.  # 返回形状为(n, 4)的张量或NumPy数组，包含[x1, y1, x2, y2]格式的边界框坐标，其中n是框的数量。

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> boxes = results[0].boxes  # 获取检测框
            >>> xyxy = boxes.xyxy  # 获取xyxy格式的边界框
            >>> print(xyxy)  # 打印边界框
        """
        return self.data[:, :4]  # 返回前四列作为边界框坐标

    @property
    def conf(self):
        """
        Returns the confidence scores for each detection box.  # 返回每个检测框的置信度分数。

        Returns:
            (torch.Tensor | numpy.ndarray): A 1D tensor or array containing confidence scores for each detection,
                with shape (N,) where N is the number of detections.  # 返回一个一维张量或数组，包含每个检测的置信度分数，形状为(N,)，其中N是检测数量。

        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))  # 创建Boxes对象
            >>> conf_scores = boxes.conf  # 获取置信度分数
            >>> print(conf_scores)  # 打印置信度分数
            tensor([0.9000])  # 打印结果示例
        """
        return self.data[:, -2]  # 返回倒数第二列作为置信度分数

    @property
    def cls(self):
        """
        Returns the class ID tensor representing category predictions for each bounding box.  # 返回表示每个边界框类别预测的类ID张量。

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the class IDs for each detection box.
                The shape is (N,), where N is the number of boxes.  # 返回一个张量或NumPy数组，包含每个检测框的类ID，形状为(N,)，其中N是框的数量。

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> boxes = results[0].boxes  # 获取检测框
            >>> class_ids = boxes.cls  # 获取类ID
            >>> print(class_ids)  # 打印类ID
            tensor([0., 2., 1.])  # 打印结果示例
        """
        return self.data[:, -1]  # 返回最后一列作为类ID

    @property
    def id(self):
        """
        Returns the tracking IDs for each detection box if available.  # 返回每个检测框的跟踪ID（如果可用）。

        Returns:
            (torch.Tensor | None): A tensor containing tracking IDs for each box if tracking is enabled,
                otherwise None. Shape is (N,) where N is the number of boxes.  # 返回一个张量，包含每个框的跟踪ID，如果未启用跟踪，则返回None。形状为(N,)，其中N是框的数量。

        Examples:
            >>> results = model.track("path/to/video.mp4")  # 使用跟踪模式运行推理
            >>> for result in results:
            ...     boxes = result.boxes  # 获取检测框
            ...     if boxes.is_track:  # 如果启用了跟踪
            ...         track_ids = boxes.id  # 获取跟踪ID
            ...         print(f"Tracking IDs: {track_ids}")  # 打印跟踪ID
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")  # 打印未启用跟踪的提示

        Notes:
            - This property is only available when tracking is enabled (i.e., when `is_track` is True).  # 此属性仅在启用跟踪时可用（即，当`is_track`为True时）。
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.  # 跟踪ID通常用于在视频分析中关联多个帧的检测。
        """
        return self.data[:, -3] if self.is_track else None  # 如果启用了跟踪，返回倒数第三列作为跟踪ID，否则返回None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice  # LRU缓存，最大大小为2
    def xywh(self):
        """
        Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.  # 将边界框从[x1, y1, x2, y2]格式转换为[x, y, 宽度, 高度]格式。

        Returns:
            (torch.Tensor | numpy.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center are the coordinates of
                the center point of the bounding box, width, height are the dimensions of the bounding box and the
                shape of the returned tensor is (N, 4), where N is the number of boxes.  # 返回格式为[x_center, y_center, 宽度, 高度]的框，其中x_center和y_center是边界框中心点的坐标，宽度和高度是边界框的尺寸，返回的张量形状为(N, 4)，其中N是框的数量。

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100], [200, 150, 300, 250]]), orig_shape=(480, 640))  # 创建Boxes对象
            >>> xywh = boxes.xywh  # 获取xywh格式的框
            >>> print(xywh)  # 打印结果
            tensor([[100.0000,  50.0000,  50.0000,  50.0000],  # 打印结果示例
                    [200.0000, 150.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)  # 调用ops模块的函数将xyxy格式转换为xywh格式

    @property
    @lru_cache(maxsize=2)  # 使用LRU缓存以提高性能
    def xyxyn(self):
        """
        Returns normalized bounding box coordinates relative to the original image size.  # 返回相对于原始图像大小的归一化边界框坐标。

        This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format,
        normalized to the range [0, 1] based on the original image dimensions.  # 此属性计算并返回以[x1, y1, x2, y2]格式表示的边界框坐标，归一化到原始图像尺寸的范围[0, 1]。

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is
                the number of boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].  # 返回形状为(N, 4)的归一化边界框坐标，其中N是框的数量。每行包含归一化到[0, 1]的[x1, y1, x2, y2]值。

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))  # 创建Boxes对象
            >>> normalized = boxes.xyxyn  # 获取归一化的边界框
            >>> print(normalized)  # 打印结果
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])  # 打印结果示例
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)  # 复制xyxy数据
        xyxy[..., [0, 2]] /= self.orig_shape[1]  # 将x坐标归一化
        xyxy[..., [1, 3]] /= self.orig_shape[0]  # 将y坐标归一化
        return xyxy  # 返回归一化后的坐标

    @property
    @lru_cache(maxsize=2)  # 使用LRU缓存以提高性能
    def xywhn(self):
        """
        Returns normalized bounding boxes in [x, y, width, height] format.  # 返回以[x, y, 宽度, 高度]格式表示的归一化边界框。

        This property calculates and returns the normalized bounding box coordinates in the format
        [x_center, y_center, width, height], where all values are relative to the original image dimensions.  # 此属性计算并返回以[x_center, y_center, 宽度, 高度]格式表示的归一化边界框坐标，其中所有值相对于原始图像尺寸。

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding boxes with shape (N, 4), where N is the
                number of boxes. Each row contains [x_center, y_center, width, height] values normalized
                to [0, 1] based on the original image dimensions.  # 返回形状为(N, 4)的归一化边界框，其中N是框的数量。每行包含归一化到[0, 1]的[x_center, y_center, 宽度, 高度]值。

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))  # 创建Boxes对象
            >>> normalized = boxes.xywhn  # 获取归一化的xywh格式框
            >>> print(normalized)  # 打印结果
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])  # 打印结果示例
        """
        xywh = ops.xyxy2xywh(self.xyxy)  # 将xyxy格式转换为xywh格式
        xywh[..., [0, 2]] /= self.orig_shape[1]  # 将x坐标归一化
        xywh[..., [1, 3]] /= self.orig_shape[0]  # 将y坐标归一化
        return xywh  # 返回归一化后的xywh格式框


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.  # 存储和操作检测掩码的类。

    This class extends BaseTensor and provides functionality for handling segmentation masks,
    including methods for converting between pixel and normalized coordinates.  # 此类扩展BaseTensor，并提供处理分割掩码的功能，包括在像素和归一化坐标之间转换的方法。

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor or array containing mask data.  # 包含掩码数据的原始张量或数组。
        orig_shape (tuple): Original image shape in (height, width) format.  # 原始图像形状（高度，宽度）。
        xy (List[numpy.ndarray]): A list of segments in pixel coordinates.  # 像素坐标的段列表。
        xyn (List[numpy.ndarray]): A list of normalized segments.  # 归一化段的列表。

    Methods:
        cpu(): Returns a copy of the Masks object with the mask tensor on CPU memory.  # 返回一个副本，掩码张量在CPU内存中。
        numpy(): Returns a copy of the Masks object with the mask tensor as a numpy array.  # 返回一个副本，掩码张量作为NumPy数组。
        cuda(): Returns a copy of the Masks object with the mask tensor on GPU memory.  # 返回一个副本，掩码张量在GPU内存中。
        to(*args, **kwargs): Returns a copy of the Masks object with the mask tensor on specified device and dtype.  # 返回一个副本，掩码张量在指定设备和数据类型上。

    Examples:
        >>> masks_data = torch.rand(1, 160, 160)  # 创建随机掩码数据
        >>> orig_shape = (720, 1280)  # 原始图像形状
        >>> masks = Masks(masks_data, orig_shape)  # 创建Masks对象
        >>> pixel_coords = masks.xy  # 获取像素坐标
        >>> normalized_coords = masks.xyn  # 获取归一化坐标
    """

    def __init__(self, masks, orig_shape) -> None:
        """
        Initialize the Masks class with detection mask data and the original image shape.  # 用检测掩码数据和原始图像形状初始化Masks类。

        Args:
            masks (torch.Tensor | np.ndarray): Detection masks with shape (num_masks, height, width).  # 形状为(num_masks, 高度, 宽度)的检测掩码。
            orig_shape (tuple): The original image shape as (height, width). Used for normalization.  # 原始图像形状（高度，宽度），用于归一化。

        Examples:
            >>> import torch  # 导入torch库
            >>> from ultralytics.engine.results import Masks  # 从ultralytics导入Masks类
            >>> masks = torch.rand(10, 160, 160)  # 10个160x160分辨率的掩码
            >>> orig_shape = (720, 1280)  # 原始图像形状
            >>> mask_obj = Masks(masks, orig_shape)  # 创建Masks对象
        """
        if masks.ndim == 2:  # 如果输入的掩码是一维的
            masks = masks[None, :]  # 将其转换为二维
        super().__init__(masks, orig_shape)  # 调用父类构造函数

    @property
    @lru_cache(maxsize=1)  # 使用LRU缓存以提高性能
    def xyn(self):
        """
        Returns normalized xy-coordinates of the segmentation masks.  # 返回分割掩码的归一化xy坐标。

        This property calculates and caches the normalized xy-coordinates of the segmentation masks. The coordinates
        are normalized relative to the original image shape.  # 此属性计算并缓存分割掩码的归一化xy坐标。坐标相对于原始图像形状进行归一化。

        Returns:
            (List[numpy.ndarray]): A list of numpy arrays, where each array contains the normalized xy-coordinates
                of a single segmentation mask. Each array has shape (N, 2), where N is the number of points in the
                mask contour.  # 返回一个NumPy数组的列表，每个数组包含单个分割掩码的归一化xy坐标。每个数组的形状为(N, 2)，其中N是掩码轮廓中的点数。

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> masks = results[0].masks  # 获取掩码对象
            >>> normalized_coords = masks.xyn  # 获取归一化坐标
            >>> print(normalized_coords[0])  # 打印第一个掩码的归一化坐标
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)  # 归一化掩码坐标
            for x in ops.masks2segments(self.data)  # 将掩码转换为段
        ]

    @property
    @lru_cache(maxsize=1)  # 使用LRU缓存以提高性能
    def xy(self):
        """
        Returns the [x, y] pixel coordinates for each segment in the mask tensor.  # 返回掩码张量中每个段的[x, y]像素坐标。

        This property calculates and returns a list of pixel coordinates for each segmentation mask in the
        Masks object. The coordinates are scaled to match the original image dimensions.  # 此属性计算并返回Masks对象中每个分割掩码的像素坐标列表。坐标被缩放以匹配原始图像尺寸。

        Returns:
            (List[numpy.ndarray]): A list of numpy arrays, where each array contains the [x, y] pixel
                coordinates for a single segmentation mask. Each array has shape (N, 2), where N is the
                number of points in the segment.  # 返回一个NumPy数组的列表，每个数组包含单个分割掩码的[x, y]像素坐标。每个数组的形状为(N, 2)，其中N是段中的点数。

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> masks = results[0].masks  # 获取掩码对象
            >>> xy_coords = masks.xy  # 获取像素坐标
            >>> print(len(xy_coords))  # 打印掩码数量
            >>> print(xy_coords[0].shape)  # 打印第一个掩码坐标的形状
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)  # 返回未归一化的坐标
            for x in ops.masks2segments(self.data)  # 将掩码转换为段
        ]


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.  # 存储和操作检测关键点的类。

    This class encapsulates functionality for handling keypoint data, including coordinate manipulation,
    normalization, and confidence values.  # 此类封装了处理关键点数据的功能，包括坐标操作、归一化和置信度值。

    Attributes:
        data (torch.Tensor): The raw tensor containing keypoint data.  # 包含关键点数据的原始张量。
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).  # 原始图像的尺寸（高度，宽度）。
        has_visible (bool): Indicates whether visibility information is available for keypoints.  # 指示关键点的可见性信息是否可用。
        xy (torch.Tensor): Keypoint coordinates in [x, y] format.  # 以[x, y]格式表示的关键点坐标。
        xyn (torch.Tensor): Normalized keypoint coordinates in [x, y] format, relative to orig_shape.  # 相对于原始形状的归一化关键点坐标，以[x, y]格式表示。
        conf (torch.Tensor): Confidence values for each keypoint, if available.  # 每个关键点的置信度值（如果可用）。

    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.  # 返回一个副本，关键点张量在CPU内存中。
        numpy(): Returns a copy of the keypoints tensor as a numpy array.  # 返回一个副本，关键点张量作为NumPy数组。
        cuda(): Returns a copy of the keypoints tensor on GPU memory.  # 返回一个副本，关键点张量在GPU内存中。
        to(*args, **kwargs): Returns a copy of the keypoints tensor with specified device and dtype.  # 返回一个副本，关键点张量在指定设备和数据类型上。

    Examples:
        >>> import torch  # 导入torch库
        >>> from ultralytics.engine.results import Keypoints  # 从ultralytics导入Keypoints类
        >>> keypoints_data = torch.rand(1, 17, 3)  # 1个检测，17个关键点（x, y, conf）
        >>> orig_shape = (480, 640)  # 原始图像形状（高度，宽度）
        >>> keypoints = Keypoints(keypoints_data, orig_shape)  # 创建Keypoints对象
        >>> print(keypoints.xy.shape)  # 访问xy坐标
        >>> print(keypoints.conf)  # 访问置信度值
        >>> keypoints_cpu = keypoints.cpu()  # 将关键点移动到CPU
    """

    @smart_inference_mode()  # avoid keypoints < conf in-place error  # 避免关键点<置信度的原地错误
    def __init__(self, keypoints, orig_shape) -> None:
        """
        Initializes the Keypoints object with detection keypoints and original image dimensions.  # 用检测关键点和原始图像尺寸初始化Keypoints对象。

        This method processes the input keypoints tensor, handling both 2D and 3D formats. For 3D tensors
        (x, y, confidence), it masks out low-confidence keypoints by setting their coordinates to zero.  # 此方法处理输入的关键点张量，处理2D和3D格式。对于3D张量（x, y, 置信度），它通过将低置信度关键点的坐标设置为零来屏蔽它们。

        Args:
            keypoints (torch.Tensor): A tensor containing keypoint data. Shape can be either:  # 包含关键点数据的张量。形状可以是：
                - (num_objects, num_keypoints, 2) for x, y coordinates only  # 仅包含x, y坐标的形状(num_objects, num_keypoints, 2)
                - (num_objects, num_keypoints, 3) for x, y coordinates and confidence scores  # 包含x, y坐标和置信度分数的形状(num_objects, num_keypoints, 3)
            orig_shape (Tuple[int, int]): The original image dimensions (height, width).  # 原始图像尺寸（高度，宽度）。

        Examples:
            >>> kpts = torch.rand(1, 17, 3)  # 1个对象，17个关键点（COCO格式），x,y,conf
            >>> orig_shape = (720, 1280)  # 原始图像高度，宽度
            >>> keypoints = Keypoints(kpts, orig_shape)  # 创建Keypoints对象
        """
        if keypoints.ndim == 2:  # 如果输入的关键点是一维的
            keypoints = keypoints[None, :]  # 将其转换为二维
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # 置信度<0.5的点（不可见）
            keypoints[..., :2][mask] = 0  # 将不可见关键点的坐标设置为0
        super().__init__(keypoints, orig_shape)  # 调用父类构造函数
        self.has_visible = self.data.shape[-1] == 3  # 判断是否包含可见性信息

    @property
    @lru_cache(maxsize=1)  # 使用LRU缓存以提高性能
    def xy(self):
        """
        Returns x, y coordinates of keypoints.  # 返回关键点的x, y坐标。

        Returns:
            (torch.Tensor): A tensor containing the x, y coordinates of keypoints with shape (N, K, 2), where N is
                the number of detections and K is the number of keypoints per detection.  # 返回一个张量，包含关键点的x, y坐标，形状为(N, K, 2)，其中N是检测数量，K是每个检测的关键点数量。

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> keypoints = results[0].keypoints  # 获取关键点对象
            >>> xy = keypoints.xy  # 获取xy坐标
            >>> print(xy.shape)  # (N, K, 2)  # 打印形状
            >>> print(xy[0])  # x, y coordinates of keypoints for first detection  # 打印第一个检测的关键点坐标

        Notes:
            - The returned coordinates are in pixel units relative to the original image dimensions.  # 返回的坐标是相对于原始图像尺寸的像素单位。
            - If keypoints were initialized with confidence values, only keypoints with confidence >= 0.5 are returned.  # 如果关键点是使用置信度值初始化的，则仅返回置信度>=0.5的关键点。
            - This property uses LRU caching to improve performance on repeated access.  # 此属性使用LRU缓存以提高重复访问的性能。
        """
        return self.data[..., :2]  # 返回前两列作为关键点坐标

    @property
    @lru_cache(maxsize=1)  # 使用LRU缓存以提高性能
    def xyn(self):
        """
        Returns normalized coordinates (x, y) of keypoints relative to the original image size.  # 返回相对于原始图像大小的归一化关键点坐标（x, y）。

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or array of shape (N, K, 2) containing normalized keypoint
                coordinates, where N is the number of instances, K is the number of keypoints, and the last
                dimension contains [x, y] values in the range [0, 1].  # 返回形状为(N, K, 2)的张量或数组，包含归一化的关键点坐标，其中N是实例数量，K是关键点数量，最后一维包含范围在[0, 1]内的[x, y]值。

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))  # 创建Keypoints对象
            >>> normalized_kpts = keypoints.xyn  # 获取归一化关键点
            >>> print(normalized_kpts.shape)  # 打印形状
            torch.Size([1, 17, 2])  # 打印结果示例
        """
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)  # 复制xy数据
        xy[..., 0] /= self.orig_shape[1]  # 将x坐标归一化
        xy[..., 1] /= self.orig_shape[0]  # 将y坐标归一化
        return xy  # 返回归一化后的坐标

    @property
    @lru_cache(maxsize=1)  # 使用LRU缓存以提高性能
    def conf(self):
        """
        Returns confidence values for each keypoint.  # 返回每个关键点的置信度值。

        Returns:
            (torch.Tensor | None): A tensor containing confidence scores for each keypoint if available,
                otherwise None. Shape is (num_detections, num_keypoints) for batched data or (num_keypoints,)
                for single detection.  # 返回一个张量，包含每个关键点的置信度分数（如果可用），否则返回None。形状为(num_detections, num_keypoints)用于批量数据，或(num_keypoints,)用于单个检测。

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1个检测，17个关键点
            >>> conf = keypoints.conf  # 获取置信度
            >>> print(conf.shape)  # torch.Size([1, 17])  # 打印形状
        """
        return self.data[..., 2] if self.has_visible else None  # 如果有可见性信息，返回第三列作为置信度，否则返回None

class Probs(BaseTensor):
    """
    A class for storing and manipulating classification probabilities.  # 存储和操作分类概率的类。

    This class extends BaseTensor and provides methods for accessing and manipulating
    classification probabilities, including top-1 and top-5 predictions.  # 此类扩展BaseTensor，并提供访问和操作分类概率的方法，包括top-1和top-5预测。

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor or array containing classification probabilities.  # 包含分类概率的原始张量或数组。
        orig_shape (tuple | None): The original image shape as (height, width). Not used in this class.  # 原始图像形状（高度，宽度），在此类中未使用。
        top1 (int): Index of the class with the highest probability.  # 具有最高概率的类的索引。
        top5 (List[int]): Indices of the top 5 classes by probability.  # 按概率排序的前5个类的索引。
        top1conf (torch.Tensor | numpy.ndarray): Confidence score of the top 1 class.  # 前1类的置信度分数。
        top5conf (torch.Tensor | numpy.ndarray): Confidence scores of the top 5 classes.  # 前5类的置信度分数。

    Methods:
        cpu(): Returns a copy of the probabilities tensor on CPU memory.  # 返回在CPU内存中的概率张量副本。
        numpy(): Returns a copy of the probabilities tensor as a numpy array.  # 返回作为NumPy数组的概率张量副本。
        cuda(): Returns a copy of the probabilities tensor on GPU memory.  # 返回在GPU内存中的概率张量副本。
        to(*args, **kwargs): Returns a copy of the probabilities tensor with specified device and dtype.  # 返回具有指定设备和数据类型的概率张量副本。

    Examples:
        >>> probs = torch.tensor([0.1, 0.3, 0.6])  # 创建一个包含分类概率的张量
        >>> p = Probs(probs)  # 创建Probs对象
        >>> print(p.top1)  # 打印具有最高概率的类的索引
        2
        >>> print(p.top5)  # 打印前5个类的索引
        [2, 1, 0]
        >>> print(p.top1conf)  # 打印前1类的置信度分数
        tensor(0.6000)
        >>> print(p.top5conf)  # 打印前5类的置信度分数
        tensor([0.6000, 0.3000, 0.1000])
    """

    def __init__(self, probs, orig_shape=None) -> None:
        """
        Initialize the Probs class with classification probabilities.  # 用分类概率初始化Probs类。

        This class stores and manages classification probabilities, providing easy access to top predictions and their
        confidences.  # 此类存储和管理分类概率，提供对前几名预测及其置信度的便捷访问。

        Args:
            probs (torch.Tensor | np.ndarray): A 1D tensor or array of classification probabilities.  # 一维张量或数组，包含分类概率。
            orig_shape (tuple | None): The original image shape as (height, width). Not used in this class but kept for
                consistency with other result classes.  # 原始图像形状（高度，宽度），在此类中未使用，但为与其他结果类保持一致而保留。

        Attributes:
            data (torch.Tensor | np.ndarray): The raw tensor or array containing classification probabilities.  # 包含分类概率的原始张量或数组。
            top1 (int): Index of the top 1 class.  # 前1类的索引。
            top5 (List[int]): Indices of the top 5 classes.  # 前5类的索引。
            top1conf (torch.Tensor | np.ndarray): Confidence of the top 1 class.  # 前1类的置信度。
            top5conf (torch.Tensor | np.ndarray): Confidences of the top 5 classes.  # 前5类的置信度。

        Examples:
            >>> import torch  # 导入torch库
            >>> probs = torch.tensor([0.1, 0.3, 0.2, 0.4])  # 创建一个包含分类概率的张量
            >>> p = Probs(probs)  # 创建Probs对象
            >>> print(p.top1)  # 打印具有最高概率的类的索引
            3
            >>> print(p.top1conf)  # 打印前1类的置信度分数
            tensor(0.4000)
            >>> print(p.top5)  # 打印前5个类的索引
            [3, 1, 2, 0]
        """
        super().__init__(probs, orig_shape)  # 调用父类构造函数

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """
        Returns the index of the class with the highest probability.  # 返回具有最高概率的类的索引。

        Returns:
            (int): Index of the class with the highest probability.  # 具有最高概率的类的索引。

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))  # 创建Probs对象
            >>> probs.top1  # 获取具有最高概率的类的索引
            2
        """
        return int(self.data.argmax())  # 返回概率最高的类的索引

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """
        Returns the indices of the top 5 class probabilities.  # 返回前5个类的概率索引。

        Returns:
            (List[int]): A list containing the indices of the top 5 class probabilities, sorted in descending order.  # 返回一个列表，包含前5个类的概率索引，按降序排列。

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))  # 创建Probs对象
            >>> print(probs.top5)  # 打印前5个类的索引
            [4, 3, 2, 1, 0]
        """
        return (-self.data).argsort(0)[:5].tolist()  # 以降序返回前5个类的索引

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """
        Returns the confidence score of the highest probability class.  # 返回具有最高概率的类的置信度分数。

        This property retrieves the confidence score (probability) of the class with the highest predicted probability
        from the classification results.  # 此属性从分类结果中获取具有最高预测概率的类的置信度分数（概率）。

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor containing the confidence score of the top 1 class.  # 返回一个张量，包含前1类的置信度分数。

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> probs = results[0].probs  # 获取分类概率
            >>> top1_confidence = probs.top1conf  # 获取前1类的置信度
            >>> print(f"Top 1 class confidence: {top1_confidence.item():.4f}")  # 打印前1类的置信度
        """
        return self.data[self.top1]  # 返回前1类的置信度分数

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """
        Returns confidence scores for the top 5 classification predictions.  # 返回前5个分类预测的置信度分数。

        This property retrieves the confidence scores corresponding to the top 5 class probabilities
        predicted by the model. It provides a quick way to access the most likely class predictions
        along with their associated confidence levels.  # 此属性获取模型预测的前5个类概率对应的置信度分数，提供快速访问最可能的类预测及其置信度的方法。

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or array containing the confidence scores for the
                top 5 predicted classes, sorted in descending order of probability.  # 返回一个张量或数组，包含前5个预测类的置信度分数，按概率降序排列。

        Examples:
            >>> results = model("image.jpg")  # 使用模型进行推理
            >>> probs = results[0].probs  # 获取分类概率
            >>> top5_conf = probs.top5conf  # 获取前5类的置信度分数
            >>> print(top5_conf)  # 打印前5类的置信度分数
        """
        return self.data[self.top5]  # 返回前5类的置信度分数
class OBB(BaseTensor):
    """
    A class for storing and manipulating Oriented Bounding Boxes (OBB).

    This class provides functionality to handle oriented bounding boxes, including conversion between
    different formats, normalization, and access to various properties of the boxes.

    Attributes:
        data (torch.Tensor): The raw OBB tensor containing box coordinates and associated data.
        orig_shape (tuple): Original image size as (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xywhr (torch.Tensor | numpy.ndarray): Boxes in [x_center, y_center, width, height, rotation] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray): Tracking IDs for each box, if available.
        xyxyxyxy (torch.Tensor | numpy.ndarray): Boxes in 8-point [x1, y1, x2, y2, x3, y3, x4, y4] format.
        xyxyxyxyn (torch.Tensor | numpy.ndarray): Normalized 8-point coordinates relative to orig_shape.
        xyxy (torch.Tensor | numpy.ndarray): Axis-aligned bounding boxes in [x1, y1, x2, y2] format.

    Methods:
        cpu(): Returns a copy of the OBB object with all tensors on CPU memory.
        numpy(): Returns a copy of the OBB object with all tensors as numpy arrays.
        cuda(): Returns a copy of the OBB object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the OBB object with tensors on specified device and dtype.

    Examples:
        >>> boxes = torch.tensor([[100, 50, 150, 100, 30, 0.9, 0]])  # xywhr, conf, cls
        >>> obb = OBB(boxes, orig_shape=(480, 640))
        >>> print(obb.xyxyxyxy)
        >>> print(obb.conf)
        >>> print(obb.cls)
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize an OBB (Oriented Bounding Box) instance with oriented bounding box data and original image shape.

        This class stores and manipulates Oriented Bounding Boxes (OBB) for object detection tasks. It provides
        various properties and methods to access and transform the OBB data.

        Args:
            boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
                with shape (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values.
                If present, the third last column contains track IDs, and the fifth column contains rotation.
            orig_shape (Tuple[int, int]): Original image size, in the format (height, width).

        Attributes:
            data (torch.Tensor | numpy.ndarray): The raw OBB tensor.
            orig_shape (Tuple[int, int]): The original image shape.
            is_track (bool): Whether the boxes include tracking IDs.

        Raises:
            AssertionError: If the number of values per box is not 7 or 8.

        Examples:
            >>> import torch
            >>> boxes = torch.rand(3, 7)  # 3 boxes with 7 values each
            >>> orig_shape = (640, 480)
            >>> obb = OBB(boxes, orig_shape)
            >>> print(obb.xywhr)  # Access the boxes in xywhr format
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self):
        """
        Returns boxes in [x_center, y_center, width, height, rotation] format.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the oriented bounding boxes with format
                [x_center, y_center, width, height, rotation]. The shape is (N, 5) where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> obb = results[0].obb
            >>> xywhr = obb.xywhr
            >>> print(xywhr.shape)
            torch.Size([3, 5])
        """
        return self.data[:, :5]

    @property
    def conf(self):
        """
        Returns the confidence scores for Oriented Bounding Boxes (OBBs).

        This property retrieves the confidence values associated with each OBB detection. The confidence score
        represents the model's certainty in the detection.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array of shape (N,) containing confidence scores
                for N detections, where each score is in the range [0, 1].

        Examples:
            >>> results = model("image.jpg")
            >>> obb_result = results[0].obb
            >>> confidence_scores = obb_result.conf
            >>> print(confidence_scores)
        """
        return self.data[:, -2]

    @property
    def cls(self):
        """
        Returns the class values of the oriented bounding boxes.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the class values for each oriented
                bounding box. The shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> result = results[0]
            >>> obb = result.obb
            >>> class_values = obb.cls
            >>> print(class_values)
        """
        return self.data[:, -1]

    @property
    def id(self):
        """
        Returns the tracking IDs of the oriented bounding boxes (if available).

        Returns:
            (torch.Tensor | numpy.ndarray | None): A tensor or numpy array containing the tracking IDs for each
                oriented bounding box. Returns None if tracking IDs are not available.

        Examples:
            >>> results = model("image.jpg", tracker=True)  # Run inference with tracking
            >>> for result in results:
            ...     if result.obb is not None:
            ...         track_ids = result.obb.id
            ...         if track_ids is not None:
            ...             print(f"Tracking IDs: {track_ids}")
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self):
        """
        Converts OBB format to 8-point (xyxyxyxy) coordinate format for rotated bounding boxes.

        Returns:
            (torch.Tensor | numpy.ndarray): Rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2), where N is
                the number of boxes. Each box is represented by 4 points (x, y), starting from the top-left corner and
                moving clockwise.

        Examples:
            >>> obb = OBB(torch.tensor([[100, 100, 50, 30, 0.5, 0.9, 0]]), orig_shape=(640, 640))
            >>> xyxyxyxy = obb.xyxyxyxy
            >>> print(xyxyxyxy.shape)
            torch.Size([1, 4, 2])
        """
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self):
        """
        Converts rotated bounding boxes to normalized xyxyxyxy format.

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2),
                where N is the number of boxes. Each box is represented by 4 points (x, y), normalized relative to
                the original image dimensions.

        Examples:
            >>> obb = OBB(torch.rand(10, 7), orig_shape=(640, 480))  # 10 random OBBs
            >>> normalized_boxes = obb.xyxyxyxyn
            >>> print(normalized_boxes.shape)
            torch.Size([10, 4, 2])
        """
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self):
        """
        Converts oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format.

        This property calculates the minimal enclosing rectangle for each oriented bounding box and returns it in
        xyxy format (x1, y1, x2, y2). This is useful for operations that require axis-aligned bounding boxes, such
        as IoU calculation with non-rotated boxes.

        Returns:
            (torch.Tensor | numpy.ndarray): Axis-aligned bounding boxes in xyxy format with shape (N, 4), where N
                is the number of boxes. Each row contains [x1, y1, x2, y2] coordinates.

        Examples:
            >>> import torch
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n-obb.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     obb = result.obb
            ...     if obb is not None:
            ...         xyxy_boxes = obb.xyxy
            ...         print(xyxy_boxes.shape)  # (N, 4)

        Notes:
            - This method approximates the OBB by its minimal enclosing rectangle.
            - The returned format is compatible with standard object detection metrics and visualization tools.
            - The property uses caching to improve performance for repeated access.
        """
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
