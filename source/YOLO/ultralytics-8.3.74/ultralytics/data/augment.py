# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math  # 导入数学库
import random  # 导入随机数库
from copy import deepcopy  # 从copy模块导入深拷贝函数
from typing import Tuple, Union  # 从typing模块导入元组和联合类型

import cv2  # 导入OpenCV库，用于计算机视觉
import numpy as np  # 导入NumPy库，用于数组和矩阵操作
import torch  # 导入PyTorch库，用于深度学习
from PIL import Image  # 从PIL库导入图像处理模块

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap  # 从Ultralytics库导入多边形转掩码的工具
from ultralytics.utils import LOGGER, colorstr  # 从Ultralytics库导入日志记录器和颜色字符串工具
from ultralytics.utils.checks import check_version  # 从Ultralytics库导入版本检查工具
from ultralytics.utils.instance import Instances  # 从Ultralytics库导入实例处理工具
from ultralytics.utils.metrics import bbox_ioa  # 从Ultralytics库导入边界框交并比计算工具
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr  # 从Ultralytics库导入分段到边界框的转换工具和坐标转换工具
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13  # 导入不同版本的TorchVision工具

DEFAULT_MEAN = (0.0, 0.0, 0.0)  # 定义默认均值
DEFAULT_STD = (1.0, 1.0, 1.0)  # 定义默认标准差
DEFAULT_CROP_FRACTION = 1.0  # 定义默认裁剪比例


class BaseTransform:
    """
    Base class for image transformations in the Ultralytics library.  # Ultralytics库中图像变换的基类

    This class serves as a foundation for implementing various image processing operations, designed to be
    compatible with both classification and semantic segmentation tasks.  # 该类作为实现各种图像处理操作的基础，旨在与分类和语义分割任务兼容

    Methods:
        apply_image: Applies image transformations to labels.  # apply_image: 将图像变换应用于标签
        apply_instances: Applies transformations to object instances in labels.  # apply_instances: 将变换应用于标签中的对象实例
        apply_semantic: Applies semantic segmentation to an image.  # apply_semantic: 将语义分割应用于图像
        __call__: Applies all label transformations to an image, instances, and semantic masks.  # __call__: 将所有标签变换应用于图像、实例和语义掩码

    Examples:
        >>> transform = BaseTransform()  # 创建BaseTransform实例
        >>> labels = {"image": np.array(...), "instances": [...], "semantic": np.array(...)}  # 定义标签字典
        >>> transformed_labels = transform(labels)  # 应用变换
    """

    def __init__(self) -> None:
        """
        Initializes the BaseTransform object.  # 初始化BaseTransform对象

        This constructor sets up the base transformation object, which can be extended for specific image
        processing tasks. It is designed to be compatible with both classification and semantic segmentation.  # 此构造函数设置基础变换对象，可以扩展用于特定的图像处理任务，旨在与分类和语义分割兼容

        Examples:
            >>> transform = BaseTransform()  # 创建BaseTransform实例
        """
        pass  # 不执行任何操作

    def apply_image(self, labels):
        """
        Applies image transformations to labels.  # 将图像变换应用于标签

        This method is intended to be overridden by subclasses to implement specific image transformation
        logic. In its base form, it returns the input labels unchanged.  # 此方法旨在被子类重写以实现特定的图像变换逻辑。在其基本形式中，它返回未改变的输入标签

        Args:
            labels (Any): The input labels to be transformed. The exact type and structure of labels may
                vary depending on the specific implementation.  # labels (Any): 要变换的输入标签。标签的确切类型和结构可能因具体实现而异

        Returns:
            (Any): The transformed labels. In the base implementation, this is identical to the input.  # 返回: 变换后的标签。在基本实现中，这与输入相同

        Examples:
            >>> transform = BaseTransform()  # 创建BaseTransform实例
            >>> original_labels = [1, 2, 3]  # 定义原始标签
            >>> transformed_labels = transform.apply_image(original_labels)  # 应用图像变换
            >>> print(transformed_labels)  # 打印变换后的标签
            [1, 2, 3]  # 输出未变的标签
        """
        pass  # 不执行任何操作

    def apply_instances(self, labels):
        """
        Applies transformations to object instances in labels.  # 将变换应用于标签中的对象实例

        This method is responsible for applying various transformations to object instances within the given
        labels. It is designed to be overridden by subclasses to implement specific instance transformation
        logic.  # 此方法负责对给定标签中的对象实例应用各种变换。它旨在被子类重写以实现特定的实例变换逻辑

        Args:
            labels (Dict): A dictionary containing label information, including object instances.  # labels (Dict): 包含标签信息的字典，包括对象实例

        Returns:
            (Dict): The modified labels dictionary with transformed object instances.  # 返回: 修改后的标签字典，包含变换后的对象实例

        Examples:
            >>> transform = BaseTransform()  # 创建BaseTransform实例
            >>> labels = {"instances": Instances(xyxy=torch.rand(5, 4), cls=torch.randint(0, 80, (5,)))}  # 定义标签字典
            >>> transformed_labels = transform.apply_instances(labels)  # 应用实例变换
        """
        pass  # 不执行任何操作

    def apply_semantic(self, labels):
        """
        Applies semantic segmentation transformations to an image.  # 将语义分割变换应用于图像

        This method is intended to be overridden by subclasses to implement specific semantic segmentation
        transformations. In its base form, it does not perform any operations.  # 此方法旨在被子类重写以实现特定的语义分割变换。在其基本形式中，它不执行任何操作

        Args:
            labels (Any): The input labels or semantic segmentation mask to be transformed.  # labels (Any): 要变换的输入标签或语义分割掩码

        Returns:
            (Any): The transformed semantic segmentation mask or labels.  # 返回: 变换后的语义分割掩码或标签

        Examples:
            >>> transform = BaseTransform()  # 创建BaseTransform实例
            >>> semantic_mask = np.zeros((100, 100), dtype=np.uint8)  # 定义语义掩码
            >>> transformed_mask = transform.apply_semantic(semantic_mask)  # 应用语义变换
        """
        pass  # 不执行任何操作

    def __call__(self, labels):
        """
        Applies all label transformations to an image, instances, and semantic masks.  # 将所有标签变换应用于图像、实例和语义掩码

        This method orchestrates the application of various transformations defined in the BaseTransform class
        to the input labels. It sequentially calls the apply_image and apply_instances methods to process the
        image and object instances, respectively.  # 此方法协调BaseTransform类中定义的各种变换应用于输入标签。它依次调用apply_image和apply_instances方法来处理图像和对象实例

        Args:
            labels (Dict): A dictionary containing image data and annotations. Expected keys include 'img' for
                the image data, and 'instances' for object instances.  # labels (Dict): 包含图像数据和注释的字典。预期的键包括'img'（图像数据）和'instances'（对象实例）

        Returns:
            (Dict): The input labels dictionary with transformed image and instances.  # 返回: 包含变换后图像和实例的输入标签字典

        Examples:
            >>> transform = BaseTransform()  # 创建BaseTransform实例
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": []}  # 定义标签字典
            >>> transformed_labels = transform(labels)  # 应用变换
        """
        self.apply_image(labels)  # 应用图像变换
        self.apply_instances(labels)  # 应用实例变换
        self.apply_semantic(labels)  # 应用语义变换


class Compose:
    """
    A class for composing multiple image transformations.  # 组合多个图像变换的类

    Attributes:
        transforms (List[Callable]): A list of transformation functions to be applied sequentially.  # 属性: transforms (List[Callable]): 一系列将按顺序应用的变换函数列表

    Methods:
        __call__: Applies a series of transformations to input data.  # __call__: 将一系列变换应用于输入数据
        append: Appends a new transform to the existing list of transforms.  # append: 将新变换附加到现有变换列表
        insert: Inserts a new transform at a specified index in the list of transforms.  # insert: 在变换列表中的指定索引处插入新变换
        __getitem__: Retrieves a specific transform or a set of transforms using indexing.  # __getitem__: 使用索引检索特定变换或一组变换
        __setitem__: Sets a specific transform or a set of transforms using indexing.  # __setitem__: 使用索引设置特定变换或一组变换
        tolist: Converts the list of transforms to a standard Python list.  # tolist: 将变换列表转换为标准Python列表

    Examples:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]  # 定义变换列表
        >>> compose = Compose(transforms)  # 创建Compose实例
        >>> transformed_data = compose(data)  # 应用组合变换
        >>> compose.append(CenterCrop((224, 224)))  # 添加中心裁剪变换
        >>> compose.insert(0, RandomFlip())  # 在开头插入随机翻转变换
    """

    def __init__(self, transforms):
        """
        Initializes the Compose object with a list of transforms.  # 使用变换列表初始化Compose对象

        Args:
            transforms (List[Callable]): A list of callable transform objects to be applied sequentially.  # Args: transforms (List[Callable]): 一系列可调用的变换对象，将按顺序应用

        Examples:
            >>> from ultralytics.data.augment import Compose, RandomHSV, RandomFlip  # 从ultralytics库导入组合和随机变换
            >>> transforms = [RandomHSV(), RandomFlip()]  # 定义变换列表
            >>> compose = Compose(transforms)  # 创建Compose实例
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]  # 如果transforms是列表，则直接赋值；否则将其放入列表中

    def __call__(self, data):
        """
        Applies a series of transformations to input data. This method sequentially applies each transformation in the
        Compose object's list of transforms to the input data.  # 将一系列变换应用于输入数据。此方法依次将Compose对象的变换列表中的每个变换应用于输入数据

        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the
                transformations in the list.  # Args: data (Any): 要变换的输入数据。根据列表中的变换类型，数据可以是任何类型

        Returns:
            (Any): The transformed data after applying all transformations in sequence.  # 返回: 应用所有变换后变换的数据

        Examples:
            >>> transforms = [Transform1(), Transform2(), Transform3()]  # 定义变换列表
            >>> compose = Compose(transforms)  # 创建Compose实例
            >>> transformed_data = compose(input_data)  # 应用组合变换
        """
        for t in self.transforms:  # 遍历所有变换
            data = t(data)  # 应用变换
        return data  # 返回变换后的数据

    def append(self, transform):
        """
        Appends a new transform to the existing list of transforms.  # 将新变换附加到现有变换列表

        Args:
            transform (BaseTransform): The transformation to be added to the composition.  # Args: transform (BaseTransform): 要添加到组合中的变换

        Examples:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])  # 创建Compose实例
            >>> compose.append(RandomHSV())  # 添加随机HSV变换
        """
        self.transforms.append(transform)  # 将变换添加到列表中

    def insert(self, index, transform):
        """
        Inserts a new transform at a specified index in the existing list of transforms.  # 在现有变换列表中的指定索引处插入新变换

        Args:
            index (int): The index at which to insert the new transform.  # Args: index (int): 插入新变换的索引
            transform (BaseTransform): The transform object to be inserted.  # Args: transform (BaseTransform): 要插入的变换对象

        Examples:
            >>> compose = Compose([Transform1(), Transform2()])  # 创建Compose实例
            >>> compose.insert(1, Transform3())  # 在索引1处插入Transform3
            >>> len(compose.transforms)  # 获取变换列表的长度
            3  # 输出变换列表的长度
        """
        self.transforms.insert(index, transform)  # 在指定索引处插入变换

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """
        Retrieves a specific transform or a set of transforms using indexing.  # 使用索引检索特定变换或一组变换

        Args:
            index (int | List[int]): Index or list of indices of the transforms to retrieve.  # Args: index (int | List[int]): 要检索的变换的索引或索引列表

        Returns:
            (Compose): A new Compose object containing the selected transform(s).  # 返回: 包含选定变换的新Compose对象

        Raises:
            AssertionError: If the index is not of type int or list.  # 抛出: AssertionError: 如果索引不是int或list类型

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]  # 定义变换列表
            >>> compose = Compose(transforms)  # 创建Compose实例
            >>> single_transform = compose[1]  # 返回仅包含RandomPerspective的Compose对象
            >>> multiple_transforms = compose[0:2]  # 返回包含RandomFlip和RandomPerspective的Compose对象
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"  # 确保索引是int或list类型
        index = [index] if isinstance(index, int) else index  # 如果是int，则转换为列表
        return Compose([self.transforms[i] for i in index])  # 返回包含选定变换的新Compose对象

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """
        Sets one or more transforms in the composition using indexing.  # 使用索引在组合中设置一个或多个变换

        Args:
            index (int | List[int]): Index or list of indices to set transforms at.  # Args: index (int | List[int]): 要设置变换的索引或索引列表
            value (Any | List[Any]): Transform or list of transforms to set at the specified index(es).  # Args: value (Any | List[Any]): 要在指定索引处设置的变换或变换列表

        Raises:
            AssertionError: If index type is invalid, value type doesn't match index type, or index is out of range.  # 抛出: AssertionError: 如果索引类型无效，值类型与索引类型不匹配，或索引超出范围

        Examples:
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])  # 创建Compose实例
            >>> compose[1] = NewTransform()  # 替换第二个变换
            >>> compose[0:2] = [NewTransform1(), NewTransform2()]  # 替换前两个变换
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"  # 确保索引是int或list类型
        if isinstance(index, list):  # 如果索引是列表
            assert isinstance(value, list), (  # 确保值也是列表
                f"The indices should be the same type as values, but got {type(index)} and {type(value)}"  # 抛出类型不匹配的错误
            )
        if isinstance(index, int):  # 如果索引是int
            index, value = [index], [value]  # 转换为列表
        for i, v in zip(index, value):  # 遍历索引和值
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."  # 确保索引在范围内
            self.transforms[i] = v  # 设置变换

    def tolist(self):
        """
        Converts the list of transforms to a standard Python list.  # 将变换列表转换为标准Python列表
    
        Returns:
            (List): A list containing all the transform objects in the Compose instance.  # 返回: 包含Compose实例中所有变换对象的列表
    
        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]  # 定义变换列表
            >>> compose = Compose(transforms)  # 创建Compose实例
            >>> transform_list = compose.tolist()  # 获取变换列表
            >>> print(len(transform_list))  # 打印变换列表的长度
            3  # 输出变换列表的长度
        """
        return self.transforms  # 返回变换列表
    
    def __repr__(self):
        """
        Returns a string representation of the Compose object.  # 返回Compose对象的字符串表示
    
        Returns:
            (str): A string representation of the Compose object, including the list of transforms.  # 返回: Compose对象的字符串表示，包括变换列表
    
        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]  # 定义变换列表
            >>> compose = Compose(transforms)  # 创建Compose实例
            >>> print(compose)  # 打印Compose对象
            Compose([  # 输出Compose对象的字符串表示
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"  # 返回Compose对象的字符串表示
 
class BaseMixTransform:
     """
     Base class for mix transformations like MixUp and Mosaic.  # 混合变换的基类，如MixUp和Mosaic
 
     This class provides a foundation for implementing mix transformations on datasets. It handles the
     probability-based application of transforms and manages the mixing of multiple images and labels.  # 此类为在数据集上实现混合变换提供基础。它处理基于概率的变换应用，并管理多个图像和标签的混合
 
     Attributes:
         dataset (Any): The dataset object containing images and labels.  # 属性: dataset (Any): 包含图像和标签的数据集对象
         pre_transform (Callable | None): Optional transform to apply before mixing.  # 属性: pre_transform (Callable | None): 在混合前应用的可选变换
         p (float): Probability of applying the mix transformation.  # 属性: p (float): 应用混合变换的概率
 
     Methods:
         __call__: Applies the mix transformation to the input labels.  # __call__: 将混合变换应用于输入标签
         _mix_transform: Abstract method to be implemented by subclasses for specific mix operations.  # _mix_transform: 抽象方法，由子类实现特定的混合操作
         get_indexes: Abstract method to get indexes of images to be mixed.  # get_indexes: 抽象方法，用于获取要混合的图像索引
         _update_label_text: Updates label text for mixed images.  # _update_label_text: 更新混合图像的标签文本
 
     Examples:
         >>> class CustomMixTransform(BaseMixTransform):  # 定义自定义混合变换类
         ...     def _mix_transform(self, labels):  # 实现混合逻辑
         ...         # Implement custom mix logic here  # 在此实现自定义混合逻辑
         ...         return labels  # 返回标签
         ...
         ...     def get_indexes(self):  # 获取要混合的图像索引
         ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]  # 随机返回三个索引
         >>> dataset = YourDataset()  # 创建数据集对象
         >>> transform = CustomMixTransform(dataset, p=0.5)  # 创建自定义混合变换实例
         >>> mixed_labels = transform(original_labels)  # 应用混合变换
     """
 
     def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
         """
         Initializes the BaseMixTransform object for mix transformations like MixUp and Mosaic.  # 初始化BaseMixTransform对象，用于混合变换，如MixUp和Mosaic
 
         This class serves as a base for implementing mix transformations in image processing pipelines.  # 此类作为在图像处理管道中实现混合变换的基础
 
         Args:
             dataset (Any): The dataset object containing images and labels for mixing.  # Args: dataset (Any): 包含要混合的图像和标签的数据集对象
             pre_transform (Callable | None): Optional transform to apply before mixing.  # Args: pre_transform (Callable | None): 在混合前应用的可选变换
             p (float): Probability of applying the mix transformation. Should be in the range [0.0, 1.0].  # Args: p (float): 应用混合变换的概率，范围应在[0.0, 1.0]之间
 
         Examples:
             >>> dataset = YOLODataset("path/to/data")  # 创建YOLO数据集对象
             >>> pre_transform = Compose([RandomFlip(), RandomPerspective()])  # 定义预处理变换
             >>> mix_transform = BaseMixTransform(dataset, pre_transform, p=0.5)  # 创建BaseMixTransform实例
         """
         self.dataset = dataset  # 将数据集对象赋值给实例属性
         self.pre_transform = pre_transform  # 将预处理变换赋值给实例属性
         self.p = p  # 将概率赋值给实例属性
 
     def __call__(self, labels):
         """
         Applies pre-processing transforms and mixup/mosaic transforms to labels data.  # 将预处理变换和混合变换应用于标签数据
 
         This method determines whether to apply the mix transform based on a probability factor. If applied, it
         selects additional images, applies pre-transforms if specified, and then performs the mix transform.  # 此方法根据概率因子确定是否应用混合变换。如果应用，则选择其他图像，应用预处理变换（如果指定），然后执行混合变换
 
         Args:
             labels (Dict): A dictionary containing label data for an image.  # Args: labels (Dict): 包含图像标签数据的字典
 
         Returns:
             (Dict): The transformed labels dictionary, which may include mixed data from other images.  # 返回: 变换后的标签字典，可能包括来自其他图像的混合数据
 
         Examples:
             >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)  # 创建BaseMixTransform实例
             >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})  # 应用混合变换
         """
         if random.uniform(0, 1) > self.p:  # 根据概率判断是否应用混合变换
             return labels  # 如果不应用，直接返回原标签
 
         # Get index of one or three other images  # 获取其他一张或三张图像的索引
         indexes = self.get_indexes()  # 获取要混合的图像索引
         if isinstance(indexes, int):  # 如果索引是整数
             indexes = [indexes]  # 转换为列表
 
         # Get images information will be used for Mosaic or MixUp  # 获取用于Mosaic或MixUp的图像信息
         mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]  # 从数据集中获取混合图像和标签
 
         if self.pre_transform is not None:  # 如果有预处理变换
             for i, data in enumerate(mix_labels):  # 遍历混合标签
                 mix_labels[i] = self.pre_transform(data)  # 应用预处理变换
         labels["mix_labels"] = mix_labels  # 将混合标签添加到原标签中
 
         # Update cls and texts  # 更新类别和文本
         labels = self._update_label_text(labels)  # 更新标签文本
         # Mosaic or MixUp  # 执行Mosaic或MixUp
         labels = self._mix_transform(labels)  # 应用混合变换
         labels.pop("mix_labels", None)  # 移除混合标签
         return labels  # 返回变换后的标签
 
     def _mix_transform(self, labels):
         """
         Applies MixUp or Mosaic augmentation to the label dictionary.  # 将MixUp或Mosaic增强应用于标签字典
 
         This method should be implemented by subclasses to perform specific mix transformations like MixUp or
         Mosaic. It modifies the input label dictionary in-place with the augmented data.  # 此方法应由子类实现，以执行特定的混合变换，如MixUp或Mosaic。它就地修改输入标签字典，添加增强数据
 
         Args:
             labels (Dict): A dictionary containing image and label data. Expected to have a 'mix_labels' key
                 with a list of additional image and label data for mixing.  # Args: labels (Dict): 包含图像和标签数据的字典。预期包含'mix_labels'键，值为要混合的其他图像和标签数据的列表
 
         Returns:
             (Dict): The modified labels dictionary with augmented data after applying the mix transform.  # 返回: 经过混合变换后，包含增强数据的修改标签字典
 
         Examples:
             >>> transform = BaseMixTransform(dataset)  # 创建BaseMixTransform实例
             >>> labels = {"image": img, "bboxes": boxes, "mix_labels": [{"image": img2, "bboxes": boxes2}]}  # 定义标签字典
             >>> augmented_labels = transform._mix_transform(labels)  # 应用混合变换
         """
         raise NotImplementedError  # 抛出未实现错误，需在子类中实现
 
     def get_indexes(self):
         """
         Gets a list of shuffled indexes for mosaic augmentation.  # 获取用于马赛克增强的随机索引列表
 
         Returns:
             (List[int]): A list of shuffled indexes from the dataset.  # 返回: 数据集中的随机索引列表
 
         Examples:
             >>> transform = BaseMixTransform(dataset)  # 创建BaseMixTransform实例
             >>> indexes = transform.get_indexes()  # 获取索引
             >>> print(indexes)  # [3, 18, 7, 2]  # 打印索引
         """
         raise NotImplementedError  # 抛出未实现错误，需在子类中实现
 
     @staticmethod
     def _update_label_text(labels):
         """
         Updates label text and class IDs for mixed labels in image augmentation.  # 更新图像增强中混合标签的标签文本和类别ID
 
         This method processes the 'texts' and 'cls' fields of the input labels dictionary and any mixed labels,
         creating a unified set of text labels and updating class IDs accordingly.  # 此方法处理输入标签字典及任何混合标签的'texts'和'cls'字段，创建统一的文本标签集，并相应更新类别ID
 
         Args:
             labels (Dict): A dictionary containing label information, including 'texts' and 'cls' fields,
                 and optionally a 'mix_labels' field with additional label dictionaries.  # Args: labels (Dict): 包含标签信息的字典，包括'texts'和'cls'字段，以及可选的'mix_labels'字段，值为额外标签字典
 
         Returns:
             (Dict): The updated labels dictionary with unified text labels and updated class IDs.  # 返回: 更新后的标签字典，包含统一的文本标签和更新的类别ID
 
         Examples:
             >>> labels = {  # 定义标签字典
             ...     "texts": [["cat"], ["dog"]],  # 包含文本标签
             ...     "cls": torch.tensor([[0], [1]]),  # 包含类别ID
             ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],  # 包含混合标签
             ... }
             >>> updated_labels = self._update_label_text(labels)  # 更新标签
             >>> print(updated_labels["texts"])  # 打印更新后的文本标签
             [['cat'], ['dog'], ['bird'], ['fish']]  # 输出更新后的文本标签
             >>> print(updated_labels["cls"])  # 打印更新后的类别ID
             tensor([[0],
                     [1]])  # 输出更新后的类别ID
             >>> print(updated_labels["mix_labels"][0]["cls"])  # 打印混合标签的类别ID
             tensor([[2],
                     [3]])  # 输出混合标签的类别ID
         """
         if "texts" not in labels:  # 如果标签中没有'texts'字段
             return labels  # 直接返回标签
 
         mix_texts = sum([labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], [])  # 合并文本标签
         mix_texts = list({tuple(x) for x in mix_texts})  # 去重
         text2id = {text: i for i, text in enumerate(mix_texts)}  # 创建文本到ID的映射
 
         for label in [labels] + labels["mix_labels"]:  # 遍历所有标签
             for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):  # 遍历类别ID
                 text = label["texts"][int(cls)]  # 获取对应的文本
                 label["cls"][i] = text2id[tuple(text)]  # 更新类别ID
             label["texts"] = mix_texts  # 更新文本标签
         return labels  # 返回更新后的标签

class MixUp(BaseMixTransform):
    """
    Applies MixUp augmentation to image datasets.
    对图像数据集应用MixUp增强。

    This class implements the MixUp augmentation technique as described in the paper "mixup: Beyond Empirical Risk
    Minimization" (https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.
    该类实现了MixUp增强技术，如论文“mixup: Beyond Empirical Risk Minimization”中所述。MixUp通过使用随机权重组合两张图像及其标签。

    Attributes:
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        dataset（Any）：MixUp增强将应用于的数据集。
        pre_transform (Callable | None): Optional transform to apply before MixUp.
        pre_transform（Callable | None）：在MixUp之前应用的可选转换。
        p (float): Probability of applying MixUp augmentation.
        p（float）：应用MixUp增强的概率。

    Methods:
        get_indexes: Returns a random index from the dataset.
        get_indexes：返回数据集中的随机索引。
        _mix_transform: Applies MixUp augmentation to the input labels.
        _mix_transform：对输入标签应用MixUp增强。

    Examples:
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """
        Initializes the MixUp augmentation object.
        初始化MixUp增强对象。

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel
        values and labels. This implementation is designed for use with the Ultralytics YOLO framework.
        MixUp是一种图像增强技术，通过对两个图像的像素值和标签进行加权求和来组合它们。此实现旨在与Ultralytics YOLO框架一起使用。

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied.
            dataset（Any）：MixUp增强将应用于的数据集。
            pre_transform (Callable | None): Optional transform to apply to images before MixUp.
            pre_transform（Callable | None）：在MixUp之前应用于图像的可选转换。
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0, 1].
            p（float）：对图像应用MixUp增强的概率。必须在[0, 1]范围内。

        Examples:
            >>> from ultralytics.data.dataset import YOLODataset
            >>> dataset = YOLODataset("path/to/data.yaml")
            >>> mixup = MixUp(dataset, pre_transform=None, p=0.5)
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """
        Get a random index from the dataset.
        从数据集中获取随机索引。

        This method returns a single random index from the dataset, which is used to select an image for MixUp
        augmentation.
        此方法返回数据集中的单个随机索引，用于选择图像进行MixUp增强。

        Returns:
            (int): A random integer index within the range of the dataset length.
            （int）：数据集长度范围内的随机整数索引。

        Examples:
            >>> mixup = MixUp(dataset)
            >>> index = mixup.get_indexes()
            >>> print(index)
            42
        """
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """
        Applies MixUp augmentation to the input labels.
        对输入标签应用MixUp增强。

        This method implements the MixUp augmentation technique as described in the paper
        "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412).
        此方法实现了MixUp增强技术，如论文“mixup: Beyond Empirical Risk Minimization”中所述。

        Args:
            labels (Dict): A dictionary containing the original image and label information.
            labels（Dict）：包含原始图像和标签信息的字典。

        Returns:
            (Dict): A dictionary containing the mixed-up image and combined label information.
            （Dict）：包含混合图像和组合标签信息的字典。

        Examples:
            >>> mixer = MixUp(dataset)
            >>> mixed_labels = mixer._mix_transform(labels)
        """
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        # mixup比例，alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        # 将两张图像按比例混合并转换为无符号8位整数
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        # 将实例标签连接在一起
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        # 将类别标签连接在一起
        return labels


class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation for image datasets.
    适用于图像数据集的马赛克增强。

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    该类通过将多张（4或9）图像组合成一张马赛克图像来执行马赛克增强。

    The augmentation is applied to a dataset with a given probability.
    该增强以给定的概率应用于数据集。

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        dataset：马赛克增强应用于的数据集。
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        imgsz（int）：单张图像经过马赛克处理后的大小（高度和宽度）。
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        p（float）：应用马赛克增强的概率。必须在0到1之间。
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        n（int）：网格大小，可以是4（2x2）或9（3x3）。
        border (Tuple[int, int]): Border size for width and height.
        border（Tuple[int, int]）：宽度和高度的边框大小。

    Methods:
        get_indexes: Returns a list of random indexes from the dataset.
        get_indexes：返回数据集中随机索引的列表。
        _mix_transform: Applies mixup transformation to the input image and labels.
        _mix_transform：对输入图像和标签应用混合转换。
        _mosaic3: Creates a 1x3 image mosaic.
        _mosaic3：创建1x3的图像马赛克。
        _mosaic4: Creates a 2x2 image mosaic.
        _mosaic4：创建2x2的图像马赛克。
        _mosaic9: Creates a 3x3 image mosaic.
        _mosaic9：创建3x3的图像马赛克。
        _update_labels: Updates labels with padding.
        _update_labels：使用填充更新标签。
        _cat_labels: Concatenates labels and clips mosaic border instances.
        _cat_labels：连接标签并裁剪马赛克边界实例。

    Examples:
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        """
        Initializes the Mosaic augmentation object.
        初始化马赛克增强对象。

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
        该类通过将多张（4或9）图像组合成一张马赛克图像来执行马赛克增强。

        The augmentation is applied to a dataset with a given probability.
        该增强以给定的概率应用于数据集。

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            dataset（Any）：马赛克增强应用于的数据集。
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            imgsz（int）：单张图像经过马赛克处理后的大小（高度和宽度）。
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            p（float）：应用马赛克增强的概率。必须在0到1之间。
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
            n（int）：网格大小，可以是4（2x2）或9（3x3）。

        Examples:
            >>> from ultralytics.data.augment import Mosaic
            >>> dataset = YourDataset(...)
            >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        """
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        # 确保概率在[0, 1]范围内
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        # 确保网格大小为4或9
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        # 边框大小，宽度和高度
        self.n = n

    def get_indexes(self, buffer=True):
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.
        返回用于马赛克增强的数据集中随机索引的列表。

        This method selects random image indexes either from a buffer or from the entire dataset, depending on
        the 'buffer' parameter. It is used to choose images for creating mosaic augmentations.
        此方法根据'buffer'参数从缓冲区或整个数据集中选择随机图像索引。用于选择用于创建马赛克增强的图像。

        Args:
            buffer (bool): If True, selects images from the dataset buffer. If False, selects from the entire
                dataset.
            buffer（bool）：如果为True，则从数据集缓冲区选择图像。如果为False，则从整个数据集中选择。

        Returns:
            (List[int]): A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).
            （List[int]）：随机图像索引的列表。列表的长度为n-1，其中n是用于马赛克的图像数量（根据n是4还是9，可能为3或8）。

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        if buffer:  # select images from buffer
            # 从缓冲区选择图像
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            # 从整个数据集中选择图像
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """
        Applies mosaic augmentation to the input image and labels.
        对输入图像和标签应用马赛克增强。

        This method combines multiple images (3, 4, or 9) into a single mosaic image based on the 'n' attribute.
        此方法根据'n'属性将多张图像（3、4或9）组合成一张马赛克图像。

        It ensures that rectangular annotations are not present and that there are other images available for
        mosaic augmentation.
        它确保没有矩形注释，并且有其他图像可用于马赛克增强。

        Args:
            labels (Dict): A dictionary containing image data and annotations. Expected keys include:
                - 'rect_shape': Should be None as rect and mosaic are mutually exclusive.
                - 'mix_labels': A list of dictionaries containing data for other images to be used in the mosaic.
            labels（Dict）：包含图像数据和注释的字典。预期的键包括：
                - 'rect_shape'：应为None，因为矩形和马赛克是互斥的。
                - 'mix_labels'：包含用于马赛克的其他图像数据的字典列表。

        Returns:
            (Dict): A dictionary containing the mosaic-augmented image and updated annotations.
            （Dict）：包含马赛克增强图像和更新注释的字典。

        Raises:
            AssertionError: If 'rect_shape' is not None or if 'mix_labels' is empty.
            引发AssertionError：如果'rect_shape'不为None或'mix_labels'为空。

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> augmented_data = mosaic._mix_transform(labels)
        """
        assert labels.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        # 确保'rect_shape'为None，因为矩形和马赛克是互斥的
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        # 确保'mix_labels'不为空
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # This code is modified for mosaic3 method.
        # 根据n的值调用相应的马赛克方法

    def _mosaic3(self, labels):
        """
        Creates a 1x3 image mosaic by combining three images.
        通过组合三张图像创建1x3的图像马赛克。

        This method arranges three images in a horizontal layout, with the main image in the center and two
        additional images on either side. It's part of the Mosaic augmentation technique used in object detection.
        此方法将三张图像以水平布局排列，主图像位于中心，两张附加图像位于两侧。它是用于目标检测的马赛克增强技术的一部分。

        Args:
            labels (Dict): A dictionary containing image and label information for the main (center) image.
                Must include 'img' key with the image array, and 'mix_labels' key with a list of two
                dictionaries containing information for the side images.
            labels（Dict）：包含主（中心）图像的图像和标签信息的字典。必须包含'img'键和图像数组，以及'mix_labels'键和包含侧面图像信息的两个字典的列表。

        Returns:
            (Dict): A dictionary with the mosaic image and updated labels. Keys include:
                - 'img' (np.ndarray): The mosaic image array with shape (H, W, C).
                - Other keys from the input labels, updated to reflect the new image dimensions.
            （Dict）：包含马赛克图像和更新标签的字典。键包括：
                - 'img'（np.ndarray）：形状为（H，W，C）的马赛克图像数组。
                - 输入标签中的其他键，已更新以反映新图像尺寸。

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=3)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(2)],
            ... }
            >>> result = mosaic._mosaic3(labels)
            >>> print(result["img"].shape)
            (640, 640, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            # 加载图像
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img3
            # 在img3中放置图像
            if i == 0:  # center
                # 中心
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 3 tiles
                # 创建一个填充为114的基础图像，大小为3*s
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
                # xmin, ymin, xmax, ymax（基础）坐标
            elif i == 1:  # right
                # 右侧
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:  # left
                # 左侧
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coordinates
            # 分配坐标

            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img3[ymin:ymax, xmin:xmax]
            # 将图像放置在img3的指定位置
            # hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            # 假设为imgsz*2的马赛克大小
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    def _mosaic4(self, labels):
        """
        Creates a 2x2 image mosaic from four input images.
        从四张输入图像创建2x2的图像马赛克。

        This method combines four images into a single mosaic image by placing them in a 2x2 grid. It also
        updates the corresponding labels for each image in the mosaic.
        此方法将四张图像组合成一张马赛克图像，按2x2网格排列。它还更新马赛克中每张图像的相应标签。

        Args:
            labels (Dict): A dictionary containing image data and labels for the base image (index 0) and three
                additional images (indices 1-3) in the 'mix_labels' key.
            labels（Dict）：包含基础图像（索引0）和'mix_labels'键中三张附加图像（索引1-3）的图像数据和标签的字典。

        Returns:
            (Dict): A dictionary containing the mosaic image and updated labels. The 'img' key contains the mosaic
                image as a numpy array, and other keys contain the combined and adjusted labels for all four images.
            （Dict）：包含马赛克图像和更新标签的字典。'img'键包含马赛克图像的numpy数组，其他键包含四张图像的组合和调整后的标签。

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(3)],
            ... }
            >>> result = mosaic._mosaic4(labels)
            >>> assert result["img"].shape == (1280, 1280, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        # 马赛克中心的x和y坐标
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            # 加载图像
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            # 在img4中放置图像
            if i == 0:  # top left
                # 左上角
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                # 创建一个填充为114的基础图像，大小为2*s
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # 大图像的xmin, ymin, xmax, ymax
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                # 小图像的xmin, ymin, xmax, ymax
            elif i == 1:  # top right
                # 右上角
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                # 左下角
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                # 右下角
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            # 将图像放置在img4的指定位置
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels):
        """
        Creates a 3x3 image mosaic from the input image and eight additional images.
        从输入图像和八张附加图像创建3x3的图像马赛克。

        This method combines nine images into a single mosaic image. The input image is placed at the center,
        and eight additional images from the dataset are placed around it in a 3x3 grid pattern.
        此方法将九张图像组合成一张马赛克图像。输入图像放置在中心，来自数据集的八张附加图像围绕它以3x3网格模式放置。

        Args:
            labels (Dict): A dictionary containing the input image and its associated labels. It should have
                the following keys:
                - 'img' (numpy.ndarray): The input image.
                - 'resized_shape' (Tuple[int, int]): The shape of the resized image (height, width).
                - 'mix_labels' (List[Dict]): A list of dictionaries containing information for the additional
                  eight images, each with the same structure as the input labels.
            labels（Dict）：包含输入图像及其相关标签的字典。它应具有以下键：
                - 'img'（numpy.ndarray）：输入图像。
                - 'resized_shape'（Tuple[int, int]）：调整大小后的图像形状（高度，宽度）。
                - 'mix_labels'（List[Dict]）：包含八张附加图像信息的字典列表，每个字典的结构与输入标签相同。

        Returns:
            (Dict): A dictionary containing the mosaic image and updated labels. It includes the following keys:
                - 'img' (numpy.ndarray): The final mosaic image.
                - Other keys from the input labels, updated to reflect the new mosaic arrangement.
            （Dict）：包含马赛克图像和更新标签的字典。它包括以下键：
                - 'img'（numpy.ndarray）：最终的马赛克图像。
                - 输入标签中的其他键，已更新以反映新的马赛克排列。

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=9)
            >>> input_labels = dataset[0]
            >>> mosaic_result = mosaic._mosaic9(input_labels)
            >>> mosaic_image = mosaic_result["img"]
        """
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1  # height, width previous
        # height, width previous
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            # 加载图像
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img9
            # 在img9中放置图像
            if i == 0:  # center
                # 中心
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                # 创建一个填充为114的基础图像，大小为3*s
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
                # xmin, ymin, xmax, ymax（基础）坐标
            elif i == 1:  # top
                # 上方
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                # 右上方
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                # 右侧
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                # 右下方
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                # 下方
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                # 左下方
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                # 左侧
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                # 左上方
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coordinates
            # 分配坐标

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img9[ymin:ymax, xmin:xmax]
            # 将图像放置在img9的指定位置
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            # 假设为imgsz*2的马赛克大小
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        """
        Updates label coordinates with padding values.
        使用填充值更新标签坐标。

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding
        values. It also denormalizes the coordinates if they were previously normalized.
        此方法通过添加填充值调整标签中对象实例的边界框坐标。如果坐标之前是归一化的，它还会将其反归一化。

        Args:
            labels (Dict): A dictionary containing image and instance information.
            labels（Dict）：包含图像和实例信息的字典。
            padw (int): Padding width to be added to the x-coordinates.
            padw（int）：要添加到x坐标的填充宽度。
            padh (int): Padding height to be added to the y-coordinates.
            padh（int）：要添加到y坐标的填充高度。

        Returns:
            (Dict): Updated labels dictionary with adjusted instance coordinates.
            （Dict）：更新的标签字典，包含调整后的实例坐标。

        Examples:
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        # 转换边界框格式为xyxy
        labels["instances"].denormalize(nw, nh)
        # 反归一化坐标
        labels["instances"].add_padding(padw, padh)
        # 添加填充
        return labels

    def _cat_labels(self, mosaic_labels):
        """
        Concatenates and processes labels for mosaic augmentation.
        连接并处理马赛克增强的标签。

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the
        mosaic border, and removes zero-area boxes.
        此方法将用于马赛克增强的多张图像的标签组合在一起，裁剪实例到马赛克边界，并移除零面积框。

        Args:
            mosaic_labels (List[Dict]): A list of label dictionaries for each image in the mosaic.
            mosaic_labels（List[Dict]）：每张马赛克图像的标签字典列表。
        
        Returns:
            (Dict): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (Tuple[int, int]): Original shape of the first image.
                - resized_shape (Tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - mosaic_border (Tuple[int, int]): Mosaic border size.
                - texts (List[str], optional): Text labels if present in the original labels.
            （Dict）：包含马赛克图像的连接和处理标签的字典，包括：
                - im_file（str）：马赛克中第一张图像的文件路径。
                - ori_shape（Tuple[int, int]）：第一张图像的原始形状。
                - resized_shape（Tuple[int, int]）：马赛克图像的形状（imgsz * 2，imgsz * 2）。
                - cls（np.ndarray）：连接的类别标签。
                - instances（Instances）：连接的实例注释。
                - mosaic_border（Tuple[int, int]）：马赛克边界大小。
                - texts（List[str]，可选）：如果原始标签中存在文本标签。

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        # 马赛克图像大小
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # 最终标签
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        # 裁剪实例到马赛克边界
        good = final_labels["instances"].remove_zero_area_boxes()
        # 移除零面积框
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):
    """
    Applies MixUp augmentation to image datasets.
    对图像数据集应用MixUp增强。

    This class implements the MixUp augmentation technique as described in the paper "mixup: Beyond Empirical Risk
    Minimization" (https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.
    该类实现了MixUp增强技术，如论文“mixup: Beyond Empirical Risk Minimization”中所述。MixUp通过使用随机权重组合两张图像及其标签。

    Attributes:
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        dataset（Any）：MixUp增强将应用于的数据集。
        pre_transform (Callable | None): Optional transform to apply before MixUp.
        pre_transform（Callable | None）：在MixUp之前应用的可选转换。
        p (float): Probability of applying MixUp augmentation.
        p（float）：应用MixUp增强的概率。

    Methods:
        get_indexes: Returns a random index from the dataset.
        get_indexes：返回数据集中的随机索引。
        _mix_transform: Applies MixUp augmentation to the input labels.
        _mix_transform：对输入标签应用MixUp增强。

    Examples:
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """
        Initializes the MixUp augmentation object.
        初始化MixUp增强对象。

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel
        values and labels. This implementation is designed for use with the Ultralytics YOLO framework.
        MixUp是一种图像增强技术，通过对两个图像的像素值和标签进行加权求和来组合它们。此实现旨在与Ultralytics YOLO框架一起使用。

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied.
            dataset（Any）：MixUp增强将应用于的数据集。
            pre_transform (Callable | None): Optional transform to apply to images before MixUp.
            pre_transform（Callable | None）：在MixUp之前应用于图像的可选转换。
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0, 1].
            p（float）：对图像应用MixUp增强的概率。必须在[0, 1]范围内。

        Examples:
            >>> from ultralytics.data.dataset import YOLODataset
            >>> dataset = YOLODataset("path/to/data.yaml")
            >>> mixup = MixUp(dataset, pre_transform=None, p=0.5)
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """
        Get a random index from the dataset.
        从数据集中获取随机索引。

        This method returns a single random index from the dataset, which is used to select an image for MixUp
        augmentation.
        此方法返回数据集中的单个随机索引，用于选择图像进行MixUp增强。

        Returns:
            (int): A random integer index within the range of the dataset length.
            （int）：数据集长度范围内的随机整数索引。

        Examples:
            >>> mixup = MixUp(dataset)
            >>> index = mixup.get_indexes()
            >>> print(index)
            42
        """
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """
        Applies MixUp augmentation to the input labels.
        对输入标签应用MixUp增强。

        This method implements the MixUp augmentation technique as described in the paper
        "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412).
        此方法实现了MixUp增强技术，如论文“mixup: Beyond Empirical Risk Minimization”中所述。

        Args:
            labels (Dict): A dictionary containing the original image and label information.
            labels（Dict）：包含原始图像和标签信息的字典。

        Returns:
            (Dict): A dictionary containing the mixed-up image and combined label information.
            （Dict）：包含混合图像和组合标签信息的字典。

        Examples:
            >>> mixer = MixUp(dataset)
            >>> mixed_labels = mixer._mix_transform(labels)
        """
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        # mixup比例，alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        # 将两张图像按比例混合并转换为无符号8位整数
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        # 将实例标签连接在一起
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        # 将类别标签连接在一起
        return labels


class RandomPerspective:
    """
    Implements random perspective and affine transformations on images and corresponding annotations.
    实现对图像及其相应注释的随机透视和仿射变换。

    This class applies random rotations, translations, scaling, shearing, and perspective transformations
    to images and their associated bounding boxes, segments, and keypoints. It can be used as part of an
    augmentation pipeline for object detection and instance segmentation tasks.
    该类对图像及其相关的边界框、分段和关键点应用随机旋转、平移、缩放、剪切和透视变换。它可以作为目标检测和实例分割任务的增强管道的一部分。

    Attributes:
        degrees (float): Maximum absolute degree range for random rotations.
        degrees（float）：随机旋转的最大绝对度数范围。
        translate (float): Maximum translation as a fraction of the image size.
        translate（float）：作为图像大小的一部分的最大平移。
        scale (float): Scaling factor range, e.g., scale=0.1 means 0.9-1.1.
        scale（float）：缩放因子范围，例如，scale=0.1表示0.9-1.1。
        shear (float): Maximum shear angle in degrees.
        shear（float）：最大剪切角度（以度为单位）。
        perspective (float): Perspective distortion factor.
        perspective（float）：透视失真因子。
        border (Tuple[int, int]): Mosaic border size as (x, y).
        border（Tuple[int, int]）：马赛克边框大小（x，y）。
        pre_transform (Callable | None): Optional transform to apply before the random perspective.
        pre_transform（Callable | None）：在随机透视之前应用的可选变换。

    Methods:
        affine_transform: Applies affine transformations to the input image.
        affine_transform：对输入图像应用仿射变换。
        apply_bboxes: Transforms bounding boxes using the affine matrix.
        apply_bboxes：使用仿射矩阵转换边界框。
        apply_segments: Transforms segments and generates new bounding boxes.
        apply_segments：转换分段并生成新的边界框。
        apply_keypoints: Transforms keypoints using the affine matrix.
        apply_keypoints：使用仿射矩阵转换关键点。
        __call__: Applies the random perspective transformation to images and annotations.
        __call__：对图像及注释应用随机透视变换。
        box_candidates: Filters transformed bounding boxes based on size and aspect ratio.
        box_candidates：根据大小和宽高比标准过滤变换后的边界框。

    Examples:
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result["img"]
        >>> transformed_instances = result["instances"]
    """

    def __init__(
        self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None
    ):
        """
        Initializes RandomPerspective object with transformation parameters.
        使用变换参数初始化RandomPerspective对象。

        This class implements random perspective and affine transformations on images and corresponding bounding boxes,
        segments, and keypoints. Transformations include rotation, translation, scaling, and shearing.
        该类对图像及其相应的边界框、分段和关键点实现随机透视和仿射变换。变换包括旋转、平移、缩放和剪切。

        Args:
            degrees (float): Degree range for random rotations.
            degrees（float）：随机旋转的度数范围。
            translate (float): Fraction of total width and height for random translation.
            translate（float）：随机平移的总宽度和高度的比例。
            scale (float): Scaling factor interval, e.g., a scale factor of 0.5 allows a resize between 50%-150%.
            scale（float）：缩放因子区间，例如，缩放因子为0.5允许在50%-150%之间调整大小。
            shear (float): Shear intensity (angle in degrees).
            shear（float）：剪切强度（以度为单位）。
            perspective (float): Perspective distortion factor.
            perspective（float）：透视失真因子。
            border (Tuple[int, int]): Tuple specifying mosaic border (top/bottom, left/right).
            border（Tuple[int, int]）：指定马赛克边框（上下、左右）的元组。
            pre_transform (Callable | None): Function/transform to apply to the image before starting the random
                transformation.
            pre_transform（Callable | None）：在开始随机变换之前应用于图像的函数/变换。

        Examples:
            >>> transform = RandomPerspective(degrees=10.0, translate=0.1, scale=0.5, shear=5.0)
            >>> result = transform(labels)  # Apply random perspective to labels
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border
        # 马赛克边框
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """
        Applies a sequence of affine transformations centered around the image center.
        应用围绕图像中心的仿射变换序列。

        This function performs a series of geometric transformations on the input image, including
        translation, perspective change, rotation, scaling, and shearing. The transformations are
        applied in a specific order to maintain consistency.
        此函数对输入图像执行一系列几何变换，包括平移、透视变化、旋转、缩放和剪切。变换以特定顺序应用，以保持一致性。

        Args:
            img (np.ndarray): Input image to be transformed.
            img（np.ndarray）：要变换的输入图像。
            border (Tuple[int, int]): Border dimensions for the transformed image.
            border（Tuple[int, int]）：变换后图像的边框尺寸。

        Returns:
            (Tuple[np.ndarray, np.ndarray, float]): A tuple containing:
                - np.ndarray: Transformed image.
                - np.ndarray: 3x3 transformation matrix.
                - float: Scale factor applied during the transformation.
            （Tuple[np.ndarray, np.ndarray, float]）：包含以下内容的元组：
                - np.ndarray：变换后的图像。
                - np.ndarray：3x3变换矩阵。
                - float：在变换过程中应用的缩放因子。

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> border = (10, 10)
            >>> transformed_img, matrix, scale = affine_transform(img, border)
        """
        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        # x平移（像素）
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
        # y平移（像素）

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        # x透视（关于y）
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)
        # y透视（关于x）

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        # x剪切（度）
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)
        # y剪切（度）

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        # x平移（像素）
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)
        # y平移（像素）

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # 变换矩阵的组合
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            # 图像发生变化
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        """
        Apply affine transformation to bounding boxes.
        对边界框应用仿射变换。

        This function applies an affine transformation to a set of bounding boxes using the provided
        transformation matrix.
        此函数使用提供的变换矩阵对一组边界框应用仿射变换。

        Args:
            bboxes (torch.Tensor): Bounding boxes in xyxy format with shape (N, 4), where N is the number
                of bounding boxes.
            bboxes（torch.Tensor）：形状为（N，4）的xyxy格式边界框，其中N是边界框的数量。
            M (torch.Tensor): Affine transformation matrix with shape (3, 3).
            M（torch.Tensor）：形状为（3，3）的仿射变换矩阵。

        Returns:
            (torch.Tensor): Transformed bounding boxes in xyxy format with shape (N, 4).
            （torch.Tensor）：形状为（N，4）的xyxy格式变换边界框。

        Examples:
            >>> bboxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
            >>> M = torch.eye(3)
            >>> transformed_bboxes = apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments, M):
        """
        Apply affine transformations to segments and generate new bounding boxes.
        对分段应用仿射变换并生成新的边界框。

        This function applies affine transformations to input segments and generates new bounding boxes based on
        the transformed segments. It clips the transformed segments to fit within the new bounding boxes.
        此函数对输入分段应用仿射变换，并根据变换后的分段生成新的边界框。它裁剪变换后的分段以适应新的边界框。

        Args:
            segments (np.ndarray): Input segments with shape (N, M, 2), where N is the number of segments and M is the
                number of points in each segment.
            segments（np.ndarray）：形状为（N，M，2）的输入分段，其中N是分段的数量，M是每个分段中的点数。
            M (np.ndarray): Affine transformation matrix with shape (3, 3).
            M（np.ndarray）：形状为（3，3）的仿射变换矩阵。

        Returns:
            (Tuple[np.ndarray, np.ndarray]): A tuple containing:
                - New bounding boxes with shape (N, 4) in xyxy format.
                - Transformed and clipped segments with shape (N, M, 2).
            （Tuple[np.ndarray, np.ndarray]）：包含以下内容的元组：
                - 形状为（N，4）的新边界框，格式为xyxy。
                - 形状为（N，M，2）的变换和裁剪后的分段。

        Examples:
            >>> segments = np.random.rand(10, 500, 2)  # 10 segments with 500 points each
            >>> M = np.eye(3)  # Identity transformation matrix
            >>> new_bboxes, new_segments = apply_segments(segments, M)
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        """
        Applies affine transformation to keypoints.
        对关键点应用仿射变换。

        This method transforms the input keypoints using the provided affine transformation matrix. It handles
        perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image
        boundaries after transformation.
        此方法使用提供的仿射变换矩阵变换输入关键点。如果需要，它会处理透视缩放，并更新在变换后超出图像边界的关键点的可见性。

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (N, 17, 3), where N is the number of instances,
                17 is the number of keypoints per instance, and 3 represents (x, y, visibility).
            keypoints（np.ndarray）：形状为（N，17，3）的关键点数组，其中N是实例的数量，17是每个实例的关键点数量，3表示（x，y，可见性）。
            M (np.ndarray): 3x3 affine transformation matrix.
            M（np.ndarray）：3x3仿射变换矩阵。

        Returns:
            (np.ndarray): Transformed keypoints array with the same shape as input (N, 17, 3).
            （np.ndarray）：变换后的关键点数组，形状与输入相同（N，17，3）。

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5 instances, 17 keypoints each
            >>> M = np.eye(3)  # Identity transformation
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        """
        Applies random perspective and affine transformations to an image and its associated labels.
        对图像及其相关标签应用随机透视和仿射变换。

        This method performs a series of transformations including rotation, translation, scaling, shearing,
        and perspective distortion on the input image and adjusts the corresponding bounding boxes, segments,
        and keypoints accordingly.
        此方法对输入图像执行一系列变换，包括旋转、平移、缩放、剪切和透视失真，并相应地调整相关的边界框、分段和关键点。

        Args:
            labels (Dict): A dictionary containing image data and annotations.
                Must include:
                    'img' (ndarray): The input image.
                    'cls' (ndarray): Class labels.
                    'instances' (Instances): Object instances with bounding boxes, segments, and keypoints.
                May include:
                    'mosaic_border' (Tuple[int, int]): Border size for mosaic augmentation.
            labels（Dict）：包含图像数据和注释的字典。必须包括：
                - 'img'（ndarray）：输入图像。
                - 'cls'（ndarray）：类别标签。
                - 'instances'（Instances）：带有边界框、分段和关键点的对象实例。
                可能包括：
                    - 'mosaic_border'（Tuple[int, int]）：马赛克增强的边框大小。

        Returns:
            (Dict): Transformed labels dictionary containing:
                - 'img' (np.ndarray): The transformed image.
                - 'cls' (np.ndarray): Updated class labels.
                - 'instances' (Instances): Updated object instances.
                - 'resized_shape' (Tuple[int, int]): New image shape after transformation.
            （Dict）：包含以下内容的变换标签字典：
                - 'img'（np.ndarray）：变换后的图像。
                - 'cls'（np.ndarray）：更新的类别标签。
                - 'instances'（Instances）：更新的对象实例。
                - 'resized_shape'（Tuple[int, int]）：变换后的新图像形状。

        Examples:
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     "img": image,
            ...     "cls": np.array([0, 1, 2]),
            ...     "instances": Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]])),
            ... }
            >>> result = transform(labels)
            >>> assert result["img"].shape[:2] == result["resized_shape"]
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad
        # 不需要比例填充

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        # 确保坐标格式正确
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M是仿射矩阵
        # 用于函数：box_candidates的缩放
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # 如果有分段，更新边界框
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # 裁剪
        new_instances.clip(*self.size)

        # 过滤实例
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # 使边界框与新边界框具有相同的缩放
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """
        Compute candidate boxes for further processing based on size and aspect ratio criteria.
        根据大小和宽高比标准计算候选框以进行进一步处理。

        This method compares boxes before and after augmentation to determine if they meet specified
        thresholds for width, height, aspect ratio, and area. It's used to filter out boxes that have
        been overly distorted or reduced by the augmentation process.
        此方法比较增强前后的边界框，以确定它们是否满足宽度、高度、宽高比和面积的指定阈值。用于过滤在增强过程中被过度扭曲或缩小的框。

        Args:
            box1 (numpy.ndarray): Original boxes before augmentation, shape (4, N) where n is the
                number of boxes. Format is [x1, y1, x2, y2] in absolute coordinates.
            box1（numpy.ndarray）：增强前的原始框，形状为（4，N），其中N是框的数量。格式为[x1，y1，x2，y2]，为绝对坐标。
            box2 (numpy.ndarray): Augmented boxes after transformation, shape (4, N). Format is
                [x1, y1, x2, y2] in absolute coordinates.
            box2（numpy.ndarray）：变换后的增强框，形状为（4，N）。格式为[x1，y1，x2，y2]，为绝对坐标。
            wh_thr (float): Width and height threshold in pixels. Boxes smaller than this in either
                dimension are rejected.
            wh_thr（float）：以像素为单位的宽度和高度阈值。小于此值的框在任一维度上都将被拒绝。
            ar_thr (float): Aspect ratio threshold. Boxes with an aspect ratio greater than this
                value are rejected.
            ar_thr（float）：宽高比阈值。宽高比大于此值的框将被拒绝。
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than
                this value are rejected.
            area_thr（float）：面积比阈值。面积比（新/旧）小于此值的框将被拒绝。
            eps (float): Small epsilon value to prevent division by zero.
            eps（float）：小的epsilon值以防止除以零。

        Returns:
            (numpy.ndarray): Boolean array of shape (n) indicating which boxes are candidates.
                True values correspond to boxes that meet all criteria.
            （numpy.ndarray）：形状为（n）的布尔数组，指示哪些框是候选框。True值对应于满足所有标准的框。

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
        

class RandomHSV:
    """
    随机调整图像的色调、饱和度和亮度（HSV）通道。

    此类对图像应用随机的HSV增强，增强的范围由hgain、sgain和vgain设置。

    Attributes:
        hgain (float): 色调的最大变化。范围通常为[0, 1]。
        sgain (float): 饱和度的最大变化。范围通常为[0, 1]。
        vgain (float): 亮度的最大变化。范围通常为[0, 1]。

    Methods:
        __call__: 对图像应用随机的HSV增强。

    Examples:
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> augmenter(labels)
        >>> augmented_image = augmented_labels["img"]
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """
        初始化RandomHSV对象以进行随机HSV（色调、饱和度、亮度）增强。

        此类在指定的限制内对图像的HSV通道进行随机调整。

        Args:
            hgain (float): 色调的最大变化。应在范围[0, 1]内。
            sgain (float): 饱和度的最大变化。应在范围[0, 1]内。
            vgain (float): 亮度的最大变化。应在范围[0, 1]内。

        Examples:
            >>> hsv_aug = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> hsv_aug(image)
        """
        self.hgain = hgain  # 设置色调的最大变化
        self.sgain = sgain  # 设置饱和度的最大变化
        self.vgain = vgain  # 设置亮度的最大变化

    def __call__(self, labels):
        """
        在预定义的限制内对图像应用随机的HSV增强。

        此方法通过随机调整输入图像的色调、饱和度和亮度（HSV）通道来修改图像。
        调整在初始化时通过hgain、sgain和vgain设置的限制内进行。

        Args:
            labels (Dict): 包含图像数据和元数据的字典。必须包含一个'img'键，其值为numpy数组形式的图像。

        Returns:
            (None): 该函数就地修改输入的'labels'字典，用HSV增强的图像更新'img'键。

        Examples:
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {"img": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> hsv_augmenter(labels)
            >>> augmented_img = labels["img"]
        """
        img = labels["img"]  # 从labels中获取图像
        if self.hgain or self.sgain or self.vgain:  # 如果有任何变化
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # 随机增益
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))  # 将图像转换为HSV并分离通道
            dtype = img.dtype  # uint8类型

            x = np.arange(0, 256, dtype=r.dtype)  # 创建一个从0到255的数组
            lut_hue = ((x * r[0]) % 180).astype(dtype)  # 色调查找表
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 饱和度查找表
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 亮度查找表

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))  # 合并调整后的通道
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # 转换回BGR格式并更新原图像
        return labels  # 返回修改后的labels


class RandomFlip:
    """
    以给定的概率对图像进行随机水平或垂直翻转。

    此类执行随机图像翻转，并更新相应的实例注释，例如边界框和关键点。

    Attributes:
        p (float): 应用翻转的概率。必须在0和1之间。
        direction (str): 翻转方向，可以是'horizontal'或'vertical'。
        flip_idx (array-like): 关键点翻转的索引映射（如果适用）。

    Methods:
        __call__: 对图像及其注释应用随机翻转变换。

    Examples:
        >>> transform = RandomFlip(p=0.5, direction="horizontal")
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        """
        使用概率和方向初始化RandomFlip类。

        此类以给定的概率对图像进行随机水平或垂直翻转。
        它还会相应地更新任何实例（边界框、关键点等）。

        Args:
            p (float): 应用翻转的概率。必须在0和1之间。
            direction (str): 应用翻转的方向。必须为'horizontal'或'vertical'。
            flip_idx (List[int] | None): 关键点翻转的索引映射（如果有）。

        Raises:
            AssertionError: 如果方向不是'horizontal'或'vertical'，或者如果p不在0到1之间。

        Examples:
            >>> flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flip_with_idx = RandomFlip(p=0.7, direction="vertical", flip_idx=[1, 0, 3, 2, 5, 4])
        """
        assert direction in {"horizontal", "vertical"}, f"支持方向为`horizontal`或`vertical`，但得到了{direction}"
        assert 0 <= p <= 1.0, f"概率应在范围[0, 1]内，但得到了{p}。"

        self.p = p  # 设置翻转的概率
        self.direction = direction  # 设置翻转的方向
        self.flip_idx = flip_idx  # 设置关键点翻转的索引映射

    def __call__(self, labels):
        """
        对图像应用随机翻转，并相应更新任何实例（如边界框或关键点）。

        此方法根据初始化的概率和方向随机翻转输入图像。它还更新相应的实例（边界框、关键点）以匹配翻转后的图像。

        Args:
            labels (Dict): 包含以下键的字典：
                'img' (numpy.ndarray): 要翻转的图像。
                'instances' (ultralytics.utils.instance.Instances): 包含边界框和可选关键点的对象。

        Returns:
            (Dict): 同一字典，包含翻转后的图像和更新后的实例：
                'img' (numpy.ndarray): 翻转后的图像。
                'instances' (ultralytics.utils.instance.Instances): 更新后的实例以匹配翻转后的图像。

        Examples:
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": Instances(...)}
            >>> random_flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flipped_labels = random_flip(labels)
        """
        img = labels["img"]  # 从labels中获取图像
        instances = labels.pop("instances")  # 从labels中获取实例并移除
        instances.convert_bbox(format="xywh")  # 转换边界框格式为xywh
        h, w = img.shape[:2]  # 获取图像的高度和宽度
        h = 1 if instances.normalized else h  # 如果实例是归一化的，则高度设为1
        w = 1 if instances.normalized else w  # 如果实例是归一化的，则宽度设为1

        # 垂直翻转
        if self.direction == "vertical" and random.random() < self.p:  # 如果方向是垂直且随机数小于概率
            img = np.flipud(img)  # 进行上下翻转
            instances.flipud(h)  # 更新实例的上下翻转
        if self.direction == "horizontal" and random.random() < self.p:  # 如果方向是水平且随机数小于概率
            img = np.fliplr(img)  # 进行左右翻转
            instances.fliplr(w)  # 更新实例的左右翻转
            # 对于关键点
            if self.flip_idx is not None and instances.keypoints is not None:  # 如果有翻转索引且实例中有关键点
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])  # 更新关键点
        labels["img"] = np.ascontiguousarray(img)  # 将翻转后的图像更新到labels中
        labels["instances"] = instances  # 将更新后的实例更新到labels中
        return labels  # 返回修改后的labels


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.
    用于检测、实例分割和姿态的图像缩放和填充。

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.
    此类将图像缩放并填充到指定形状，同时保持纵横比。它还更新相应的标签和边界框。

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        new_shape（元组）：目标形状（高度，宽度）用于缩放。
        auto (bool): Whether to use minimum rectangle.
        auto（布尔值）：是否使用最小矩形。
        scaleFill (bool): Whether to stretch the image to new_shape.
        scaleFill（布尔值）：是否将图像拉伸到new_shape。
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        scaleup（布尔值）：是否允许放大。如果为False，则仅缩小。
        stride (int): Stride for rounding padding.
        stride（整数）：用于四舍五入填充的步幅。
        center (bool): Whether to center the image or align to top-left.
        center（布尔值）：是否将图像居中或对齐到左上角。

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.
        __call__：缩放和填充图像，更新标签和边界框。

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """
        Initialize LetterBox object for resizing and padding images.
        初始化LetterBox对象以缩放和填充图像。

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.
        此类旨在为目标检测、实例分割和姿态估计任务缩放和填充图像。它支持各种缩放模式，包括自动缩放、填充缩放和信箱填充。

        Args:
            new_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            new_shape（元组[int, int]）：缩放后图像的目标大小（高度，宽度）。
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            auto（布尔值）：如果为True，则使用最小矩形进行缩放。如果为False，则直接使用new_shape。
            scaleFill (bool): If True, stretch the image to new_shape without padding.
            scaleFill（布尔值）：如果为True，则在不填充的情况下将图像拉伸到new_shape。
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            scaleup（布尔值）：如果为True，则允许放大。如果为False，则仅缩小。
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            center（布尔值）：如果为True，则将图像居中。如果为False，则将图像放置在左上角。
            stride (int): Stride of the model (e.g., 32 for YOLOv5).
            stride（整数）：模型的步幅（例如，YOLOv5的步幅为32）。

        Attributes:
            new_shape (Tuple[int, int]): Target size for the resized image.
            new_shape（元组[int, int]）：缩放后图像的目标大小。
            auto (bool): Flag for using minimum rectangle resizing.
            auto（布尔值）：使用最小矩形缩放的标志。
            scaleFill (bool): Flag for stretching image without padding.
            scaleFill（布尔值）：在不填充的情况下拉伸图像的标志。
            scaleup (bool): Flag for allowing upscaling.
            scaleup（布尔值）：允许放大的标志。
            stride (int): Stride value for ensuring image size is divisible by stride.
            stride（整数）：确保图像大小可被步幅整除的步幅值。

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape  # 设置目标形状
        self.auto = auto  # 设置是否使用最小矩形
        self.scaleFill = scaleFill  # 设置是否拉伸图像
        self.scaleup = scaleup  # 设置是否允许放大
        self.stride = stride  # 设置步幅
        self.center = center  # 设置图像是否居中

    def __call__(self, labels=None, image=None):
        """
        Resizes and pads an image for object detection, instance segmentation, or pose estimation tasks.
        缩放和填充图像以用于目标检测、实例分割或姿态估计任务。

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. 
        此方法对输入图像应用信箱填充，这涉及在保持纵横比的同时缩放图像，并添加填充以适应新形状。
        It also updates any associated labels accordingly.
        它还相应地更新任何相关标签。

        Args:
            labels (Dict | None): A dictionary containing image data and associated labels, or empty dict if None.
            labels（字典 | None）：包含图像数据和相关标签的字典，或如果为None则为空字典。
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.
            image（np.ndarray | None）：输入图像，作为numpy数组。如果为None，则从'labels'中获取图像。

        Returns:
            (Dict | Tuple): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns a tuple containing the resized
                and padded image, and a tuple of (ratio, (left_pad, top_pad)).
            （字典 | 元组）：如果提供了'labels'，则返回更新后的字典，包含缩放和填充后的图像、更新的标签和附加元数据。
                如果'labels'为空，则返回一个元组，包含缩放和填充后的图像，以及一个元组（ratio, (left_pad, top_pad)）。

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:  # 如果没有提供标签
            labels = {}  # 初始化为空字典
        img = labels.get("img") if image is None else image  # 获取图像，如果提供了图像，则使用提供的图像
        shape = img.shape[:2]  # 当前形状[高度，宽度]
        new_shape = labels.pop("rect_shape", self.new_shape)  # 从标签中获取新形状，或使用默认的新形状
        if isinstance(new_shape, int):  # 如果新形状是整数
            new_shape = (new_shape, new_shape)  # 将其转换为元组形式

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例（新尺寸/旧尺寸）
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)  # 如果不允许放大，则确保比例不超过1.0

        # Compute padding
        ratio = r, r  # 宽度和高度的比例
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 计算新的未填充尺寸
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算宽度和高度的填充
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # 确保填充符合步幅
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0  # 如果拉伸，则填充为0
            new_unpad = (new_shape[1], new_shape[0])  # 更新未填充尺寸
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 更新宽度和高度的比例

        if self.center:  # 如果居中
            dw /= 2  # 将填充分为两侧
            dh /= 2

        if shape[::-1] != new_unpad:  # 如果当前形状与新未填充尺寸不同
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # 缩放图像
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))  # 计算上下填充
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))  # 计算左右填充
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # 添加边框
        if labels.get("ratio_pad"):  # 如果标签中有比例填充
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # 更新比例填充信息

        if len(labels):  # 如果标签不为空
            labels = self._update_labels(labels, ratio, left, top)  # 更新标签
            labels["img"] = img  # 将缩放后的图像存储在标签中
            labels["resized_shape"] = new_shape  # 更新标签中的缩放后形状
            return labels  # 返回更新后的标签
        else:
            return img  # 返回缩放后的图像

    @staticmethod
    def _update_labels(labels, ratio, padw, padh):
        """
        Updates labels after applying letterboxing to an image.
        在对图像应用信箱填充后更新标签。

        This method modifies the bounding box coordinates of instances in the labels
        to account for resizing and padding applied during letterboxing.
        此方法修改标签中实例的边界框坐标，以考虑在信箱填充过程中应用的缩放和填充。

        Args:
            labels (Dict): A dictionary containing image labels and instances.
            labels（字典）：包含图像标签和实例的字典。
            ratio (Tuple[float, float]): Scaling ratios (width, height) applied to the image.
            ratio（元组[浮点数，浮点数]）：应用于图像的缩放比例（宽度，高度）。
            padw (float): Padding width added to the image.
            padw（浮点数）：添加到图像的填充宽度。
            padh (float): Padding height added to the image.
            padh（浮点数）：添加到图像的填充高度。

        Returns:
            (Dict): Updated labels dictionary with modified instance coordinates.
            （字典）：更新的标签字典，包含修改后的实例坐标。

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels["instances"].convert_bbox(format="xyxy")  # 将边界框转换为xyxy格式
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])  # 反归一化实例
        labels["instances"].scale(*ratio)  # 根据比例缩放实例
        labels["instances"].add_padding(padw, padh)  # 添加填充
        return labels  # 返回更新后的标签


class CopyPaste(BaseMixTransform):
    """
    CopyPaste类用于对图像数据集应用Copy-Paste增强。

    此类实现了Copy-Paste增强技术，具体描述见论文“Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation”（https://arxiv.org/abs/2012.07177）。它结合来自不同图像的对象以创建新的训练样本。

    Attributes:
        dataset (Any): 将应用Copy-Paste增强的数据集。
        pre_transform (Callable | None): 可选的在Copy-Paste之前应用的变换。
        p (float): 应用Copy-Paste增强的概率。

    Methods:
        get_indexes: 返回数据集中随机索引。
        _mix_transform: 将Copy-Paste增强应用于输入标签。
        __call__: 对图像及其注释应用Copy-Paste变换。

    Examples:
        >>> from ultralytics.data.augment import CopyPaste
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> copypaste = CopyPaste(dataset, p=0.5)
        >>> augmented_labels = copypaste(original_labels)
    """

    def __init__(self, dataset=None, pre_transform=None, p=0.5, mode="flip") -> None:
        """初始化CopyPaste对象，包含数据集、预变换和应用MixUp的概率。"""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)  # 调用父类构造函数
        assert mode in {"flip", "mixup"}, f"Expected `mode` to be `flip` or `mixup`, but got {mode}."  # 确保模式正确
        self.mode = mode  # 设置模式

    def get_indexes(self):
        """返回数据集中用于CopyPaste增强的随机索引列表。"""
        return random.randint(0, len(self.dataset) - 1)  # 返回随机索引

    def _mix_transform(self, labels):
        """将Copy-Paste增强应用于将另一个图像的对象合并到当前图像中。"""
        labels2 = labels["mix_labels"][0]  # 获取混合标签
        return self._transform(labels, labels2)  # 应用变换

    def __call__(self, labels):
        """对图像及其标签应用Copy-Paste增强。"""
        if len(labels["instances"].segments) == 0 or self.p == 0:  # 如果没有实例或概率为0
            return labels  # 返回原标签
        if self.mode == "flip":  # 如果模式为翻转
            return self._transform(labels)  # 应用翻转变换

        # 获取其他图像的索引
        indexes = self.get_indexes()  # 获取索引
        if isinstance(indexes, int):  # 如果是单个索引
            indexes = [indexes]  # 转换为列表

        # 获取用于Mosaic或MixUp的图像信息
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]  # 获取混合标签

        if self.pre_transform is not None:  # 如果有预变换
            for i, data in enumerate(mix_labels):  # 遍历混合标签
                mix_labels[i] = self.pre_transform(data)  # 应用预变换
        labels["mix_labels"] = mix_labels  # 更新混合标签

        # 更新类别和文本
        labels = self._update_label_text(labels)  # 更新标签文本
        # Mosaic或MixUp
        labels = self._mix_transform(labels)  # 应用混合变换
        labels.pop("mix_labels", None)  # 移除混合标签
        return labels  # 返回更新后的标签

    def _transform(self, labels1, labels2={}):
        """将Copy-Paste增强应用于将另一个图像的对象合并到当前图像中。"""
        im = labels1["img"]  # 获取当前图像
        cls = labels1["cls"]  # 获取当前类别
        h, w = im.shape[:2]  # 获取图像的高度和宽度
        instances = labels1.pop("instances")  # 移除并获取实例
        instances.convert_bbox(format="xyxy")  # 转换边界框格式为xyxy
        instances.denormalize(w, h)  # 反归一化实例

        im_new = np.zeros(im.shape, np.uint8)  # 创建新图像
        instances2 = labels2.pop("instances", None)  # 移除并获取第二个实例
        if instances2 is None:  # 如果没有第二个实例
            instances2 = deepcopy(instances)  # 深拷贝当前实例
            instances2.fliplr(w)  # 水平翻转实例
        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)  # 计算面积交集（N, M）
        indexes = np.nonzero((ioa < 0.30).all(1))[0]  # 找到交集小于30%的索引
        n = len(indexes)  # 获取索引数量
        sorted_idx = np.argsort(ioa.max(1)[indexes])  # 对最大交集进行排序
        indexes = indexes[sorted_idx]  # 更新索引

        for j in indexes[: round(self.p * n)]:  # 遍历选中的索引
            cls = np.concatenate((cls, labels2.get("cls", cls)[[j]]), axis=0)  # 更新类别
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)  # 合并实例
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)  # 绘制轮廓

        result = labels2.get("img", cv2.flip(im, 1))  # 获取增强的图像
        i = im_new.astype(bool)  # 将新图像转换为布尔数组
        im[i] = result[i]  # 将增强的图像合并到当前图像中

        labels1["img"] = im  # 更新标签中的图像
        labels1["cls"] = cls  # 更新标签中的类别
        labels1["instances"] = instances  # 更新标签中的实例
        return labels1  # 返回更新后的标签


class Albumentations:
    """
    Albumentations图像增强变换。

    此类使用Albumentations库应用各种图像变换。它包括模糊、中值模糊、转换为灰度、对比度限制自适应直方图均衡（CLAHE）、随机亮度和对比度变化、随机伽马，以及通过压缩降低图像质量。

    Attributes:
        p (float): 应用变换的概率。
        transform (albumentations.Compose): 组合的Albumentations变换。
        contains_spatial (bool): 指示变换是否包含空间操作。

    Methods:
        __call__: 将Albumentations变换应用于输入标签。

    Examples:
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    Notes:
        - 使用此类必须安装Albumentations包。
        - 如果未安装该包或初始化过程中发生错误，则transform将被设置为None。
        - 空间变换的处理方式不同，需要对边界框进行特殊处理。
    """

    def __init__(self, p=1.0):
        """
        初始化Albumentations变换对象，适用于YOLO边界框格式参数。

        此类使用Albumentations库应用各种图像增强，包括模糊、中值模糊、转换为灰度、对比度限制自适应直方图均衡、随机亮度和对比度变化、随机伽马，以及通过压缩降低图像质量。

        Args:
            p (float): 应用增强的概率。必须在0到1之间。

        Attributes:
            p (float): 应用增强的概率。
            transform (albumentations.Compose): 组合的Albumentations变换。
            contains_spatial (bool): 指示变换是否包含空间变换。

        Raises:
            ImportError: 如果未安装Albumentations包。
            Exception: 初始化过程中发生的其他错误。

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            >>> augmented_image = augmented["image"]
            >>> augmented_bboxes = augmented["bboxes"]

        Notes:
            - 需要Albumentations版本1.0.3或更高版本。
            - 空间变换的处理方式不同，以确保与边界框的兼容性。
            - 一些变换默认以非常低的概率（0.01）应用。
        """
        self.p = p  # 设置应用增强的概率
        self.transform = None  # 初始化变换为None
        prefix = colorstr("albumentations: ")  # 设置前缀

        try:
            import albumentations as A  # 导入Albumentations库

            check_version(A.__version__, "1.0.3", hard=True)  # 检查版本要求

            # 可能的空间变换列表
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # 来源于 https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

            # 变换列表
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_range=(75, 100), p=0.0),
            ]

            # 组合变换
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)  # 检查是否包含空间变换
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                # 对于albumentations>=1.4.21所需的确定性变换
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))  # 记录信息
        except ImportError:  # 包未安装，跳过
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")  # 记录异常信息

    def __call__(self, labels):
        """
        将Albumentations变换应用于输入标签。

        此方法使用Albumentations库应用一系列图像增强。它可以对输入图像及其相应标签执行空间和非空间变换。

        Args:
            labels (Dict): 包含图像数据和注释的字典。预期的键有：
                - 'img': numpy.ndarray表示图像
                - 'cls': numpy.ndarray类别标签
                - 'instances': 包含边界框和其他实例信息的对象

        Returns:
            (Dict): 输入字典，包含增强后的图像和更新的注释。

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]])),
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        Notes:
            - 该方法以概率self.p应用变换。
            - 空间变换更新边界框，而非空间变换仅修改图像。
            - 需要安装Albumentations库。
        """
        if self.transform is None or random.random() > self.p:  # 如果变换未定义或随机数大于概率
            return labels  # 返回原标签

        if self.contains_spatial:  # 如果包含空间变换
            cls = labels["cls"]  # 获取类别
            if len(cls):  # 如果类别不为空
                im = labels["img"]  # 获取图像
                labels["instances"].convert_bbox("xywh")  # 转换边界框格式为xywh
                labels["instances"].normalize(*im.shape[:2][::-1])  # 反归一化实例
                bboxes = labels["instances"].bboxes  # 获取边界框
                # TODO: 添加对分段和关键点的支持
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # 应用变换
                if len(new["class_labels"]) > 0:  # 如果新图像中有边界框
                    labels["img"] = new["image"]  # 更新图像
                    labels["cls"] = np.array(new["class_labels"])  # 更新类别
                    bboxes = np.array(new["bboxes"], dtype=np.float32)  # 更新边界框
                labels["instances"].update(bboxes=bboxes)  # 更新实例
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]  # 应用非空间变换

        return labels  # 返回更新后的标签


class Format:
    """
    用于对象检测、实例分割和姿态估计任务的图像注释格式化类。

    此类标准化图像和实例注释，以便在PyTorch DataLoader的`collate_fn`中使用。

    Attributes:
        bbox_format (str): 边界框格式。选项为'xywh'或'xyxy'。
        normalize (bool): 是否对边界框进行归一化。
        return_mask (bool): 是否返回实例掩码以进行分割。
        return_keypoint (bool): 是否返回姿态估计的关键点。
        return_obb (bool): 是否返回定向边界框。
        mask_ratio (int): 掩码的下采样比例。
        mask_overlap (bool): 是否允许掩码重叠。
        batch_idx (bool): 是否保留批次索引。
        bgr (float): 返回BGR图像的概率。

    Methods:
        __call__: 格式化包含图像、类别、边界框的标签字典，并可选地返回掩码和关键点。
        _format_img: 将图像从Numpy数组转换为PyTorch张量。
        _format_segments: 将多边形点转换为位图掩码。

    Examples:
        >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels["img"]
        >>> bboxes = formatted_labels["bboxes"]
        >>> masks = formatted_labels["masks"]
    """

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        """
        使用给定参数初始化Format类以格式化图像和实例注释。

        此类标准化对象检测、实例分割和姿态估计任务的图像和实例注释，为PyTorch DataLoader的`collate_fn`做准备。

        Args:
            bbox_format (str): 边界框格式。选项为'xywh'、'xyxy'等。
            normalize (bool): 是否将边界框归一化到[0,1]。
            return_mask (bool): 如果为True，则返回用于分割任务的实例掩码。
            return_keypoint (bool): 如果为True，则返回用于姿态估计的关键点。
            return_obb (bool): 如果为True，则返回定向边界框。
            mask_ratio (int): 掩码的下采样比例。
            mask_overlap (bool): 如果为True，则允许掩码重叠。
            batch_idx (bool): 如果为True，则保留批次索引。
            bgr (float): 返回BGR图像而非RGB图像的概率。

        Attributes:
            bbox_format (str): 边界框格式。
            normalize (bool): 边界框是否被归一化。
            return_mask (bool): 是否返回实例掩码。
            return_keypoint (bool): 是否返回关键点。
            return_obb (bool): 是否返回定向边界框。
            mask_ratio (int): 掩码的下采样比例。
            mask_overlap (bool): 掩码是否可以重叠。
            batch_idx (bool): 是否保留批次索引。
            bgr (float): 返回BGR图像的概率。

        Examples:
            >>> format = Format(bbox_format="xyxy", return_mask=True, return_keypoint=False)
            >>> print(format.bbox_format)
            xyxy
        """
        self.bbox_format = bbox_format  # 设置边界框格式
        self.normalize = normalize  # 设置是否归一化
        self.return_mask = return_mask  # 设置是否返回掩码，训练检测时应为False
        self.return_keypoint = return_keypoint  # 设置是否返回关键点
        self.return_obb = return_obb  # 设置是否返回定向边界框
        self.mask_ratio = mask_ratio  # 设置掩码的下采样比例
        self.mask_overlap = mask_overlap  # 设置是否允许掩码重叠
        self.batch_idx = batch_idx  # 设置是否保留批次索引
        self.bgr = bgr  # 设置返回BGR图像的概率

    def __call__(self, labels):
        """
        格式化图像注释以用于对象检测、实例分割和姿态估计任务。

        此方法标准化图像和实例注释，以便在PyTorch DataLoader的`collate_fn`中使用。它处理输入标签字典，将注释转换为指定格式，并在需要时应用归一化。

        Args:
            labels (Dict): 包含图像和注释数据的字典，预期的键有：
                - 'img': 作为Numpy数组的输入图像。
                - 'cls': 实例的类别标签。
                - 'instances': 包含边界框、分段和关键点的Instances对象。

        Returns:
            (Dict): 一个字典，包含格式化的数据，包括：
                - 'img': 格式化的图像张量。
                - 'cls': 类别标签的张量。
                - 'bboxes': 指定格式的边界框张量。
                - 'masks': 实例掩码张量（如果return_mask为True）。
                - 'keypoints': 关键点张量（如果return_keypoint为True）。
                - 'batch_idx': 批次索引张量（如果batch_idx为True）。

        Examples:
            >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
            >>> labels = {"img": np.random.rand(640, 640, 3), "cls": np.array([0, 1]), "instances": Instances(...)}
            >>> formatted_labels = formatter(labels)
            >>> print(formatted_labels.keys())
        """
        img = labels.pop("img")  # 从标签中移除并获取图像
        h, w = img.shape[:2]  # 获取图像的高度和宽度
        cls = labels.pop("cls")  # 从标签中移除并获取类别
        instances = labels.pop("instances")  # 从标签中移除并获取实例
        instances.convert_bbox(format=self.bbox_format)  # 转换边界框格式
        instances.denormalize(w, h)  # 反归一化实例
        nl = len(instances)  # 获取实例数量

        if self.return_mask:  # 如果需要返回掩码
            if nl:  # 如果有实例
                masks, instances, cls = self._format_segments(instances, cls, w, h)  # 格式化分段
                masks = torch.from_numpy(masks)  # 将掩码转换为PyTorch张量
            else:  # 如果没有实例
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )  # 创建全零掩码
            labels["masks"] = masks  # 将掩码添加到标签中
        labels["img"] = self._format_img(img)  # 格式化图像
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)  # 将类别转换为张量
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))  # 将边界框转换为张量
        if self.return_keypoint:  # 如果需要返回关键点
            labels["keypoints"] = torch.from_numpy(instances.keypoints)  # 将关键点转换为张量
            if self.normalize:  # 如果需要归一化
                labels["keypoints"][..., 0] /= w  # 归一化x坐标
                labels["keypoints"][..., 1] /= h  # 归一化y坐标
        if self.return_obb:  # 如果需要返回定向边界框
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )  # 转换为定向边界框
        # NOTE: 需要在xywhr格式中归一化obb以确保宽高一致性
        if self.normalize:  # 如果需要归一化
            labels["bboxes"][:, [0, 2]] /= w  # 归一化左上角和右下角的x坐标
            labels["bboxes"][:, [1, 3]] /= h  # 归一化左上角和右下角的y坐标
        # 然后我们可以使用collate_fn
        if self.batch_idx:  # 如果需要保留批次索引
            labels["batch_idx"] = torch.zeros(nl)  # 创建全零的批次索引
        return labels  # 返回格式化后的标签

    def _format_img(self, img):
        """
        将图像格式化为YOLO格式，从Numpy数组转换为PyTorch张量。

        此函数执行以下操作：
        1. 确保图像具有3个维度（如果需要，则添加通道维度）。
        2. 将图像从HWC格式转换为CHW格式。
        3. 可选地将颜色通道从RGB翻转为BGR。
        4. 将图像转换为连续数组。
        5. 将Numpy数组转换为PyTorch张量。

        Args:
            img (np.ndarray): 输入图像，形状为(H, W, C)或(H, W)。

        Returns:
            (torch.Tensor): 格式化后的图像，形状为(C, H, W)。

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> formatted_img = self._format_img(img)
            >>> print(formatted_img.shape)
            torch.Size([3, 100, 100])
        """
        if len(img.shape) < 3:  # 如果图像维度少于3
            img = np.expand_dims(img, -1)  # 添加通道维度
        img = img.transpose(2, 0, 1)  # 转换为CHW格式
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)  # 可选地翻转颜色通道
        img = torch.from_numpy(img)  # 转换为PyTorch张量
        return img  # 返回格式化后的图像

    def _format_segments(self, instances, cls, w, h):
        """
        将多边形分段转换为位图掩码。

        Args:
            instances (Instances): 包含分段信息的对象。
            cls (numpy.ndarray): 每个实例的类别标签。
            w (int): 图像的宽度。
            h (int): 图像的高度。

        Returns:
            masks (numpy.ndarray): 位图掩码，形状为(N, H, W)或(1, H, W)（如果mask_overlap为True）。
            instances (Instances): 更新的实例对象，如果mask_overlap为True，则带有排序的分段。
            cls (numpy.ndarray): 更新的类别标签，如果mask_overlap为True，则已排序。

        Notes:
            - 如果self.mask_overlap为True，则掩码重叠并按面积排序。
            - 如果self.mask_overlap为False，则每个掩码单独表示。
            - 掩码根据self.mask_ratio进行下采样。
        """
        segments = instances.segments  # 获取分段信息
        if self.mask_overlap:  # 如果允许掩码重叠
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)  # 转换为掩码
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]  # 更新实例
            cls = cls[sorted_idx]  # 更新类别标签
        else:  # 如果不允许掩码重叠
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)  # 转换为掩码

        return masks, instances, cls  # 返回掩码、实例和类别标签


class RandomLoadText:
    """
    随机采样正负文本并相应更新类别索引的类。

    此类负责从给定的类别文本集中采样文本，包括正样本（图像中存在）和负样本（图像中不存在）。它更新类别索引以反映采样的文本，并可以选择性地将文本列表填充到固定长度。

    Attributes:
        prompt_format (str): 文本提示的格式字符串。
        neg_samples (Tuple[int, int]): 随机采样负文本的范围。
        max_samples (int): 一幅图像中不同文本样本的最大数量。
        padding (bool): 是否将文本填充到max_samples。
        padding_value (str): 填充时使用的文本，当padding为True时。

    Methods:
        __call__: 处理输入标签并返回更新后的类别和文本。

    Examples:
        >>> loader = RandomLoadText(prompt_format="Object: {}", neg_samples=(5, 10), max_samples=20)
        >>> labels = {"cls": [0, 1, 2], "texts": [["cat"], ["dog"], ["bird"]], "instances": [...]}
        >>> updated_labels = loader(labels)
        >>> print(updated_labels["texts"])
        ['Object: cat', 'Object: dog', 'Object: bird', 'Object: elephant', 'Object: car']
    """

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: Tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: str = "",
    ) -> None:
        """
        初始化RandomLoadText类以随机采样正负文本。

        此类旨在随机采样正文本和负文本，并相应更新类别索引以匹配样本数量。它可用于基于文本的对象检测任务。

        Args:
            prompt_format (str): 提示的格式字符串。默认是'{}'。格式字符串应包含一对大括号{}，文本将插入其中。
            neg_samples (Tuple[int, int]): 随机采样负文本的范围。第一个整数指定负样本的最小数量，第二个整数指定最大数量。默认是(80, 80)。
            max_samples (int): 一幅图像中不同文本样本的最大数量。默认是80。
            padding (bool): 是否将文本填充到max_samples。如果为True，则文本数量将始终等于max_samples。默认是False。
            padding_value (str): 填充时使用的文本。默认是空字符串。

        Attributes:
            prompt_format (str): 提示的格式字符串。
            neg_samples (Tuple[int, int]): 采样负文本的范围。
            max_samples (int): 最大文本样本数量。
            padding (bool): 是否启用填充。
            padding_value (str): 填充时使用的值。

        Examples:
            >>> random_load_text = RandomLoadText(prompt_format="Object: {}", neg_samples=(50, 100), max_samples=120)
            >>> random_load_text.prompt_format
            'Object: {}'
            >>> random_load_text.neg_samples
            (50, 100)
            >>> random_load_text.max_samples
            120
        """
        self.prompt_format = prompt_format  # 设置提示格式
        self.neg_samples = neg_samples  # 设置负样本范围
        self.max_samples = max_samples  # 设置最大样本数量
        self.padding = padding  # 设置是否填充
        self.padding_value = padding_value  # 设置填充文本

    def __call__(self, labels: dict) -> dict:
        """
        随机采样正负文本并相应更新类别索引。

        此方法根据图像中现有的类别标签采样正文本，并从剩余类别中随机选择负文本。然后更新类别索引以匹配新采样的文本顺序。

        Args:
            labels (Dict): 包含图像标签和元数据的字典。必须包含'texts'和'cls'键。

        Returns:
            (Dict): 更新后的标签字典，包含新的'cls'和'texts'条目。

        Examples:
            >>> loader = RandomLoadText(prompt_format="A photo of {}", neg_samples=(5, 10), max_samples=20)
            >>> labels = {"cls": np.array([[0], [1], [2]]), "texts": [["dog"], ["cat"], ["bird"]]}
            >>> updated_labels = loader(labels)
        """
        assert "texts" in labels, "No texts found in labels."  # 确保标签中包含'texts'
        class_texts = labels["texts"]  # 获取类别文本
        num_classes = len(class_texts)  # 获取类别数量
        cls = np.asarray(labels.pop("cls"), dtype=int)  # 从标签中移除并获取类别
        pos_labels = np.unique(cls).tolist()  # 获取唯一的正标签

        if len(pos_labels) > self.max_samples:  # 如果正标签数量超过最大样本数量
            pos_labels = random.sample(pos_labels, k=self.max_samples)  # 随机选择正标签

        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))  # 计算负样本数量
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]  # 获取负标签
        neg_labels = random.sample(neg_labels, k=neg_samples)  # 随机选择负标签

        sampled_labels = pos_labels + neg_labels  # 合并正负标签
        random.shuffle(sampled_labels)  # 随机打乱标签顺序

        label2ids = {label: i for i, label in enumerate(sampled_labels)}  # 创建标签到索引的映射
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)  # 初始化有效索引
        new_cls = []  # 新类别列表
        for i, label in enumerate(cls.squeeze(-1).tolist()):  # 遍历类别
            if label not in label2ids:  # 如果标签不在映射中
                continue  # 跳过
            valid_idx[i] = True  # 标记为有效
            new_cls.append([label2ids[label]])  # 更新新类别

        labels["instances"] = labels["instances"][valid_idx]  # 更新实例
        labels["cls"] = np.array(new_cls)  # 更新类别

        # 随机选择一个提示，当有多个提示时
        texts = []  # 初始化文本列表
        for label in sampled_labels:  # 遍历采样标签
            prompts = class_texts[label]  # 获取对应的提示文本
            assert len(prompts) > 0  # 确保提示不为空
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])  # 随机选择一个提示
            texts.append(prompt)  # 添加到文本列表

        if self.padding:  # 如果需要填充
            valid_labels = len(pos_labels) + len(neg_labels)  # 计算有效标签数量
            num_padding = self.max_samples - valid_labels  # 计算需要填充的数量
            if num_padding > 0:  # 如果需要填充
                texts += [self.padding_value] * num_padding  # 添加填充文本

        labels["texts"] = texts  # 更新标签中的文本
        return labels  # 返回更新后的标签


def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for training.
    应用一系列图像变换以进行训练。

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    此函数创建图像增强技术的组合，以准备YOLO训练的图像。
    
    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        dataset (Dataset): 包含图像数据和注释的数据集对象。
        imgsz (int): The target image size for resizing.
        imgsz (int): 目标图像大小，用于调整大小。
        hyp (Namespace): A dictionary of hyperparameters controlling various aspects of the transformations.
        hyp (Namespace): 控制变换各个方面的超参数字典。
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.
        stretch (bool): 如果为True，则对图像应用拉伸。如果为False，则使用LetterBox调整大小。

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.
        (Compose): 应用于数据集的图像变换组合。

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)  # 创建马赛克变换实例
    affine = RandomPerspective(
        degrees=hyp.degrees,  # 随机透视变换的角度
        translate=hyp.translate,  # 随机平移
        scale=hyp.scale,  # 随机缩放
        shear=hyp.shear,  # 随机剪切
        perspective=hyp.perspective,  # 随机透视
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),  # 如果stretch为False，则使用LetterBox调整大小
    )

    pre_transform = Compose([mosaic, affine])  # 组合预处理变换
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))  # 插入CopyPaste变换
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),  # 组合马赛克和透视变换
                p=hyp.copy_paste,  # CopyPaste的概率
                mode=hyp.copy_paste_mode,  # CopyPaste的模式
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation 用于关键点增强
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)  # 获取关键点形状
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")  # 警告：未定义'flip_idx'数组
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")  # 抛出错误：flip_idx长度必须等于kpt_shape[0]

    return Compose(
        [
            pre_transform,  # 预处理变换
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),  # 混合增强
            Albumentations(p=1.0),  # 应用Albumentations变换
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),  # 随机调整HSV值
            RandomFlip(direction="vertical", p=hyp.flipud),  # 随机垂直翻转
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),  # 随机水平翻转
        ]
    )  # transforms


# Classification augmentations -----------------------------------------------------------------------------------------
def classify_transforms(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation="BILINEAR",
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    Creates a composition of image transforms for classification tasks.
    创建图像变换的组合以用于分类任务。

    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.
    此函数生成适合在评估或推理期间对图像进行预处理的torchvision变换序列。变换包括调整大小、中心裁剪、转换为张量和归一化。

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        size (int | tuple): 变换后图像的目标大小。如果是整数，则定义最短边。如果是元组，则定义（高度，宽度）。
        mean (tuple): Mean values for each RGB channel used in normalization.
        mean (tuple): 用于归一化的每个RGB通道的均值。
        std (tuple): Standard deviation values for each RGB channel used in normalization.
        std (tuple): 用于归一化的每个RGB通道的标准差值。
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        interpolation (str): 插值方法，可以是'NEAREST'、'BILINEAR'或'BICUBIC'。
        crop_fraction (float): Fraction of the image to be cropped.
        crop_fraction (float): 要裁剪的图像的比例。

    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.
        (torchvision.transforms.Compose): torchvision变换的组合。

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'
    import math  # 导入数学库以进行数学计算

    if isinstance(size, (tuple, list)):
        assert len(size) == 2, f"'size' tuples must be length 2, not length {len(size)}"  # 确保size元组长度为2
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)  # 根据crop_fraction计算缩放大小
    else:
        scale_size = math.floor(size / crop_fraction)  # 计算缩放大小
        scale_size = (scale_size, scale_size)  # 转换为元组

    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    # 保持纵横比，在图像中裁剪中心，不添加边框，图像会丢失
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        # 简单情况，使用torchvision内置的Resize调整最短边（标量大小参数）
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]  # 调整图像大小
    else:
        # Resize the shortest edge to matching target dim for non-square target
        # 将最短边调整为匹配非正方形目标的目标尺寸
        tfl = [T.Resize(scale_size)]  # 调整图像大小
    tfl.extend(
        [
            T.CenterCrop(size),  # 中心裁剪
            T.ToTensor(),  # 转换为张量
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),  # 归一化
        ]
    )
    return T.Compose(tfl)  # 返回变换的组合


# Classification training augmentations --------------------------------------------------------------------------------
def classify_augmentations(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter=False,
    erasing=0.0,
    interpolation="BILINEAR",
):
    """
    Creates a composition of image augmentation transforms for classification tasks.
    创建图像变换的组合以用于分类任务。

    This function generates a set of image transformations suitable for training classification models. It includes
    options for resizing, flipping, color jittering, auto augmentation, and random erasing.
    此函数生成适合训练分类模型的一组图像变换。它包括调整大小、翻转、颜色抖动、自动增强和随机擦除的选项。

    Args:
        size (int): Target size for the image after transformations.
        size (int): 变换后图像的目标大小。
        mean (tuple): Mean values for normalization, one per channel.
        mean (tuple): 用于归一化的每个通道的均值。
        std (tuple): Standard deviation values for normalization, one per channel.
        std (tuple): 用于归一化的每个通道的标准差值。
        scale (tuple | None): Range of size of the origin size cropped.
        scale (tuple | None): 原始裁剪大小的范围。
        ratio (tuple | None): Range of aspect ratio of the origin aspect ratio cropped.
        ratio (tuple | None): 原始裁剪的纵横比范围。
        hflip (float): Probability of horizontal flip.
        hflip (float): 水平翻转的概率。
        vflip (float): Probability of vertical flip.
        vflip (float): 垂直翻转的概率。
        auto_augment (str | None): Auto augmentation policy. Can be 'randaugment', 'augmix', 'autoaugment' or None.
        auto_augment (str | None): 自动增强策略。可以是'randaugment'、'augmix'、'autoaugment'或None。
        hsv_h (float): Image HSV-Hue augmentation factor.
        hsv_h (float): 图像HSV-色调增强因子。
        hsv_s (float): Image HSV-Saturation augmentation factor.
        hsv_s (float): 图像HSV-饱和度增强因子。
        hsv_v (float): Image HSV-Value augmentation factor.
        hsv_v (float): 图像HSV-值增强因子。
        force_color_jitter (bool): Whether to apply color jitter even if auto augment is enabled.
        force_color_jitter (bool): 是否在启用自动增强时仍然应用颜色抖动。
        erasing (float): Probability of random erasing.
        erasing (float): 随机擦除的概率。
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        interpolation (str): 插值方法，可以是'NEAREST'、'BILINEAR'或'BICUBIC'。

    Returns:
        (torchvision.transforms.Compose): A composition of image augmentation transforms.
        (torchvision.transforms.Compose): 图像增强变换的组合。

    Examples:
        >>> transforms = classify_augmentations(size=224, auto_augment="randaugment")
        >>> augmented_image = transforms(original_image)
    """
    # Transforms to apply if Albumentations not installed
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} must be integer, not (list, tuple)")  # 确保大小是整数
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range 默认的imagenet缩放范围
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range 默认的imagenet纵横比范围
    interpolation = getattr(T.InterpolationMode, interpolation)  # 获取插值模式
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]  # 随机调整大小裁剪
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))  # 添加水平翻转变换
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))  # 添加垂直翻转变换

    secondary_tfl = []  # 初始化次要变换列表
    disable_color_jitter = False  # 禁用颜色抖动标志
    if auto_augment:
        assert isinstance(auto_augment, str), f"Provided argument should be string, but got type {type(auto_augment)}"  # 确保自动增强参数是字符串
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not force_color_jitter  # 确定是否禁用颜色抖动

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))  # 添加随机增强变换
            else:
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')  # 警告：需要torchvision >= 0.11.0

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))  # 添加AugMix变换
            else:
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')  # 警告：需要torchvision >= 0.13.0

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))  # 添加自动增强变换
            else:
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')  # 警告：需要torchvision >= 0.10.0

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'  # 抛出错误：无效的自动增强策略
            )

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))  # 添加颜色抖动变换

    final_tfl = [
        T.ToTensor(),  # 转换为张量
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),  # 归一化
        T.RandomErasing(p=erasing, inplace=True),  # 随机擦除
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)  # 返回变换的组合


# NOTE: keep this class for backward compatibility
class ClassifyLetterBox:
    """
    A class for resizing and padding images for classification tasks.
    用于调整大小和填充图像以进行分类任务的类。

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]).
    此类旨在成为变换管道的一部分，例如：T.Compose([LetterBox(size), ToTensor()])。
    It resizes and pads images to a specified size while maintaining the original aspect ratio.
    它将图像调整大小并填充到指定大小，同时保持原始纵横比。

    Attributes:
        h (int): Target height of the image.
        h (int): 图像的目标高度。
        w (int): Target width of the image.
        w (int): 图像的目标宽度。
        auto (bool): If True, automatically calculates the short side using stride.
        auto (bool): 如果为True，则使用步幅自动计算短边。
        stride (int): The stride value, used when 'auto' is True.
        stride (int): 当'auto'为True时使用的步幅值。

    Methods:
        __call__: Applies the letterbox transformation to an input image.
        __call__: 将letterbox变换应用于输入图像。

    Examples:
        >>> transform = ClassifyLetterBox(size=(640, 640), auto=False, stride=32)
        >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = transform(img)
        >>> print(result.shape)
        (640, 640, 3)
    """

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes the ClassifyLetterBox object for image preprocessing.
        初始化用于图像预处理的ClassifyLetterBox对象。

        This class is designed to be part of a transformation pipeline for image classification tasks. It resizes and
        pads images to a specified size while maintaining the original aspect ratio.
        此类旨在成为图像分类任务的变换管道的一部分。它将图像调整大小并填充到指定大小，同时保持原始纵横比。

        Args:
            size (int | Tuple[int, int]): Target size for the letterboxed image. If an int, a square image of
                (size, size) is created. If a tuple, it should be (height, width).
            size (int | Tuple[int, int]): 用于letterbox图像的目标大小。如果是整数，则创建一个大小为(size, size)的正方形图像。如果是元组，则应为(高度, 宽度)。
            auto (bool): If True, automatically calculates the short side based on stride. Default is False.
            auto (bool): 如果为True，则根据步幅自动计算短边。默认值为False。
            stride (int): The stride value, used when 'auto' is True. Default is 32.
            stride (int): 当'auto'为True时使用的步幅值。默认值为32。

        Attributes:
            h (int): Target height of the letterboxed image.
            h (int): letterbox图像的目标高度。
            w (int): Target width of the letterboxed image.
            w (int): letterbox图像的目标宽度。
            auto (bool): Flag indicating whether to automatically calculate short side.
            auto (bool): 指示是否自动计算短边的标志。
            stride (int): Stride value for automatic short side calculation.
            stride (int): 用于自动短边计算的步幅值。

        Examples:
            >>> transform = ClassifyLetterBox(size=224)
            >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> result = transform(img)
            >>> print(result.shape)
            (224, 224, 3)
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size  # 设置目标高度和宽度
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes and pads an image using the letterbox method.
        使用letterbox方法调整大小和填充图像。

        This method resizes the input image to fit within the specified dimensions while maintaining its aspect ratio,
        then pads the resized image to match the target size.
        此方法将输入图像调整为适合指定尺寸，同时保持其纵横比，然后填充调整大小的图像以匹配目标大小。

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C).
            im (numpy.ndarray): 输入图像，形状为(H, W, C)的numpy数组。

        Returns:
            (numpy.ndarray): Resized and padded image as a numpy array with shape (hs, ws, 3), where hs and ws are
                the target height and width respectively.
            (numpy.ndarray): 调整大小和填充后的图像，形状为(hs, ws, 3)的numpy数组，其中hs和ws分别是目标高度和宽度。

        Examples:
            >>> letterbox = ClassifyLetterBox(size=(640, 640))
            >>> image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> resized_image = letterbox(image)
            >>> print(resized_image.shape)
            (640, 640, 3)
        """
        imh, imw = im.shape[:2]  # 获取输入图像的高度和宽度
        r = min(self.h / imh, self.w / imw)  # ratio of new/old dimensions 计算新旧维度的比例
        h, w = round(imh * r), round(imw * r)  # resized image dimensions 计算调整大小后的图像维度

        # Calculate padding dimensions 计算填充维度
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)  # 根据步幅计算填充维度
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)  # 计算填充的顶部和左侧边距

        # Create padded image 创建填充图像
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)  # 创建填充图像，填充值为114
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)  # 调整图像大小并填充
        return im_out  # 返回调整大小和填充后的图像

# NOTE: keep this class for backward compatibility
class CenterCrop:
    """
    Applies center cropping to images for classification tasks.
    对图像进行中心裁剪以用于分类任务。

    This class performs center cropping on input images, resizing them to a specified size while maintaining the aspect
    ratio. It is designed to be part of a transformation pipeline, e.g., T.Compose([CenterCrop(size), ToTensor()]).
    此类对输入图像进行中心裁剪，将其调整为指定大小，同时保持纵横比。它旨在成为变换管道的一部分，例如：T.Compose([CenterCrop(size), ToTensor()]).

    Attributes:
        h (int): Target height of the cropped image.
        h (int): 裁剪图像的目标高度。
        w (int): Target width of the cropped image.
        w (int): 裁剪图像的目标宽度。

    Methods:
        __call__: Applies the center crop transformation to an input image.
        __call__: 将中心裁剪变换应用于输入图像。

    Examples:
        >>> transform = CenterCrop(640)
        >>> image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> cropped_image = transform(image)
        >>> print(cropped_image.shape)
        (640, 640, 3)
    """

    def __init__(self, size=640):
        """
        Initializes the CenterCrop object for image preprocessing.
        初始化用于图像预处理的CenterCrop对象。

        This class is designed to be part of a transformation pipeline, e.g., T.Compose([CenterCrop(size), ToTensor()]).
        此类旨在成为变换管道的一部分，例如：T.Compose([CenterCrop(size), ToTensor()]).
        It performs a center crop on input images to a specified size.
        它对输入图像进行中心裁剪，调整为指定大小。

        Args:
            size (int | Tuple[int, int]): The desired output size of the crop. If size is an int, a square crop
                (size, size) is made. If size is a sequence like (h, w), it is used as the output size.
            size (int | Tuple[int, int]): 裁剪的期望输出大小。如果size是整数，则创建一个正方形裁剪（size, size）。如果size是像（h, w）的序列，则将其用作输出大小。

        Returns:
            (None): This method initializes the object and does not return anything.
            (None): 此方法初始化对象，不返回任何内容。

        Examples:
            >>> transform = CenterCrop(224)
            >>> img = np.random.rand(300, 300, 3)
            >>> cropped_img = transform(img)
            >>> print(cropped_img.shape)
            (224, 224, 3)
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size  # 设置目标高度和宽度

    def __call__(self, im):
        """
        Applies center cropping to an input image.
        对输入图像应用中心裁剪。

        This method resizes and crops the center of the image using a letterbox method. It maintains the aspect
        ratio of the original image while fitting it into the specified dimensions.
        此方法使用letterbox方法调整大小并裁剪图像的中心。它保持原始图像的纵横比，同时将其适应指定的尺寸。

        Args:
            im (numpy.ndarray | PIL.Image.Image): The input image as a numpy array of shape (H, W, C) or a
                PIL Image object.
            im (numpy.ndarray | PIL.Image.Image): 输入图像，形状为(H, W, C)的numpy数组或PIL图像对象。

        Returns:
            (numpy.ndarray): The center-cropped and resized image as a numpy array of shape (self.h, self.w, C).
            (numpy.ndarray): 中心裁剪和调整大小后的图像，形状为(self.h, self.w, C)的numpy数组。

        Examples:
            >>> transform = CenterCrop(size=224)
            >>> image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            >>> cropped_image = transform(image)
            >>> assert cropped_image.shape == (224, 224, 3)
        """
        if isinstance(im, Image.Image):  # convert from PIL to numpy array if required
            im = np.asarray(im)  # 如果需要，将PIL图像转换为numpy数组
        imh, imw = im.shape[:2]  # 获取输入图像的高度和宽度
        m = min(imh, imw)  # min dimension 计算最小维度
        top, left = (imh - m) // 2, (imw - m) // 2  # 计算顶部和左侧的填充位置
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)  # 调整图像大小并裁剪

# NOTE: keep this class for backward compatibility
class ToTensor:
    """
    Converts an image from a numpy array to a PyTorch tensor.
    将图像从numpy数组转换为PyTorch张量。

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]).
    此类旨在成为变换管道的一部分，例如：T.Compose([LetterBox(size), ToTensor()]).
    It converts numpy arrays or PIL Images to PyTorch tensors, with an option for half-precision (float16) conversion.
    它将numpy数组或PIL图像转换为PyTorch张量，并提供半精度（float16）转换的选项。

    Attributes:
        half (bool): If True, converts the image to half precision (float16).
        half (bool): 如果为True，则将图像转换为半精度（float16）。

    Methods:
        __call__: Applies the tensor conversion to an input image.
        __call__: 将张量转换应用于输入图像。

    Examples:
        >>> transform = ToTensor(half=True)
        >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> tensor_img = transform(img)
        >>> print(tensor_img.shape, tensor_img.dtype)
        torch.Size([3, 640, 640]) torch.float16

    Notes:
        The input image is expected to be in BGR format with shape (H, W, C).
        输入图像应为形状为(H, W, C)的BGR格式。
        The output tensor will be in RGB format with shape (C, H, W), normalized to [0, 1].
        输出张量将为形状为(C, H, W)的RGB格式，归一化到[0, 1]。
    """

    def __init__(self, half=False):
        """
        Initializes the ToTensor object for converting images to PyTorch tensors.
        初始化用于将图像转换为PyTorch张量的ToTensor对象。

        This class is designed to be used as part of a transformation pipeline for image preprocessing in the
        Ultralytics YOLO framework. It converts numpy arrays or PIL Images to PyTorch tensors, with an option
        for half-precision (float16) conversion.
        此类旨在作为Ultralytics YOLO框架中图像预处理的变换管道的一部分。它将numpy数组或PIL图像转换为PyTorch张量，并提供半精度（float16）转换的选项。

        Args:
            half (bool): If True, converts the tensor to half precision (float16). Default is False.
            half (bool): 如果为True，则将张量转换为半精度（float16）。默认值为False。

        Examples:
            >>> transform = ToTensor(half=True)
            >>> img = np.random.rand(640, 640, 3)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.dtype)
            torch.float16
        """
        super().__init__()
        self.half = half  # 设置半精度标志

    def __call__(self, im):
        """
        Transforms an image from a numpy array to a PyTorch tensor.
        将图像从numpy数组转换为PyTorch张量。

        This method converts the input image from a numpy array to a PyTorch tensor, applying optional
        half-precision conversion and normalization. The image is transposed from HWC to CHW format and
        the color channels are reversed from BGR to RGB.
        此方法将输入图像从numpy数组转换为PyTorch张量，应用可选的半精度转换和归一化。图像从HWC格式转置为CHW格式，并将颜色通道从BGR反转为RGB。

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C) in BGR order.
            im (numpy.ndarray): 输入图像，形状为(H, W, C)的numpy数组，BGR顺序。

        Returns:
            (torch.Tensor): The transformed image as a PyTorch tensor in float32 or float16, normalized
                to [0, 1] with shape (C, H, W) in RGB order.
            (torch.Tensor): 转换后的图像作为PyTorch张量，类型为float32或float16，归一化到[0, 1]，形状为(C, H, W)的RGB顺序。

        Examples:
            >>> transform = ToTensor(half=True)
            >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.shape, tensor_img.dtype)
            torch.Size([3, 640, 640]) torch.float16
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # 转换为torch张量
        im = im.half() if self.half else im.float()  # uint8转换为fp16/32
        im /= 255.0  # 0-255转换为0.0-1.0
        return im  # 返回转换后的图像