# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import inspect
import numpy as np
import pprint
from typing import Any, List, Optional, Tuple, Union
from fvcore.transforms.transform import Transform, TransformList

"""
See "Data Augmentation" tutorial for an overview of the system:
https://detectron2.readthedocs.io/tutorials/augmentation.html
"""


# 导出的公共接口列表
__all__ = [
    "Augmentation",      # 数据增强的基类
    "AugmentationList",  # 数据增强序列类
    "AugInput",         # 数据增强输入类
    "TransformGen",     # Augmentation的别名
    "apply_transform_gens", # 应用数据增强的函数
    "StandardAugInput",    # AugInput的别名
    "apply_augmentations", # 应用数据增强的函数
]


def _check_img_dtype(img):
    # 检查输入图像是否为numpy数组
    assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    # 检查图像数据类型是否为uint8或浮点型
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    # 检查图像维度是否为2维(灰度图)或3维(彩色图)
    assert img.ndim in [2, 3], img.ndim


def _get_aug_input_args(aug, aug_input) -> List[Any]:
    """
    Get the arguments to be passed to ``aug.get_transform`` from the input ``aug_input``.
    从aug_input中获取传递给aug.get_transform的参数
    """
    if aug.input_args is None:
        # Decide what attributes are needed automatically
        # 自动决定需要哪些属性
        prms = list(inspect.signature(aug.get_transform).parameters.items())
        # The default behavior is: if there is one parameter, then its "image"
        # (work automatically for majority of use cases, and also avoid BC breaking),
        # Otherwise, use the argument names.
        # 默认行为：如果只有一个参数，则为"image"(适用于大多数情况，且避免破坏向后兼容)
        # 否则，使用参数名称
        if len(prms) == 1:
            names = ("image",)
        else:
            names = []
            for name, prm in prms:
                # 检查是否使用了可变长度参数(*args, **kwargs)
                if prm.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise TypeError(
                        f""" \
The default implementation of `{type(aug)}.__call__` does not allow \
`{type(aug)}.get_transform` to use variable-length arguments (*args, **kwargs)! \
If arguments are unknown, reimplement `__call__` instead. \
"""
                    )
                names.append(name)
        # 将参数名称列表转换为元组并存储
        aug.input_args = tuple(names)

    # 从aug_input中获取所需的参数值
    args = []
    for f in aug.input_args:
        try:
            args.append(getattr(aug_input, f))
        except AttributeError as e:
            # 如果缺少所需的属性，则抛出异常
            raise AttributeError(
                f"{type(aug)}.get_transform needs input attribute '{f}', "
                f"but it is not an attribute of {type(aug_input)}!"
            ) from e
    return args


class Augmentation:
    """
    Augmentation defines (often random) policies/strategies to generate :class:`Transform`
    from data. It is often used for pre-processing of input data.
    数据增强类定义了(通常是随机的)策略来从数据生成Transform类。它通常用于数据的预处理。

    A "policy" that generates a :class:`Transform` may, in the most general case,
    need arbitrary information from input data in order to determine what transforms
    to apply. Therefore, each :class:`Augmentation` instance defines the arguments
    needed by its :meth:`get_transform` method. When called with the positional arguments,
    the :meth:`get_transform` method executes the policy.
    生成Transform的"策略"在最一般的情况下，可能需要从输入数据中获取任意信息来决定应用什么转换。
    因此，每个Augmentation实例都定义了其get_transform方法所需的参数。
    当使用位置参数调用时，get_transform方法执行该策略。

    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`,
    but not how to execute the actual transform operations to those data.
    Its :meth:`__call__` method will use :meth:`AugInput.transform` to execute the transform.
    注意，Augmentation定义了创建Transform的策略，但不定义如何对数据执行实际的转换操作。
    它的__call__方法将使用AugInput.transform来执行转换。

    The returned `Transform` object is meant to describe deterministic transformation, which means
    it can be re-applied on associated data, e.g. the geometry of an image and its segmentation
    masks need to be transformed together.
    (If such re-application is not needed, then determinism is not a crucial requirement.)
    返回的Transform对象用于描述确定性转换，这意味着它可以重新应用于相关数据，
    例如，图像的几何变换和其分割掩码需要一起转换。
    (如果不需要这种重新应用，那么确定性就不是一个关键要求。)
    """

    input_args: Optional[Tuple[str]] = None
    """
    Stores the attribute names needed by :meth:`get_transform`, e.g.  ``("image", "sem_seg")``.
    By default, it is just a tuple of argument names in :meth:`self.get_transform`, which often only
    contain "image". As long as the argument name convention is followed, there is no need for
    users to touch this attribute.
    存储get_transform方法所需的属性名称，例如("image", "sem_seg")。
    默认情况下，它只是self.get_transform中的参数名称元组，通常只包含"image"。
    只要遵循参数名称约定，用户就不需要修改这个属性。
    """

    def _init(self, params=None):
        # 初始化方法，用于设置实例属性
        if params:
            # 遍历参数字典，将非私有属性设置为实例属性
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def get_transform(self, *args) -> Transform:
        """
        Execute the policy based on input data, and decide what transform to apply to inputs.
        基于输入数据执行策略，并决定对输入应用什么转换。

        Args:
            args: Any fixed-length positional arguments. By default, the name of the arguments
                should exist in the :class:`AugInput` to be used.

        Returns:
            Transform: Returns the deterministic transform to apply to the input.
            Transform: 返回要应用于输入的确定性转换。

        Examples:
        ::
            class MyAug:
                # if a policy needs to know both image and semantic segmentation
                # 如果策略需要同时知道图像和语义分割信息
                def get_transform(image, sem_seg) -> T.Transform:
                    pass
            tfm: Transform = MyAug().get_transform(image, sem_seg)
            new_image = tfm.apply_image(image)

        Notes:
            Users can freely use arbitrary new argument names in custom
            :meth:`get_transform` method, as long as they are available in the
            input data. In detectron2 we use the following convention:

            * image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
              floating point in range [0, 1] or [0, 255].
            * boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
              of N instances. Each is in XYXY format in unit of absolute coordinates.
            * sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.

            We do not specify convention for other types and do not include builtin
            :class:`Augmentation` that uses other types in detectron2.
        """
        raise NotImplementedError

    def __call__(self, aug_input) -> Transform:
        """
        Augment the given `aug_input` **in-place**, and return the transform that's used.
        对给定的`aug_input`进行原地数据增强，并返回使用的变换。

        This method will be called to apply the augmentation. In most augmentation, it
        is enough to use the default implementation, which calls :meth:`get_transform`
        using the inputs. But a subclass can overwrite it to have more complicated logic.
        这个方法将被调用来应用数据增强。对于大多数增强操作来说，使用默认实现就足够了，
        默认实现会使用输入调用:meth:`get_transform`。但是子类可以重写它来实现更复杂的逻辑。

        Args:
            aug_input (AugInput): an object that has attributes needed by this augmentation
                (defined by ``self.get_transform``). Its ``transform`` method will be called
                to in-place transform it.
                一个具有此增强所需属性的对象（由``self.get_transform``定义）。
                它的``transform``方法将被调用来进行原地变换。

        Returns:
            Transform: the transform that is applied on the input.
            返回应用于输入的变换。
        """
        # 获取增强操作所需的参数
        args = _get_aug_input_args(self, aug_input)
        # 使用参数获取变换
        tfm = self.get_transform(*args)
        # 确保返回的是Transform或TransformList类型
        assert isinstance(tfm, (Transform, TransformList)), (
            f"{type(self)}.get_transform must return an instance of Transform! "
            f"Got {type(tfm)} instead."
        )
        # 对输入进行变换
        aug_input.transform(tfm)
        # 返回使用的变换
        return tfm

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        生成low和high之间的均匀分布随机浮点数。
        """
        # 如果没有指定high，则将low作为上限，0作为下限
        if high is None:
            low, high = 0, low
        # 如果没有指定size，则使用空列表
        if size is None:
            size = []
        # 返回指定范围内的随机数
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        生成类似这样的字符串表示：
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            # 获取__init__方法的签名
            sig = inspect.signature(self.__init__)
            # 获取类名
            classname = type(self).__name__
            # 用于存储参数字符串的列表
            argstr = []
            # 遍历所有参数
            for name, param in sig.parameters.items():
                # 不支持*args和**kwargs参数
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                # 确保属性存在
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                # 获取属性值
                attr = getattr(self, name)
                # 获取参数默认值
                default = param.default
                # 如果属性值等于默认值，则跳过
                if default is attr:
                    continue
                # 格式化属性值
                attr_str = pprint.pformat(attr)
                # 如果格式化后的字符串包含换行符，则使用省略号
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    # 如果pformat决定使用多行，则不显示
                    attr_str = "..."
                # 添加参数字符串
                argstr.append("{}={}".format(name, attr_str))
            # 返回最终的字符串表示
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            # 如果发生断言错误，则使用父类的__repr__
            return super().__repr__()

    # 将__str__方法设置为与__repr__相同
    __str__ = __repr__


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    将Transform包装成Augmentation。
    私有方法，在内部用于实现增强操作。
    """
    # 确保输入是Transform或Augmentation类型
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    # 如果已经是Augmentation类型，则直接返回
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:
        # 定义一个内部类来包装Transform
        class _TransformToAug(Augmentation):
            def __init__(self, tfm: Transform):
                self.tfm = tfm

            def get_transform(self, *args):
                return self.tfm

            def __repr__(self):
                return repr(self.tfm)

            __str__ = __repr__

        # 返回包装后的对象
        return _TransformToAug(tfm_or_aug)


class AugmentationList(Augmentation):
    """
    Apply a sequence of augmentations.
    应用一系列的数据增强操作。

    It has ``__call__`` method to apply the augmentations.
    它有``__call__``方法来应用这些增强操作。

    Note that :meth:`get_transform` method is impossible (will throw error if called)
    for :class:`AugmentationList`, because in order to apply a sequence of augmentations,
    the kth augmentation must be applied first, to provide inputs needed by the (k+1)th
    augmentation.
    注意:class:`AugmentationList`不可能有:meth:`get_transform`方法（如果调用会抛出错误），
    因为为了应用一系列增强操作，第k个增强必须先被应用，以提供第k+1个增强所需的输入。
    """

    def __init__(self, augs):
        """
        Args:
            augs (list[Augmentation or Transform]):
            增强操作或变换的列表
        """
        # 调用父类的初始化方法
        super().__init__()
        # 将所有输入转换为Augmentation类型
        self.augs = [_transform_to_aug(x) for x in augs]

    def __call__(self, aug_input) -> Transform:
        # 用于存储所有变换的列表
        tfms = []
        # 依次应用每个增强操作
        for x in self.augs:
            tfm = x(aug_input)
            tfms.append(tfm)
        # 返回变换列表
        return TransformList(tfms)

    def __repr__(self):
        # 获取每个增强操作的字符串表示
        msgs = [str(x) for x in self.augs]
        # 返回最终的字符串表示
        return "AugmentationList[{}]".format(", ".join(msgs))

    # 将__str__方法设置为与__repr__相同
    __str__ = __repr__


class AugInput:
    """
    Input that can be used with :meth:`Augmentation.__call__`.
    This is a standard implementation for the majority of use cases.
    This class provides the standard attributes **"image", "boxes", "sem_seg"**
    defined in :meth:`__init__` and they may be needed by different augmentations.
    Most augmentation policies do not need attributes beyond these three.
    可以与:meth:`Augmentation.__call__`一起使用的输入。
    这是大多数用例的标准实现。
    这个类提供了在:meth:`__init__`中定义的标准属性**"image", "boxes", "sem_seg"**，
    这些属性可能被不同的增强操作需要。
    大多数增强策略不需要这三个属性之外的属性。

    After applying augmentations to these attributes (using :meth:`AugInput.transform`),
    the returned transforms can then be used to transform other data structures that users have.
    在对这些属性应用增强操作后（使用:meth:`AugInput.transform`），
    返回的变换可以用于变换用户拥有的其他数据结构。

    Examples:
    ::
        input = AugInput(image, boxes=boxes)
        tfms = augmentation(input)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may implement augmentation policies
    that need other inputs. An algorithm may need to transform inputs in a way different
    from the standard approach defined in this class. In those rare situations, users can
    implement a class similar to this class, that satify the following condition:
    处理新数据类型的扩展项目可能需要实现需要其他输入的增强策略。
    算法可能需要以不同于此类中定义的标准方法来变换输入。
    在这些罕见的情况下，用户可以实现一个类似于这个类的类，需要满足以下条件：

    * The input must provide access to these data in the form of attribute access
      (``getattr``).  For example, if an :class:`Augmentation` to be applied needs "image"
      and "sem_seg" arguments, its input must have the attribute "image" and "sem_seg".
    * 输入必须以属性访问的形式提供对这些数据的访问（``getattr``）。
      例如，如果要应用的:class:`Augmentation`需要"image"和"sem_seg"参数，
      其输入必须具有"image"和"sem_seg"属性。

    * The input must have a ``transform(tfm: Transform) -> None`` method which
      in-place transforms all its attributes.
    * 输入必须有一个``transform(tfm: Transform) -> None``方法，
      该方法会原地变换其所有属性。
    """

    # TODO maybe should support more builtin data types here
    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. The meaning of C is up
                to users.
                (H,W)或(H,W,C)形状的ndarray，类型为uint8，范围在[0, 255]之间，
                或者浮点型，范围在[0, 1]或[0, 255]之间。C的含义由用户决定。

            boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
                Nx4形状的float32类型边界框，采用XYXY_ABS模式

            sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
                is an integer label of pixel.
                HxW形状的uint8类型语义分割掩码。每个元素是像素的整数标签。
        """
        # 检查图像数据类型
        _check_img_dtype(image)
        # 设置属性
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.
        原地变换此类的所有属性。

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        "原地"意味着在调用此方法后，访问诸如``self.image``这样的属性将返回变换后的数据。
        """
        # 变换图像
        self.image = tfm.apply_image(self.image)
        # 如果存在边界框，则变换边界框
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        # 如果存在语义分割掩码，则变换掩码
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Equivalent of ``AugmentationList(augmentations)(self)``
        等同于``AugmentationList(augmentations)(self)``
        """
        # 创建AugmentationList并应用于自身
        return AugmentationList(augmentations)(self)


def apply_augmentations(augmentations: List[Union[Transform, Augmentation]], inputs):
    """
    Use ``T.AugmentationList(augmentations)(inputs)`` instead.
    请使用``T.AugmentationList(augmentations)(inputs)``替代。
    """
    # 如果输入是numpy数组，处理常见的仅图像增强情况
    if isinstance(inputs, np.ndarray):
        # handle the common case of image-only Augmentation, also for backward compatibility
        # 处理仅图像增强的常见情况，同时保持向后兼容性
        image_only = True
        inputs = AugInput(inputs)
    else:
        image_only = False
    # 应用增强操作
    tfms = inputs.apply_augmentations(augmentations)
    # 根据是否仅处理图像返回相应结果
    return inputs.image if image_only else inputs, tfms


apply_transform_gens = apply_augmentations
"""
Alias for backward-compatibility.
用于向后兼容的别名。
"""

TransformGen = Augmentation
"""
Alias for Augmentation, since it is something that generates :class:`Transform`s
Augmentation的别名，因为它是用来生成:class:`Transform`的
"""

StandardAugInput = AugInput
"""
Alias for compatibility. It's not worth the complexity to have two classes.
用于兼容性的别名。没有必要使用两个类增加复杂性。
"""
