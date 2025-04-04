# Copyright (c) Facebook, Inc. and its affiliates.
import copy  # 导入复制模块，用于对象的深拷贝
import itertools  # 导入迭代工具模块
import numpy as np  # 导入numpy，用于数值计算
from typing import Any, Iterator, List, Union  # 导入类型提示相关模块
import pycocotools.mask as mask_util  # 导入COCO工具中的mask工具
import torch  # 导入PyTorch
from torch import device  # 导入device类型

from detectron2.layers.roi_align import ROIAlign  # 导入ROI对齐层
from detectron2.utils.memory import retry_if_cuda_oom  # 导入CUDA内存不足时重试的工具

from .boxes import Boxes  # 从当前包导入Boxes类


def polygon_area(x, y):
    # Using the shoelace formula
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    # 使用鞋带公式计算多边形面积
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)
    
    参数：
        polygons (list[ndarray])：每个数组形状为 (Nx2,)，表示多边形的顶点坐标
        height, width (int)：输出位掩码的高度和宽度

    Returns:
        ndarray: a bool mask of shape (height, width)
        
    返回：
        ndarray：形状为 (height, width) 的布尔掩码
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        # COCOAPI 不支持空多边形
        return np.zeros((height, width)).astype(np.bool)
    rles = mask_util.frPyObjects(polygons, height, width)  # 将多边形转换为RLE（行程长度编码）格式
    rle = mask_util.merge(rles)  # 合并多个RLE
    return mask_util.decode(rle).astype(np.bool)  # 将RLE解码为布尔掩码


def rasterize_polygons_within_box(
    polygons: List[np.ndarray], box: np.ndarray, mask_size: int
) -> torch.Tensor:
    """
    Rasterize the polygons into a mask image and
    crop the mask content in the given box.
    The cropped mask is resized to (mask_size, mask_size).
    
    将多边形栅格化为掩码图像，并在给定框内裁剪掩码内容。
    裁剪后的掩码会被调整为 (mask_size, mask_size) 的尺寸。

    This function is used when generating training targets for mask head in Mask R-CNN.
    Given original ground-truth masks for an image, new ground-truth mask
    training targets in the size of `mask_size x mask_size`
    must be provided for each predicted box. This function will be called to
    produce such targets.
    
    此函数用于为Mask R-CNN中的掩码头生成训练目标。
    给定图像的原始真实掩码，必须为每个预测框提供大小为`mask_size x mask_size`的新真实掩码训练目标。
    该函数将被调用以生成此类目标。

    Args:
        polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
        box: 4-element numpy array
        mask_size (int):
        
    参数：
        polygons (list[ndarray[float]])：表示一个实例的多边形列表。
        box: 包含4个元素的numpy数组，表示边界框坐标 [x1, y1, x2, y2]
        mask_size (int)：输出掩码的大小

    Returns:
        Tensor: BoolTensor of shape (mask_size, mask_size)
        
    返回：
        Tensor：形状为 (mask_size, mask_size) 的布尔张量
    """
    # 1. Shift the polygons w.r.t the boxes
    # 1. 相对于框移动多边形的坐标
    w, h = box[2] - box[0], box[3] - box[1]  # 计算框的宽度和高度

    polygons = copy.deepcopy(polygons)  # 深拷贝多边形，避免修改原始数据
    for p in polygons:
        p[0::2] = p[0::2] - box[0]  # 将多边形x坐标相对于框左上角偏移
        p[1::2] = p[1::2] - box[1]  # 将多边形y坐标相对于框左上角偏移

    # 2. Rescale the polygons to the new box size
    # max() to avoid division by small number
    # 2. 将多边形缩放到新的框大小
    # 使用max()避免除以小数字
    ratio_h = mask_size / max(h, 0.1)  # 计算高度缩放比例
    ratio_w = mask_size / max(w, 0.1)  # 计算宽度缩放比例

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h  # 如果宽高比例相同，直接缩放所有坐标
    else:
        for p in polygons:
            p[0::2] *= ratio_w  # 缩放x坐标
            p[1::2] *= ratio_h  # 缩放y坐标

    # 3. Rasterize the polygons with coco api
    # 3. 使用COCO API将多边形栅格化
    mask = polygons_to_bitmask(polygons, mask_size, mask_size)  # 将多边形转换为位掩码
    mask = torch.from_numpy(mask)  # 将numpy数组转换为PyTorch张量
    return mask


class BitMasks:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.
    
    该类以位图形式存储一张图像中所有对象的分割掩码。

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
        
    属性：
        tensor: 形状为N,H,W的布尔张量，表示图像中的N个实例。
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
            
        参数：
            tensor: 形状为N,H,W的布尔张量，表示图像中的N个实例。
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")  # 确定设备类型
        tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)  # 确保张量类型为布尔值
        assert tensor.dim() == 3, tensor.size()  # 确保张量是3维的
        self.image_size = tensor.shape[1:]  # 存储图像尺寸 (H, W)
        self.tensor = tensor  # 存储掩码张量

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "BitMasks":
        return BitMasks(self.tensor.to(*args, **kwargs))  # 将掩码转移到新设备或转换类型

    @property
    def device(self) -> torch.device:
        return self.tensor.device  # 返回掩码张量所在的设备

    @torch.jit.unused
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BitMasks":
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.
            
        返回：
            BitMasks：通过索引创建一个新的 :class:`BitMasks` 对象。

        The following usage are allowed:
        
        允许以下使用方式：

        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
           `new_masks = masks[3]`: 返回仅包含一个掩码的 `BitMasks`。
           
        2. `new_masks = masks[2:10]`: return a slice of masks.
           `new_masks = masks[2:10]`: 返回掩码的一个切片。
           
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.
           `new_masks = masks[vector]`，其中vector是长度为`len(masks)`的torch.BoolTensor。
           向量中非零元素将被选择。

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        
        请注意，返回的对象可能与此对象共享存储，取决于PyTorch的索引语义。
        """
        if isinstance(item, int):
            return BitMasks(self.tensor[item].unsqueeze(0))  # 如果是整数索引，返回单个掩码
        m = self.tensor[item]  # 应用索引
        assert m.dim() == 3, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )  # 确保结果仍是3维的
        return BitMasks(m)  # 返回新的BitMasks对象

    @torch.jit.unused
    def __iter__(self) -> torch.Tensor:
        yield from self.tensor  # 迭代每个掩码

    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s  # 对象的字符串表示，显示实例数量

    def __len__(self) -> int:
        return self.tensor.shape[0]  # 返回掩码数量

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.
        
        查找非空掩码。

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
                
        返回：
            Tensor：一个布尔张量，表示每个掩码是空(False)还是非空(True)。
        """
        return self.tensor.flatten(1).any(dim=1)  # 检查每个掩码是否包含任何True值

    @staticmethod
    def from_polygon_masks(
        polygon_masks: Union["PolygonMasks", List[List[np.ndarray]]], height: int, width: int
    ) -> "BitMasks":
        """
        Args:
            polygon_masks (list[list[ndarray]] or PolygonMasks)
            height, width (int)
            
        参数：
            polygon_masks (list[list[ndarray]] 或 PolygonMasks)：多边形掩码
            height, width (int)：输出位掩码的高度和宽度
        """
        if isinstance(polygon_masks, PolygonMasks):
            polygon_masks = polygon_masks.polygons  # 如果是PolygonMasks对象，获取其多边形
        masks = [polygons_to_bitmask(p, height, width) for p in polygon_masks]  # 将每个多边形转换为位掩码
        if len(masks):
            return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))  # 堆叠所有掩码
        else:
            return BitMasks(torch.empty(0, height, width, dtype=torch.bool))  # 创建空的BitMasks

    @staticmethod
    def from_roi_masks(roi_masks: "ROIMasks", height: int, width: int) -> "BitMasks":
        """
        Args:
            roi_masks:
            height, width (int):
            
        参数：
            roi_masks: ROI掩码对象
            height, width (int): 输出位掩码的高度和宽度
        """
        return roi_masks.to_bitmasks(height, width)  # 将ROI掩码转换为位掩码

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        """
        Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.
        It has less reconstruction error compared to rasterization with polygons.
        However we observe no difference in accuracy,
        but BitMasks requires more memory to store all the masks.
        
        根据给定的框裁剪每个位掩码，并将结果调整为 (mask_size, mask_size)。
        这可用于准备Mask R-CNN的训练目标。
        与多边形栅格化相比，它具有更少的重建误差。
        然而，我们观察到准确性没有差异，
        但BitMasks需要更多内存来存储所有掩码。

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.
            
        参数：
            boxes (Tensor)：存储每个掩码的框的Nx4张量
            mask_size (int)：栅格化掩码的大小

        Returns:
            Tensor:
                A bool tensor of shape (N, mask_size, mask_size), where
                N is the number of predicted boxes for this image.
                
        返回：
            Tensor：
                形状为 (N, mask_size, mask_size) 的布尔张量，其中
                N 是此图像的预测框数量。
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))  # 确保框和掩码数量一致
        device = self.tensor.device  # 获取设备

        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]  # 创建批次索引
        rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5，合并批次索引和框坐标

        bit_masks = self.tensor.to(dtype=torch.float32)  # 将掩码转换为浮点类型
        rois = rois.to(device=device)  # 确保ROI在正确的设备上
        output = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(bit_masks[:, None, :, :], rois)
            .squeeze(1)
        )  # 应用ROIAlign操作
        output = output >= 0.5  # 将输出转换回布尔类型
        return output

    def get_bounding_boxes(self) -> Boxes:
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
            
        返回：
            Boxes：围绕位掩码的紧密边界框。
            如果掩码为空，其边界框将全为零。
        """
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)  # 初始化边界框张量
        x_any = torch.any(self.tensor, dim=1)  # 沿高度维度检查任何True值
        y_any = torch.any(self.tensor, dim=2)  # 沿宽度维度检查任何True值
        for idx in range(self.tensor.shape[0]):
            x = torch.where(x_any[idx, :])[0]  # 找到x轴上有掩码的位置
            y = torch.where(y_any[idx, :])[0]  # 找到y轴上有掩码的位置
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )  # 设置边界框 [x1, y1, x2, y2]
        return Boxes(boxes)  # 返回Boxes对象

    @staticmethod
    def cat(bitmasks_list: List["BitMasks"]) -> "BitMasks":
        """
        Concatenates a list of BitMasks into a single BitMasks
        
        将BitMasks列表连接成单个BitMasks

        Arguments:
            bitmasks_list (list[BitMasks])
            
        参数：
            bitmasks_list (list[BitMasks])：要连接的BitMasks列表

        Returns:
            BitMasks: the concatenated BitMasks
            
        返回：
            BitMasks：连接后的BitMasks
        """
        assert isinstance(bitmasks_list, (list, tuple))  # 确保输入是列表或元组
        assert len(bitmasks_list) > 0  # 确保列表非空
        assert all(isinstance(bitmask, BitMasks) for bitmask in bitmasks_list)  # 确保所有元素都是BitMasks

        cat_bitmasks = type(bitmasks_list[0])(torch.cat([bm.tensor for bm in bitmasks_list], dim=0))  # 连接所有掩码张量
        return cat_bitmasks  # 返回连接后的BitMasks


class PolygonMasks:
    """
    This class stores the segmentation masks for all objects in one image, in the form of polygons.
    
    该类以多边形形式存储一张图像中所有对象的分割掩码。

    Attributes:
        polygons: list[list[ndarray]]. Each ndarray is a float64 vector representing a polygon.
        
    属性:
        polygons: list[list[ndarray]]。每个ndarray是一个表示多边形的float64向量。
    """

    def __init__(self, polygons: List[List[Union[torch.Tensor, np.ndarray]]]):
        """
        Arguments:
            polygons (list[list[np.ndarray]]): The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                instance, and the third level to the polygon coordinates.
                The third level array should have the format of
                [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                
        参数:
            polygons (list[list[np.ndarray]]): 列表的第一层
                对应于单个实例，第二层对应于组成该实例的
                所有多边形，第三层对应于多边形坐标。
                第三层数组应具有格式
                [x0, y0, x1, y1, ..., xn, yn] (n >= 3)。
        """
        if not isinstance(polygons, list):
            raise ValueError(
                "Cannot create PolygonMasks: Expect a list of list of polygons per image. "
                "Got '{}' instead.".format(type(polygons))
            )  # 确保输入是列表类型

        def _make_array(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
            # Use float64 for higher precision, because why not?
            # Always put polygons on CPU (self.to is a no-op) since they
            # are supposed to be small tensors.
            # May need to change this assumption if GPU placement becomes useful
            # 使用float64以获得更高精度，为什么不呢？
            # 始终将多边形放在CPU上（self.to是一个无操作），因为它们
            # 应该是小型张量。
            # 如果GPU放置变得有用，可能需要改变这个假设
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()  # 如果是张量，转换为numpy数组
            return np.asarray(t).astype("float64")  # 转换为float64类型的numpy数组

        def process_polygons(
            polygons_per_instance: List[Union[torch.Tensor, np.ndarray]]
        ) -> List[np.ndarray]:
            if not isinstance(polygons_per_instance, list):
                raise ValueError(
                    "Cannot create polygons: Expect a list of polygons per instance. "
                    "Got '{}' instead.".format(type(polygons_per_instance))
                )  # 确保每个实例的多边形是列表类型
            # transform each polygon to a numpy array
            # 将每个多边形转换为numpy数组
            polygons_per_instance = [_make_array(p) for p in polygons_per_instance]  # 转换所有多边形为numpy数组
            for polygon in polygons_per_instance:
                if len(polygon) % 2 != 0 or len(polygon) < 6:
                    raise ValueError(f"Cannot create a polygon from {len(polygon)} coordinates.")  # 确保每个多边形坐标点数量正确
            return polygons_per_instance

        self.polygons: List[List[np.ndarray]] = [
            process_polygons(polygons_per_instance) for polygons_per_instance in polygons
        ]  # 处理所有实例的多边形

    def to(self, *args: Any, **kwargs: Any) -> "PolygonMasks":
        return self  # 由于多边形始终在CPU上，所以to方法只是返回自身

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")  # 多边形掩码始终在CPU上

    def get_bounding_boxes(self) -> Boxes:
        """
        Returns:
            Boxes: tight bounding boxes around polygon masks.
            
        返回:
            Boxes: 围绕多边形掩码的紧密边界框。
        """
        boxes = torch.zeros(len(self.polygons), 4, dtype=torch.float32)  # 初始化边界框张量
        for idx, polygons_per_instance in enumerate(self.polygons):
            minxy = torch.as_tensor([float("inf"), float("inf")], dtype=torch.float32)  # 初始化最小xy坐标为无穷大
            maxxy = torch.zeros(2, dtype=torch.float32)  # 初始化最大xy坐标为零
            for polygon in polygons_per_instance:
                coords = torch.from_numpy(polygon).view(-1, 2).to(dtype=torch.float32)  # 将多边形坐标重塑为Nx2形状
                minxy = torch.min(minxy, torch.min(coords, dim=0).values)  # 更新最小xy坐标
                maxxy = torch.max(maxxy, torch.max(coords, dim=0).values)  # 更新最大xy坐标
            boxes[idx, :2] = minxy  # 设置边界框左上角坐标
            boxes[idx, 2:] = maxxy  # 设置边界框右下角坐标
        return Boxes(boxes)  # 返回Boxes对象

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.
        
        查找非空掩码。

        Returns:
            Tensor:
                a BoolTensor which represents whether each mask is empty (False) or not (True).
                
        返回:
            Tensor:
                一个BoolTensor，表示每个掩码是否为空(False)或非空(True)。
        """
        keep = [1 if len(polygon) > 0 else 0 for polygon in self.polygons]  # 检查每个实例是否有多边形
        return torch.from_numpy(np.asarray(keep, dtype=np.bool))  # 返回布尔张量

    def __getitem__(self, item: Union[int, slice, List[int], torch.BoolTensor]) -> "PolygonMasks":
        """
        Support indexing over the instances and return a `PolygonMasks` object.
        `item` can be:
        
        支持对实例进行索引，并返回一个`PolygonMasks`对象。
        `item`可以是:

        1. An integer. It will return an object with only one instance.
           一个整数。它将返回只有一个实例的对象。
           
        2. A slice. It will return an object with the selected instances.
           一个切片。它将返回包含所选实例的对象。
           
        3. A list[int]. It will return an object with the selected instances,
           correpsonding to the indices in the list.
           一个list[int]。它将返回包含所选实例的对象，
           对应于列表中的索引。
           
        4. A vector mask of type BoolTensor, whose length is num_instances.
           It will return an object with the instances whose mask is nonzero.
           一个类型为BoolTensor的向量掩码，其长度为num_instances。
           它将返回具有掩码为非零的实例的对象。
        """
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]  # 选择单个实例的多边形
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]  # 选择一个切片的实例
        elif isinstance(item, list):
            selected_polygons = [self.polygons[i] for i in item]  # 选择指定索引的实例
        elif isinstance(item, torch.Tensor):
            # Polygons is a list, so we have to move the indices back to CPU.
            # 多边形是一个列表，所以我们必须将索引移回CPU。
            if item.dtype == torch.bool:
                assert item.dim() == 1, item.shape
                item = item.nonzero().squeeze(1).cpu().numpy().tolist()  # 将布尔掩码转换为索引列表
            elif item.dtype in [torch.int32, torch.int64]:
                item = item.cpu().numpy().tolist()  # 将整数张量转换为列表
            else:
                raise ValueError("Unsupported tensor dtype={} for indexing!".format(item.dtype))  # 不支持的张量类型
            selected_polygons = [self.polygons[i] for i in item]  # 选择指定索引的实例
        return PolygonMasks(selected_polygons)  # 返回新的PolygonMasks对象

    def __iter__(self) -> Iterator[List[np.ndarray]]:
        """
        Yields:
            list[ndarray]: the polygons for one instance.
            Each Tensor is a float64 vector representing a polygon.
            
        生成:
            list[ndarray]: 一个实例的多边形。
            每个Tensor是一个表示多边形的float64向量。
        """
        return iter(self.polygons)  # 迭代所有实例的多边形

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.polygons))
        return s  # 返回类的字符串表示

    def __len__(self) -> int:
        return len(self.polygons)  # 返回实例数量

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        """
        Crop each mask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.
        
        根据给定的框裁剪每个掩码，并将结果调整为(mask_size, mask_size)。
        这可用于准备Mask R-CNN的训练目标。

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.
            
        参数:
            boxes (Tensor): 存储每个掩码的框的Nx4张量
            mask_size (int): 栅格化掩码的大小。

        Returns:
            Tensor: A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
            
        返回:
            Tensor: 形状为(N, mask_size, mask_size)的布尔张量，其中
            N是此图像的预测框数量。
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))  # 确保框和掩码数量一致

        device = boxes.device
        # Put boxes on the CPU, as the polygon representation is not efficient GPU-wise
        # (several small tensors for representing a single instance mask)
        # 将框放在CPU上，因为多边形表示在GPU上效率不高
        # (多个小张量表示单个实例掩码)
        boxes = boxes.to(torch.device("cpu"))  # 将框转移到CPU

        results = [
            rasterize_polygons_within_box(poly, box.numpy(), mask_size)
            for poly, box in zip(self.polygons, boxes)
        ]  # 对每个(多边形,框)对进行栅格化
        """
        poly: list[list[float]], the polygons for one instance
        box: a tensor of shape (4,)
        
        poly: list[list[float]], 一个实例的多边形
        box: 形状为(4,)的张量
        """
        if len(results) == 0:
            return torch.empty(0, mask_size, mask_size, dtype=torch.bool, device=device)  # 如果没有结果，返回空张量
        return torch.stack(results, dim=0).to(device=device)  # 堆叠所有结果并转移到指定设备

    def area(self):
        """
        Computes area of the mask.
        Only works with Polygons, using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        
        计算掩码的面积。
        仅适用于多边形，使用鞋带公式：
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Returns:
            Tensor: a vector, area for each instance
            
        返回:
            Tensor: 一个向量，每个实例的面积
        """

        area = []
        for polygons_per_instance in self.polygons:
            area_per_instance = 0
            for p in polygons_per_instance:
                area_per_instance += polygon_area(p[0::2], p[1::2])  # 使用鞋带公式计算多边形面积
            area.append(area_per_instance)  # 添加实例的总面积

        return torch.tensor(area)  # 返回面积张量

    @staticmethod
    def cat(polymasks_list: List["PolygonMasks"]) -> "PolygonMasks":
        """
        Concatenates a list of PolygonMasks into a single PolygonMasks
        
        将PolygonMasks列表连接成单个PolygonMasks

        Arguments:
            polymasks_list (list[PolygonMasks])
            
        参数:
            polymasks_list (list[PolygonMasks]): 要连接的PolygonMasks列表

        Returns:
            PolygonMasks: the concatenated PolygonMasks
            
        返回:
            PolygonMasks: 连接后的PolygonMasks
        """
        assert isinstance(polymasks_list, (list, tuple))  # 确保输入是列表或元组
        assert len(polymasks_list) > 0  # 确保列表非空
        assert all(isinstance(polymask, PolygonMasks) for polymask in polymasks_list)  # 确保所有元素都是PolygonMasks

        cat_polymasks = type(polymasks_list[0])(
            list(itertools.chain.from_iterable(pm.polygons for pm in polymasks_list))
        )  # 使用itertools.chain连接所有多边形列表
        return cat_polymasks  # 返回连接后的PolygonMasks


class ROIMasks:
    """
    Represent masks by N smaller masks defined in some ROIs. Once ROI boxes are given,
    full-image bitmask can be obtained by "pasting" the mask on the region defined
    by the corresponding ROI box.
    
    通过在某些ROI中定义的N个较小掩码来表示掩码。一旦给定ROI框，
    可以通过将掩码"粘贴"到相应ROI框定义的区域上来获得全图位掩码。
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor: (N, M, M) mask tensor that defines the mask within each ROI.
            
        参数:
            tensor: (N, M, M)掩码张量，定义每个ROI内的掩码。
        """
        if tensor.dim() != 3:
            raise ValueError("ROIMasks must take a masks of 3 dimension.")  # 确保输入是3维张量
        self.tensor = tensor  # 存储掩码张量

    def to(self, device: torch.device) -> "ROIMasks":
        return ROIMasks(self.tensor.to(device))  # 将掩码转移到指定设备

    @property
    def device(self) -> device:
        return self.tensor.device  # 返回掩码张量所在的设备

    def __len__(self):
        return self.tensor.shape[0]  # 返回ROI掩码的数量

    def __getitem__(self, item) -> "ROIMasks":
        """
        Returns:
            ROIMasks: Create a new :class:`ROIMasks` by indexing.
            
        返回:
            ROIMasks: 通过索引创建一个新的:class:`ROIMasks`。

        The following usage are allowed:
        
        允许以下用法:

        1. `new_masks = masks[2:10]`: return a slice of masks.
           `new_masks = masks[2:10]`: 返回掩码的一个切片。
           
        2. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.
           `new_masks = masks[vector]`，其中vector是一个torch.BoolTensor，
           `length = len(masks)`。向量中的非零元素将被选择。

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        
        请注意，返回的对象可能与此对象共享存储，
        取决于PyTorch的索引语义。
        """
        t = self.tensor[item]  # 应用索引
        if t.dim() != 3:
            raise ValueError(
                f"Indexing on ROIMasks with {item} returns a tensor with shape {t.shape}!"
            )  # 确保结果仍是3维的
        return ROIMasks(t)  # 返回新的ROIMasks对象

    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s  # 返回类的字符串表示

    @torch.jit.unused
    def to_bitmasks(self, boxes: torch.Tensor, height, width, threshold=0.5):
        """
        Args:
            boxes: 将ROI掩码粘贴到图像上的位置
            height, width: 输出位掩码的尺寸
            threshold: 二值化阈值
        """
        from detectron2.layers import paste_masks_in_image  # 导入掩码粘贴函数

        paste = retry_if_cuda_oom(paste_masks_in_image)  # 如果CUDA内存不足则重试的掩码粘贴函数
        bitmasks = paste(
            self.tensor,
            boxes,
            (height, width),
            threshold=threshold,
        )  # 将ROI掩码粘贴到全图上
        return BitMasks(bitmasks)  # 返回BitMasks对象
