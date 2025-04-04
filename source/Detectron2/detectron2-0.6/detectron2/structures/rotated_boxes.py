# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List, Tuple
import torch

from detectron2.layers.rotated_boxes import pairwise_iou_rotated

from .boxes import Boxes


class RotatedBoxes(Boxes):
    """
    This structure stores a list of rotated boxes as a Nx5 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)
    
    该结构将一系列旋转框存储为 Nx5 的 torch.Tensor。
    它支持一些关于框的常用方法
    (`area`, `clip`, `nonempty` 等)，
    并且行为类似于张量
    (支持索引、`to(device)`、`.device` 以及对所有框的迭代)
    
    Attributes:
        tensor (torch.Tensor): float matrix of Nx5. Each row is
            (x_center, y_center, width, height, angle),
            in which angle is represented in degrees.
            While there's no strict range limitation for the angle,
            it's recommended to normalize the angle to [-180, 180) or [0, 360)
            for better readability and easier visualization.
            
    属性:
        tensor (torch.Tensor): Nx5 的浮点矩阵。每行表示
            (x_center, y_center, width, height, angle)，
            其中角度以度为单位表示。
            虽然角度没有严格的范围限制，
            但建议将角度归一化到 [-180, 180) 或 [0, 360) 范围内，
            以便更好地可读性和更容易的可视化。
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx5 matrix.  Each row is
                (x_center, y_center, width, height, angle),
                in which angle is represented in degrees.
                While there's no strict range restriction for it,
                the recommended principal range is between [-180, 180) degrees.

        Assume we have a horizontal box B = (x_center, y_center, width, height),
        where width is along the x-axis and height is along the y-axis.
        The rotated box B_rot (x_center, y_center, width, height, angle)
        can be seen as:

        1. When angle == 0:
           B_rot == B
        2. When angle > 0:
           B_rot is obtained by rotating B w.r.t its center by :math:`|angle|` degrees CCW;
        3. When angle < 0:
           B_rot is obtained by rotating B w.r.t its center by :math:`|angle|` degrees CW.

        Mathematically, since the right-handed coordinate system for image space
        is (y, x), where y is top->down and x is left->right, the 4 vertices of the
        rotated rectangle :math:`(yr_i, xr_i)` (i = 1, 2, 3, 4) can be obtained from
        the vertices of the horizontal rectangle :math:`(y_i, x_i)` (i = 1, 2, 3, 4)
        in the following way (:math:`\\theta = angle*\\pi/180` is the angle in radians,
        :math:`(y_c, x_c)` is the center of the rectangle):

        .. math::

            yr_i = \\cos(\\theta) (y_i - y_c) - \\sin(\\theta) (x_i - x_c) + y_c,

            xr_i = \\sin(\\theta) (y_i - y_c) + \\cos(\\theta) (x_i - x_c) + x_c,

        which is the standard rigid-body rotation transformation.

        Intuitively, the angle is
        (1) the rotation angle from y-axis in image space
        to the height vector (top->down in the box's local coordinate system)
        of the box in CCW, and
        (2) the rotation angle from x-axis in image space
        to the width vector (left->right in the box's local coordinate system)
        of the box in CCW.

        More intuitively, consider the following horizontal box ABCD represented
        in (x1, y1, x2, y2): (3, 2, 7, 4),
        covering the [3, 7] x [2, 4] region of the continuous coordinate system
        which looks like this:

        .. code:: none

            O--------> x
            |
            |  A---B
            |  |   |
            |  D---C
            |
            v y

        Note that each capital letter represents one 0-dimensional geometric point
        instead of a 'square pixel' here.

        In the example above, using (x, y) to represent a point we have:

        .. math::

            O = (0, 0), A = (3, 2), B = (7, 2), C = (7, 4), D = (3, 4)

        We name vector AB = vector DC as the width vector in box's local coordinate system, and
        vector AD = vector BC as the height vector in box's local coordinate system. Initially,
        when angle = 0 degree, they're aligned with the positive directions of x-axis and y-axis
        in the image space, respectively.

        For better illustration, we denote the center of the box as E,

        .. code:: none

            O--------> x
            |
            |  A---B
            |  | E |
            |  D---C
            |
            v y

        where the center E = ((3+7)/2, (2+4)/2) = (5, 3).

        Also,

        .. math::

            width = |AB| = |CD| = 7 - 3 = 4,
            height = |AD| = |BC| = 4 - 2 = 2.

        Therefore, the corresponding representation for the same shape in rotated box in
        (x_center, y_center, width, height, angle) format is:

        (5, 3, 4, 2, 0),

        Now, let's consider (5, 3, 4, 2, 90), which is rotated by 90 degrees
        CCW (counter-clockwise) by definition. It looks like this:

        .. code:: none

            O--------> x
            |   B-C
            |   | |
            |   |E|
            |   | |
            |   A-D
            v y

        The center E is still located at the same point (5, 3), while the vertices
        ABCD are rotated by 90 degrees CCW with regard to E:
        A = (4, 5), B = (4, 1), C = (6, 1), D = (6, 5)

        Here, 90 degrees can be seen as the CCW angle to rotate from y-axis to
        vector AD or vector BC (the top->down height vector in box's local coordinate system),
        or the CCW angle to rotate from x-axis to vector AB or vector DC (the left->right
        width vector in box's local coordinate system).

        .. math::

            width = |AB| = |CD| = 5 - 1 = 4,
            height = |AD| = |BC| = 6 - 4 = 2.

        Next, how about (5, 3, 4, 2, -90), which is rotated by 90 degrees CW (clockwise)
        by definition? It looks like this:

        .. code:: none

            O--------> x
            |   D-A
            |   | |
            |   |E|
            |   | |
            |   C-B
            v y

        The center E is still located at the same point (5, 3), while the vertices
        ABCD are rotated by 90 degrees CW with regard to E:
        A = (6, 1), B = (6, 5), C = (4, 5), D = (4, 1)

        .. math::

            width = |AB| = |CD| = 5 - 1 = 4,
            height = |AD| = |BC| = 6 - 4 = 2.

        This covers exactly the same region as (5, 3, 4, 2, 90) does, and their IoU
        will be 1. However, these two will generate different RoI Pooling results and
        should not be treated as an identical box.

        On the other hand, it's easy to see that (X, Y, W, H, A) is identical to
        (X, Y, W, H, A+360N), for any integer N. For example (5, 3, 4, 2, 270) would be
        identical to (5, 3, 4, 2, -90), because rotating the shape 270 degrees CCW is
        equivalent to rotating the same shape 90 degrees CW.

        We could rotate further to get (5, 3, 4, 2, 180), or (5, 3, 4, 2, -180):

        .. code:: none

            O--------> x
            |
            |  C---D
            |  | E |
            |  B---A
            |
            v y

        .. math::

            A = (7, 4), B = (3, 4), C = (3, 2), D = (7, 2),

            width = |AB| = |CD| = 7 - 3 = 4,
            height = |AD| = |BC| = 4 - 2 = 2.

        Finally, this is a very inaccurate (heavily quantized) illustration of
        how (5, 3, 4, 2, 60) looks like in case anyone wonders:

        .. code:: none

            O--------> x
            |     B\
            |    /  C
            |   /E /
            |  A  /
            |   `D
            v y

        It's still a rectangle with center of (5, 3), width of 4 and height of 2,
        but its angle (and thus orientation) is somewhere between
        (5, 3, 4, 2, 0) and (5, 3, 4, 2, 90).
        
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")  # 确定设备：如果tensor是torch.Tensor类型则使用其设备，否则使用CPU
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)  # 将输入转换为指定类型和设备的张量
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            # 使用reshape，这样我们就不会创建一个不依赖于输入的新张量（从而使jit混淆）
            tensor = tensor.reshape((0, 5)).to(dtype=torch.float32, device=device)  # 如果张量为空，将其重塑为(0,5)形状
        assert tensor.dim() == 2 and tensor.size(-1) == 5, tensor.size()  # 确保张量是2维且最后一维大小为5

        self.tensor = tensor  # 存储张量

    def clone(self) -> "RotatedBoxes":
        """
        Clone the RotatedBoxes.

        Returns:
            RotatedBoxes
            
        克隆旋转框。

        返回:
            RotatedBoxes
        """
        return RotatedBoxes(self.tensor.clone())  # 返回tensor克隆后的新RotatedBoxes实例

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        # 假设框是float32类型，不支持to(dtype)
        return RotatedBoxes(self.tensor.to(device=device))  # 返回转移到指定设备的新RotatedBoxes实例

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
            
        计算所有框的面积。

        返回:
            torch.Tensor: 包含每个框面积的向量。
        """
        box = self.tensor  # 获取框张量
        area = box[:, 2] * box[:, 3]  # 计算面积：宽 * 高
        return area  # 返回面积

    def normalize_angles(self) -> None:
        """
        Restrict angles to the range of [-180, 180) degrees
        
        将角度限制在 [-180, 180) 度的范围内
        """
        self.tensor[:, 4] = (self.tensor[:, 4] + 180.0) % 360.0 - 180.0  # 对角度进行归一化处理

    def clip(self, box_size: Tuple[int, int], clip_angle_threshold: float = 1.0) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].
        
        通过将x坐标限制在[0, width]范围内和y坐标限制在[0, height]范围内来裁剪（原地操作）框。

        For RRPN:
        Only clip boxes that are almost horizontal with a tolerance of
        clip_angle_threshold to maintain backward compatibility.
        
        对于RRPN：
        只裁剪那些几乎水平的框，角度误差在clip_angle_threshold之内，以保持向后兼容性。

        Rotated boxes beyond this threshold are not clipped for two reasons:
        
        超出此阈值的旋转框不被裁剪，原因有两个：

        1. There are potentially multiple ways to clip a rotated box to make it
           fit within the image.
           有多种可能的方式来裁剪旋转框以使其适合图像。
           
        2. It's tricky to make the entire rectangular box fit within the image
           and still be able to not leave out pixels of interest.
           让整个矩形框适合图像并且不遗漏感兴趣的像素是很棘手的。

        Therefore we rely on ops like RoIAlignRotated to safely handle this.
        
        因此，我们依靠RoIAlignRotated这样的操作来安全地处理这种情况。

        Args:
            box_size (height, width): The clipping box's size.
            clip_angle_threshold:
                Iff. abs(normalized(angle)) <= clip_angle_threshold (in degrees),
                we do the clipping as horizontal boxes.
                
        参数：
            box_size (height, width)：裁剪框的尺寸。
            clip_angle_threshold：
                当且仅当abs(normalized(angle)) <= clip_angle_threshold（以度为单位）时，
                我们将其作为水平框进行裁剪。
        """
        h, w = box_size  # 获取高度和宽度

        # normalize angles to be within (-180, 180] degrees
        # 将角度归一化到(-180, 180]度范围内
        self.normalize_angles()  # 归一化角度

        idx = torch.where(torch.abs(self.tensor[:, 4]) <= clip_angle_threshold)[0]  # 找出角度小于阈值的框的索引

        # convert to (x1, y1, x2, y2)
        # 转换为(x1, y1, x2, y2)格式
        x1 = self.tensor[idx, 0] - self.tensor[idx, 2] / 2.0  # 计算左上角x坐标
        y1 = self.tensor[idx, 1] - self.tensor[idx, 3] / 2.0  # 计算左上角y坐标
        x2 = self.tensor[idx, 0] + self.tensor[idx, 2] / 2.0  # 计算右下角x坐标
        y2 = self.tensor[idx, 1] + self.tensor[idx, 3] / 2.0  # 计算右下角y坐标

        # clip
        # 裁剪
        x1.clamp_(min=0, max=w)  # 将x1限制在[0,w]范围内
        y1.clamp_(min=0, max=h)  # 将y1限制在[0,h]范围内
        x2.clamp_(min=0, max=w)  # 将x2限制在[0,w]范围内
        y2.clamp_(min=0, max=h)  # 将y2限制在[0,h]范围内

        # convert back to (xc, yc, w, h)
        # 转换回(xc, yc, w, h)格式
        self.tensor[idx, 0] = (x1 + x2) / 2.0  # 计算中心x坐标
        self.tensor[idx, 1] = (y1 + y2) / 2.0  # 计算中心y坐标
        # make sure widths and heights do not increase due to numerical errors
        # 确保宽度和高度不会因为数值误差而增加
        self.tensor[idx, 2] = torch.min(self.tensor[idx, 2], x2 - x1)  # 更新宽度
        self.tensor[idx, 3] = torch.min(self.tensor[idx, 3], y2 - y1)  # 更新高度

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.
        
        找出非空的框。
        如果框的任一边长不大于阈值，则认为该框为空。

        Returns:
            Tensor: a binary vector which represents
            whether each box is empty (False) or non-empty (True).
            
        返回：
            Tensor：一个二元向量，表示每个框是否为空（False）或非空（True）。
        """
        box = self.tensor  # 获取框张量
        widths = box[:, 2]  # 获取所有框的宽度
        heights = box[:, 3]  # 获取所有框的高度
        keep = (widths > threshold) & (heights > threshold)  # 宽度和高度都大于阈值的框为非空
        return keep  # 返回非空框的掩码

    def __getitem__(self, item) -> "RotatedBoxes":
        """
        Returns:
            RotatedBoxes: Create a new :class:`RotatedBoxes` by indexing.
            
        返回：
            RotatedBoxes：通过索引创建一个新的:class:`RotatedBoxes`。

        The following usage are allowed:
        
        允许以下用法：

        1. `new_boxes = boxes[3]`: return a `RotatedBoxes` which contains only one box.
           `new_boxes = boxes[3]`：返回一个只包含一个框的`RotatedBoxes`。
           
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
           `new_boxes = boxes[2:10]`：返回一个框的切片。
           
        3. `new_boxes = boxes[vector]`, where vector is a torch.ByteTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.
           `new_boxes = boxes[vector]`，其中vector是一个torch.ByteTensor，
           `length = len(boxes)`。向量中的非零元素将被选中。

        Note that the returned RotatedBoxes might share storage with this RotatedBoxes,
        subject to Pytorch's indexing semantics.
        
        请注意，返回的RotatedBoxes可能与此RotatedBoxes共享存储，
        这取决于PyTorch的索引语义。
        """
        if isinstance(item, int):
            return RotatedBoxes(self.tensor[item].view(1, -1))  # 如果索引是整数，返回单个框
        b = self.tensor[item]  # 应用索引
        assert b.dim() == 2, "Indexing on RotatedBoxes with {} failed to return a matrix!".format(
            item
        )  # 确保结果是2维的
        return RotatedBoxes(b)  # 返回新的RotatedBoxes对象

    def __len__(self) -> int:
        return self.tensor.shape[0]  # 返回框的数量

    def __repr__(self) -> str:
        return "RotatedBoxes(" + str(self.tensor) + ")"  # 返回字符串表示

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box covering
                [0, width] x [0, height]
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".
                
        参数：
            box_size (height, width)：覆盖[0, width] x [0, height]的参考框的尺寸
            boundary_threshold (int)：超出参考框边界超过boundary_threshold的框被视为"外部"。

        For RRPN, it might not be necessary to call this function since it's common
        for rotated box to extend to outside of the image boundaries
        (the clip function only clips the near-horizontal boxes)
        
        对于RRPN，可能不需要调用此函数，因为旋转框通常会延伸到图像边界之外
        （clip函数只裁剪接近水平的框）

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
            
        返回：
            一个二元向量，指示每个框是否在参考框内。
        """
        height, width = box_size  # 获取参考框的高度和宽度

        cnt_x = self.tensor[..., 0]  # 获取框中心的x坐标
        cnt_y = self.tensor[..., 1]  # 获取框中心的y坐标
        half_w = self.tensor[..., 2] / 2.0  # 计算框宽度的一半
        half_h = self.tensor[..., 3] / 2.0  # 计算框高度的一半
        a = self.tensor[..., 4]  # 获取框的角度
        c = torch.abs(torch.cos(a * math.pi / 180.0))  # 计算角度的余弦绝对值
        s = torch.abs(torch.sin(a * math.pi / 180.0))  # 计算角度的正弦绝对值
        # This basically computes the horizontal bounding rectangle of the rotated box
        # 这基本上计算旋转框的水平边界矩形
        max_rect_dx = c * half_w + s * half_h  # 计算水平边界矩形宽度的一半
        max_rect_dy = c * half_h + s * half_w  # 计算水平边界矩形高度的一半

        inds_inside = (
            (cnt_x - max_rect_dx >= -boundary_threshold)  # 左边界检查
            & (cnt_y - max_rect_dy >= -boundary_threshold)  # 上边界检查
            & (cnt_x + max_rect_dx < width + boundary_threshold)  # 右边界检查
            & (cnt_y + max_rect_dy < height + boundary_threshold)  # 下边界检查
        )  # 判断框是否在参考框内

        return inds_inside  # 返回框是否在参考框内的掩码

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
            
        返回：
            框中心的Nx2数组(x, y)。
        """
        return self.tensor[:, :2]  # 返回所有框的中心坐标

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the rotated box with horizontal and vertical scaling factors
        Note: when scale_factor_x != scale_factor_y,
        the rotated box does not preserve the rectangular shape when the angle
        is not a multiple of 90 degrees under resize transformation.
        Instead, the shape is a parallelogram (that has skew)
        Here we make an approximation by fitting a rotated rectangle to the parallelogram.
        
        使用水平和垂直缩放因子缩放旋转框
        注意：当scale_factor_x != scale_factor_y时，
        如果角度不是90度的倍数，则在调整大小变换下旋转框不会保持矩形形状。
        相反，形状是一个平行四边形（具有倾斜）
        这里我们通过将旋转矩形拟合到平行四边形来进行近似。
        """
        self.tensor[:, 0] *= scale_x  # 缩放中心x坐标
        self.tensor[:, 1] *= scale_y  # 缩放中心y坐标
        theta = self.tensor[:, 4] * math.pi / 180.0  # 将角度转换为弧度
        c = torch.cos(theta)  # 计算角度的余弦值
        s = torch.sin(theta)  # 计算角度的正弦值

        # In image space, y is top->down and x is left->right
        # Consider the local coordintate system for the rotated box,
        # where the box center is located at (0, 0), and the four vertices ABCD are
        # A(-w / 2, -h / 2), B(w / 2, -h / 2), C(w / 2, h / 2), D(-w / 2, h / 2)
        # the midpoint of the left edge AD of the rotated box E is:
        # E = (A+D)/2 = (-w / 2, 0)
        # the midpoint of the top edge AB of the rotated box F is:
        # F(0, -h / 2)
        # To get the old coordinates in the global system, apply the rotation transformation
        # (Note: the right-handed coordinate system for image space is yOx):
        # (old_x, old_y) = (s * y + c * x, c * y - s * x)
        # E(old) = (s * 0 + c * (-w/2), c * 0 - s * (-w/2)) = (-c * w / 2, s * w / 2)
        # F(old) = (s * (-h / 2) + c * 0, c * (-h / 2) - s * 0) = (-s * h / 2, -c * h / 2)
        # After applying the scaling factor (sfx, sfy):
        # E(new) = (-sfx * c * w / 2, sfy * s * w / 2)
        # F(new) = (-sfx * s * h / 2, -sfy * c * h / 2)
        # The new width after scaling tranformation becomes:

        # w(new) = |E(new) - O| * 2
        #        = sqrt[(sfx * c * w / 2)^2 + (sfy * s * w / 2)^2] * 2
        #        = sqrt[(sfx * c)^2 + (sfy * s)^2] * w
        # i.e., scale_factor_w = sqrt[(sfx * c)^2 + (sfy * s)^2]
        #
        # For example,
        # when angle = 0 or 180, |c| = 1, s = 0, scale_factor_w == scale_factor_x;
        # when |angle| = 90, c = 0, |s| = 1, scale_factor_w == scale_factor_y
        self.tensor[:, 2] *= torch.sqrt((scale_x * c) ** 2 + (scale_y * s) ** 2)  # 根据公式缩放宽度

        # h(new) = |F(new) - O| * 2
        #        = sqrt[(sfx * s * h / 2)^2 + (sfy * c * h / 2)^2] * 2
        #        = sqrt[(sfx * s)^2 + (sfy * c)^2] * h
        # i.e., scale_factor_h = sqrt[(sfx * s)^2 + (sfy * c)^2]
        #
        # For example,
        # when angle = 0 or 180, |c| = 1, s = 0, scale_factor_h == scale_factor_y;
        # when |angle| = 90, c = 0, |s| = 1, scale_factor_h == scale_factor_x
        
        # h(new) = |F(new) - O| * 2
        #        = sqrt[(sfx * s * h / 2)^2 + (sfy * c * h / 2)^2] * 2
        #        = sqrt[(sfx * s)^2 + (sfy * c)^2] * h
        # 即，scale_factor_h = sqrt[(sfx * s)^2 + (sfy * c)^2]
        #
        # 例如，
        # 当angle = 0或180时，|c| = 1, s = 0, scale_factor_h == scale_factor_y;
        # 当|angle| = 90时，c = 0, |s| = 1, scale_factor_h == scale_factor_x
        self.tensor[:, 3] *= torch.sqrt((scale_x * s) ** 2 + (scale_y * c) ** 2)  # 根据公式缩放高度

        # The angle is the rotation angle from y-axis in image space to the height
        # vector (top->down in the box's local coordinate system) of the box in CCW.
        #
        # angle(new) = angle_yOx(O - F(new))
        #            = angle_yOx( (sfx * s * h / 2, sfy * c * h / 2) )
        #            = atan2(sfx * s * h / 2, sfy * c * h / 2)
        #            = atan2(sfx * s, sfy * c)
        #
        # For example,
        # when sfx == sfy, angle(new) == atan2(s, c) == angle(old)
        
        # 角度是从图像空间中的y轴到框的高度向量
        # (在框的局部坐标系中从上到下)的逆时针旋转角度。
        #
        # angle(new) = angle_yOx(O - F(new))
        #            = angle_yOx( (sfx * s * h / 2, sfy * c * h / 2) )
        #            = atan2(sfx * s * h / 2, sfy * c * h / 2)
        #            = atan2(sfx * s, sfy * c)
        #
        # 例如，
        # 当sfx == sfy时，angle(new) == atan2(s, c) == angle(old)
        self.tensor[:, 4] = torch.atan2(scale_x * s, scale_y * c) * 180 / math.pi  # 计算缩放后的新角度

    @classmethod
    def cat(cls, boxes_list: List["RotatedBoxes"]) -> "RotatedBoxes":
        """
        Concatenates a list of RotatedBoxes into a single RotatedBoxes

        Arguments:
            boxes_list (list[RotatedBoxes])

        Returns:
            RotatedBoxes: the concatenated RotatedBoxes
            
        将RotatedBoxes列表连接成单个RotatedBoxes

        参数:
            boxes_list (list[RotatedBoxes])：要连接的RotatedBoxes列表

        返回:
            RotatedBoxes：连接后的RotatedBoxes
        """
        assert isinstance(boxes_list, (list, tuple))  # 确保输入是列表或元组
        if len(boxes_list) == 0:
            return cls(torch.empty(0))  # 如果列表为空，返回空的RotatedBoxes
        assert all([isinstance(box, RotatedBoxes) for box in boxes_list])  # 确保所有元素都是RotatedBoxes类型

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        # 使用torch.cat（而不是layers.cat）以确保返回的框永远不会与输入共享存储
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))  # 连接所有框的张量
        return cat_boxes  # 返回连接后的RotatedBoxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device  # 返回张量所在的设备

    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (5,) at a time.
        
        每次产生一个形状为(5,)的框张量。
        """
        yield from self.tensor  # 逐个产生张量中的每一行


def pairwise_iou(boxes1: RotatedBoxes, boxes2: RotatedBoxes) -> torch.Tensor:
    """
    Given two lists of rotated boxes of size N and M,
    compute the IoU (intersection over union)
    between **all** N x M pairs of boxes.
    The box order must be (x_center, y_center, width, height, angle).

    Args:
        boxes1, boxes2 (RotatedBoxes):
            two `RotatedBoxes`. Contains N & M rotated boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
        
    给定两个大小为N和M的旋转框列表，
    计算**所有** N x M对框之间的IoU（交并比）。
    框的顺序必须是(x_center, y_center, width, height, angle)。

    参数:
        boxes1, boxes2 (RotatedBoxes):
            两个`RotatedBoxes`对象。分别包含N和M个旋转框。

    返回:
        Tensor: IoU，大小为[N,M]。
    """

    return pairwise_iou_rotated(boxes1.tensor, boxes2.tensor)  # 调用底层函数计算IoU
