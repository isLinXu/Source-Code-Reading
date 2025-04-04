import math
import torch


def diou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Distance Intersection over Union Loss (Zhaohui Zheng et. al)
    距离交并比损失函数 (作者：Zhaohui Zheng等人)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        # boxes1, boxes2 (Tensor): 边界框坐标，采用XYXY格式，形状为(N, 4)或(4,)。
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 # 'none': 不对输出进行任何降维处理。
                 'mean': The output will be averaged.
                 # 'mean': 对输出取平均值。
                 'sum': The output will be summed.
                 # 'sum': 对输出求和。
        eps (float): small number to prevent division by zero
        # eps (float): 防止除以零的小数值
    """

    # 将边界框拆分为x1, y1, x2, y2坐标
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    # 确保边界框格式正确：x2应大于x1，y2应大于y1
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    # 计算交集区域的关键点
    xkis1 = torch.max(x1, x1g)  # 交集左上角x坐标
    ykis1 = torch.max(y1, y1g)  # 交集左上角y坐标
    xkis2 = torch.min(x2, x2g)  # 交集右下角x坐标
    ykis2 = torch.min(y2, y2g)  # 交集右下角y坐标

    # 计算交集面积
    intsct = torch.zeros_like(x1)  # 初始化交集面积为0
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)  # 创建有效交集的掩码
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])  # 计算交集面积
    # 计算并集面积 = 第一个框面积 + 第二个框面积 - 交集面积 + eps
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union  # 计算IoU

    # smallest enclosing box
    # 计算最小包围框
    xc1 = torch.min(x1, x1g)  # 包围框左上角x坐标
    yc1 = torch.min(y1, y1g)  # 包围框左上角y坐标
    xc2 = torch.max(x2, x2g)  # 包围框右下角x坐标
    yc2 = torch.max(y2, y2g)  # 包围框右下角y坐标
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps  # 包围框对角线长度的平方

    # centers of boxes
    # 计算两个框的中心点
    x_p = (x2 + x1) / 2  # 第一个框的中心x坐标
    y_p = (y2 + y1) / 2  # 第一个框的中心y坐标
    x_g = (x1g + x2g) / 2  # 第二个框的中心x坐标
    y_g = (y1g + y2g) / 2  # 第二个框的中心y坐标
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)  # 中心点距离的平方

    # Eqn. (7)
    # 公式(7)：DIoU损失计算
    loss = 1 - iou + (distance / diag_len)  # DIoU损失 = 1 - IoU + 中心点距离/对角线长度
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()  # 对损失取平均
    elif reduction == "sum":
        loss = loss.sum()  # 对损失求和

    return loss


def ciou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Complete Intersection over Union Loss (Zhaohui Zheng et. al)
    完整交并比损失函数 (作者：Zhaohui Zheng等人)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        # boxes1, boxes2 (Tensor): 边界框坐标，采用XYXY格式，形状为(N, 4)或(4,)。
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 # 'none': 不对输出进行任何降维处理。
                 'mean': The output will be averaged.
                 # 'mean': 对输出取平均值。
                 'sum': The output will be summed.
                 # 'sum': 对输出求和。
        eps (float): small number to prevent division by zero
        # eps (float): 防止除以零的小数值
    """

    # 将边界框拆分为x1, y1, x2, y2坐标
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    # 确保边界框格式正确：x2应大于x1，y2应大于y1
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    # 计算交集区域的关键点
    xkis1 = torch.max(x1, x1g)  # 交集左上角x坐标
    ykis1 = torch.max(y1, y1g)  # 交集左上角y坐标
    xkis2 = torch.min(x2, x2g)  # 交集右下角x坐标
    ykis2 = torch.min(y2, y2g)  # 交集右下角y坐标

    # 计算交集面积
    intsct = torch.zeros_like(x1)  # 初始化交集面积为0
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)  # 创建有效交集的掩码
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])  # 计算交集面积
    # 计算并集面积 = 第一个框面积 + 第二个框面积 - 交集面积 + eps
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union  # 计算IoU

    # smallest enclosing box
    # 计算最小包围框
    xc1 = torch.min(x1, x1g)  # 包围框左上角x坐标
    yc1 = torch.min(y1, y1g)  # 包围框左上角y坐标
    xc2 = torch.max(x2, x2g)  # 包围框右下角x坐标
    yc2 = torch.max(y2, y2g)  # 包围框右下角y坐标
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps  # 包围框对角线长度的平方

    # centers of boxes
    # 计算两个框的中心点
    x_p = (x2 + x1) / 2  # 第一个框的中心x坐标
    y_p = (y2 + y1) / 2  # 第一个框的中心y坐标
    x_g = (x1g + x2g) / 2  # 第二个框的中心x坐标
    y_g = (y1g + y2g) / 2  # 第二个框的中心y坐标
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)  # 中心点距离的平方

    # width and height of boxes
    # 计算两个框的宽度和高度
    w_pred = x2 - x1  # 第一个框的宽度
    h_pred = y2 - y1  # 第一个框的高度
    w_gt = x2g - x1g  # 第二个框的宽度
    h_gt = y2g - y1g  # 第二个框的高度
    # 计算纵横比一致性项v：使用框的宽高比差异的反正切值
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)  # 计算平衡参数alpha，用梯度停止来平衡项的权重

    # Eqn. (10)
    # 公式(10)：CIoU损失计算
    loss = 1 - iou + (distance / diag_len) + alpha * v  # CIoU损失 = 1 - IoU + 中心点距离项 + 纵横比一致性项
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()  # 对损失取平均
    elif reduction == "sum":
        loss = loss.sum()  # 对损失求和

    return loss
