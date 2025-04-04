# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch

from detectron2.layers import nonzero_tuple


# TODO: the name is too general
class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    这个类将每个预测的"元素

    def __init__(
        self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False
    ):
        
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
                用于将预测分层的阈值列表。
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
                用于标记每个层级预测的值列表。标签可以是{-1, 0, 1}中的一个，分别表示
                {忽略, 负类, 正类}。
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.
                如果为True，会为最大匹配质量低于高阈值的预测生成额外的匹配。
                详见set_low_quality_matches_方法。

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
                例如：
                当阈值=[0.3, 0.5]，标签=[0, -1, 1]时：
                所有IoU < 0.3的预测将被标记为0，在训练时被视为假阳性。
                所有0.3 <= IoU < 0.5的预测将被标记为-1，因此被忽略。
                所有0.5 <= IoU的预测将被标记为1，在训练时被视为真阳性。
        """
        # Add -inf and +inf to first and last position in thresholds
        # 在阈值列表的首尾位置添加负无穷和正无穷
        thresholds = thresholds[:]
        assert thresholds[0] > 0  # 确保第一个阈值大于0
        thresholds.insert(0, -float("inf"))  # 插入负无穷作为最小阈值
        thresholds.append(float("inf"))  # 添加正无穷作为最大阈值
        # Currently torchscript does not support all + generator
        # 目前torchscript不支持all+生成器语法
        assert all([low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])])  # 确保阈值递增
        assert all([l in [-1, 0, 1] for l in labels])  # 确保标签值在{-1,0,1}中
        assert len(labels) == len(thresholds) - 1  # 确保标签数量正确
        self.thresholds = thresholds  # 存储阈值列表
        self.labels = labels  # 存储标签列表
        self.allow_low_quality_matches = allow_low_quality_matches  # 存储是否允许低质量匹配的标志

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).
                一个MxN的张量，包含M个真实标注元素和N个预测元素之间的成对质量值。
                所有元素必须 >= 0（因为在set_low_quality_matches_方法中使用了torch.nonzero）。

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
                长度为N的向量，其中matches[i]是匹配的真实标注索引，范围在[0, M)内
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
                长度为N的向量，其中pred_labels[i]表示预测是真阳性、假阳性还是被忽略
        """
        assert match_quality_matrix.dim() == 2  # 确保输入矩阵是2维的
        if match_quality_matrix.numel() == 0:  # 如果矩阵为空（没有元素）
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )  # 创建一个全0的默认匹配索引向量
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            # 当没有真实标注框时，我们定义IoU = 0，因此将标签设置为self.labels[0]（通常默认为背景类0）
            # 如果想选择忽略这些预测，可以设置labels=[-1,0,-1,1]并设置适当的阈值
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )  # 创建一个默认标签向量，值为第一个标签值
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)  # 确保所有质量值都非负

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # 匹配质量矩阵维度为 M（真实标注数）x N（预测数）
        # 在真实标注维度上取最大值，为每个预测找到最佳的真实标注候选
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)  # 初始化标签全为1

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)  # 找出质量值在当前阈值范围内的预测
            match_labels[low_high] = l  # 为这些预测分配对应的标签

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.
        为仅有低质量匹配的预测生成额外的匹配。具体来说，对于每个真实标注框G，找到与其具有最大重叠
        （包括平局）的预测集合；对于该集合中的每个预测，如果它未匹配，则将其匹配到真实标注框G。

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        此函数实现了Faster R-CNN论文3.1.2节中的RPN分配情况(i)。
        """
        # For each gt, find the prediction with which it has highest quality
        # 对于每个真实标注框，找到与其具有最高质量的预测
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        # 找到可用的最高质量匹配，即使质量很低，也包括平局情况。
        # 注意：由于使用了torch.nonzero，匹配质量必须为正。
        _, pred_inds_with_highest_quality = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
        # This follows the implementation in Detectron, and is found to have no significant impact.
        # 如果一个锚框仅因为与真实框A的低质量匹配而被标记为正样本，
        # 但它与真实框B有更大的重叠，它的匹配索引仍将是真实框B。
        # 这遵循Detectron的实现，并且发现没有显著影响。
        match_labels[pred_inds_with_highest_quality] = 1
