import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor, nn

# 导入必要的模块和函数
from detectron2.data.detection_utils import convert_image_to_rgb  # 用于图像格式转换
from detectron2.modeling import Backbone  # 导入骨干网络基类
from detectron2.structures import Boxes, ImageList, Instances  # 导入检测相关的数据结构
from detectron2.utils.events import get_event_storage  # 用于获取事件存储器

from ..postprocessing import detector_postprocess  # 导入检测后处理函数


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    将张量从(N, (Ai x K), H, W)的形状重排为(N, (HxWxAi), K)的形状
    """
    # 确保输入张量是4维的
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    # 重塑张量维度，引入K维度
    tensor = tensor.view(N, -1, K, H, W)
    # 调整维度顺序
    tensor = tensor.permute(0, 3, 4, 1, 2)
    # 最终重塑为目标形状，其中HWA表示特征图的空间位置和锚点数的乘积
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


class DenseDetector(nn.Module):
    """
    Base class for dense detector. We define a dense detector as a fully-convolutional model that
    makes per-pixel (i.e. dense) predictions.
    密集检测器的基类。我们将密集检测器定义为一个全卷积模型，它能够进行逐像素（即密集）预测。
    """

    def __init__(
        self,
        backbone: Backbone,
        head: nn.Module,
        head_in_features: Optional[List[str]] = None,
        *,
        pixel_mean,
        pixel_std,
    ):
        """
        Args:
            backbone: backbone module
            head: head module
            head_in_features: backbone features to use in head. Default to all backbone features.
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        参数说明：
            backbone: 骨干网络模块，用于提取图像特征
            head: 检测头模块，用于处理骨干网络提取的特征
            head_in_features: 用于检测头的骨干网络特征，默认使用所有骨干网络特征
            pixel_mean: 用于图像归一化的均值（BGR顺序）
                对于不同通道数的图像，可以设置不同的均值和标准差
                默认使用ImageNet的均值：[103.53, 116.28, 123.675]
            pixel_std: 用于图像归一化的标准差
                当使用Detectron1或MSRA的预训练模型时，标准差已被整合到conv1的权重中，此时需要设置为1
                否则可以使用ImageNet的标准差：[57.375, 57.120, 58.395]
        """
        # 调用父类初始化
        super().__init__()

        # 设置骨干网络和检测头
        self.backbone = backbone
        self.head = head
        # 如果未指定head_in_features，则使用所有特征层，并按stride排序
        if head_in_features is None:
            shapes = self.backbone.output_shape()
            self.head_in_features = sorted(shapes.keys(), key=lambda x: shapes[x].stride)
        else:
            self.head_in_features = head_in_features

        # 注册图像归一化参数为模型的buffer
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        
        参数：
            batched_inputs: 一个列表，包含:class:`DatasetMapper`的批处理输出。
                列表中的每个元素包含一张图像的输入数据。
                目前，列表中的每个元素是一个字典，包含：

                * image: 张量，格式为(C, H, W)的图像数据
                * instances: 实例标注数据

                原始字典中还包含其他信息，例如：

                * "height", "width" (int): 模型的输出分辨率，用于推理。
                  详见:meth:`postprocess`。

        返回值：
            在训练时，返回dict[str, Tensor]：从损失名称到存储损失值的张量的映射。
            在推理时，返回标准输出格式，详见:doc:`/tutorials/models`。
        """
        # 预处理输入图像
        images = self.preprocess_image(batched_inputs)
        # 通过骨干网络提取特征
        features = self.backbone(images.tensor)
        # 选择用于检测头的特征
        features = [features[f] for f in self.head_in_features]
        # 通过检测头生成预测结果
        predictions = self.head(features)

        if self.training:
            # 确保不在TorchScript模式下运行
            assert not torch.jit.is_scripting(), "Not supported"
            # 确保训练数据包含实例标注
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            # 获取真实标注数据并转移到对应设备
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # 执行训练时的前向传播
            return self.forward_training(images, features, predictions, gt_instances)
        else:
            # 执行推理时的前向传播
            results = self.forward_inference(images, features, predictions)
            if torch.jit.is_scripting():
                return results

            # 处理每张图像的预测结果
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                # 获取输出图像的高度和宽度
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                # 对预测结果进行后处理
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def forward_training(self, images, features, predictions, gt_instances):
        raise NotImplementedError()

    def preprocess_image(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        对输入图像进行归一化、填充和批处理。
        """
        # 将输入图像转移到指定设备
        images = [x["image"].to(self.device) for x in batched_inputs]
        # 使用预定义的均值和标准差进行图像归一化
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # 将图像转换为ImageList格式，处理不同尺寸的图像
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _transpose_dense_predictions(
        self, predictions: List[List[Tensor]], dims_per_anchor: List[int]
    ) -> List[List[Tensor]]:
        """
        Transpose the dense per-level predictions.
        转置每个层级的密集预测结果。

        Args:
            predictions: a list of outputs, each is a list of per-level
                predictions with shape (N, Ai x K, Hi, Wi), where N is the
                number of images, Ai is the number of anchors per location on
                level i, K is the dimension of predictions per anchor.
            dims_per_anchor: the value of K for each predictions. e.g. 4 for
                box prediction, #classes for classification prediction.

        Returns:
            List[List[Tensor]]: each prediction is transposed to (N, Hi x Wi x Ai, K).
        
        参数：
            predictions: 输出列表，每个元素是每个层级的预测结果列表，
                形状为(N, Ai x K, Hi, Wi)，其中N是图像数量，
                Ai是第i层每个位置的锚点数量，K是每个锚点的预测维度。
            dims_per_anchor: 每个预测的K值，例如边界框预测为4，
                分类预测为类别数量。

        返回：
            List[List[Tensor]]: 每个预测结果被转置为(N, Hi x Wi x Ai, K)形状。
        """
        # 确保预测结果和维度数量匹配
        assert len(predictions) == len(dims_per_anchor)
        res: List[List[Tensor]] = []
        # 对每个预测结果进行维度转置
        for pred, dim_per_anchor in zip(predictions, dims_per_anchor):
            pred = [permute_to_N_HWA_K(x, dim_per_anchor) for x in pred]
            res.append(pred)
        return res

    def _ema_update(self, name: str, value: float, initial_value: float, momentum: float = 0.9):
        """
        Apply EMA update to `self.name` using `value`.
        使用指定的value对self.name进行指数移动平均(EMA)更新。

        This is mainly used for loss normalizer. In Detectron1, loss is normalized by number
        of foreground samples in the batch. When batch size is 1 per GPU, #foreground has a
        large variance and using it lead to lower performance. Therefore we maintain an EMA of
        #foreground to stabilize the normalizer.
        这主要用于损失归一化。在Detectron1中，损失是通过批次中前景样本的数量来归一化的。
        当每个GPU的批次大小为1时，前景样本数量的方差较大，这会导致性能下降。
        因此，我们维护一个前景样本数量的EMA来稳定归一化器。

        Args:
            name: name of the normalizer
                 归一化器的名称
            value: the new value to update
                   需要更新的新值
            initial_value: the initial value to start with
                          初始值
            momentum: momentum of EMA
                      EMA的动量参数

        Returns:
            float: the updated EMA value
                  更新后的EMA值
        """
        # 检查是否已存在该属性
        if hasattr(self, name):
            old = getattr(self, name)
        else:
            old = initial_value
        # 计算EMA更新值：新值 = 动量 * 旧值 + (1-动量) * 当前值
        new = old * momentum + value * (1 - momentum)
        # 更新属性值
        setattr(self, name, new)
        return new

    def _decode_per_level_predictions(
        self,
        anchors: Boxes,
        pred_scores: Tensor,
        pred_deltas: Tensor,
        score_thresh: float,
        topk_candidates: int,
        image_size: Tuple[int, int],
    ) -> Instances:
        """
        Decode boxes and classification predictions of one featuer level, by
        the following steps:
        1. filter the predictions based on score threshold and top K scores.
        2. transform the box regression outputs
        3. return the predicted scores, classes and boxes
        解码单个特征层的边界框和分类预测结果，步骤如下：
        1. 基于分数阈值和top K分数过滤预测结果
        2. 转换边界框回归输出
        3. 返回预测的分数、类别和边界框

        Args:
            anchors: Boxes, anchor for this feature level
                    当前特征层的锚框
            pred_scores: HxWxA,K
                        预测分数，形状为HxWxA,K
            pred_deltas: HxWxA,4
                        预测的边界框回归值，形状为HxWxA,4

        Returns:
            Instances: with field "scores", "pred_boxes", "pred_classes".
                      包含"scores"、"pred_boxes"、"pred_classes"字段的实例对象
        """
        # Apply two filtering to make NMS faster.
        # 1. Keep boxes with confidence score higher than threshold
        # 应用两次过滤来加速NMS
        # 1. 保留置信度分数高于阈值的边界框
        keep_idxs = pred_scores > score_thresh
        pred_scores = pred_scores[keep_idxs]
        topk_idxs = torch.nonzero(keep_idxs)  # Kx2

        # 2. Keep top k top scoring boxes only
        # 2. 仅保留得分最高的前k个边界框
        num_topk = min(topk_candidates, topk_idxs.size(0))
        # torch.sort is actually faster than .topk (https://github.com/pytorch/pytorch/issues/22812)
        # 使用sort比topk更快
        pred_scores, idxs = pred_scores.sort(descending=True)
        pred_scores = pred_scores[:num_topk]
        topk_idxs = topk_idxs[idxs[:num_topk]]

        # 解包索引，获取锚框索引和类别索引
        anchor_idxs, classes_idxs = topk_idxs.unbind(dim=1)

        # 应用边界框变换，将预测的回归值转换为实际的边界框坐标
        pred_boxes = self.box2box_transform.apply_deltas(
            pred_deltas[anchor_idxs], anchors.tensor[anchor_idxs]
        )
        # 创建并返回包含预测结果的实例对象
        return Instances(
            image_size, pred_boxes=Boxes(pred_boxes), scores=pred_scores, pred_classes=classes_idxs
        )

    def _decode_multi_level_predictions(
        self,
        anchors: List[Boxes],
        pred_scores: List[Tensor],
        pred_deltas: List[Tensor],
        score_thresh: float,
        topk_candidates: int,
        image_size: Tuple[int, int],
    ) -> Instances:
        """
        Run `_decode_per_level_predictions` for all feature levels and concat the results.
        对所有特征层运行`_decode_per_level_predictions`并连接结果。
        """
        # 对每个特征层的预测结果进行解码
        predictions = [
            self._decode_per_level_predictions(
                anchors_i,
                box_cls_i,
                box_reg_i,
                self.test_score_thresh,
                self.test_topk_candidates,
                image_size,
            )
            # Iterate over every feature level
            # 遍历每个特征层
            for box_cls_i, box_reg_i, anchors_i in zip(pred_scores, pred_deltas, anchors)
        ]
        # 连接所有特征层的预测结果
        return predictions[0].cat(predictions)  # 'Instances.cat' is not scriptale but this is

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.
        用于可视化真实图像和最终网络预测结果的函数。
        在原始图像上显示真实边界框和最多20个预测的目标边界框。

        Args:
            batched_inputs (list): a list that contains input to the model.
                                  包含模型输入的列表
            results (List[Instances]): a list of #images elements returned by forward_inference().
                                      forward_inference()返回的图像实例列表
        """
        # 导入可视化工具
        from detectron2.utils.visualizer import Visualizer

        # 确保输入和结果的数量匹配
        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        # 仅可视化单张图像
        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        # 转换图像格式为RGB
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        # 创建可视化器并绘制真实边界框
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        # 处理预测结果
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        # 创建可视化器并绘制预测边界框
        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        # 垂直堆叠真实框和预测框的可视化结果
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        # 设置可视化图像的名称并保存
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)
