# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import thop
import torch

from ultralytics.nn.modules import (  # 从 ultralytics.nn.modules 导入各种模块
    AIFI,  # AIFI 模块
    C1,  # C1 模块
    C2,  # C2 模块
    C2PSA,  # C2PSA 模块
    C3,  # C3 模块
    C3TR,  # C3TR 模块
    ELAN1,  # ELAN1 模块
    OBB,  # OBB 模块
    PSA,  # PSA 模块
    SPP,  # SPP 模块
    SPPELAN,  # SPPELAN 模块
    SPPF,  # SPPF 模块
    AConv,  # AConv 模块
    ADown,  # ADown 模块
    Bottleneck,  # Bottleneck 模块
    BottleneckCSP,  # BottleneckCSP 模块
    C2f,  # C2f 模块
    C2fAttn,  # C2fAttn 模块
    C2fCIB,  # C2fCIB 模块
    C2fPSA,  # C2fPSA 模块
    C3Ghost,  # C3Ghost 模块
    C3k2,  # C3k2 模块
    C3x,  # C3x 模块
    CBFuse,  # CBFuse 模块
    CBLinear,  # CBLinear 模块
    Classify,  # Classify 模块
    Concat,  # Concat 模块
    Conv,  # Conv 模块
    Conv2,  # Conv2 模块
    ConvTranspose,  # ConvTranspose 模块
    Detect,  # Detect 模块
    DWConv,  # DWConv 模块
    DWConvTranspose2d,  # DWConvTranspose2d 模块
    Focus,  # Focus 模块
    GhostBottleneck,  # GhostBottleneck 模块
    GhostConv,  # GhostConv 模块
    HGBlock,  # HGBlock 模块
    HGStem,  # HGStem 模块
    ImagePoolingAttn,  # ImagePoolingAttn 模块
    Index,  # Index 模块
    Pose,  # Pose 模块
    RepC3,  # RepC3 模块
    RepConv,  # RepConv 模块
    RepNCSPELAN4,  # RepNCSPELAN4 模块
    RepVGGDW,  # RepVGGDW 模块
    ResNetLayer,  # ResNetLayer 模块
    RTDETRDecoder,  # RTDETRDecoder 模块
    SCDown,  # SCDown 模块
    Segment,  # Segment 模块
    TorchVision,  # TorchVision 模块
    WorldDetect,  # WorldDetect 模块
    v10Detect,  # v10Detect 模块
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load  # 从 ultralytics.utils 导入工具
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml  # 从 ultralytics.utils.checks 导入检查工具
from ultralytics.utils.loss import (  # 从 ultralytics.utils.loss 导入各种损失函数
    E2EDetectLoss,  # E2EDetectLoss 损失
    v8ClassificationLoss,  # v8ClassificationLoss 损失
    v8DetectionLoss,  # v8DetectionLoss 损失
    v8OBBLoss,  # v8OBBLoss 损失
    v8PoseLoss,  # v8PoseLoss 损失
    v8SegmentationLoss,  # v8SegmentationLoss 损失
)
from ultralytics.utils.ops import make_divisible  # 从 ultralytics.utils.ops 导入 make_divisible 函数
from ultralytics.utils.plotting import feature_visualization  # 从 ultralytics.utils.plotting 导入特征可视化函数
from ultralytics.utils.torch_utils import (  # 从 ultralytics.utils.torch_utils 导入各种 PyTorch 工具
    fuse_conv_and_bn,  # 融合 Conv 和 BatchNorm 层
    fuse_deconv_and_bn,  # 融合反卷积和 BatchNorm 层
    initialize_weights,  # 初始化权重
    intersect_dicts,  # 交集字典
    model_info,  # 模型信息
    scale_img,  # 缩放图像
    time_sync,  # 时间同步
)


class BaseModel(torch.nn.Module):  # BaseModel 类继承自 torch.nn.Module
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""  # BaseModel 类作为 Ultralytics YOLO 系列模型的基类

    def forward(self, x, *args, **kwargs):  # 定义前向传播方法
        """
        Perform forward pass of the model for either training or inference.  # 执行模型的前向传播，用于训练或推理。

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.  # 如果 x 是字典，则计算并返回训练损失；否则返回推理预测。

        Args:  # 参数：
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.  # x（torch.Tensor | dict）：用于推理的输入张量，或包含图像张量和标签的字典。
            *args (Any): Variable length argument list.  # *args（任意类型）：可变长度参数列表。
            **kwargs (Any): Arbitrary keyword arguments.  # **kwargs（任意类型）：任意关键字参数。

        Returns:  # 返回：
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).  # （torch.Tensor）：如果 x 是字典（训练），则返回损失；否则返回网络预测（推理）。
        """
        if isinstance(x, dict):  # for cases of training and validating while training.  # 用于训练和验证的情况。
            return self.loss(x, *args, **kwargs)  # 计算损失
        return self.predict(x, *args, **kwargs)  # 进行推理预测

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):  # 定义预测方法
        """
        Perform a forward pass through the network.  # 在网络中执行前向传播。

        Args:  # 参数：
            x (torch.Tensor): The input tensor to the model.  # x（torch.Tensor）：模型的输入张量。
            profile (bool):  Print the computation time of each layer if True, defaults to False.  # profile（bool）：如果为 True，则打印每层的计算时间，默认为 False。
            visualize (bool): Save the feature maps of the model if True, defaults to False.  # visualize（bool）：如果为 True，则保存模型的特征图，默认为 False。
            augment (bool): Augment image during prediction, defaults to False.  # augment（bool）：在预测期间增强图像，默认为 False。
            embed (list, optional): A list of feature vectors/embeddings to return.  # embed（列表，可选）：要返回的特征向量/嵌入列表。

        Returns:  # 返回：
            (torch.Tensor): The last output of the model.  # （torch.Tensor）：模型的最后输出。
        """
        if augment:  # 如果启用了增强
            return self._predict_augment(x)  # 执行增强预测
        return self._predict_once(x, profile, visualize, embed)  # 执行一次预测

    def _predict_once(self, x, profile=False, visualize=False, embed=None):  # 定义一次预测的方法
        """
        Perform a forward pass through the network.  # 在网络中执行前向传播。

        Args:  # 参数：
            x (torch.Tensor): The input tensor to the model.  # x（torch.Tensor）：模型的输入张量。
            profile (bool):  Print the computation time of each layer if True, defaults to False.  # profile（bool）：如果为 True，则打印每层的计算时间，默认为 False。
            visualize (bool): Save the feature maps of the model if True, defaults to False.  # visualize（bool）：如果为 True，则保存模型的特征图，默认为 False。
            embed (list, optional): A list of feature vectors/embeddings to return.  # embed（列表，可选）：要返回的特征向量/嵌入列表。

        Returns:  # 返回：
            (torch.Tensor): The last output of the model.  # （torch.Tensor）：模型的最后输出。
        """
        y, dt, embeddings = [], [], []  # outputs  # 初始化输出列表
        for m in self.model:  # 遍历模型中的每一层
            if m.f != -1:  # if not from previous layer  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers  # 从早期层获取输入
            if profile:  # 如果启用了性能分析
                self._profile_one_layer(m, x, dt)  # 分析当前层的性能
            x = m(x)  # run  # 执行当前层的前向传播
            y.append(x if m.i in self.save else None)  # save output  # 如果当前层需要保存输出，则保存
            if visualize:  # 如果启用了可视化
                feature_visualization(x, m.type, m.i, save_dir=visualize)  # 保存特征图
            if embed and m.i in embed:  # 如果需要嵌入并且当前层在嵌入列表中
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten  # 执行自适应平均池化并展平
                if m.i == max(embed):  # 如果当前层是嵌入列表中的最大层
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)  # 返回嵌入
        return x  # 返回最后的输出

    def _predict_augment(self, x):  # 定义增强预测的方法
        """Perform augmentations on input image x and return augmented inference."""  # 对输入图像 x 执行增强并返回增强推理。
        LOGGER.warning(  # 记录警告信息
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "  # "{self.__class__.__name__} 不支持 'augment=True' 预测。"
            f"Reverting to single-scale prediction."  # "恢复为单尺度预测。"
        )
        return self._predict_once(x)  # 执行一次预测

    def _profile_one_layer(self, m, x, dt):  # 定义单层性能分析的方法
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:  # 参数：
            m (torch.nn.Module): The layer to be profiled.  # m（torch.nn.Module）：要分析的层。
            x (torch.Tensor): The input data to the layer.  # x（torch.Tensor）：层的输入数据。
            dt (list): A list to store the computation time of the layer.  # dt（列表）：用于存储层计算时间的列表。
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix  # 如果是最后一层列表，复制输入以进行就地修复
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs  # 计算当前层的 FLOPs
        t = time_sync()  # 获取当前时间
        for _ in range(10):  # 执行 10 次前向传播以测量时间
            m(x.copy() if c else x)  # run  # 执行当前层的前向传播
        dt.append((time_sync() - t) * 100)  # 计算并存储当前层的计算时间
        if m == self.model[0]:  # 如果是第一层
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")  # 打印表头
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")  # 打印当前层的计算时间、FLOPs 和参数数量
        if c:  # 如果是最后一层
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")  # 打印总计算时间

    def fuse(self, verbose=True):  # 定义融合方法
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:  # 返回：
            (torch.nn.Module): The fused model is returned.  # （torch.nn.Module）：返回融合后的模型。
        """
        if not self.is_fused():  # 如果模型尚未融合
            for m in self.model.modules():  # 遍历模型中的每个模块
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):  # 如果是卷积层且有 BatchNorm
                    if isinstance(m, Conv2):  # 如果是 Conv2 层
                        m.fuse_convs()  # 融合卷积
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv  # 更新卷积层
                    delattr(m, "bn")  # remove batchnorm  # 删除 BatchNorm 属性
                    m.forward = m.forward_fuse  # update forward  # 更新前向传播方法
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):  # 如果是反卷积层且有 BatchNorm
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)  # 融合反卷积和 BatchNorm
                    delattr(m, "bn")  # remove batchnorm  # 删除 BatchNorm 属性
                    m.forward = m.forward_fuse  # update forward  # 更新前向传播方法
                if isinstance(m, RepConv):  # 如果是 RepConv 层
                    m.fuse_convs()  # 融合卷积
                    m.forward = m.forward_fuse  # update forward  # 更新前向传播方法
                if isinstance(m, RepVGGDW):  # 如果是 RepVGGDW 层
                    m.fuse()  # 融合
                    m.forward = m.forward_fuse  # 更新前向传播方法
            self.info(verbose=verbose)  # 打印模型信息

        return self  # 返回当前模型

    def is_fused(self, thresh=10):  # 定义检查是否融合的方法
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:  # 参数：
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.  # thresh（整数，可选）：BatchNorm 层的阈值数量。默认为 10。

        Returns:  # 返回：
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.  # （布尔值）：如果模型中的 BatchNorm 层数量少于阈值，则返回 True，否则返回 False。
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()  # 获取归一化层，例如 BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model  # 如果模型中的 BatchNorm 层数量少于阈值，则返回 True

    def info(self, detailed=False, verbose=True, imgsz=640):  # 定义打印模型信息的方法
        """
        Prints model information.

        Args:  # 参数：
            detailed (bool): if True, prints out detailed information about the model. Defaults to False  # detailed（布尔值）：如果为 True，则打印模型的详细信息。默认为 False。
            verbose (bool): if True, prints out the model information. Defaults to False  # verbose（布尔值）：如果为 True，则打印模型信息。默认为 False。
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640  # imgsz（整数）：模型将训练的图像大小。默认为 640。
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)  # 返回模型信息

    def _apply(self, fn):  # 定义应用函数的方法
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:  # 参数：
            fn (function): the function to apply to the model  # fn（函数）：要应用于模型的函数

        Returns:  # 返回：
            (BaseModel): An updated BaseModel object.  # （BaseModel）：更新后的 BaseModel 对象。
        """
        self = super()._apply(fn)  # 调用父类的 _apply 方法
        m = self.model[-1]  # Detect()  # 获取最后一层
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect  # 包含所有 Detect 子类，如 Segment、Pose、OBB、WorldDetect
            m.stride = fn(m.stride)  # 应用函数到步幅
            m.anchors = fn(m.anchors)  # 应用函数到锚点
            m.strides = fn(m.strides)  # 应用函数到步幅列表
        return self  # 返回当前模型

    def load(self, weights, verbose=True):  # 定义加载权重的方法
        """
        Load the weights into the model.

        Args:  # 参数：
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.  # weights（字典 | torch.nn.Module）：要加载的预训练权重。
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.  # verbose（布尔值，可选）：是否记录转移进度。默认为 True。
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts  # 如果是字典，则获取模型；否则直接使用 weights
        csd = model.float().state_dict()  # checkpoint state_dict as FP32  # 将检查点的 state_dict 转换为 FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect  # 取交集
        self.load_state_dict(csd, strict=False)  # load  # 加载状态字典
        if verbose:  # 如果启用了详细模式
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")  # 记录转移进度

    def loss(self, batch, preds=None):  # 定义计算损失的方法
        """
        Compute loss.

        Args:  # 参数：
            batch (dict): Batch to compute loss on  # batch（字典）：要计算损失的批次
            preds (torch.Tensor | List[torch.Tensor]): Predictions.  # preds（torch.Tensor | List[torch.Tensor]）：预测结果。
        """
        if getattr(self, "criterion", None) is None:  # 如果 criterion 尚未定义
            self.criterion = self.init_criterion()  # 初始化损失标准

        preds = self.forward(batch["img"]) if preds is None else preds  # 如果 preds 为空，则通过前向传播计算预测
        return self.criterion(preds, batch)  # 计算并返回损失

    def init_criterion(self):  # 定义初始化损失标准的方法
        """Initialize the loss criterion for the BaseModel."""  # 初始化 BaseModel 的损失标准。
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")  # 抛出未实现错误，提示需要在任务头中实现 compute_loss()

class DetectionModel(BaseModel):  # DetectionModel 类继承自 BaseModel
    """YOLO detection model."""  # YOLO 检测模型。

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes  # 初始化 YOLO 检测模型，给定配置和参数
        """Initialize the YOLO detection model with the given config and parameters."""  # 使用给定的配置和参数初始化 YOLO 检测模型。
        super().__init__()  # 调用父类构造函数
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict  # 如果 cfg 是字典，则直接使用；否则加载 YAML 配置
        if self.yaml["backbone"][0][2] == "Silence":  # 如果骨干网络使用 "Silence" 模块
            LOGGER.warning(  # 记录警告信息
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "  # "YOLOv9 `Silence` 模块已弃用，建议使用 torch.nn.Identity。"
                "Please delete local *.pt file and re-download the latest model checkpoint."  # "请删除本地 *.pt 文件并重新下载最新的模型检查点。"
            )
            self.yaml["backbone"][0][2] = "nn.Identity"  # 将 "Silence" 替换为 "nn.Identity"

        # Define model  # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels  # 获取输入通道
        if nc and nc != self.yaml["nc"]:  # 如果提供了类别数量并且与 YAML 中的不同
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")  # 记录信息，覆盖类别数量
            self.yaml["nc"] = nc  # override YAML value  # 覆盖 YAML 中的类别数量
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist  # 解析模型并保存
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict  # 创建默认类名字典
        self.inplace = self.yaml.get("inplace", True)  # 获取是否就地操作的标志
        self.end2end = getattr(self.model[-1], "end2end", False)  # 获取是否为端到端模型的标志

        # Build strides  # 构建步幅
        m = self.model[-1]  # Detect()  # 获取最后一层
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect  # 包含所有 Detect 子类，如 Segment、Pose、OBB、WorldDetect
            s = 256  # 2x min stride  # 设置最小步幅
            m.inplace = self.inplace  # 设置是否就地操作

            def _forward(x):  # 定义前向传播的方法
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""  # 在模型中执行前向传播，处理不同的 Detect 子类类型。
                if self.end2end:  # 如果是端到端模型
                    return self.forward(x)["one2many"]  # 返回一对多的输出
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)  # 根据模型类型返回输出

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward  # 计算步幅
            self.stride = m.stride  # 设置步幅
            m.bias_init()  # only run once  # 仅运行一次初始化偏置
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR  # 默认步幅

        # Init weights, biases  # 初始化权重和偏置
        initialize_weights(self)  # 初始化权重
        if verbose:  # 如果启用了详细模式
            self.info()  # 打印模型信息
            LOGGER.info("")  # 打印空行

    def _predict_augment(self, x):  # 定义增强预测的方法
        """Perform augmentations on input image x and return augmented inference and train outputs."""  # 对输入图像 x 执行增强并返回增强推理和训练输出。
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":  # 如果是端到端模型或不是 DetectionModel
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")  # 记录警告信息
            return self._predict_once(x)  # 执行一次预测
        img_size = x.shape[-2:]  # height, width  # 获取图像的高度和宽度
        s = [1, 0.83, 0.67]  # scales  # 定义缩放比例
        f = [None, 3, None]  # flips (2-ud, 3-lr)  # 定义翻转方式
        y = []  # outputs  # 初始化输出列表
        for si, fi in zip(s, f):  # 遍历缩放比例和翻转方式
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))  # 缩放图像
            yi = super().predict(xi)[0]  # forward  # 执行前向预测
            yi = self._descale_pred(yi, fi, si, img_size)  # 反缩放预测结果
            y.append(yi)  # 添加输出
        y = self._clip_augmented(y)  # clip augmented tails  # 裁剪增强输出
        return torch.cat(y, -1), None  # augmented inference, train  # 返回增强推理和训练输出

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):  # 定义反缩放预测的方法
        """De-scale predictions following augmented inference (inverse operation)."""  # 执行增强推理后的反缩放预测（逆操作）。
        p[:, :4] /= scale  # de-scale  # 反缩放
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)  # 拆分预测结果
        if flips == 2:  # 如果是上下翻转
            y = img_size[0] - y  # de-flip ud  # 反转 y 坐标
        elif flips == 3:  # 如果是左右翻转
            x = img_size[1] - x  # de-flip lr  # 反转 x 坐标
        return torch.cat((x, y, wh, cls), dim)  # 返回拼接后的预测结果

    def _clip_augmented(self, y):  # 定义裁剪增强输出的方法
        """Clip YOLO augmented inference tails."""  # 裁剪 YOLO 增强推理的尾部。
        nl = self.model[-1].nl  # number of detection layers (P3-P5)  # 检测层的数量
        g = sum(4**x for x in range(nl))  # grid points  # 计算网格点
        e = 1  # exclude layer count  # 排除层计数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices  # 计算索引
        y[0] = y[0][..., :-i]  # large  # 裁剪大输出
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices  # 计算索引
        y[-1] = y[-1][..., i:]  # small  # 裁剪小输出
        return y  # 返回裁剪后的输出

    def init_criterion(self):  # 定义初始化损失标准的方法
        """Initialize the loss criterion for the DetectionModel."""  # 初始化 DetectionModel 的损失标准。
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)  # 返回相应的损失标准


class OBBModel(DetectionModel):  # OBBModel 类继承自 DetectionModel
    """YOLO Oriented Bounding Box (OBB) model."""  # YOLO 定向边界框（OBB）模型。

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):  # 初始化 OBB 模型
        """Initialize YOLO OBB model with given config and parameters."""  # 使用给定的配置和参数初始化 YOLO OBB 模型。
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # 调用父类构造函数

    def init_criterion(self):  # 定义初始化损失标准的方法
        """Initialize the loss criterion for the model."""  # 初始化模型的损失标准。
        return v8OBBLoss(self)  # 返回 OBB 损失标准


class SegmentationModel(DetectionModel):  # SegmentationModel 类继承自 DetectionModel
    """YOLO segmentation model."""  # YOLO 分割模型。

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):  # 初始化分割模型
        """Initialize YOLOv8 segmentation model with given config and parameters."""  # 使用给定的配置和参数初始化 YOLOv8 分割模型。
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # 调用父类构造函数

    def init_criterion(self):  # 定义初始化损失标准的方法
        """Initialize the loss criterion for the SegmentationModel."""  # 初始化 SegmentationModel 的损失标准。
        return v8SegmentationLoss(self)  # 返回分割损失标准


class PoseModel(DetectionModel):  # PoseModel 类继承自 DetectionModel
    """YOLO pose model."""  # YOLO 姿态模型。

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):  # 初始化姿态模型
        """Initialize YOLOv8 Pose model."""  # 初始化 YOLOv8 姿态模型。
        if not isinstance(cfg, dict):  # 如果 cfg 不是字典
            cfg = yaml_model_load(cfg)  # load model YAML  # 加载模型 YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):  # 如果提供了关键点形状并且与配置中的不同
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")  # 记录信息，覆盖关键点形状
            cfg["kpt_shape"] = data_kpt_shape  # 更新配置中的关键点形状
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)  # 调用父类构造函数

    def init_criterion(self):  # 定义初始化损失标准的方法
        """Initialize the loss criterion for the PoseModel."""  # 初始化 PoseModel 的损失标准。
        return v8PoseLoss(self)  # 返回姿态损失标准


class ClassificationModel(BaseModel):  # ClassificationModel 类继承自 BaseModel
    """YOLO classification model."""  # YOLO 分类模型。

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):  # 初始化分类模型
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""  # 使用 YAML、通道、类别数量和详细标志初始化 ClassificationModel。
        super().__init__()  # 调用父类构造函数
        self._from_yaml(cfg, ch, nc, verbose)  # 从 YAML 加载模型配置

    def _from_yaml(self, cfg, ch, nc, verbose):  # 定义从 YAML 加载配置的方法
       


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.
    RTDETR（实时检测和跟踪使用变压器）检测模型类。

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.
    此类负责构建RTDETR架构，定义损失函数，并促进训练和推理过程。RTDETR是一个对象检测和跟踪模型，扩展自DetectionModel基类。

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        init_criterion: 初始化用于损失计算的标准。
        loss: Computes and returns the loss during training.
        loss: 计算并返回训练期间的损失。
        predict: Performs a forward pass through the network and returns the output.
        predict: 通过网络执行前向传递并返回输出。
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.
        初始化RTDETRDetectionModel。

        Args:
            cfg (str): Configuration file name or path.
            cfg (str): 配置文件名称或路径。
            ch (int): Number of input channels.
            ch (int): 输入通道的数量。
            nc (int, optional): Number of classes. Defaults to None.
            nc (int, optional): 类别数量，默认为None。
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
            verbose (bool, optional): 在初始化期间打印额外信息，默认为True。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel.
        初始化RTDETRDetectionModel的损失标准。"""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.
        计算给定数据批次的损失。

        Args:
            batch (dict): Dictionary containing image and label data.
            batch (dict): 包含图像和标签数据的字典。
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.
            preds (torch.Tensor, optional): 预计算的模型预测，默认为None。

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
            (tuple): 一个元组，包含总损失和主要三个损失的张量。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        # 注意：将gt_bbox和gt_labels预处理为列表。
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        # 注意：RTDETR中有大约12个损失，反向传播时使用所有损失，但只显示主要的三个损失。
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.
        通过模型执行前向传递。

        Args:
            x (torch.Tensor): The input tensor.
            x (torch.Tensor): 输入张量。
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            profile (bool, optional): 如果为True，分析每层的计算时间，默认为False。
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            visualize (bool, optional): 如果为True，保存特征图以供可视化，默认为False。
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            batch (dict, optional): 用于评估的真实数据，默认为None。
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            augment (bool, optional): 如果为True，在推理期间执行数据增强，默认为False。
            embed (list, optional): A list of feature vectors/embeddings to return.
            embed (list, optional): 要返回的特征向量/嵌入的列表。

        Returns:
            (torch.Tensor): Model's output tensor.
            (torch.Tensor): 模型的输出张量。
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model.
    YOLOv8世界模型。"""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters.
        使用给定的配置和参数初始化YOLOv8世界模型。"""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.
        预先设置类，以便模型可以在没有CLIP模型的情况下进行离线推理。"""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.
        通过模型执行前向传递。

        Args:
            x (torch.Tensor): The input tensor.
            x (torch.Tensor): 输入张量。
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            profile (bool, optional): 如果为True，分析每层的计算时间，默认为False。
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            visualize (bool, optional): 如果为True，保存特征图以供可视化，默认为False。
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            txt_feats (torch.Tensor): 文本特征，如果给定则使用，默认为None。
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            augment (bool, optional): 如果为True，在推理期间执行数据增强，默认为False。
            embed (list, optional): A list of feature vectors/embeddings to return.
            embed (list, optional): 要返回的特征向量/嵌入的列表。

        Returns:
            (torch.Tensor): Model's output tensor.
            (torch.Tensor): 模型的输出张量。
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.
        计算损失。

        Args:
            batch (dict): Batch to compute loss on.
            batch (dict): 要计算损失的批次。
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
            preds (torch.Tensor | List[torch.Tensor]): 预测。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models.
    模型的集合。"""

    def __init__(self):
        """Initialize an ensemble of models.
        初始化模型的集合。"""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer.
        函数生成YOLO网络的最终层。"""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output

# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).
    上下文管理器，用于临时添加或修改Python模块缓存（`sys.modules`）中的模块。

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.
    此函数可用于在运行时更改模块路径。在重构代码时非常有用，当您将模块从一个位置移动到另一个位置时，但仍希望支持旧的导入路径以保持向后兼容性。

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        modules (dict, optional): 一个字典，将旧模块路径映射到新模块路径。
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.
        attributes (dict, optional): 一个字典，将旧模块属性映射到新模块属性。

    Example:
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```
    示例：
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # 这将现在导入 new.module
            from old.module import attribute  # 这将现在导入 new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        注意：更改仅在上下文管理器内部生效，并在上下文管理器退出后撤销。
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
        请注意，直接操作 `sys.modules` 可能会导致不可预测的结果，尤其是在较大的应用程序或库中。请谨慎使用此函数。
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        # 在 sys.modules 中以旧名称设置属性
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        # 在 sys.modules 中以旧名称设置模块
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        # 移除临时模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling.
    一个占位符类，用于在反序列化期间替换未知类。"""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments.
        初始化 SafeClass 实例，忽略所有参数。"""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments.
        运行 SafeClass 实例，忽略所有参数。"""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass.
    自定义 Unpickler，将未知类替换为 SafeClass。"""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules.
        尝试查找类，如果不在安全模块中，则返回 SafeClass。"""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
            # 添加其他被认为是安全的模块
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    尝试使用 torch.load() 函数加载 PyTorch 模型。如果引发 ModuleNotFoundError，则捕获该错误，记录警告消息，并尝试通过 check_requirements() 函数安装缺失的模块。
    安装后，该函数再次尝试使用 torch.load() 加载模型。

    Args:
        weight (str): The file path of the PyTorch model.
        weight (str): PyTorch 模型的文件路径。
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.
        safe_only (bool): 如果为 True，在加载期间将未知类替换为 SafeClass。

    Example:
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```
    示例：
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    Returns:
        ckpt (dict): The loaded model checkpoint.
        ckpt (dict): 加载的模型检查点。
        file (str): The loaded filename
        file (str): 加载的文件名
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    # 如果本地缺失，则在线搜索
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                # 通过自定义 pickle 模块加载
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        # e.name 是缺失模块的名称
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        # 安装缺失的模块
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        # 文件可能是一个 YOLO 实例，使用例如 torch.save(model, "saved_model.pt") 保存
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file

def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a.
    加载模型的集合 weights=[a,b,c] 或单个模型 weights=[a] 或 weights=a。"""
    ensemble = Ensemble()  # 创建一个模型集合
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # 加载检查点
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # 组合参数
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 模型

        # Model compatibility updates
        # 模型兼容性更新
        model.args = args  # 将参数附加到模型
        model.pt_path = w  # 将 *.pt 文件路径附加到模型
        model.task = guess_model_task(model)  # 猜测模型任务
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])  # 默认步幅

        # Append
        # 添加到集合中
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # 模型设置为评估模式

    # Module updates
    # 模块更新
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace  # 设置就地操作
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # 兼容 torch 1.11.0

    # Return model
    # 返回模型
    if len(ensemble) == 1:
        return ensemble[-1]  # 如果只有一个模型，返回该模型

    # Return ensemble
    # 返回模型集合
    LOGGER.info(f"Ensemble created with {weights}\n")  # 日志记录集合创建信息
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))  # 将属性从第一个模型复制到集合
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride  # 设置步幅
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"  # 确保所有模型的类别数量相同
    return ensemble  # 返回模型集合


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights.
    加载单个模型的权重。"""
    ckpt, weight = torch_safe_load(weight)  # 加载检查点
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # 组合模型和默认参数，优先使用模型参数
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 模型

    # Model compatibility updates
    # 模型兼容性更新
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # 将参数附加到模型
    model.pt_path = weight  # 将 *.pt 文件路径附加到模型
    model.task = guess_model_task(model)  # 猜测模型任务
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])  # 默认步幅

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # 模型设置为评估模式

    # Module updates
    # 模块更新
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace  # 设置就地操作
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # 兼容 torch 1.11.0

    # Return model and ckpt
    # 返回模型和检查点
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model.
    将 YOLO model.yaml 字典解析为 PyTorch 模型。"""
    import ast

    # Args
    # 参数
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")  # 最大通道数
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))  # 获取类别数、激活函数和缩放
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))  # 获取深度、宽度和关键点形状
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]  # 默认缩放
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")  # 日志警告
        depth, width, max_channels = scales[scale]  # 根据缩放获取深度、宽度和最大通道数

    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活函数
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印激活函数信息

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")  # 打印模型信息
    ch = [ch]  # 输入通道列表
    layers, save, c2 = [], [], ch[-1]  # 初始化层、保存列表和输出通道
    base_modules = frozenset(  # 基础模块集合
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
        }
    )
    repeat_modules = frozenset(  # 带有“repeat”参数的模块
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # 从、数量、模块、参数
        m = (
            getattr(torch.nn, m[3:])  # 获取模块
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )
        for j, a in enumerate(args):  # 遍历参数
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)  # 将字符串参数转换为对应的对象
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # 深度增益
        if m in base_modules:
            c1, c2 = ch[f], args[0]  # 输入和输出通道
            if c2 != nc:  # 如果输出通道不等于类别数
                c2 = make_divisible(min(c2, max_channels) * width, 8)  # 确保输出通道符合要求
            if m is C2fAttn:  # 设置嵌入通道和头数
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]  # 更新参数
            if m in repeat_modules:
                args.insert(2, n)  # 添加重复次数
                n = 1
            if m is C3k2:  # 对于 M/L/X 尺寸
                legacy = False
                if scale in "mlx":
                    args[3] = True  # 设置标志

        elif m is AIFI:
            args = [ch[f], *args]  # 更新参数
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]  # 输入、通道和输出通道
            args = [c1, cm, c2, *args[2:]]  # 更新参数
            if m is HGBlock:
                args.insert(4, n)  # 添加重复次数
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4  # 更新输出通道
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]  # 更新参数
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # 更新输出通道
        elif m in frozenset({Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}):
            args.append([ch[x] for x in f])  # 将输入通道添加到参数
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)  # 更新参数
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy  # 设置兼容性

        elif m is RTDETRDecoder:  # 特殊情况，通道参数必须在索引 1 中传递
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]  # 更新参数
        elif m is CBFuse:
            c2 = ch[f[-1]]  # 更新输出通道
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]  # 更新参数
        else:
            c2 = ch[f]  # 更新输出通道

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 创建模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型
        m_.np = sum(x.numel() for x in m_.parameters())  # 参数数量
        m_.i, m_.f, m_.type = i, f, t  # 附加索引、来源索引和类型
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # 打印模块信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)  # 添加模块到层
        if i == 0:
            ch = []  # 清空通道列表
        ch.append(c2)  # 添加输出通道
    return torch.nn.Sequential(*layers), sorted(save)  # 返回模型和保存列表


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file.
    从 YAML 文件加载 YOLOv8 模型。"""
    path = Path(path)  # 将路径转换为 Path 对象
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):  # 检查模型是否为 P6
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)  # 重命名模型
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")  # 日志警告
        path = path.with_name(new_stem + path.suffix)  # 更新路径

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # 统一路径
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)  # 检查 YAML 文件
    d = yaml_load(yaml_file)  # 加载模型字典
    d["scale"] = guess_model_scale(path)  # 猜测模型缩放
    d["yaml_file"] = str(path)  # 保存 YAML 文件路径
    return d  # 返回模型字典


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    将 YOLO 模型的 YAML 文件路径作为输入，并提取模型缩放的大小字符。该函数使用正则表达式匹配在 YAML 文件名中查找模型缩放的模式，该模式由 n、s、m、l 或 x 表示。该函数返回模型缩放的大小字符作为字符串。

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.
        model_path (str | Path): YOLO 模型的 YAML 文件路径。

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
        (str): 模型缩放的大小字符，可以是 n、s、m、l 或 x。
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # noqa, returns n, s, m, l, or x
    except AttributeError:
        return ""  # 返回空字符串


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.
    根据模型的架构或配置猜测 PyTorch 模型的任务。

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.
        model (torch.nn.Module | dict): PyTorch 模型或 YAML 格式的模型配置。

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').
        (str): 模型的任务（'detect'、'segment'、'classify'、'pose'）。

    Raises:
        SyntaxError: If the task of the model could not be determined.
        SyntaxError: 如果无法确定模型的任务。
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary.
        从 YAML 字典中猜测任务。"""
        m = cfg["head"][-1][-2].lower()  # 输出模块名称
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"  # 返回分类任务
        if "detect" in m:
            return "detect"  # 返回检测任务
        if m == "segment":
            return "segment"  # 返回分割任务
        if m == "pose":
            return "pose"  # 返回姿态任务
        if m == "obb":
            return "obb"  # 返回边界框任务

    # Guess from model cfg
    # 从模型配置中猜测任务
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)  # 返回任务

    # Guess from PyTorch model
    # 从 PyTorch 模型中猜测任务
    if isinstance(model, torch.nn.Module):  # PyTorch 模型
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]  # 返回任务
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))  # 返回任务
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"  # 返回分割任务
            elif isinstance(m, Classify):
                return "classify"  # 返回分类任务
            elif isinstance(m, Pose):
                return "pose"  # 返回姿态任务
            elif isinstance(m, OBB):
                return "obb"  # 返回边界框任务
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"  # 返回检测任务

    # Guess from model filename
    # 从模型文件名中猜测任务
    if isinstance(model, (str, Path)):
        model = Path(model)  # 将模型路径转换为 Path 对象
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"  # 返回分割任务
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"  # 返回分类任务
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"  # 返回姿态任务
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"  # 返回边界框任务
        elif "detect" in model.parts:
            return "detect"  # 返回检测任务

    # Unable to determine task from model
    # 无法从模型中确定任务
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # 假设为检测任务