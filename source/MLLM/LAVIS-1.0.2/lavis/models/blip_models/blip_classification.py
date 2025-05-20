"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可证声明

from copy import deepcopy # 从 copy 模块导入 deepcopy 函数，用于深度复制对象

import torch # 导入 PyTorch 深度学习库
import torch.nn.functional as F # 导入 PyTorch 的函数式接口，通常用于激活函数、损失函数等
from lavis.common.registry import registry # 从 lavis.common.registry 导入 registry，用于注册模型
from lavis.models.base_model import MomentumDistilationMixin # 从 lavis.models.base_model 导入 MomentumDistilationMixin，提供动量蒸馏的功能
from lavis.models.blip_models.blip import BlipBase # 从 lavis.models.blip_models.blip 导入 BlipBase 类，作为 BLIP 模型的基础类
from lavis.models.blip_models.blip_outputs import ( # 从 lavis.models.blip_models.blip_outputs 导入 BLIP 模型的输出类
    BlipIntermediateOutput, # BLIP 模型的中间输出类
    BlipOutputWithLogits, # 包含 logits 的 BLIP 模型输出类
)
from lavis.models.med import XBertEncoder # 从 lavis.models.med 导入 XBertEncoder，用于文本编码器
from lavis.models.vit import VisionTransformerEncoder # 从 lavis.models.vit 导入 VisionTransformerEncoder，用于视觉编码器
from torch import nn # 导入 PyTorch 的神经网络模块


# 使用 registry 注册模型，名称为 "blip_classification"
@registry.register_model("blip_classification")
# 定义 BlipClassification 类，继承自 BlipBase 和 MomentumDistilationMixin
class BlipClassification(BlipBase, MomentumDistilationMixin):
    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_classification_base.yaml", # base 模型对应的配置文件路径
    }

    # 类的初始化方法
    def __init__(
        self,
        image_encoder, # 图像编码器
        text_encoder, # 文本编码器
        num_classes, # 类别数量
        momentum=0.995, # 动量参数
        alpha=0.4, # alpha 参数，用于动量蒸馏中的软目标权重
        max_txt_len=40, # 最大文本长度
        use_distill=True, # 是否使用动量蒸馏
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化分词器
        self.tokenizer = self.init_tokenizer()

        # 存储是否使用动量蒸馏的标志
        self.use_distill = use_distill

        # 初始化视觉编码器和文本编码器
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        # 获取文本编码器的隐藏层大小
        hidden_size = text_encoder.config.hidden_size
        # 创建分类头部，包含两个全连接层和一个 ReLU 激活函数
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # 第一个全连接层
            nn.ReLU(), # ReLU 激活函数
            nn.Linear(hidden_size, num_classes), # 第二个全连接层，输出维度为类别数量
        )

        # 如果使用动量蒸馏
        if self.use_distill:
            # 深度复制视觉编码器作为动量视觉编码器
            self.visual_encoder_m = deepcopy(self.visual_encoder)
            # 深度复制文本编码器作为动量文本编码器
            self.text_encoder_m = deepcopy(self.text_encoder)
            # 深度复制分类头部作为动量分类头部
            self.cls_head_m = deepcopy(self.cls_head)

            # 存储动量参数和 alpha 参数
            self.momentum = momentum
            self.alpha = alpha

            # 创建模型对列表，用于动量更新
            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m], # 视觉编码器对
                [self.text_encoder, self.text_encoder_m], # 文本编码器对
                [self.cls_head, self.cls_head_m], # 分类头部对
            ]

            # 复制参数到动量模型
            self.copy_params()

        # 存储最大文本长度
        self.max_txt_len = max_txt_len

    # 计算 ramp-up 因子的方法，用于控制动量蒸馏的权重
    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        # 计算当前迭代次数占总迭代次数的比例，并限制在 [0, 1] 之间
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    # 模型的前向传播方法
    def forward(self, samples, is_train=True):
        # 获取文本输入
        sentences = samples["text_input"]
        # 使用分词器处理文本输入
        sentences = self.tokenizer(
            sentences, # 文本列表
            padding="longest", # 填充到最长文本的长度
            truncation=True, # 截断超过最大长度的文本
            max_length=self.max_txt_len, # 最大长度
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(self.device) # 将张量移动到设备
        # 将 token 化的文本添加到 samples 字典中
        samples.update({"tokenized_text": sentences})

        # 获取目标标签
        targets = samples["label"]

        # 通过视觉编码器获取图像嵌入
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        # 通过文本编码器进行前向传播 (使用 automask 模式)，结合图像嵌入
        encoder_output = self.text_encoder.forward_automask(
            samples["tokenized_text"], image_embeds
        )

        # 通过分类头部获取预测结果 (使用 [CLS] token 的输出)
        prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :])

        # 如果是训练模式
        if is_train:
            # 如果使用动量蒸馏
            if self.use_distill:
                with torch.no_grad(): # 在 no_grad 模式下进行动量模型的计算
                    # 更新动量模型的参数
                    self._momentum_update()

                    # 通过动量视觉编码器获取图像嵌入
                    image_embeds_m = self.visual_encoder_m(samples["image"])
                    # 通过动量文本编码器进行前向传播 (使用 automask 模式)，结合动量图像嵌入
                    encoder_output_m = self.text_encoder_m.forward_automask(
                        samples["tokenized_text"], image_embeds_m
                    )

                    # 通过动量分类头部获取预测结果
                    prediction_m = self.cls_head_m(
                        encoder_output_m.last_hidden_state[:, 0, :]
                    )

                # 计算 ramp-up 因子控制的 alpha 值
                alpha = self.alpha * self._rampup_factor(
                    epoch=samples["epoch"], # 当前 epoch
                    iters=samples["iters"], # 当前迭代次数
                    num_iters_per_epoch=samples["num_iters_per_epoch"], # 每个 epoch 的迭代次数
                )

                # 计算损失：交叉熵损失 (硬目标) 和 KL 散度损失 (软目标) 的加权和
                loss = (1 - alpha) * F.cross_entropy(
                    prediction, targets # 硬目标损失
                ) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1), # KL 散度损失
                    dim=1,
                ).mean() # 在批次维度上取平均
            else:
                # 如果不使用动量蒸馏，只计算交叉熵损失
                loss = F.cross_entropy(prediction, targets)

            # return {"loss": loss} # 返回损失字典 (注释掉)
            # 返回 BlipOutputWithLogits 对象，包含损失、中间输出和 logits
            return BlipOutputWithLogits(
                loss=loss, # 总损失
                intermediate_output=BlipIntermediateOutput( # 中间输出
                    image_embeds=image_embeds, # 图像嵌入
                    image_embeds_m=image_embeds_m, # 动量图像嵌入
                    encoder_output=encoder_output, # 编码器输出
                    encoder_output_m=encoder_output_m, # 动量编码器输出
                ),
                logits=prediction, # 预测 logits
                logits_m=prediction_m, # 动量模型预测 logits
            )

        # 如果不是训练模式 (评估或预测模式)
        else:
            # 返回包含预测结果和目标标签的字典
            return {"predictions": prediction, "targets": targets}

    # 预测方法，调用 forward 方法并设置 is_train=False
    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    # 类方法，从配置字典创建模型实例
    @classmethod
    def from_config(cls, cfg=None):
        # 从配置中创建视觉编码器
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder # 文本编码器 + 多模态编码器
        # 从配置中创建文本编码器
        text_encoder = XBertEncoder.from_config(cfg)
        # 从配置中获取是否使用动量蒸馏的标志，如果不存在则使用默认值 True
        use_distill = cfg.get("use_distill", True)
        # 从配置中获取动量参数，如果不存在则使用默认值 0.995
        momentum = cfg.get("momentum", 0.995)
        # 从配置中获取类别数量，如果不存在则使用默认值 -1
        num_classes = cfg.get("num_classes", -1)
        # 从配置中获取 alpha 参数，如果不存在则使用默认值 0.4
        alpha = cfg.get("alpha", 0.4)
        # 从配置中获取最大文本长度，如果不存在则使用默认值 40
        max_txt_len = cfg.get("max_txt_len", 40)

        # 断言类别数量大于 1，否则抛出错误
        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        ) # 提供的类别数量无效，找到 {}

        # 创建模型实例
        model = cls(
            image_encoder=image_encoder, # 图像编码器
            text_encoder=text_encoder, # 文本编码器
            use_distill=use_distill, # 是否使用动量蒸馏
            alpha=alpha, # alpha 参数
            num_classes=num_classes, # 类别数量
            momentum=momentum, # 动量参数
            max_txt_len=max_txt_len, # 最大文本长度
        )

        # load pre-trained weights # 加载预训练权重
        # 从配置中获取预训练权重路径
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            # 如果路径存在，加载预训练权重
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)

        # 返回创建的模型实例
        return model
