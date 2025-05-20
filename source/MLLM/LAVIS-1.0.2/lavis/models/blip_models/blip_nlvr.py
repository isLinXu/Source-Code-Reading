"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可证声明

import os # 导入操作系统模块

import torch # 导入 PyTorch 深度学习库
import torch.nn.functional as F # 导入 PyTorch 的函数式接口，通常用于激活函数、损失函数等
from lavis.common.dist_utils import download_cached_file # 从 lavis.common.dist_utils 导入 download_cached_file 函数，用于下载缓存文件
from lavis.common.registry import registry # 从 lavis.common.registry 导入 registry，用于注册模型
from lavis.common.utils import get_abs_path, is_url # 从 lavis.common.utils 导入 get_abs_path 和 is_url 函数，用于获取绝对路径和判断是否是 URL
from lavis.models.base_model import MomentumDistilationMixin # 从 lavis.models.base_model 导入 MomentumDistilationMixin，提供动量蒸馏的功能
from lavis.models.blip_models.blip import BlipBase # 从 lavis.models.blip_models.blip 导入 BlipBase 类，作为 BLIP 模型的基础类
from lavis.models.blip_models.blip_outputs import BlipIntermediateOutput, BlipOutput # 从 lavis.models.blip_models.blip_outputs 导入 BLIP 模型的中间输出类和最终输出类
from lavis.models.blip_models.nlvr_encoder import BertModel # 从 lavis.models.blip_models.nlvr_encoder 导入 BertModel，用于 NLVR 任务的文本编码器
from lavis.models.vit import VisionTransformerEncoder, interpolate_pos_embed # 从 lavis.models.vit 导入 VisionTransformerEncoder 和 interpolate_pos_embed 函数，用于视觉编码器和插值位置嵌入
from torch import nn # 导入 PyTorch 的神经网络模块
from transformers import BertConfig # 从 transformers 库导入 BertConfig，用于配置 Bert 模型


# 使用 registry 注册模型，名称为 "blip_nlvr"
@registry.register_model("blip_nlvr")
# 定义 BlipNLVR 类，继承自 BlipBase 和 MomentumDistilationMixin
class BlipNLVR(BlipBase, MomentumDistilationMixin):
    """
    Class for BLIP NLVR model. # BLIP NLVR 模型类。

    Supported model types: # 支持的模型类型：
        - base: model with pre-trained BLIP weights, used as initialization for fine-tuning. # base: 带有预训练 BLIP 权重的模型，用作微调的初始化。
        - nlvr: finetuned model on NLVR2 dataset. # nlvr: 在 NLVR2 数据集上微调的模型。

    Usage: # 用法：
        >>> from lavis.models import load_model
        >>> model = load_model("blip_nlvr", "nlvr")
    """

    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "nlvr": "configs/models/blip_nlvr.yaml", # nlvr 模型对应的配置文件路径
    }

    # 类的初始化方法
    def __init__(self, image_encoder, text_encoder, num_classes):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化分词器
        self.tokenizer = self.init_tokenizer()
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

    # 模型的前向传播方法
    def forward(self, samples, is_train=True):
        """
        Forward function for training and evaluation. # 用于训练和评估的前向传播函数。

        Args: # 参数：
            samples (dict): a dict of input samples, which contains the following keys: # 输入样本字典，包含以下键：
                - image0 (torch.Tensor): input image 0, shape (batch_size, 3, H, W), default H=384, W=384. # 输入图像 0，形状为 (batch_size, 3, H, W)，默认 H=384, W=384。
                - image1 (torch.Tensor): input image 1, shape (batch_size, 3, H, W), default H=384, W=384. # 输入图像 1，形状为 (batch_size, 3, H, W)，默认 H=384, W=384。
                - text_input (list): list of strings, each string is a natural language sentence. # 字符串列表，每个字符串是一个自然语言句子。
                - label (torch.LongTensor): ground truth label with shape (batch_size,). # 形状为 (batch_size,) 的真实标签。
            is_train (bool): whether the model is in training mode. # 模型是否处于训练模式。
                If True, the model will return the loss; # 如果为 True，模型将返回损失；
                If False, the model will return the prediction. # 如果为 False，模型将返回预测结果。

        Examples: # 示例：
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("blip_nlvr", "nlvr")
            >>> samples = {
            ...     "image0": torch.randn(2, 3, 384, 384),
            ...     "image1": torch.randn(2, 3, 384, 384),
            ...     "text_input": ["there is a ferret in tall grass", "there are lips in one of the images"],
            ...     "label": torch.tensor([0, 1]),
            ... }
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['intermediate_output', 'loss'])
        """
        # 获取文本输入
        text = samples["text_input"]
        # 使用分词器处理文本输入，填充到最长长度，返回 PyTorch 张量，并移动到设备
        text = self.tokenizer(text, padding="longest", return_tensors="pt").to(
            self.device
        )
        # 将第一个 token (通常是 [CLS]) 替换为编码器 token [ENC]
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        # 获取目标标签
        targets = samples["label"]

        # 获取图像 0 和图像 1
        image0 = samples["image0"]
        image1 = samples["image1"]
        # 将图像 0 和图像 1 沿批次维度拼接
        images = torch.cat([image0, image1], dim=0)

        # 通过视觉编码器获取图像嵌入
        image_embeds = self.visual_encoder.forward_features(images)
        # 创建图像注意力掩码，所有位置都设置为 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # 将拼接后的图像嵌入分割回图像 0 和图像 1 的嵌入
        image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))

        # 通过文本编码器进行前向传播，将图像 0 和图像 1 的嵌入作为 encoder_hidden_states
        encoder_output = self.text_encoder(
            text.input_ids, # 编码器输入 ids
            attention_mask=text.attention_mask, # 注意力掩码
            encoder_hidden_states=[image0_embeds, image1_embeds], # 编码器隐藏状态 (图像嵌入列表)
            encoder_attention_mask=[ # 编码器注意力掩码 (图像注意力掩码列表)
                image_atts[: image0_embeds.size(0)], # 图像 0 的注意力掩码
                image_atts[image0_embeds.size(0) :], # 图像 1 的注意力掩码
            ],
            return_dict=True, # 返回字典格式的输出
        )

        # 通过分类头部获取预测结果 (使用 [CLS] token 的输出)
        prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :])

        # 如果是训练模式
        if is_train:
            # 计算交叉熵损失
            loss = F.cross_entropy(prediction, targets)
            # return {"loss": loss} # 返回损失字典 (注释掉)
            # 返回 BlipOutput 对象，包含损失和中间输出
            return BlipOutput(
                loss=loss, # 损失
                intermediate_output=BlipIntermediateOutput( # 中间输出
                    image_embeds=torch.stack([image0_embeds, image1_embeds], dim=0), # 堆叠图像嵌入
                    encoder_output=encoder_output, # 编码器输出
                ),
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
        # 从配置文件路径加载 Bert 配置
        bert_config = BertConfig.from_json_file(get_abs_path(cfg["med_config_path"]))
        # 创建 BertModel 作为文本编码器，不添加 pooling 层
        text_encoder = BertModel(config=bert_config, add_pooling_layer=False)

        # 从配置中获取类别数量，如果不存在则使用默认值 3
        num_classes = cfg.get("num_classes", 3)

        # 断言类别数量大于 1，否则抛出错误
        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        ) # 提供的类别数量无效，找到 {}

        # 创建模型实例
        model = cls(
            image_encoder=image_encoder, # 图像编码器
            text_encoder=text_encoder, # 文本编码器
            num_classes=num_classes, # 类别数量
        )

        # 从配置中加载 checkpoint
        model.load_checkpoint_from_config(cfg)

        # 返回创建的模型实例
        return model

    # 从预训练权重加载模型的方法
    def load_from_pretrained(self, url_or_filename):
        # 判断是 URL 还是本地文件
        if is_url(url_or_filename):
            # 如果是 URL，下载缓存文件
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            # 加载 checkpoint 文件到 CPU
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            # 如果是本地文件，直接加载到 CPU
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            # 如果路径无效，抛出错误
            raise RuntimeError("checkpoint url or path is invalid") # checkpoint url 或路径无效

        # 获取 state_dict
        state_dict = checkpoint["model"]

        # 插值视觉编码器的位置嵌入，以适应不同的图像尺寸
        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )

        # 遍历 state_dict 中的键
        for key in list(state_dict.keys()):
            # 如果键包含 "crossattention.self." (交叉注意力层的 self attention 部分)
            if "crossattention.self." in key:
                # 创建新的键，分别对应图像 0 和图像 1 的交叉注意力层
                new_key0 = key.replace("self", "self0")
                new_key1 = key.replace("self", "self1")
                # 将原始权重复制到新的键
                state_dict[new_key0] = state_dict[key]
                state_dict[new_key1] = state_dict[key]
            # 如果键包含 "crossattention.output.dense." (交叉注意力层的输出全连接层部分)
            elif "crossattention.output.dense." in key:
                # 创建新的键，分别对应图像 0 和图像 1 的交叉注意力输出层
                new_key0 = key.replace("dense", "dense0")
                new_key1 = key.replace("dense", "dense1")
                # 将原始权重复制到新的键
                state_dict[new_key0] = state_dict[key]
                state_dict[new_key1] = state_dict[key]

        # 加载 state_dict 到当前模型，strict=False 表示允许部分加载，忽略不匹配的键
        msg = self.load_state_dict(state_dict, strict=False)
        # 打印加载 checkpoint 的路径
        print("load checkpoint from %s" % url_or_filename)
        # 打印加载时缺失的键
        print(f"missing keys {msg.missing_keys}")
        # 返回加载结果信息
        return msg
