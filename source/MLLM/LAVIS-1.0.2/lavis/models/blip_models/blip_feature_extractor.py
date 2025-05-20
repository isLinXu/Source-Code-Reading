"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可证声明

import warnings # 导入 warnings 模块，用于发出警告

import torch # 导入 PyTorch 深度学习库
import torch.nn.functional as F # 导入 PyTorch 的函数式接口，通常用于激活函数、损失函数等
from lavis.common.registry import registry # 从 lavis.common.registry 导入 registry，用于注册模型
from lavis.models.blip_models.blip import BlipBase # 从 lavis.models.blip_models.blip 导入 BlipBase 类，作为 BLIP 模型的基础类
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures # 从 lavis.models.blip_models.blip_outputs 导入 BlipOutputFeatures 类，用于存储特征提取的输出
from lavis.models.med import XBertEncoder # 从 lavis.models.med 导入 XBertEncoder，用于文本编码器
from lavis.models.vit import VisionTransformerEncoder # 从 lavis.models.vit 导入 VisionTransformerEncoder，用于视觉编码器
from torch import nn # 导入 PyTorch 的神经网络模块

# 使用 registry 注册模型，名称为 "blip_feature_extractor"
@registry.register_model("blip_feature_extractor")
# 定义 BlipFeatureExtractor 类，继承自 BlipBase
class BlipFeatureExtractor(BlipBase):
    """
    Class for BLIP feature extractor. # BLIP 特征提取器类。

    Supported model types: # 支持的模型类型：
        - base: BLIP base model with pre-trained weights from capfilt by BLIP large model. # base: 使用 BLIP large 模型通过 capfilt 预训练权重的 BLIP base 模型。

    Usage: # 用法：
        >>> from lavis.models import load_model
        >>> model = load_model("blip_feature_extractor", "base")
    """

    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_feature_extractor_base.yaml", # base 模型对应的配置文件路径
        # "large": "configs/models/blip_feature_extractor_large.yaml", # large 模型对应的配置文件路径 (注释掉)
    }

    # 类的初始化方法
    def __init__(self, image_encoder, text_encoder, embed_dim, max_txt_len=40):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化分词器
        self.tokenizer = self.init_tokenizer()

        # 初始化视觉编码器和文本编码器
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        # creating projection layers for ITC # 创建用于 ITC (Image-Text Contrastive) 任务的投影层
        text_width = text_encoder.config.hidden_size # 获取文本编码器的隐藏层大小
        vision_width = image_encoder.vision_width # 获取图像编码器的视觉宽度

        self.vision_proj = nn.Linear(vision_width, embed_dim) # 视觉投影层
        self.text_proj = nn.Linear(text_width, embed_dim) # 文本投影层

        # 存储最大文本长度
        self.max_txt_len = max_txt_len

        # 定义温度参数 temp，用于对比学习，初始化为 0.07 的可学习参数
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    # 特征提取方法，使用 torch.no_grad() 装饰器，表示不计算梯度
    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples. # 提取多模态或单模态样本的特征。

        Args: # 参数：
            samples (dict): A dictionary of samples, containing the following keys: # 一个样本字典，包含以下键：
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image. # 形状为 (B, C, H, W) 的张量，包含图像。
                    Raw images should be preprocessed before being passed to feature extractor. # 原始图像在传递给特征提取器之前应该进行预处理。
                - text_input (list): A list of strings containing the text, length B. # 包含文本的字符串列表，长度为 B。
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image". # 特征提取模式。可以是 "multimodal", "text" 或 "image"。
                If "multimodal", return image features and multimodal features; # 如果是 "multimodal"，返回图像特征和多模态特征；
                if "text", return text features; # 如果是 "text"，返回文本特征；
                if "image", return image features. # 如果是 "image"，返回图像特征。
                Default: "multimodal". # 默认值："multimodal"。

        Returns: # 返回：
            BlipOutputFeatures: A BlipOutputFeatures object containing the features. # 一个 BlipOutputFeatures 对象，包含特征。
                See lavis/models/blip_models/blip_outputs.py for more details. # 更多详情请参阅 lavis/models/blip_models/blip_outputs.py。

        Examples: # 示例：
        ```python
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> caption = "a large fountain spewing water into the air"
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_feature_extractor", is_eval=True)
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> text_input = txt_processors["eval"](caption)

            >>> sample = {"image": image, "text_input": [text_input]}

            >>> features_multimodal = model.extract_features(sample)
            >>> features_multimodal.keys()
            odict_keys(['image_embeds', 'multimodal_embeds'])
            >>> features_multimodal.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_multimodal.multimodal_embeds.shape
            torch.Size([1, 12, 768])

            >>> features_text = model.extract_features(sample, mode="text")
            >>> features_text.keys()
            odict_keys(['text_embeds', 'text_features'])
            >>> features_text.text_embeds.shape
            torch.Size([1, 12, 768])
            >>> features_text.text_features.shape
            torch.Size([1, 12, 256])

            >>> features_image = model.extract_features(sample, mode="image")
            >>> features_image.keys()
            odict_keys(['image_embeds', 'image_features'])
            >>> features_image.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_image.image_features.shape
            torch.Size([1, 197, 256])
        ```
        """
        # 从 samples 字典中获取图像和文本输入
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal" # 断言 mode 必须是 "image", "text", "multimodal" 之一
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'" # mode 必须是 'image', 'text', 'multimodal' 之一

        # initalize output # 初始化输出变量
        image_embeds, text_embeds, multimodal_embeds = None, None, None # 图像嵌入、文本嵌入、多模态嵌入
        image_features, text_features = None, None # 图像特征、文本特征

        # 如果模式是 "image"
        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'" # 在 'image' 或 'multimodal' 模式下未提供图像
            # return image features # 返回图像特征
            # 通过视觉编码器获取图像嵌入
            image_embeds = self.visual_encoder.forward_features(image)

            # 通过视觉投影层获取图像特征，并进行 L2 归一化
            image_features = self.vision_proj(image_embeds)
            image_features = F.normalize(image_features, dim=-1)

        # 如果模式是 "text"
        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'" # 在 'text' 或 'multimodal' 模式下文本输入为 None

            # 使用分词器处理文本输入
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            # return text features # 返回文本特征
            # 通过文本编码器获取文本输出 (文本模式)
            text_output = self.text_encoder(
                text.input_ids, # 文本输入 ids
                attention_mask=text.attention_mask, # 注意力掩码
                return_dict=True, # 返回字典格式的输出
                mode="text", # 指定为文本模式
            )
            # 获取文本嵌入
            text_embeds = text_output.last_hidden_state

            # 通过文本投影层获取文本特征，并进行 L2 归一化
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        # 如果模式是 "multimodal"
        elif mode == "multimodal":
            # return multimodel features # 返回多模态特征
            # 通过视觉编码器获取图像嵌入
            image_embeds = self.visual_encoder.forward_features(image)
            # 创建图像注意力掩码，所有位置都设置为 1
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )

            # 使用分词器处理文本输入
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            # 将第一个 token (通常是 [CLS]) 替换为编码器 token [ENC]
            text.input_ids[:, 0] = self.tokenizer.enc_token_id

            # 通过文本编码器进行前向传播，将图像嵌入作为 encoder_hidden_states
            output = self.text_encoder(
                text.input_ids, # 编码器输入 ids
                attention_mask=text.attention_mask, # 注意力掩码
                encoder_hidden_states=image_embeds, # 编码器隐藏状态 (图像嵌入)
                encoder_attention_mask=image_atts, # 编码器注意力掩码 (图像注意力)
                return_dict=True, # 返回字典格式的输出
            )

            # 获取多模态嵌入 (文本编码器的输出)
            multimodal_embeds = output.last_hidden_state

        # 返回 BlipOutputFeatures 对象，包含提取的各种特征
        return BlipOutputFeatures(
            image_embeds=image_embeds, # 图像嵌入
            image_embeds_proj=image_features, # 投影后的图像特征
            text_embeds=text_embeds, # 文本嵌入
            text_embeds_proj=text_features, # 投影后的文本特征
            multimodal_embeds=multimodal_embeds, # 多模态嵌入
        )

    # 类方法，从配置字典创建模型实例
    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased' # 设置 from_pretrained=True 以加载 'bert-base-uncased' 的权重
        # 从配置中创建视觉编码器
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        # 从配置中创建文本编码器
        text_encoder = XBertEncoder.from_config(cfg)

        # 从配置中获取模型参数，如果不存在则使用默认值
        embed_dim = cfg.get("embed_dim", 256) # 嵌入维度
        max_txt_len = cfg.get("max_txt_len", 30) # 最大文本长度

        # 创建模型实例
        model = cls(
            image_encoder=image_encoder, # 图像编码器
            text_encoder=text_encoder, # 文本编码器
            embed_dim=embed_dim, # 嵌入维度
            max_txt_len=max_txt_len, # 最大文本长度
        )

        # load pre-trained weights # 加载预训练权重
        pretrain_path = cfg.get("pretrained", None) # 从配置中获取预训练权重路径
        if pretrain_path is not None:
            # 如果路径存在，加载预训练权重
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)
        else:
            # 如果路径不存在，发出警告
            warnings.warn("No pretrained weights are loaded.") # 未加载预训练权重。

        # 返回创建的模型实例
        return model
