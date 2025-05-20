"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 版权信息和许可证声明

import torch # 导入 PyTorch 深度学习库
from lavis.common.registry import registry # 从 lavis.common.registry 导入 registry，用于注册模型

from lavis.models.blip_models.blip import BlipBase # 从 lavis.models.blip_models.blip 导入 BlipBase 类，作为 BLIP 模型的基础类
from lavis.models.blip_models.blip_outputs import ( # 从 lavis.models.blip_models.blip_outputs 导入 BLIP 模型的输出类
    BlipOutput, # BLIP 模型的最终输出类
    BlipIntermediateOutput, # BLIP 模型的中间输出类
)
from lavis.models.med import XBertLMHeadDecoder # 从 lavis.models.med 导入 XBertLMHeadDecoder，用于文本解码器
from lavis.models.vit import VisionTransformerEncoder # 从 lavis.models.vit 导入 VisionTransformerEncoder，用于视觉编码器


# 使用 registry 注册模型，名称为 "blip_caption"
@registry.register_model("blip_caption")
# 定义 BlipCaption 类，继承自 BlipBase
class BlipCaption(BlipBase):
    """
    BLIP captioning model. # BLIP 图像字幕生成模型。

    Supported model types: # 支持的模型类型：
        - base_coco: fine-tuned BLIP base model on COCO caption dataset (Karparthy split). # base_coco: 在 COCO 字幕数据集 (Karparthy 分割) 上微调的 BLIP base 模型。
        - large_coco: fine-tuned BLIP large model on COCO caption dataset (Karparthy split). # large_coco: 在 COCO 字幕数据集 (Karparthy 分割) 上微调的 BLIP large 模型。

    Usage: # 用法：
        >>> from lavis.models import load_model
        >>> model = load_model("blip_caption", "base_coco")
        >>> model = load_model("blip_caption", "large_coco")
    """

    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_coco": "configs/models/blip_caption_base_coco.yaml", # base_coco 模型对应的配置文件路径
        "large_coco": "configs/models/blip_caption_large_coco.yaml", # large_coco 模型对应的配置文件路径
    }

    # 类的初始化方法
    def __init__(self, image_encoder, text_decoder, prompt=None, max_txt_len=40):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化分词器
        self.tokenizer = self.init_tokenizer()

        # 初始化视觉编码器和文本解码器
        self.visual_encoder = image_encoder
        self.text_decoder = text_decoder

        # 存储 prompt (提示文本)
        self.prompt = prompt
        # 计算 prompt 的长度 (token 数)，减去末尾的 [SEP] token
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # 存储最大文本长度
        self.max_txt_len = max_txt_len

    # 前向传播：编码器部分
    def forward_encoder(self, samples):
        # 通过视觉编码器获取图像嵌入
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        # 返回图像嵌入
        return image_embeds

    # 前向传播：解码器部分
    def forward_decoder(self, samples, image_embeds):
        # prepare inputs for forwarding decoder # 准备用于解码器前向传播的输入
        raw_text = samples["text_input"] # 获取原始文本输入
        # 使用分词器处理文本输入
        text = self.tokenizer(
            raw_text, # 原始文本列表
            padding="longest", # 填充到最长文本的长度
            truncation=True, # 截断超过最大长度的文本
            max_length=self.max_txt_len, # 最大长度
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(self.device) # 将张量移动到设备

        # 将第一个 token (通常是 [CLS]) 替换为解码器 token [DEC] (bos_token)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # prepare targets for forwarding decoder # 准备用于解码器前向传播的目标标签
        # 创建解码器目标标签，将填充 token (-100) 忽略计算损失
        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        # 将 prompt 部分的标签设置为 -100，表示不计算 prompt 部分的损失
        decoder_targets[:, : self.prompt_length] = -100

        # forward decoder # 解码器前向传播
        # 创建图像注意力掩码，所有位置都设置为 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # 通过文本解码器进行前向传播
        decoder_output = self.text_decoder(
            input_ids=text.input_ids, # 解码器输入 ids
            attention_mask=text.attention_mask, # 注意力掩码
            encoder_hidden_states=image_embeds, # 编码器隐藏状态 (图像嵌入)
            encoder_attention_mask=image_atts, # 编码器注意力掩码 (图像注意力)
            labels=decoder_targets, # 目标标签
            return_dict=True, # 返回字典格式的输出
        )

        # 返回解码器输出和目标标签
        return decoder_output, decoder_targets

    # 模型的主前向传播方法
    def forward(self, samples):
        r"""
        Args: # 参数：
            samples (dict): A dictionary containing the following keys: # 一个字典，包含以下键：
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W) # 形状为 (batch_size, 3, H, W) 的张量
                - text_input (list): A list of strings of length batch_size. # 长度为 batch_size 的字符串列表。
        Returns: # 返回：
            output (BlipOutput): A BlipOutput object containing the following # 一个 BlipOutput 对象，包含以下属性：
                attributes: # 属性：
                - loss (torch.Tensor): A scalar tensor containing the total loss. For BlipCaption, this is the same as the LM loss. # 包含总损失的标量张量。对于 BlipCaption，这与 LM 损失相同。
                - loss_lm (torch.Tensor): A scalar tensor containing the LM loss. # 包含 LM 损失的标量张量。
                - intermediate_outputs (BlipIntermediateOutput): A BlipIntermediateOutput object containing intermediate outputs. # 一个 BlipIntermediateOutput 对象，包含中间输出。
                  see :class:`lavis.models.blip_models.blip_outputs.BlipOutput` for more details. # 更多详情请参阅 :class:`lavis.models.blip_models.blip_outputs.BlipOutput`。

        Example: # 示例：
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> text_input = ["a large statue of a person spraying water from a fountain"]
        >>> samples = {"image": image, "text_input": text_input}
        >>> output = model(samples)
        >>> output.keys()
        odict_keys(['intermediate_output', 'loss', 'loss_lm'])
        >>> output.intermediate_output.image_embeds.shape
        torch.Size([1, 577, 768])
        >>> output.intermediate_output.decoder_labels.shape
        torch.Size([1, 13])
        ```"""

        # 通过编码器获取图像嵌入
        image_embeds = self.forward_encoder(samples)
        # 通过解码器获取解码器输出和目标标签
        decoder_output, decoder_targets = self.forward_decoder(samples, image_embeds)

        # return decoder_out # 返回解码器输出 (注释掉)
        # 返回 BlipOutput 对象，包含损失和中间输出
        return BlipOutput(
            loss=decoder_output.loss, # 总损失 (等于 LM 损失)
            loss_lm=decoder_output.loss, # LM 损失
            intermediate_output=BlipIntermediateOutput( # 中间输出
                image_embeds=image_embeds, # 图像嵌入
                decoder_output=decoder_output, # 解码器输出
                decoder_labels=decoder_targets, # 解码器目标标签
            ),
        )

    # 文本生成方法
    def generate(
        self,
        samples,
        use_nucleus_sampling=False, # 是否使用 nucleus 采样
        num_beams=3, # beam search 的 beam 数量
        max_length=30, # 生成序列的最大长度
        min_length=10, # 生成序列的最小长度
        top_p=0.9, # nucleus 采样的累积概率阈值
        repetition_penalty=1.0, # 重复惩罚参数
        num_captions=1, # 为每张图像生成的字幕数量
    ):
        """
        Args: # 参数：
            samples (dict): A dictionary containing the following keys: # 一个字典，包含以下键：
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W) # 形状为 (batch_size, 3, H, W) 的张量
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling. # 是否使用 nucleus 采样。如果为 False，则使用 top-k 采样。
            num_beams (int): Number of beams for beam search. 1 means no beam search. # beam search 的 beam 数量。1 表示不使用 beam search。
            max_length (int): The maximum length of the sequence to be generated. # 生成序列的最大长度。
            min_length (int): The minimum length of the sequence to be generated. # 生成序列的最小长度。
            top_p (float): The cumulative probability for nucleus sampling. # nucleus 采样的累积概率。
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty. # 重复惩罚参数。1.0 表示没有惩罚。
            num_captions (int): Number of captions to be generated for each image. # 为每张图像生成的字幕数量。
        Returns: # 返回：
            captions (list): A list of strings of length batch_size * num_captions. # 长度为 batch_size * num_captions 的字符串列表。

        Example: # 示例：
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> samples = {"image": image}
        >>> captions = model.generate(samples)
        >>> captions
        ['a large statue of a person spraying water from a fountain']
        >>> captions = model.generate(samples, use_nucleus_sampling=True, num_captions=3)
        >>> captions # example output, results may vary due to randomness # 示例输出，结果可能因随机性而异
        ['singapore showing the view of some building',
        'the singapore harbor in twilight, as the weather is going down',
        'the famous singapore fountain at sunset']
        """
        # prepare inputs for decoder generation. # 准备用于解码器生成的输入。
        # 通过编码器获取图像嵌入
        encoder_out = self.forward_encoder(samples)
        # 将图像嵌入重复 num_captions 次，以生成多个字幕
        image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)

        # 创建 prompt 列表，每个样本对应一个 prompt
        prompt = [self.prompt] * image_embeds.size(0)
        # 使用分词器处理 prompt
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # 将第一个 token 替换为 bos_token_id
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        # 移除最后一个 token (通常是 [SEP])
        prompt.input_ids = prompt.input_ids[:, :-1]

        # get decoded text # 获取解码后的文本
        # 调用文本解码器的 generate_from_encoder 方法进行文本生成
        decoder_out = self.text_decoder.generate_from_encoder(
            tokenized_prompt=prompt, # token 化的 prompt
            visual_embeds=image_embeds, # 视觉嵌入
            sep_token_id=self.tokenizer.sep_token_id, # 分隔 token 的 ID
            pad_token_id=self.tokenizer.pad_token_id, # 填充 token 的 ID
            use_nucleus_sampling=use_nucleus_sampling, # 是否使用 nucleus 采样
            num_beams=num_beams, # beam 数量
            max_length=max_length, # 最大长度
            min_length=min_length, # 最小长度
            top_p=top_p, # top_p 参数
            repetition_penalty=repetition_penalty, # 重复惩罚参数
        )

        # 使用分词器将生成的 token ID 序列解码为字符串，跳过特殊 token
        outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        # 移除 prompt 部分，只保留生成的字幕部分
        captions = [output[len(self.prompt) :] for output in outputs]

        # 返回生成的字幕列表
        return captions

    # 类方法，从配置字典创建模型实例
    @classmethod
    def from_config(cls, cfg):
        # vision encoder # 视觉编码器
        # 从配置中创建视觉编码器
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        # text encoder + multimodal decoder # 文本编码器 + 多模态解码器
        # 从配置中创建文本解码器
        text_decoder = XBertLMHeadDecoder.from_config(cfg)

        # 从配置中获取 prompt 和最大文本长度，如果不存在则使用默认值
        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 40)

        # 创建模型实例
        model = cls(image_encoder, text_decoder, prompt=prompt, max_txt_len=max_txt_len)
        # 从配置中加载 checkpoint
        model.load_checkpoint_from_config(cfg)

        # 返回创建的模型实例
        return model
