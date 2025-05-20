"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的 functional 模块
import torch.nn.functional as F
# 导入 LAVIS 的 registry 模块
from lavis.common.registry import registry
# 从 blip2_qformer 模块导入 Blip2Qformer 类
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer


# 使用 registry 注册模型，名称为 "blip2_image_text_matching"
@registry.register_model("blip2_image_text_matching")
# 定义 Blip2ITM 类，继承自 Blip2Qformer
class Blip2ITM(Blip2Qformer):
    """
    BLIP Image-Text Matching (ITM) model.
    BLIP 图像-文本匹配 (ITM) 模型。
    Supported model types:
    支持的模型类型：
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
    用法：
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_text_matching", "pretrained")
        >>> model = load_model("blip2_image_text_matching", "coco")
    """

    # 类的初始化方法
    def __init__(
        self,
        vit_model="eva_clip_g", # ViT 模型名称，默认为 "eva_clip_g"
        img_size=224, # 图像尺寸，默认为 224
        drop_path_rate=0, # drop path rate，默认为 0
        use_grad_checkpoint=False, # 是否使用梯度检查点，默认为 False
        vit_precision="fp16", # ViT 的精度，默认为 "fp16"
        freeze_vit=True, # 是否冻结 ViT，默认为 True
        num_query_token=32, # query token 的数量，默认为 32
        cross_attention_freq=2, # 交叉注意力的频率，默认为 2
        embed_dim=256, # 嵌入维度，默认为 256
        max_txt_len=32, # 最大文本长度，默认为 32
    ):
        # 调用父类的初始化方法
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

    # 前向传播方法
    def forward(self, samples, match_head="itm"):
        # 获取图像和文本输入
        image = samples["image"]
        caption = samples["text_input"]

        # 使用自动混合精度计算图像 embedding
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        # 将图像 embedding 转换为 float 类型
        image_embeds = image_embeds.float()
        # 创建图像 attention mask
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 使用 tokenizer 对文本进行编码
        text = self.tokenizer(
            caption,
            truncation=True, # 截断
            max_length=self.max_txt_len, # 最大长度
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(image.device) # 移动到设备

        # 如果匹配头是 "itm" (Image-Text Matching)
        if match_head == "itm":
            # 扩展 query tokens 以匹配批次大小
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # 创建 query tokens 的 attention mask
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            # 拼接 query tokens 和文本的 attention mask
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            # 通过 Q-Former 的 bert 模型计算 ITM 输出
            output_itm = self.Qformer.bert(
                text.input_ids, # 文本 ids
                query_embeds=query_tokens, # query embeddings
                attention_mask=attention_mask, # attention mask
                encoder_hidden_states=image_embeds, # 图像 embeddings 作为 encoder hidden states
                encoder_attention_mask=image_atts, # 图像 attention mask
                return_dict=True, # 返回字典
            )
            # 获取 ITM 输出中对应于 query tokens 的 hidden states
            itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
            # 通过 ITM 头部计算 ITM logits
            itm_logit = self.itm_head(itm_embeddings)
            # 在 query token 维度上取平均
            itm_logit = itm_logit.mean(dim=1)

            # 返回 ITM logit
            return itm_logit

        # 如果匹配头是 "itc" (Image-Text Contrastive)
        elif match_head == "itc":
            # 扩展 query tokens 以匹配批次大小
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            # 通过 Q-Former 的 bert 模型处理 query tokens 和图像 embedding
            query_output = self.Qformer.bert(
                query_embeds=query_tokens, # query embeddings
                encoder_hidden_states=image_embeds, # 图像 embeddings
                encoder_attention_mask=image_atts, # 图像 attention mask
                return_dict=True, # 返回字典
            )
            # 对 Q-Former 输出的 last_hidden_state 进行视觉投影并归一化，得到图像特征
            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )

            # 通过 Q-Former 的 bert 模型处理文本 tokens
            text_output = self.Qformer.bert(
                text.input_ids, # 文本 ids
                attention_mask=text.attention_mask, # 文本 attention mask
                return_dict=True, # 返回字典
            )
            # 对文本输出的第一个 token (CLS token) 的 hidden state 进行文本投影并归一化，得到文本特征
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            # 计算图像特征和文本特征之间的相似度
            sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
            # 取最大相似度
            sim, _ = torch.max(sims, dim=1)

            # 返回相似度
            return sim