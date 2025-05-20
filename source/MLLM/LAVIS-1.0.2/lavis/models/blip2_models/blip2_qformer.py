"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 导入 logging 模块用于日志记录
import logging

# 导入 PyTorch 库
import torch
# 导入 PyTorch 分布式训练模块
import torch.distributed as dist
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 从 torch.cuda.amp 导入 autocast 用于自动混合精度训练
from torch.cuda.amp import autocast as autocast
# 从 torch.nn 导入 functional 模块
from torch.nn import functional as F

# 导入 LAVIS 内部模块
from lavis.common.registry import registry
# 导入分布式训练相关的工具函数
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
# 从 blip2 模块导入 Blip2Base, compute_sim_matrix, disabled_train
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
# 导入 Blip 输出相关的类
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


# 使用 registry 注册模型，名称为 "blip2" 和 "blip2_feature_extractor"
@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
# 定义 Blip2Qformer 类，继承自 Blip2Base
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    带有 Q-former 和 ViT 的 BLIP2 第一阶段模型。
    Supported model types:
    支持的模型类型：
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
    用法：
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

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
        super().__init__()

        # 初始化 tokenizer
        self.tokenizer = self.init_tokenizer()

        # 初始化视觉编码器和 LayerNorm 层
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        # 如果冻结 ViT
        if freeze_vit:
            # 遍历视觉编码器的参数
            for name, param in self.visual_encoder.named_parameters():
                # 设置参数不需要梯度
                param.requires_grad = False
            # 将视觉编码器设置为评估模式
            self.visual_encoder = self.visual_encoder.eval()
            # 将视觉编码器的 train 方法替换为 disabled_train
            self.visual_encoder.train = disabled_train
            # 记录日志，表示冻结了视觉编码器
            logging.info("freeze vision encoder") # 冻结视觉编码器
        # 初始化 Q-Former 和 query tokens
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        # 调整 Q-Former 的 token 嵌入大小以匹配 tokenizer 的词汇表大小
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        # 获取 Q-Former 的状态字典
        state_dict = self.Qformer.state_dict()
        # 遍历 Q-Former 的参数
        for name, param in self.Qformer.named_parameters():
            # 如果参数名包含 "_query"
            if "_query" in name:
                # 获取原始参数名
                key_orig = name.replace("_query", "")
                # 将原始参数的权重复制给带有 "_query" 的参数
                param.data.copy_(state_dict[key_orig])

        # 定义视觉投影层
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        # 定义文本投影层
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # 定义 ITM (Image-Text Matching) 头部，输出维度为 2 (匹配/不匹配)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # 定义可学习的温度参数，用于对比学习
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # 设置最大文本长度
        self.max_txt_len = max_txt_len

    # 前向传播方法
    def forward(self, samples):
        # 获取图像和文本输入
        image = samples["image"]
        text = samples["text_input"]

        # 通过视觉编码器和 LayerNorm 获取图像 embedding
        image_embeds = self.ln_vision(self.visual_encoder(image))
        # 创建图像 attention mask
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 扩展 query tokens 以匹配批次大小
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # 通过 Q-Former 的 bert 模型处理 query tokens 和图像 embedding
        query_output = self.Qformer.bert(
            query_embeds=query_tokens, # query tokens 作为 query embedding
            encoder_hidden_states=image_embeds, # 图像 embedding 作为 encoder hidden states
            encoder_attention_mask=image_atts, # 图像 attention mask
            use_cache=True, # 使用缓存
            return_dict=True, # 返回字典格式的输出
        )

        # 对 Q-Former 输出的 last_hidden_state 进行视觉投影并归一化，得到图像特征
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        # 使用 tokenizer 对文本进行编码
        text_tokens = self.tokenizer(
            text,
            padding="max_length", # 填充到最大长度
            truncation=True, # 截断
            max_length=self.max_txt_len, # 最大长度
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(image.device) # 移动到设备

        # 通过 Q-Former 的 bert 模型处理文本 tokens
        text_output = self.Qformer.bert(
            text_tokens.input_ids, # 文本 ids
            attention_mask=text_tokens.attention_mask, # 文本 attention mask
            return_dict=True, # 返回字典
        )
        # 对文本输出的第一个 token (CLS token) 的 hidden state 进行文本投影并归一化，得到文本特征
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        # ============== 图像-文本对比学习 ===================###
        # 在所有 GPU 之间收集图像特征
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim] # [批次大小*GPU数量, query token 数量, 嵌入维度]
        # 在所有 GPU 之间收集文本特征
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim] # [批次大小*GPU数量, 嵌入维度]

        # 计算 query-to-text 相似度
        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens] # [批次大小, 批次大小*GPU数量, query token 数量]

        # image-text similarity: aggregate across all query tokens
        # 图像-文本相似度：在所有 query tokens 上进行聚合
        sim_i2t, _ = sim_q2t.max(-1) # 取最大相似度
        sim_i2t = sim_i2t / self.temp # 除以温度参数

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # 文本-query 相似度：[批次大小, 批次大小*GPU数量, query token 数量]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        # 文本-图像相似度：在所有 query tokens 上进行聚合
        sim_t2i, _ = sim_t2q.max(-1) # 取最大相似度
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu] # [批次大小, 批次大小*GPU数量]

        # 获取当前进程的排名
        rank = dist.get_rank()
        # 获取当前批次大小
        bs = image.size(0)
        # 创建目标标签，对应于当前进程的样本索引
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        # 计算图像-文本对比损失
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) # image-to-text 损失
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1) # text-to-image 损失
        ) / 2 # 取平均

        ###============== Image-text Matching ===================###
        # ============== 图像-文本匹配 ===================###
        # 在所有 GPU 之间收集文本 ids
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        # 在所有 GPU 之间收集文本 attention masks
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        # 在所有 GPU 之间收集图像 embeddings (带梯度)
        image_embeds_world = all_gather_with_grad(image_embeds)
        # 在 no_grad 模式下计算负样本权重
        with torch.no_grad():
            # 计算 text-to-image 相似度的 softmax 权重
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4 # 加上一个小的 epsilon 避免零概率
            # 将当前进程对应的对角线权重设为 0 (排除正样本)
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            # 计算 image-to-text 相似度的 softmax 权重
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4 # 加上一个小的 epsilon
            # 将当前进程对应的对角线权重设为 0 (排除正样本)
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        # 为每个文本选择一个负样本图像
        image_embeds_neg = []
        for b in range(bs):
            # 根据权重采样一个负样本索引
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            # 获取对应的负样本图像 embedding
            image_embeds_neg.append(image_embeds_world[neg_idx])
        # 将负样本图像 embedding 堆叠起来
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        # 为每个图像选择一个负样本文本
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            # 根据权重采样一个负样本索引
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            # 获取对应的负样本文本 ids 和 attention mask
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        # 将负样本文本 ids 和 attention masks 堆叠起来
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # 拼接文本 ids：正样本，正样本，负样本
        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        # 拼接文本 attention masks：正样本，正样本，负样本
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        # 扩展 query tokens 用于 ITM 计算
        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # 创建 query tokens 的 attention mask
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # 拼接 query tokens 和文本的 attention mask
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        # 拼接图像 embeddings：正样本，负样本，正样本
        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        # 创建图像 attention mask
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 通过 Q-Former 的 bert 模型计算 ITM 分数
        output_itm = self.Qformer.bert(
            text_ids_all, # 文本 ids
            query_embeds=query_tokens_itm, # query embeddings
            attention_mask=attention_mask_all, # attention mask
            encoder_hidden_states=image_embeds_all, # 图像 embeddings 作为 encoder hidden states
            encoder_attention_mask=image_atts_all, # 图像 attention mask
            return_dict=True, # 返回字典
        )

        # 获取 ITM 输出中对应于 query tokens 的 hidden states
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # 通过 ITM 头部计算 ITM logits
        vl_output = self.itm_head(vl_embeddings)
        # 对 ITM logits 在 query token 维度上取平均
        logits = vl_output.mean(dim=1)

        # 创建 ITM 目标标签：前 bs 个是正样本 (1)，后 2*bs 个是负样本 (0)
        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device) # 移动到设备
        # 计算 ITM 交叉熵损失
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        # ================= 图像字幕生成 ========================##
        # 复制文本 ids 作为 decoder 输入
        decoder_input_ids = text_tokens.input_ids.clone()
        # 将第一个 token 替换为 bos_token_id
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # 创建标签，将 pad_token_id 替换为 -100 (在计算损失时忽略)
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        # 创建 query tokens 的 attention mask
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # 拼接 query tokens 和文本的 attention mask
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        # 通过 Q-Former 计算语言模型损失
        lm_output = self.Qformer(
            decoder_input_ids, # decoder 输入 ids
            attention_mask=attention_mask, # attention mask
            past_key_values=query_output.past_key_values, # 使用之前计算的 past_key_values
            return_dict=True, # 返回字典
            labels=labels, # 标签
        )

        # 获取语言模型损失
        loss_lm = lm_output.loss

        # 返回 BlipOutput 对象，包含总损失和各项损失
        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm, # 总损失
            loss_itc=loss_itc, # ITC 损失
            loss_itm=loss_itm, # ITM 损失
            loss_lm=loss_lm, # LM 损失
        )

    # 生成字幕的方法 (不计算梯度)
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False, # 是否使用 nucleus sampling
        num_beams=3, # beam search 的 beam 数量
        max_length=30, # 最大生成长度
        min_length=10, # 最小生成长度
        top_p=0.9, # nucleus sampling 的 top_p
        repetition_penalty=1.0, # 重复惩罚
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # 获取图像数据
        image = samples["image"]
        # 通过视觉编码器和 LayerNorm 获取图像 embedding
        image_embeds = self.ln_vision(self.visual_encoder(image))

        # 如果不使用 nucleus sampling
        if not use_nucleus_sampling:
            # 将图像 embedding 重复 num_beams 次
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            # 如果使用 nucleus sampling，beam 数量设为 1
            num_beams = 1
        # 创建图像 attention mask
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 构建模型生成所需的 kwargs
        model_kwargs = {
            "encoder_hidden_states": image_embeds, # 图像 embedding 作为 encoder hidden states
            "encoder_attention_mask": image_atts, # 图像 attention mask
        }

        # 创建 decoder 的初始输入 ids (bos_token_id)
        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        # 扩展 query tokens 以匹配批次大小
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # 调用 Q-Former 的 generate 方法生成字幕
        outputs = self.Qformer.generate(
            input_ids=input_ids, # 初始输入 ids
            query_embeds=query_tokens, # query embeddings
            max_length=max_length, # 最大长度
            min_length=min_length, # 最小长度
            num_beams=num_beams, # beam 数量
            do_sample=use_nucleus_sampling, # 是否采样
            top_p=top_p, # top_p 参数
            eos_token_id=self.tokenizer.sep_token_id, # 结束 token id
            pad_token_id=self.tokenizer.pad_token_id, # 填充 token id
            **model_kwargs # 其他模型参数
        )
        # 使用 tokenizer 解码生成的 ids 为文本
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 返回生成的字幕列表
        return captions

    # 前向传播图像的方法
    def forward_image(self, image):
        # 通过视觉编码器和 LayerNorm 获取图像 embedding
        image_embeds = self.ln_vision(self.visual_encoder(image))
        # 创建图像 attention mask
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 扩展 query tokens 以匹配批次大小
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # 通过 Q-Former 的 bert 模型处理 query tokens 和图像 embedding
        query_output = self.Qformer.bert(
            query_embeds=query_tokens, # query embeddings
            encoder_hidden_states=image_embeds, # 图像 embeddings
            encoder_attention_mask=image_atts, # 图像 attention mask
            return_dict=True, # 返回字典
        )
        # 返回 Q-Former 输出的 last_hidden_state 和原始图像 embedding
        return query_output.last_hidden_state, image_embeds

    # 前向传播文本的方法
    def forward_text(self, text_tokens):
        # 通过 Q-Former 的 bert 模型处理文本 tokens
        text_output = self.Qformer.bert(
            text_tokens.input_ids, # 文本 ids
            attention_mask=text_tokens.attention_mask, # 文本 attention mask
            return_dict=True, # 返回字典
        )
        # 返回文本输出的第一个 token (CLS token) 的 hidden state
        return text_output.last_hidden_state[:, 0, :]

    # 计算 ITM 分数的方法
    def compute_itm(self, image_inputs, text_ids, text_atts):
        # 创建图像 attention mask
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        # 扩展 query tokens
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        # 创建 query tokens 的 attention mask
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        # 拼接 query tokens 和文本的 attention mask
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        # 通过 Q-Former 的 bert 模型计算 ITM 输出
        output_itm = self.Qformer.bert(
            text_ids, # 文本 ids
            query_embeds=query_tokens, # query embeddings
            attention_mask=attention_mask, # attention mask
            encoder_hidden_states=image_inputs, # 图像 inputs 作为 encoder hidden states
            encoder_attention_mask=image_atts, # 图像 attention mask
            return_dict=True, # 返回字典
        )
        # 获取 ITM 输出中对应于 query tokens 的 hidden states
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        # 通过 ITM 头部计算 ITM logits
        itm_logit = self.itm_head(vl_embeddings)
        # 取 ITM logits 中表示匹配 (索引 1) 的部分，并在 query token 维度上取平均
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        # 返回 ITM logit
        return itm_logit

    # 提取特征的方法 (不计算梯度)
    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        提取多模态或单模态样本的特征。
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        # 获取图像和文本输入
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        # 断言 mode 必须是 "image", "text", "multimodal" 之一
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'" # mode 必须是 'image', 'text', 'multimodal' 之一

        # initalize output
        # 初始化输出变量
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        # 如果 mode 是 "image"
        if mode == "image":
            # 断言图像不为空
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'" # 在 'image' 或 'multimodal' 模式下未提供图像
            # return query features
            # 返回 query 特征
            # 使用自动混合精度计算图像 embedding
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            # 将图像 embedding 转换为 float 类型
            image_embeds_frozen = image_embeds_frozen.float()
            # 创建图像 attention mask
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            # 扩展 query tokens
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            # 通过 Q-Former 的 bert 模型处理 query tokens 和图像 embedding
            query_output = self.Qformer.bert(
                query_embeds=query_tokens, # query embeddings
                encoder_hidden_states=image_embeds_frozen, # 图像 embeddings
                encoder_attention_mask=image_atts, # 图像 attention mask
                return_dict=True, # 返回字典
            )
            # 获取图像 embedding (Q-Former 输出的 last_hidden_state)
            image_embeds = query_output.last_hidden_state
            # 对图像 embedding 进行视觉投影并归一化，得到图像特征
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        # 如果 mode 是 "text"
        elif mode == "text":
            # 断言文本输入不为空
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'" # 在 'text' 或 'multimodal' 模式下文本输入为空

            # return text features
            # 返回文本特征
            # 使用 tokenizer 对文本进行编码
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            # 通过 Q-Former 的 bert 模型处理文本 tokens
            text_output = self.Qformer.bert(
                text.input_ids, # 文本 ids
                attention_mask=text.attention_mask, # 文本 attention mask
                return_dict=True, # 返回字典
            )
            # 获取文本 embedding (Q-Former 输出的 last_hidden_state)
            text_embeds = text_output.last_hidden_state
            # 对文本 embedding 进行文本投影
            text_features = self.text_proj(text_embeds)
            # 对文本特征进行归一化
            text_features = F.normalize(text_features, dim=-1)

        # 如果 mode 是 "multimodal"
        elif mode == "multimodal":
            # return multimodel query features
            # 返回多模态 query 特征
            # 使用自动混合精度计算图像 embedding
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            # 将图像 embedding 转换为 float 类型
            image_embeds_frozen = image_embeds_frozen.float()
            # 创建图像 attention mask
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            # 扩展 query tokens
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            # 创建 query tokens 的 attention mask
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            # 使用 tokenizer 对文本进行编码
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            # 拼接 query tokens 和文本的 attention mask
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            # 通过 Q-Former 的 bert 模型处理文本 tokens, query tokens 和图像 embedding
            output = self.Qformer.bert(
                text.input_ids, # 文本 ids
                query_embeds=query_tokens, # query embeddings
                attention_mask=attention_mask, # attention mask
                encoder_hidden_states=image_embeds_frozen, # 图像 embeddings 作为 encoder hidden states
                encoder_attention_mask=image_atts, # 图像 attention mask
                return_dict=True, # 返回字典
            )

            # 获取多模态 embedding (Q-Former 输出中对应于 query tokens 的 hidden states)
            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        # 返回 BlipOutputFeatures 对象，包含提取的各种特征
        return BlipOutputFeatures(
            image_embeds=image_embeds, # 图像 embedding
            image_embeds_proj=image_features, # 投影后的图像特征
            text_embeds=text_embeds, # 文本 embedding
            text_embeds_proj=text_features, # 投影后的文本特征
            multimodal_embeds=multimodal_embeds, # 多模态 embedding
        )

    # 类方法，从配置字典创建模型实例
    @classmethod
    def from_config(cls, cfg):
        # 从配置中获取模型参数，如果不存在则使用默认值
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        # 创建模型实例
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        # 从配置中加载 checkpoint
        model.load_checkpoint_from_config(cfg)

        # 返回创建的模型实例
        return model

    # 计算相似度矩阵的方法
    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        计算给定数据加载器的 i2t, t2i 相似度矩阵。
        """
        # 从任务配置中获取 k_test 参数
        k_test = task_cfg.k_test

        # 调用 compute_sim_matrix 函数计算相似度矩阵
        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)