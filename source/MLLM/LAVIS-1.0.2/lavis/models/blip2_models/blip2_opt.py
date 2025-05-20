"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast  # 自动混合精度
import torch.nn as nn

from lavis.common.registry import registry  # 模型注册器
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train  # BLIP2基础类
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig  # OPT模型
from transformers import AutoTokenizer  # 自动分词器


@registry.register_model("blip2_opt")  # 注册模型为blip2_opt
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """
    # BLIP2 OPT模型，支持多种预训练和微调模型

    PRETRAINED_MODEL_CONFIG_DICT = {  # 预训练模型配置字典
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",  # 视觉模型类型，默认eva_clip_g
        img_size=224,  # 图像大小，默认224
        drop_path_rate=0,  # drop path rate，默认0
        use_grad_checkpoint=False,  # 是否使用梯度检查点，默认False
        vit_precision="fp16",  # 视觉模型精度，默认fp16
        freeze_vit=True,  # 是否冻结视觉模型，默认True
        num_query_token=32,  # 查询token数量，默认32
        opt_model="facebook/opt-2.7b",  # OPT模型路径，默认facebook/opt-2.7b
        prompt="",  # 提示文本，默认空
        max_txt_len=32,  # 最大文本长度，默认32
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()  # 初始化分词器

        # 初始化视觉编码器
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:  # 如果冻结视觉模型
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False  # 冻结所有参数
            self.visual_encoder = self.visual_encoder.eval()  # 设置为评估模式
            self.visual_encoder.train = disabled_train  # 禁用训练
            logging.info("freeze vision encoder")  # 记录日志

        # 初始化Qformer
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None  # 移除分类头
        self.Qformer.bert.embeddings.word_embeddings = None  # 移除词嵌入
        self.Qformer.bert.embeddings.position_embeddings = None  # 移除位置嵌入
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None  # 移除输出层
            layer.intermediate = None  # 移除中间层

        # 初始化OPT分词器和模型
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16  # 使用float16精度
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False  # 冻结OPT模型参数
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]  # 获取换行符的token id

        # 初始化投影层
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len  # 最大文本长度
        self.prompt = prompt  # 提示文本
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")  # 将提示文本转换为token
        self.prompt_length = prompt_tokens.attention_mask.sum(1)  # 计算提示文本长度

    def forward(self, samples):
        image = samples["image"]  # 获取输入图像
        with self.maybe_autocast():  # 使用自动混合精度
            image_embeds = self.ln_vision(self.visual_encoder(image))  # 通过视觉编码器获取图像嵌入
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )  # 创建图像注意力掩码

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # 扩展查询token
        query_output = self.Qformer.bert(  # 通过Qformer获取查询输出
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)  # 投影到OPT模型维度
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)  # 创建OPT注意力掩码

        self.opt_tokenizer.padding_side = "right"  # 设置填充方向为右侧

        text = [t + "\n" for t in samples["text_input"]]  # 在文本输入后添加换行符

        opt_tokens = self.opt_tokenizer(  # 将文本转换为token
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(  # 创建目标token，将填充token标记为-100
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:  # 如果有提示文本
            targets[:, : self.prompt_length] = -100  # 不对提示文本计算损失

        empty_targets = (  # 创建空目标
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)  # 拼接空目标和目标token

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)  # 获取输入嵌入
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)  # 拼接输入嵌入和OPT输入
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)  # 拼接注意力掩码

        with self.maybe_autocast():  # 使用自动混合精度
            outputs = self.opt_model(  # 通过OPT模型获取输出
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss  # 计算损失

        return {"loss": loss}  # 返回损失

    @torch.no_grad()  # 禁用梯度计算，用于推理阶段
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,  # 是否使用nucleus采样，默认为False
        num_beams=5,  # beam search的beam数量，默认为5
        max_length=30,  # 生成序列的最大长度，默认为30
        min_length=1,  # 生成序列的最小长度，默认为1
        top_p=0.9,  # nucleus采样的累积概率阈值，默认为0.9
        repetition_penalty=1.0,  # 重复惩罚系数，默认为1.0
        length_penalty=1.0,  # 长度惩罚系数，默认为1.0
        num_captions=1,  # 每张图像生成的caption数量，默认为1
        temperature=1,  # 采样温度，默认为1
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
        image = samples["image"]  # 从输入中获取图像张量
        with self.maybe_autocast():  # 使用自动混合精度
            image_embeds = self.ln_vision(self.visual_encoder(image))  # 通过视觉编码器获取图像嵌入
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )  # 创建图像注意力掩码，形状为(batch_size, num_patches)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # 扩展查询token以匹配batch size
            query_output = self.Qformer.bert(  # 通过Qformer获取查询输出
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)  # 将Qformer输出投影到OPT模型维度
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )  # 创建OPT注意力掩码

            if "prompt" in samples.keys():  # 如果输入中包含自定义prompt
                prompt = samples["prompt"]
            else:
                prompt = self.prompt  # 否则使用默认prompt

            prompt = [prompt] * image.size(0)  # 将prompt扩展到batch size

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(
                image.device
            )  # 将prompt转换为token
            input_ids = opt_tokens.input_ids  # 获取输入token ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)  # 拼接注意力掩码

            if use_nucleus_sampling:  # 如果使用nucleus采样
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)  # 重复查询嵌入以匹配caption数量
                num_beams = 1  # nucleus采样时不使用beam search
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)  # 重复查询嵌入以匹配beam数量

            outputs = self.opt_model.generate(  # 使用OPT模型生成文本
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,  # 是否使用采样
                top_p=top_p,  # nucleus采样参数
                temperature=temperature,  # 采样温度
                num_beams=num_beams,  # beam search的beam数量
                max_new_tokens=max_length,  # 最大生成token数量
                min_length=min_length,  # 最小生成token数量
                eos_token_id=self.eos_token_id,  # 结束符token id
                repetition_penalty=repetition_penalty,  # 重复惩罚系数
                length_penalty=length_penalty,  # 长度惩罚系数
                num_return_sequences=num_captions,  # 返回的序列数量
            )

            prompt_length = opt_tokens.input_ids.shape[1]  # 计算prompt的长度
            output_text = self.opt_tokenizer.batch_decode(  # 将生成的token解码为文本
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]  # 去除文本前后的空白字符
            return output_text  # 返回生成的caption列表

    @classmethod  # 类方法，用于从配置创建模型实例
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")  # 获取视觉模型类型，默认为eva_clip_g
        img_size = cfg.get("image_size")  # 获取图像大小
        num_query_token = cfg.get("num_query_token")  # 获取查询token数量
        opt_model = cfg.get("opt_model")  # 获取OPT模型路径

        drop_path_rate = cfg.get("drop_path_rate", 0)  # 获取drop path rate，默认为0
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)  # 是否使用梯度检查点，默认为False
        vit_precision = cfg.get("vit_precision", "fp16")  # 视觉模型精度，默认为fp16
        freeze_vit = cfg.get("freeze_vit", True)  # 是否冻结视觉模型，默认为True

        prompt = cfg.get("prompt", "")  # 获取提示文本，默认为空
        max_txt_len = cfg.get("max_txt_len", 32)  # 获取最大文本长度，默认为32

        model = cls(  # 创建模型实例
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)  # 从配置加载预训练权重

        return model  # 返回模型实例
