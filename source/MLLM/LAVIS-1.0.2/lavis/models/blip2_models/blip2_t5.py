"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 导入必要的库
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast  # 导入自动混合精度训练功能
from transformers import T5TokenizerFast  # 导入T5分词器

from lavis.common.registry import registry  # 导入模型注册表
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train  # 导入BLIP2基础类和禁用训练函数
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration  # 导入T5模型配置和条件生成类


@registry.register_model("blip2_t5")  # 将模型注册到LAVIS模型注册表中
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """
    # BLIP2 T5模型，支持多种预训练和微调配置，包括不同大小的FlanT5模型

    # 预训练模型配置字典，包含各种模型配置的YAML文件路径
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",  # 使用的视觉Transformer模型
        img_size=224,  # 输入图像大小
        drop_path_rate=0,  # 随机路径丢弃率
        use_grad_checkpoint=False,  # 是否使用梯度检查点以节省内存
        vit_precision="fp16",  # 视觉模型精度
        freeze_vit=True,  # 是否冻结视觉编码器
        num_query_token=32,  # Q-Former查询token数量
        t5_model="google/flan-t5-xl",  # 使用的T5模型
        prompt="",  # 默认提示词
        max_txt_len=32,  # 最大文本长度
        apply_lemmatizer=False,  # 是否应用词形还原器
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        # apply_lemmatizer: 设为True时，使用词形还原器对predict_answers()的结果进行后处理
        """
        super().__init__()

        # 初始化BERT分词器
        self.tokenizer = self.init_tokenizer()

        # 初始化视觉编码器和层归一化
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        
        # 如果冻结视觉编码器，则禁用其梯度更新和训练模式
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # 初始化Q-Former和查询token
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        
        # 移除Q-Former中不需要的组件以减少内存占用
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # 初始化T5分词器和模型
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"  # 设置密集层激活函数为GELU
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        # 冻结T5模型并转换为bfloat16精度
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        # 添加从Q-Former到T5的投影层
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        # 设置最大文本长度和提示词
        self.max_txt_len = max_txt_len
        self.prompt = prompt

        # 词形还原器相关设置
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):
        """前向传播函数，用于训练过程"""
        image = samples["image"]

        # 使用视觉编码器处理图像
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        
        # 为图像嵌入创建注意力掩码
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 扩展查询token以匹配批次大小
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        # 通过Q-Former处理查询token和图像嵌入
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 将Q-Former输出投影到T5空间
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        # 使用bfloat16精度处理T5部分
        with self.maybe_autocast(dtype=torch.bfloat16):
            # 对输入和输出文本进行分词
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            # 合并视觉和文本注意力掩码
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            # 创建目标，将填充token替换为-100（忽略计算损失）
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            # 获取输入嵌入并与视觉嵌入合并
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            # 运行T5模型并计算损失
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,  # 是否使用核采样
        num_beams=5,  # 束搜索的束数
        max_length=30,  # 生成文本的最大长度
        min_length=1,  # 生成文本的最小长度
        top_p=0.9,  # 核采样的累积概率阈值
        repetition_penalty=1.0,  # 重复惩罚系数
        length_penalty=1.0,  # 长度惩罚系数
        num_captions=1,  # 为每张图像生成的描述数量
        temperature=1,  # 采样温度
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
        # 从样本中获取图像
        image = samples["image"]

        # 处理图像并获取嵌入
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 处理查询token
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 投影到T5空间
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        # 获取提示词
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        # 确保提示词数量与批次大小匹配
        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(
                0
            ), "The number of prompts must be equal to the batch size."

        # 对提示词进行分词
        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        # 合并视觉和文本注意力掩码
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        # 使用bfloat16精度生成文本
        with self.maybe_autocast(dtype=torch.bfloat16):
            # 获取输入嵌入并与视觉嵌入合并
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            # 使用T5模型生成文本
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            # 将token ID解码为文本
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,  # 束搜索的束数
        inference_method="generate",  # 推理方法
        max_len=10,  # 最大生成长度
        min_len=1,  # 最小生成长度
        num_ans_candidates=128,  # 答案候选数量
        answer_list=None,  # 可选的答案列表
        prompt="",  # 提示词
        length_penalty=-1,  # 长度惩罚
        **kwargs
    ):
        """预测问题的答案"""
        image = samples["image"]
        # 处理图像并获取嵌入
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 处理查询token
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 投影到T5空间
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        # 处理输入文本
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        # 对输入文本进行分词
        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        # 合并视觉和文本注意力掩码
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        # 使用bfloat16精度生成答案
        with self.maybe_autocast(dtype=torch.bfloat16):
            # 获取输入嵌入并与视觉嵌入合并
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            # 使用T5模型生成答案
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            # 将token ID解码为文本
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        # 如果需要，应用词形还原器
        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        """对答案应用词形还原"""
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                # 对名词和动词进行词形还原，保留其他词原样
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        """懒加载词形还原器"""
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        """从配置文件创建模型实例"""
        # 从配置中获取各参数
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        # 创建模型实例
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        # 从配置加载检查点
        model.load_checkpoint_from_config(cfg)

        return model
