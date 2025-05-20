"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# 导入所需的库
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

# 导入 LAVIS 内部模块
import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
# 导入 transformers 库中的 BertTokenizer
from transformers import BertTokenizer


# 定义 Blip2Base 类，继承自 BaseModel
class Blip2Base(BaseModel):
    # 类方法，用于初始化 tokenizer
    @classmethod
    def init_tokenizer(cls):
        # 从预训练的 bert-base-uncased 模型加载 tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # 添加特殊 token "[DEC]" 作为 bos_token (beginning of sequence token)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # 返回初始化后的 tokenizer
        return tokenizer

    # 方法，用于根据设备类型决定是否使用 autocast (自动混合精度)
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # 如果在 CPU 上，不使用 autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        # 如果在 GPU 上，如果提供了 dtype 则使用该 dtype 进行 autocast，否则使用 torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            # 如果启用 autocast，返回 torch.cuda.amp.autocast 上下文管理器
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            # 如果不启用 autocast，返回 contextlib.nullcontext (空上下文管理器)
            return contextlib.nullcontext()

    # 类方法，用于初始化 Q-Former
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        # 从预训练的 bert-base-uncased 模型加载 BertConfig
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        # 设置 encoder 的宽度为视觉特征的宽度
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        # 每隔一个块插入交叉注意力层
        encoder_config.add_cross_attention = True
        # 设置交叉注意力的频率
        encoder_config.cross_attention_freq = cross_attention_freq
        # 设置 query token 的长度
        encoder_config.query_length = num_query_token
        # 从预训练的 bert-base-uncased 模型加载 BertLMHeadModel，并应用修改后的配置
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        # 创建可学习的 query tokens 参数
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        # 使用正态分布初始化 query tokens
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        # 返回 Q-Former 模型和 query tokens
        return Qformer, query_tokens

    # 类方法，用于初始化视觉编码器
    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        # 检查模型名称是否有效
        assert model_name in [
            "eva_clip_g",
            "clip_L",
        ], "vit model must be eva_clip_g or clip_L" # vit 模型必须是 eva_clip_g 或 clip_L
        # 如果模型名称是 eva_clip_g
        if model_name == "eva_clip_g":
            # 创建 EVA-ViT-G 视觉编码器
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        # 如果模型名称是 clip_L
        elif model_name == "clip_L":
            # 创建 CLIP-ViT-L 视觉编码器
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        # 创建 LayerNorm 层，用于对视觉编码器的输出进行归一化
        ln_vision = LayerNorm(visual_encoder.num_features)
        # 返回视觉编码器和 LayerNorm 层
        return visual_encoder, ln_vision

    # 方法，用于从预训练的 URL 或文件加载模型权重
    def load_from_pretrained(self, url_or_filename):
        # 检查是否是 URL
        if is_url(url_or_filename):
            # 如果是 URL，下载缓存文件
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            # 加载 checkpoint 文件到 CPU
            checkpoint = torch.load(cached_file, map_location="cpu")
        # 检查是否是文件路径
        elif os.path.isfile(url_or_filename):
            # 如果是文件路径，直接加载 checkpoint 文件到 CPU
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            # 如果 URL 或文件路径无效，抛出运行时错误
            raise RuntimeError("checkpoint url or path is invalid") # checkpoint url 或路径无效

        # 从 checkpoint 中获取模型的状态字典
        state_dict = checkpoint["model"]

        # 加载状态字典到当前模型，strict=False 表示允许部分匹配
        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys)) # 记录缺失的键
        # 记录从哪个位置加载了 checkpoint
        logging.info("load checkpoint from %s" % url_or_filename) # 从 %s 加载 checkpoint

        # 返回加载状态字典的结果信息
        return msg


# 定义一个函数，用于禁用模型的训练模式
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    # 用这个函数覆盖 model.train，以确保训练/评估模式不再改变。
    # 返回 self，表示模型本身
    return self


# 定义 LayerNorm 类，继承自 nn.LayerNorm
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    # 继承 torch 的 LayerNorm 以处理 fp16。

    # 前向传播方法
    def forward(self, x: torch.Tensor):
        # 记录输入张量的原始数据类型
        orig_type = x.dtype
        # 将输入张量转换为 float32 类型进行 LayerNorm 计算
        ret = super().forward(x.type(torch.float32))
        # 将计算结果转换回原始数据类型并返回
        return ret.type(orig_type)


# 定义计算相似度矩阵的函数
def compute_sim_matrix(model, data_loader, **kwargs):
    # 从 kwargs 中获取 k_test 参数
    k_test = kwargs.pop("k_test")

    # 初始化 MetricLogger 用于记录评估指标
    metric_logger = MetricLogger(delimiter="  ")
    # 设置日志头部信息
    header = "Evaluation:" # 评估：

    # 记录开始计算特征的信息
    logging.info("Computing features for evaluation...") # 正在计算评估特征...
    # 记录开始时间
    start_time = time.time()

    # 获取数据集中的所有文本
    texts = data_loader.dataset.text
    # 获取文本数量
    num_text = len(texts)
    # 设置文本批次大小
    text_bs = 256
    # 初始化列表用于存储文本 ids, 文本 embeddings 和文本 attention masks
    text_ids = []
    text_embeds = []
    text_atts = []
    # 遍历文本，按批次处理
    for i in range(0, num_text, text_bs):
        # 获取当前批次的文本
        text = texts[i : min(num_text, i + text_bs)]
        # 使用模型 tokenizer 对文本进行编码
        text_input = model.tokenizer(
            text,
            padding="max_length", # 填充到最大长度
            truncation=True, # 截断超过最大长度的部分
            max_length=35, # 最大长度设置为 35
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(model.device) # 将张量移动到模型所在的设备

        # 通过模型的前向传播获取文本特征
        text_feat = model.forward_text(text_input)
        # 对文本特征进行投影并归一化，得到文本 embedding
        text_embed = F.normalize(model.text_proj(text_feat))
        # 将文本 embedding 添加到列表中
        text_embeds.append(text_embed)
        # 将文本 ids 添加到列表中
        text_ids.append(text_input.input_ids)
        # 将文本 attention masks 添加到列表中
        text_atts.append(text_input.attention_mask)

    # 将所有批次的文本 embedding 拼接起来
    text_embeds = torch.cat(text_embeds, dim=0)
    # 将所有批次的文本 ids 拼接起来
    text_ids = torch.cat(text_ids, dim=0)
    # 将所有批次的文本 attention masks 拼接起来
    text_atts = torch.cat(text_atts, dim=0)

    # 初始化列表用于存储 ViT 特征和图像 embeddings
    vit_feats = []
    image_embeds = []
    # 遍历数据加载器中的样本
    for samples in data_loader:
        # 获取图像数据
        image = samples["image"]

        # 将图像数据移动到模型所在的设备
        image = image.to(model.device)
        # 通过模型的前向传播获取图像特征和 ViT 特征
        image_feat, vit_feat = model.forward_image(image)
        # 对图像特征进行投影，得到图像 embedding
        image_embed = model.vision_proj(image_feat)
        # 对图像 embedding 进行归一化
        image_embed = F.normalize(image_embed, dim=-1)

        # 将 ViT 特征移动到 CPU 并添加到列表中
        vit_feats.append(vit_feat.cpu())
        # 将图像 embedding 添加到列表中
        image_embeds.append(image_embed)

    # 将所有批次的 ViT 特征拼接起来
    vit_feats = torch.cat(vit_feats, dim=0)
    # 将所有批次的图像 embedding 拼接起来
    image_embeds = torch.cat(image_embeds, dim=0)

    # 初始化列表用于存储相似度矩阵
    sims_matrix = []
    # 遍历每个图像 embedding
    for image_embed in image_embeds:
        # 计算图像 embedding 与所有文本 embedding 的相似度 (query-to-text)
        sim_q2t = image_embed @ text_embeds.t()
        # 获取每个文本对应的最大相似度 (image-to-text)
        sim_i2t, _ = sim_q2t.max(0)
        # 将 image-to-text 相似度添加到列表中
        sims_matrix.append(sim_i2t)
    # 将所有 image-to-text 相似度堆叠成矩阵
    sims_matrix = torch.stack(sims_matrix, dim=0)

    # 初始化 score_matrix_i2t，用于存储 image-to-text 的最终得分
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0 # 初始化为 -100.0
    ).to(model.device) # 将矩阵移动到模型所在的设备

    # 获取分布式训练的任务数量
    num_tasks = dist_utils.get_world_size()
    # 获取当前任务的排名
    rank = dist_utils.get_rank()
    # 计算每个任务处理的步长
    step = sims_matrix.size(0) // num_tasks + 1
    # 计算当前任务的起始索引
    start = rank * step
    # 计算当前任务的结束索引
    end = min(sims_matrix.size(0), start + step)

    # 遍历当前任务负责的相似度矩阵部分，并记录日志
    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header) # 每 50 步记录一次日志
    ):
        # 对于每个图像，获取与其相似度最高的 k_test 个文本的相似度和索引
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # 复制对应的 ViT 特征，用于计算 ITM 分数
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        # 计算 ITM (Image-Text Matching) 分数
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx], # 对应的文本 ids
            text_atts=text_atts[topk_idx], # 对应的文本 attention masks
        ).float() # 转换为 float 类型
        # 将 ITM 分数与相似度相加，更新 score_matrix_i2t
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    # 转置相似度矩阵，用于计算 text-to-image 相似度
    sims_matrix = sims_matrix.t()
    # 初始化 score_matrix_t2i，用于存储 text-to-image 的最终得分
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0 # 初始化为 -100.0
    ).to(model.device) # 将矩阵移动到模型所在的设备

    # 计算每个任务处理的步长
    step = sims_matrix.size(0) // num_tasks + 1
    # 计算当前任务的起始索引
    start = rank * step
    # 计算当前任务的结束索引
    end = min(sims_matrix.size(0), start + step)

    # 遍历当前任务负责的相似度矩阵部分，并记录日志
    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header) # 每 50 步记录一次日志
    ):
        # 对于每个文本，获取与其相似度最高的 k_test 个图像的相似度和索引
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # 获取对应的 ViT 特征，用于计算 ITM 分数
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        # 计算 ITM (Image-Text Matching) 分数
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1), # 对应的文本 ids (重复 k_test 次)
            text_atts=text_atts[start + i].repeat(k_test, 1), # 对应的文本 attention masks (重复 k_test 次)
        ).float() # 转换为 float 类型
        # 将 ITM 分数与相似度相加，更新 score_matrix_t2i
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    # 如果是分布式训练环境
    if dist_utils.is_dist_avail_and_initialized():
        # 等待所有进程完成
        dist.barrier()
        # 对 score_matrix_i2t 进行全局求和
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        # 对 score_matrix_t2i 进行全局求和
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    # 计算总评估时间
    total_time = time.time() - start_time
    # 将总时间格式化为字符串
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # 记录评估时间
    logging.info("Evaluation time {}".format(total_time_str)) # 评估时间 {}

    # 返回 image-to-text 和 text-to-image 的最终得分矩阵 (转换为 NumPy 数组并移到 CPU)
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()