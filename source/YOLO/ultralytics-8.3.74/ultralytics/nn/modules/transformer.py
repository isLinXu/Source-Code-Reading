# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer modules."""
# 变换器模块

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "TransformerEncoderLayer",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
)

class RTDETRDecoder(nn.Module):
    """Defines a Real-Time Deformable Transformer Decoder for object detection."""
    # 定义实时可变形变换解码器，用于目标检测

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters."""
        # 使用给定参数初始化RTDETRDecoder模块
        super().__init__()  # 调用父类的初始化方法
        self.hidden_dim = hd  # 隐藏层维度
        self.nhead = nh  # 头数
        self.nl = len(ch)  # 层数
        self.nc = nc  # 类别数量
        self.num_queries = nq  # 查询数量
        self.num_decoder_layers = ndl  # 解码器层数量

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # 创建输入投影模块列表，包含卷积层和批量归一化层
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        # 创建可变形变换器解码器层
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl)
        # 创建可变形变换器解码器

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)  # 创建去噪分类嵌入
        self.num_denoising = nd  # 去噪数量
        self.label_noise_ratio = label_noise_ratio  # 标签噪声比率
        self.box_noise_scale = box_noise_scale  # 边框噪声比例

        # Decoder embedding
        self.learnt_init_query = learnt_init_query  # 是否学习初始查询
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)  # 创建目标嵌入
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)  # 创建查询位置头

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))  # 创建编码器输出
        self.enc_score_head = nn.Linear(hd, nc)  # 创建编码器得分头
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)  # 创建编码器边框头

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])  # 创建解码器得分头
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])  # 创建解码器边框头

        self._reset_parameters()  # 重置参数

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        # 执行模块的前向传播，返回输入的边框和分类得分
        from ultralytics.models.utils.ops import get_cdn_group  # 导入函数

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)  # 获取编码器输入

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        # 准备去噪训练

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        # 获取解码器输入

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        # 执行解码器

        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta  # 组合输出
        if self.training:
            return x  # 如果是训练模式，返回输出
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)  # 拼接边框和得分
        return y if self.export else (y, x)  # 根据导出模式返回结果

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        # 为给定形状生成锚框边界框，并进行验证
        anchors = []  # 初始化锚框列表
        for i, (h, w) in enumerate(shapes):  # 遍历形状
            sy = torch.arange(end=h, dtype=dtype, device=device)  # 创建y坐标
            sx = torch.arange(end=w, dtype=dtype, device=device)  # 创建x坐标
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)  # 创建有效宽高张量
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)  # 计算宽高
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))  # 计算锚框的对数
        anchors = anchors.masked_fill(~valid_mask, float("inf"))  # 用无效值填充
        return anchors, valid_mask  # 返回锚框和有效掩码

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # 通过获取输入的投影特征处理并返回编码器输入
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]  # 获取投影特征
        # Get encoder inputs
        feats = []  # 初始化特征列表
        shapes = []  # 初始化形状列表
        for feat in x:  # 遍历特征
            h, w = feat.shape[2:]  # 获取高度和宽度
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))  # 将特征展平并调整维度
            # [nl, 2]
            shapes.append([h, w])  # 记录形状

        # [b, h*w, c]
        feats = torch.cat(feats, 1)  # 拼接特征
        return feats, shapes  # 返回特征和形状

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        # 从提供的特征和形状生成并准备解码器所需的输入
        bs = feats.shape[0]  # 获取批次大小
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)  # 生成锚框
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256  # 通过编码器输出处理特征

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)  # 获取编码器输出得分

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)  # 获取top k索引
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)  # 创建批次索引

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)  # 获取top k特征
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)  # 获取top k锚框

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors  # 计算参考边框

        enc_bboxes = refer_bbox.sigmoid()  # 对边框应用sigmoid
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)  # 如果有去噪边框，拼接
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)  # 获取编码器得分

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        # 如果学习初始查询，重复目标嵌入；否则使用top k特征
        if self.training:
            refer_bbox = refer_bbox.detach()  # 在训练时分离边框
            if not self.learnt_init_query:
                embeddings = embeddings.detach()  # 在训练时分离嵌入
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)  # 如果有去噪嵌入，拼接

        return embeddings, refer_bbox, enc_bboxes, enc_scores  # 返回嵌入、参考边框、编码边框和编码得分

    # TODO
    def _reset_parameters(self):
        """Reset module parameters."""
        # 重置模块参数
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc  # 初始化类偏置
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.sampling_offsets.weight.data, 0.0)  # 设置采样偏移的权重
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)  # 计算theta
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # 创建网格初始化
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])  # 归一化网格
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):  # 遍历每个点
            grid_init[:, :, i, :] *= i + 1  # 调整网格
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))  # 设置偏置
        constant_(self.attention_weights.weight.data, 0.0)  # 设置注意力权重
        constant_(self.attention_weights.bias.data, 0.0)  # 设置注意力偏置
        xavier_uniform_(self.value_proj.weight.data)  # 使用Xavier均匀分布初始化值投影权重
        constant_(self.value_proj.bias.data, 0.0)  # 设置值投影偏置
        xavier_uniform_(self.output_proj.weight.data)  # 使用Xavier均匀分布初始化输出投影权重
        constant_(self.output_proj.bias.data, 0.0)  # 设置输出投影偏置

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.
        # 执行多尺度可变形注意力的前向传播

        Args:
            query (torch.Tensor): [bs, query_length, C]
            # 查询张量，形状为[bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            # 参考边框，形状为[bs, query_length, n_levels, 2]，范围在[0, 1]，左上角为(0,0)，右下角为(1, 1)，包括填充区域
            value (torch.Tensor): [bs, value_length, C]
            # 值张量，形状为[bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            # 值形状列表，形状为[n_levels, 2]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements
            # 值掩码，形状为[bs, value_length]，非填充元素为True，填充元素为False

        Returns:
            output (Tensor): [bs, Length_{query}, C]
            # 输出张量，形状为[bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]  # 获取批次大小和查询长度
        len_v = value.shape[1]  # 获取值的长度
        assert sum(s[0] * s[1] for s in value_shapes) == len_v  # 确保值的长度与形状匹配

        value = self.value_proj(value)  # 通过值投影处理值
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))  # 用0填充掩码位置
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)  # 调整值的形状
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)  # 计算采样偏移
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)  # 计算注意力权重
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)  # 应用softmax计算注意力权重
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]  # 获取参考边框的点数
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)  # 创建偏移归一化器
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]  # 计算偏移
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add  # 计算采样位置
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5  # 计算偏移
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add  # 计算采样位置
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")  # 检查点数是否合法
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)  # 执行多尺度可变形注意力
        return self.output_proj(output)  # 通过输出投影处理输出


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.
    # 可变形变换器解码器层，灵感来自PaddleDetection和Deformable-DETR实现

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.0, act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        # 使用给定参数初始化可变形变换器解码器层
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)  # 创建自注意力层
        self.dropout1 = nn.Dropout(dropout)  # 创建dropout层
        self.norm1 = nn.LayerNorm(d_model)  # 创建层归一化层

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)  # 创建交叉注意力层
        self.dropout2 = nn.Dropout(dropout)  # 创建dropout层
        self.norm2 = nn.LayerNorm(d_model)  # 创建层归一化层

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)  # 创建前馈网络的第一层
        self.act = act  # 激活函数
        self.dropout3 = nn.Dropout(dropout)  # 创建dropout层
        self.linear2 = nn.Linear(d_ffn, d_model)  # 创建前馈网络的第二层
        self.dropout4 = nn.Dropout(dropout)  # 创建dropout层
        self.norm3 = nn.LayerNorm(d_model)  # 创建层归一化层

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        # 如果提供位置嵌入，则将其添加到输入张量
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        # 执行前馈网络部分的前向传播
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))  # 通过前馈网络处理目标
        tgt = tgt + self.dropout4(tgt2)  # 添加dropout
        return self.norm3(tgt)  # 返回归一化后的目标

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""
        # 执行整个解码器层的前向传播
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)  # 获取查询和键
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)  # 执行自注意力
        embed = embed + self.dropout1(tgt)  # 添加dropout
        embed = self.norm1(embed)  # 归一化

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )  # 执行交叉注意力
        embed = embed + self.dropout2(tgt)  # 添加dropout
        embed = self.norm2(embed)  # 归一化

        # FFN
        return self.forward_ffn(embed)  # 执行前馈网络的前向传播


class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.
    # 基于PaddleDetection实现的可变形变换器解码器

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        # 使用给定参数初始化可变形变换器解码器
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)  # 克隆解码器层
        self.num_layers = num_layers  # 解码器层数量
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx  # 评估索引

    def forward(
        self,
        embed,  # decoder embeddings
        refer_bbox,  # anchor
        feats,  # image features
        shapes,  # feature shapes
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        """Perform the forward pass through the entire decoder."""
        # 执行整个解码器的前向传播
        output = embed  # 初始化输出
        dec_bboxes = []  # 初始化解码边框列表
        dec_cls = []  # 初始化解码分类列表
        last_refined_bbox = None  # 初始化最后的精炼边框
        refer_bbox = refer_bbox.sigmoid()  # 对参考边框应用sigmoid
        for i, layer in enumerate(self.layers):  # 遍历解码器层
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))  # 执行层的前向传播

            bbox = bbox_head[i](output)  # 获取边框
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))  # 精炼边框

            if self.training:
                dec_cls.append(score_head[i](output))  # 获取分类得分
                if i == 0:
                    dec_bboxes.append(refined_bbox)  # 如果是第一层，添加精炼边框
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))  # 否则，使用上一个精炼边框
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))  # 获取分类得分
                dec_bboxes.append(refined_bbox)  # 添加精炼边框
                break

            last_refined_bbox = refined_bbox  # 更新最后的精炼边框
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox  # 更新参考边框

        return torch.stack(dec_bboxes), torch.stack(dec_cls)  # 返回堆叠的解码边框和分类得分