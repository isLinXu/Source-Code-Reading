#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:
    """
    Llava的元模型类，用于处理与多模态相关的模型组件。

    参数:
        config: 配置对象，包含模型的各种配置参数。

    返回:
        None
    """
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # 根据配置构建视觉塔和投影器，如果配置中包含mm_vision_tower属性
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            # 如果配置中的mm_patch_merge_type包含'unpad'，则初始化
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        """
        获取视觉塔模型。

        参数:
            无

        返回:
            vision_tower: 视觉塔模型实例，如果存在的话。
        """
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        初始化视觉模块。

        参数:
            model_args: 模型参数对象，用于构建视觉塔和投影器。
            fsdp: 可选参数，用于处理分布式训练。

        返回:
            None
        """
        # 从模型参数中提取视觉塔和相关配置
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        # 配置视觉塔
        self.config.mm_vision_tower = vision_tower

        # 检查并构建视觉塔
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            # 根据fsdp配置视觉塔
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            # 如果视觉塔已存在，根据fsdp获取视觉塔实例
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            # 加载视觉塔模型
            vision_tower.load_model()

        # 配置多模态投影器
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        # 构建或重置多模态投影器
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            # 初始化newline参数
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # 解冻投影器参数
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # 预训练多模态MLP适配器
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.

    恢复原始尺寸的图像。

    该函数用于将填充后的图像张量恢复到其原始尺寸。它通过计算原始尺寸和当前张量尺寸之间的比例，
    然后根据比例缩放图像并去除多余的填充来实现这一点。

    参数:
    tensor: Tensor, 填充后的图像张量。
    original_size: Tuple[int, int], 图像的原始尺寸 (宽度, 高度)。

    返回:
    unpadded_tensor: Tensor, 恢复到原始尺寸的图像张量。
    """
    # 获取原始图像尺寸
    original_width, original_height = original_size
    # 获取当前图像张量的尺寸
    current_height, current_width = tensor.shape[1:]

    # 计算原始图像的宽高比
    original_aspect_ratio = original_width / original_height
    # 计算当前张量的宽高比
    current_aspect_ratio = current_width / current_height

    # 根据宽高比判断应该如何去除填充
    if original_aspect_ratio > current_aspect_ratio:
        # 如果原始图像的宽高比大于当前张量的宽高比，根据宽度计算缩放因子
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        # 计算上下边的填充量，并去除
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        # 如果原始图像的宽高比小于等于当前张量的宽高比，根据高度计算缩放
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        # 计算左右边的填充量，并去除
        unpadded_tensor = tensor[:, :, padding:current_width - padding]
    # 返回去除填充后的图像张
    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    """
    这个类定义了一个具有视觉能力的因果语言模型的抽象基类。
    它旨在被具体实现的子类继承以提供详细的行为。
    """
    @abstractmethod
    def get_model(self):
        """
        抽象方法，用于获取具有视觉能力的语言模型实例。
        此方法应由子类实现以返回具体的模型实例。

        返回:
            具有视觉能力的语言模型的实现。
        """
        pass

    def get_vision_tower(self):
        """
        获取模型的视觉塔组件。
        此方法通过调用模型的 get_vision_tower 方法来返回其视觉塔组件。

        返回:
            模型的视觉塔组件。
        """
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """
        编码图像特征。

        该方法首先通过视觉塔模型提取图像的基础特征，然后通过一个多模态投影器将这些特征转换到另一个空间，
        以便与其它模态的数据（如文本）进行融合或比较。

        参数:
        images: 输入的图像数据，可以是图像的张量或类似对象。

        返回:
        编码后的图像特征。
        """
        # 通过视觉塔模型提取图像特征
        image_features = self.get_model().get_vision_tower()(images)

        # 通过多模态投影器转换图像特征
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        """
        准备多模态模型的输入和标签数据。

        此函数的主要作用是处理输入数据，使其适应多模态模型的要求。特别是对图像数据进行预处理，
        如编码和重塑，以适应模型的架构和计算需求。

        参数:
        - self: 实例对象。
        - input_ids: 文本输入的标识。
        - position_ids: 位置标识。
        - attention_mask: 注意力掩码，用于指示哪些位置应该被模型忽略。
        - past_key_values: 之前的键值对，用于加速解码过程。
        - labels: 输入标签。
        - images: 图像数据，可以是单个图像或图像列表。
        - image_sizes: 图像的尺寸信息，如果提供，将用于图像处理中。

        返回:
        - input_ids, position_ids, attention_mask, past_key_values, 处理后的图像特征, labels
        """
        # 获取视觉塔实例
        vision_tower = self.get_vision_tower()
        # 如果视觉塔不存在，或者图像数据不存在，或者文本输入的序列长度为1，则直接返回原始输入数据
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # 对图像数据进行处理
        if type(images) is list or images.ndim == 5:
            # 如果图像是列表形式，或者维度为5，则进行特定的处理
            if type(images) is list:
                # 对列表中的每个图像进行处理，确保其维度正确
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            # 将所有图像连接在一起，以便进行批量处理
            concat_images = torch.cat([image for image in images], dim=0)
            # 对图像进行编码，提取特征
            image_features = self.encode_images(concat_images)
            # 根据每个图像的尺寸，将图像特征分割开
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # 根据配置决定如何合并图像补丁特征
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            # 根据配置决定如何合并图像补丁特征
            if mm_patch_merge_type == 'flat':
                # 如果是'flat'类型，直接将图像特征展平
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                # 如果是'spatial'类型，进行空间维度的处理
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        # 对每个图像特征进行处理，包括重塑和可能的去垫
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            # 根据任意分辨率的图像网格形状重塑图像特征
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            # 如果包含'unpad'操作，对图像特征进行去垫处理
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            # 如果不包含'unpad'操作，直接重塑图像特征
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            # 如果包含'unpad'操作，添加newline token并进行去垫处
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                # 如果合并类型不是预期的类型，抛出错误
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # 如果图像数据不是列表形式，直接进行编码
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        # 如果未提供 position_ids，则生成 position_ids
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        # 如果未提供 labels，则生成 labels，并填充 IGNORE_INDEX
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # 使用 attention_mask 去除 padding -- FIXME: 这是一个待优化的占位符
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # 处理批量中的每个序列
        # 遍历输入ID列表，处理每个批次的数据
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 计算当前批次中图像令牌的数量，以确定图像的数量
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # 如果当前批次中没有图像令牌，则直接处理并继续下一个批次
            if num_images == 0:
                # 如果没有图像，直接嵌入 input_ids
                cur_image_features = image_features[cur_image_idx]
                # 将当前批次的输入ID嵌入到模型中，获取嵌入表示
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # 将输入嵌入和图像特征进行拼接，即使图像特征为空
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                # 将处理后的输入嵌入和标签添加到新的列表中
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                # 更新图像索引，准备处理下一个批次的图像
                cur_image_idx += 1
                # 继续处理下一个批次
                continue

            # 如果存在图像，处理 input_ids 和 labels
            # 在处理输入数据时，首先定位所有图像令牌的索引位置，并将其作为分割点
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

            # 初始化不包含图像令牌的输入ID和标签列表
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            # 根据图像令牌的索引，分割输入ID和标签，排除图像令牌
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            # 计算每个分割后标签的长度，用于后续的重新组合
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            # 将分割后的输入ID转换为嵌入表示
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))

            # 根据每个分割的长度，将输入嵌入分割开
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            # 初始化新的输入嵌入和标签列表，用于存储处理后的数据
            cur_new_input_embeds = []
            cur_new_labels = []

            # 遍历每个分割，并根据需要添加图像特征和对应的忽略标签
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            # 将新的输入嵌入转移到适当的设备上
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # 将所有新的输入嵌入和标签连接起来，形成最终的输入和标签数据
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # 将处理后的数据添加到总的输入和标签列表中
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # 将序列截断到最大长度，因为图像嵌入可能会使序列变长
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        # 合并它们
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        # 初始化填充后的输入嵌入、标签、注意力掩码和位置ID
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # 遍历输入嵌入和标签，进行填充
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            # 根据配置决定填充的方式（左侧或右侧）
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                # 左侧填充
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # 右侧填充
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        # 将填充后的输入嵌入堆叠成批次
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # 根据传入的参数决定是否返回标签和注意力掩码
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        # 返回处理后的数据
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """
        初始化视觉令牌化器。

        根据模型参数配置，向令牌化器中添加新的令牌，用于处理图像输入，并相应地调整嵌入层的大小和训练配置。

        参数:
        - model_args: 模型参数配置对象，包含与令牌化相关的设置。
        - tokenizer: 文本令牌化器对象，将被添加新的令牌。

        返回值:
        无返回值，但会修改tokenizer和模型的内部状态。
        """
        # 如果配置了使用图像补丁令牌，则添加默认的图像补丁令牌，并调整令牌嵌入层的大小。
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        # 如果配置了使用图像开始和结束令牌，则添加默认的图像开始和结束令牌，并调整令牌嵌入层的大小。
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # 当有新的令牌数大于0时，意味着需要对输入和输出嵌入层进行扩展
            if num_new_tokens > 0:
                # 获取输入嵌入层的权重数据
                input_embeddings = self.get_input_embeddings().weight.data
                # 获取输出嵌入层的权重数据
                output_embeddings = self.get_output_embeddings().weight.data

                # 计算输入嵌入层中已有令牌的平均权重
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                # 计算输出嵌入层中已有令牌的平均权重
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                # 将新令牌的输入嵌入权重设置为已有令牌的平均值
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                # 将新令牌的输出嵌入权重设置为已有令牌的平均值
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # 如果配置了调整MLP适配器，则设置输入和输出嵌入层的梯度要求。
            # 将输入嵌入层的参数设为可训练，将输出嵌入层的参数设为不可训练。
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            # 如果配置了预训练MLP适配器，则加载预训练权重并应用到输入嵌入层。
            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                # 断言新增的token数量为2，以确保后续的赋值操作不会出错
                assert num_new_tokens == 2
                # 根据input_embeddings和embed_tokens_weight的形状来决定如何赋值
                if input_embeddings.shape == embed_tokens_weight.shape:
                    # 如果形状完全相同，则直接复制最后num_new_tokens个权重
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    # 如果embed_tokens_weight的token数量与新增的token数量相同，则直接赋值
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    # 如果两者的形状都不匹配，则抛出错误
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        # 如果仅使用图像补丁令牌且配置了调整MLP适配器，则冻结输入和输出嵌入层的梯度。
        elif model_args.mm_use_im_patch_token:
            # 如果调整MLP适配器
            if model_args.tune_mm_mlp_adapter:
                # 冻结输入嵌入层的参数
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                # 冻结输出嵌入层的参数
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
