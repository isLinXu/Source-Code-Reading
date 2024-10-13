import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    """
    CLIP视觉塔模型，用于处理图像数据并提取特征。
    """
    def __init__(self, vision_tower, args, delay_load=False):
        """
        初始化函数，用于配置视觉塔模型的参数。

        :param vision_tower: 视觉塔的名称，用于识别和加载预训练模型。
        :param args: 包含各种配置参数的命名空间对象。
        :param delay_load: 是否延迟加载模型。如果为True，会在需要时才加载模型。
        """
        super().__init__()
        # 标记模型是否已加载，默认为False
        self.is_loaded = False
        # 视觉塔的名称和配置参数
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # 根据配置和是否延迟加载，决定是否在初始化时加载模型
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # 如果延迟加载且不需要解冻模型，则只加载配置
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        """
        加载视觉塔模型。

        :param device_map: 指定模型加载到的设备映射。
        """
        if self.is_loaded:
            # 如果模型已加载，打印提示信息并跳过加载
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # 加载图像处理器和视觉塔模型
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """
        从图像前向传播的输出中选择特定的特征。

        :param image_forward_outs: 图像前向传播的输出。
        :return: 选定的图像特征。
        """
        # 从隐藏状态中根据选择的层提取特征
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # 如果选择patch特征，则去除CLS token，保留patch特征
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            # 如果选择包含CLS的patch特征，则直接返回
            image_features = image_features
        else:
            # 如果选择的特征类型不支持，抛出错误
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        """
        对输入的图像数据进行前向传播计算，提取图像特征。

        参数:
        - images: 可以是单个图像张量或图像张量列表，表示输入的图像数据。

        返回:
        - image_features: 图像特征张量或图像特征张量列表，取决于输入images的类型。

        该方法首先检查输入的图像数据类型。如果输入的是一个图像张量列表，它将逐个图像进行前向传播，
        提取每个图像的特征，并将这些特征收集到一个列表中返回。如果输入的是单个图像张量，它将直接进行前向传播，
        提取图像特征，并返回这个特征张量。
        """
        # 当输入的图像数据是列表时，对每个图像进行处理，并收集每个图像的特征
        if type(images) is list:
            image_features = []
            for image in images:
                # 将图像数据转移到指定的设备和类型，并增加一个维度以适应模型输入要求
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                # 选择并提取图像特征
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                # 将提取的图像特征添加到特征列表中
                image_features.append(image_feature)
        else:
            # 当输入的图像数据是单个张量时，直接进行处理
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # 提取图像特征并转换回输入图像的数据类型
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        # 返回图像特征，类型取决于输入数据的类型
        return image_features

    @property
    def dummy_feature(self):
        """
        生成一个虚拟的特征张量。

        这个方法主要用于在模型初始化或者测试时生成一个固定大小的虚拟特征张量，
        以便进行后续的操作或者测试。该方法返回的张量大小为1xhidden_size，类型和设备
        与当前实例的配置一致。

        Returns:
            torch.Tensor: 大小为1xhidden_size的零张量，类型和设备与实例配置一致。
        """
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """
        返回视觉塔的数据类型。

        :return: 视觉塔的数据类型
        """
        return self.vision_tower.dtype

    @property
    def device(self):
        """
        返回视觉塔所在的设备，例如CPU或GPU。

        :return: 视觉塔所在的设备
        """
        return self.vision_tower.device

    @property
    def config(self):
        """
        根据视觉塔是否已加载，返回相应的配置。
        如果视觉塔已加载，返回其配置；否则，返回备份配置。

        :return: 视觉塔的配置或备份配置
        """
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """
        返回配置中的隐藏层大小。

        :return: 配置中的隐藏层大小
        """
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        """
        计算并返回图像每一面（宽度和高度）的补丁数量。
        通过将图像尺寸除以补丁尺寸得到。

        :return: 每一面的补丁数量
        """
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        """
        计算并返回图像总的补丁数量。
        通过计算每一面的补丁数量的平方得到。

        :return: 总的补丁数量
        """
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    """
    CLIPVisionTowerS2类继承自CLIPVisionTower，用于对图像进行多尺度处理。

    该类的主要功能是在预处理阶段对图像进行多尺度调整，并使用特定的前向传播函数来处理这些不同尺度的图像。

    参数:
    - vision_tower: CLIPVisionTower的实例，通常用于图像编码。
    - args: 包含额外配置的命名空间对象，例如多尺度设置。
    - delay_load: 是否延迟加载部分组件，默认为False。
    """
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)
        # 初始化多尺度设置
        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        # 尝试导入多尺度前向传播函数
        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        # 更新图像预处理大小，使其与s2_scale中最大的图像尺寸相匹配
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        """
        加载模型。

        参数:
            device_map (str, optional): 设备映射，用于指定模型加载的设备。默认为None。

        返回:
            None

        说明:
            此方法用于加载图像处理器和视觉塔模型。如果模型已经加载，则不会重复加载。
            它首先检查模型是否已经加载，然后初始化图像处理器和视觉塔模型。
            可以通过device_map参数指定模型加载到特定设备。最后，更新图像处理器的配置，
            并设置模型为不可训练状态（不需要梯度）。
        """
        # 检查模型是否已经加载，如果已经加载，则打印消息并跳过加载
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        # 初始化图像处理器
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # 初始化视觉塔模型，并加载到指定设备
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # 设置模型为不需要梯度状态
        self.vision_tower.requires_grad_(False)

        # 更新图像处理器的配置
        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size
        # 设置模型为已加载状态
        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        """
        通过视觉塔处理图像并提取特征。

        参数:
        - images: 输入的图像数据，需要转换到模型的设备和数据类型中。

        返回:
        - image_features: 提取的图像特征，数据类型与输入图像相同。
        """
        # 通过视觉塔前向传播图像数据，获取包括隐藏状态的输出
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        # 从视觉塔的输出中选择并提取图像特征
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        """
        对输入的单图像或图像列表进行多尺度特征提取。

        参数:
        - images: 单个图像或图像列表，是特征提取的输入。

        返回:
        - image_features: 多尺度提取的图像特征。
        """
        # 判断输入是否为图像列表，如果是，对每个图像分别进行特征提取
        if type(images) is list:
            image_features = []
            for image in images:
                # 使用多尺度方法对单个图像进行特征提取
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            # 如果输入不是列表，则直接对输入进行多尺度特征提取
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        """
        计算并返回隐藏层的总尺寸。

        返回:
        - 隐藏层的总尺寸，等于配置的隐藏尺寸乘以多尺度的数量。
        """
        # 返回隐藏层总尺寸
        return self.config.hidden_size * len(self.s2_scales)
