from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    选择最佳分辨率。根据原始尺寸从一组可能的分辨率中选择最佳分辨率。

    参数:
        original_size (tuple): 图像的原始尺寸，格式为 (width, height)。
        possible_resolutions (list): 可能的分辨率列表，格式为 [(width1, height1), (width2, height2), ...]。

    返回:
        tuple: 最适合的分辨率，格式为 (width, height)。
    """
    # 解析原始尺寸
    original_width, original_height = original_size
    # 初始化最佳分辨率变量
    best_fit = None
    # 初始化最大有效分辨率和最小浪费分辨率变量
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')
    # 遍历可能的分辨率列表
    for width, height in possible_resolutions:
        # 计算缩放比例，确保图像不失真
        scale = min(width / original_width, height / original_height)
        # 计算缩小后的图像尺寸
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        # 计算有效分辨率，即缩小后的分辨率和原始分辨率的较小值
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        # 计算浪费的分辨率，即总分辨率减去有效分辨率
        wasted_resolution = (width * height) - effective_resolution
        # 更新最大有效分辨率和最小浪费分辨率以及最佳分辨率
        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)
    # 返回最佳分辨率
    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    按照目标分辨率调整并填充图像，同时保持原始宽高比。

    参数:
        image (PIL.Image.Image): 输入图像。
        target_resolution (tuple): 目标分辨率 (宽度, 高度)。

    返回:
        PIL.Image.Image: 调整大小并填充后的图像。
    """
    # 获取原始图像尺寸
    original_width, original_height = image.size
    # 获取目标分辨率尺寸
    target_width, target_height = target_resolution
    # 计算宽度和高度的缩放比例
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    # 根据保持宽高比确定使用的缩放比例
    if scale_w < scale_h:
        # 如果宽度的缩放比例较小，则根据宽度进行缩放
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # 如果高度的缩放比例较小或相等，则根据高度进行缩放
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    # 缩放图像
    resized_image = image.resize((new_width, new_height))
    # 创建一个新的图像，目标分辨率为黑色背景
    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    # 计算粘贴缩放后图像的位置以居中
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    # 将缩放后的图像粘贴到新图像上
    new_image.paste(resized_image, (paste_x, paste_y))
    # 返回调整大小并填充后的图像
    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    将图像分割为指定大小的块。

    该函数接收一个PIL图像对象和一个块大小作为参数，然后将图像分割为多个非重叠的块，
    每个块的大小为patch_size x patch_size。函数返回一个包含所有块的列表。

    Args:
        image (PIL.Image.Image): 输入的PIL图像对象。
        patch_size (int): 每个块的大小。

    Returns:
        list: 一个包含所有块的列表，每个块都是一个PIL.Image.Image对象。
    """
    # 初始化一个空列表，用于存储分割后的块
    patches = []
    # 获取图像的宽度和高度
    width, height = image.size
    # 遍历图像的每一个可能的起始位置，步长为patch_size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            # 计算当前块的边界框
            box = (j, i, j + patch_size, i + patch_size)
            # 使用边界框从图像中裁剪出当前块
            patch = image.crop(box)
            # 将当前块添加到列表中
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    计算任意分辨率图像预处理后的图像补丁网格形状。

    参数:
        image_size (tuple): 输入图像的大小，格式为 (宽度, 高度)。
        grid_pinpoints (str): 可能的分辨率列表的字符串表示形式。
        patch_size (int): 每个图像补丁的大小。

    返回:
        tuple: 图像补丁网格的形状，格式为 (宽度, 高度)。
    """
    # 确定 grid_pinpoints 的格式，并在必要时将其转换为列表
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    # 从可能的分辨率中选择最合适的分辨率
    width, height = select_best_resolution(image_size, possible_resolutions)
    # 计算并返回图像补丁网格的形状
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.

    处理具有可变分辨率的图像。

    该函数根据提供的可能分辨率列表，选择最适合的分辨率来调整输入图像的大小。
    然后将图像分割成多个补丁，并按处理器要求的尺寸重新调整原始图像大小。
    最后，预处理所有图像补丁并以张量形式返回。

    参数:
        image (PIL.Image.Image): 要处理的输入图像。
        processor: 图像处理器对象，用于预处理图像并获取裁剪和尺寸信息。
        grid_pinpoints (str): 可能分辨率的字符串表示。

    返回:
        torch.Tensor: 包含处理后的图像补丁的张量。
    """
    # 根据grid_pinpoints的类型确定可能的分辨率列表
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    # 根据图像大小和可能的分辨率列表选择最佳分辨率
    best_resolution = select_best_resolution(image.size, possible_resolutions)

    # 将图像调整到最佳分辨率并填充
    image_padded = resize_and_pad_image(image, best_resolution)

    # 根据处理器指定的裁剪高度将填充后的图像分割成补丁
    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    # 将原始图像调整到处理器的最短边尺寸
    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    # 将调整大小后的原始图像与补丁合并成一个列表以统一处理
    image_patches = [image_original_resize] + patches

    # 预处理每个图像补丁并转换为像素值张量
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    # 沿新维度堆叠所有图像补丁并作为张量返回
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    """
    从Base64编码的字符串加载图像。

    参数:
    image(str): Base64编码的图像字符串。

    返回:
    Image: 加载的图像对象。
    """
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    """
    将给定的PIL图像对象转换为正方形。

    如果图像本身就是正方形，则直接返回。如果图像不是正方形，
    则在必要的一侧添加背景颜色以使其成为正方形。

    参数:
    pil_img(Image): 需要转换的PIL图像对象。
    background_color(str或tuple): 用于填充的背景颜色，可以是颜色名称或RGB元组。

    返回:
    Image: 转换后的正方形图像对象。
    """
    # 获取PIL图像的宽度和高度
    width, height = pil_img.size
    # 判断图像的宽度和高度是否相等
    if width == height:
        # 如果宽度和高度相等，则直接返回原图像
        return pil_img
    elif width > height:
        # 如果宽度大于高度，创建一个新的图像，其宽度与原图像相同，高度填充背景色
        result = Image.new(pil_img.mode, (width, width), background_color)
        # 将原图像粘贴到新图像的中心位置，垂直居中
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        # 如果高度大于宽度，创建一个新的图像，其高度与原图像相同，宽度填充背景色
        result = Image.new(pil_img.mode, (height, height), background_color)
        # 将原图像粘贴到新图像的中心位置，水平居中
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    """
    处理一组图像，使其满足模型配置中指定的宽高比要求。

    此函数根据模型的配置处理图像数据，以确保图像的宽高比符合模型的要求。
    它支持不同的宽高比处理方式，包括填充.square)和任意分辨率(anyres)。
    如果图像已经符合要求的格式，则直接返回处理后的图像张量。

    参数:
    - images: List[Image]，需要处理的原始图像列表。
    - image_processor: ImageProcessor，用于处理图像的对象，包含图像预处理方法。
    - model_cfg: ModelConfig，模型配置对象，包含模型所需的图像宽高比等信息。

    返回:
    - torch.Tensor，处理后的图像数据张量。
    """
    # 获取模型配置中的图像宽高比设置
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)

    # 初始化存储处理后图像的列表
    new_images = []

    # 如果模型配置要求使用填充方式处理图像宽高比
    if image_aspect_ratio == 'pad':
        for image in images:
            # 将图像填充为正方形，使用图像处理器的平均颜色作为背景颜色填充
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            # 对图像进行预处理，转换为模型所需的张量格式
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # 将处理后的图像添加到列表中
            new_images.append(image)

    # 如果模型配置允许任意分辨率的图像
    elif image_aspect_ratio == "anyres":
        for image in images:
            # 根据模型配置中的网格点处理图像
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            # 将处理后的图像添加到列表中
            new_images.append(image)
    # 如果模型配置中没有指定特殊宽高比处理方式
    else:
        # 直接使用图像处理器对所有图像进行批量预处理，并返回张量
        return image_processor(images, return_tensors='pt')['pixel_values']
    # 如果所有处理后的图像形状相同
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    # 返回处理后的图像数据
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """
    对包含图像标记的提示进行 tokenization。

    该函数的目的是处理一个包含特殊 <image> 标记的提示字符串，将其转换为模型可以处理的 token 序列。
    它首先将提示字符串分割成若干部分，然后对每一部分进行 tokenization，并在每部分之间插入图像 token，
    最后根据需要返回不同格式的 token 序列。

    参数:
    - prompt: 待 tokenization 的提示字符串，可能包含 <image> 标记。
    - tokenizer: 用于 tokenization 的工具，通常是 Hugging Face 的 tokenizer 对象。
    - image_token_index: 图像 token 在词汇表中的索引，默认为 IMAGE_TOKEN_INDEX。
    - return_tensors: 返回的 tensor 类型，支持 'pt'（PyTorch）或其他，None 表示不返回 tensor。

    返回:
    - tokenized_input_ids: token 化后的输入 ID 列表或根据 return_tensors 指定的 tensor 类型。
    """
    # 将提示字符串分割成若干部分，每部分进行 tokenization
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        """
        在序列 X 的每个元素之间插入分隔符 sep。

        用于在 token 之间插入特殊的 token（如图像 token）。
        """
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    # 如果提示的开始有 BOS（开始）token，则保留这个 token
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    # 在 token 化的提示部分之间插入图像 token
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    # 根据需要返回不同格式的 token 序列
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    """
    从模型路径中提取模型名称。

    该函数首先去除路径末尾的斜杠（如果有的话），然后根据'/'分割路径，
    并根据分割后的最后一段路径名是否以'checkpoint-'开头来决定如何组合模型名称。
    如果是，返回前两个部分的组合；如果不是，直接返回最后一段作为模型名称。

    参数:
    model_path(str): 模型的完整路径。

    返回:
    str: 提取或组合后的模型名称。
    """
    # 去除路径末尾的斜杠，以避免在处理路径时出现不必要的错误
    model_path = model_path.strip("/")
    # 使用'/'分割路径，获取所有部分
    model_paths = model_path.split("/")
    # 检查路径的最后一部分是否以'checkpoint-'开头，如果是，则返回前两个部分的组合
    if model_paths[-1].startswith('checkpoint-'):
        # 如果是，组合倒数第二部分和最后一部分作为模型名称
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        # 如果不是，直接使用最后一部分作为模型名称
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    """
    基于关键词的停止标准类。该类用于在文本生成过程中，
    当生成的文本包含指定的关键词时停止生成。

    参数:
    - keywords: 指定的关键词列表，用于停止条件的判断。
    - tokenizer: 用于文本编码和解码的工具。
    - input_ids: 输入文本的编码表示。
    """
    def __init__(self, keywords, tokenizer, input_ids):
        # 初始化关键词和相关配置
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            # 跳过可能的开始标记
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            # 更新最大关键词长度
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        检查一批输出中是否包含关键词，如果有则停止生成。

        参数:
        - output_ids: 一批输出的编码表示。
        - scores: 当前输出的分数，未使用但必须包含在签名中以符合接口要求。

        返回:
        - bool: 如果输出中包含关键词，则返回True，否则返回False。
        """
        # 考虑到输出长度，计算一个截断偏移量
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        # 确保关键词ids在正确的设备上
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        # 检查是否生成了关键词
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        # 解码输出并检查关键词是否在输出中
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        对于一批输出，检查每个输出中是否包含关键词，如果有则停止生成。

        参数:
        - output_ids: 一批输出的编码表示。
        - scores: 当前输出的分数，未使用但必须包含在签名中以符合接口要求。

        返回:
        - bool: 如果所有输出中都不包含关键词，则返回True，否则返回False。
        """
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
