import ast
import math
from PIL import Image

def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    计算任意分辨率图像预处理后的图像块网格形状。

    参数:
        image_size (tuple): 输入图像的大小，格式为 (宽度, 高度)。
        grid_pinpoints (str): 可能分辨率的字符串表示形式的列表。
        patch_size (int): 每个图像块的大小。

    返回:
        tuple: 图像块网格的形状，格式为 (宽度, 高度)。
    """
    # 如果grid_pinpoints是列表类型，则直接使用
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        # 否则将字符串转换为列表
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    # 选择最佳分辨率
    width, height = select_best_resolution(image_size, possible_resolutions)
    # 返回图像块网格的形状
    return width // patch_size, height // patch_size

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).

    根据原始尺寸从可能的分辨率列表中选择最佳分辨率。

    参数:
        original_size (tuple): 图像的原始尺寸，格式为 (宽度, 高度)。
        possible_resolutions (list): 可能的分辨率列表，格式为 [(宽度1, 高度1), (宽度2, 高度2), ...]。

    返回:
        tuple: 最佳适配分辨率，格式为 (宽度, 高度)
    """
    original_width, original_height = original_size                                                                                                                 # 解构原始尺寸
    best_fit = None                                                                                                                                                 # 初始化最佳适配为None
    max_effective_resolution = 0                                                                                                                                    # 初始化最大有效分辨率为0
    min_wasted_resolution = float('inf')                                                                                                                            # 初始化最小浪费分辨率为无穷大

    for width, height in possible_resolutions:                                                                                                                      # 遍历所有可能的分辨率
        scale = min(width / original_width, height / original_height)                                                                                               # 计算缩放比例
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)                                                             # 计算缩放后的尺寸
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)                                                          # 计算有效分辨率
        wasted_resolution = (width * height) - effective_resolution                                                                                                 # 计算浪费的分辨率

        # 更新最佳适配分辨率的条件
        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution         # 更新最大有效分辨率
            min_wasted_resolution = wasted_resolution               # 更新最小浪费分辨率
            best_fit = (width, height)                              # 更新最佳适配分辨率

    return best_fit                                                 # 返回最佳适配分辨率

## added by llava-1.6
def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.

    将图像分割成指定大小的补丁。

    参数:
        image (PIL.Image.Image): 输入图像。
        patch_size (int): 每个补丁的大小。

    返回:
        list: 代表补丁的 PIL.Image.Image 对象列表。
    """
    patches = []                                                # 初始化一个空列表用于存储补丁
    width, height = image.size                                  # 获取图像的宽度和高度
    # 以patch_size为步长，遍历图像的高度
    for i in range(0, height, patch_size):
        # 以patch_size为步长，遍历图像的宽度
        for j in range(0, width, patch_size):
            # 定义一个矩形框，用于裁剪图像
            box = (j, i, j + patch_size, i + patch_size)
            # 使用矩形框裁剪图像，得到一个补丁
            patch = image.crop(box)
            # 将补丁添加到列表中
            patches.append(patch)

    return patches                                              # 返回包含所有补丁的列表

## added by llava-1.6
def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.

    调整图像大小并填充到目标分辨率，同时保持宽高比。
    Args:
        image (PIL.Image.Image): 输入的图像。
        target_resolution (tuple): 目标分辨率 (宽度, 高度)。

    Returns:
        PIL.Image.Image: 调整大小并填充后的图像。
    """
    original_width, original_height = image.size                                    # 获取原始图像的宽度和高度
    target_width, target_height = target_resolution                                 # 获取目标分辨率的宽度和高度

    scale_w = target_width / original_width                                         # 计算宽度缩放比例
    scale_h = target_height / original_height                                       # 计算高度缩放比例
    # 根据宽度和高度的缩放比例，决定新的宽度和高度
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image  # 调整图像大小
    resized_image = image.resize((new_width, new_height))
    # 创建一个新的图像，大小为目标分辨率，背景色为黑色
    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2                                       # 计算粘贴位置的x坐标
    paste_y = (target_height - new_height) // 2                                     # 计算粘贴位置的y坐标
    new_image.paste(resized_image, (paste_x, paste_y))                              # 将调整大小后的图像粘贴到新图像中心位置

    return new_image                                                                # 返回调整大小并填充后的图像

def get_value_from_kwargs(kwargs, name):
    """
    从kwargs字典中获取指定名称的值，并从字典中移除该键值对。

    参数:
    kwargs (dict): 包含键值对的字典。
    name (str): 需要获取值的键名称。

    返回:
    object: 如果键存在于字典中，则返回对应的值；否则返回None。
    """
    if name in kwargs:                          # 检查键是否存在于字典中
        return kwargs.pop(name)                 # 获取值并从字典中移除键值对
    else:
        return None                             # 键不存在时返回None