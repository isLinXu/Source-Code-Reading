import torch  # 导入PyTorch库
from PIL import Image  # 从PIL导入Image模块，用于图像处理


def load_image(filename, size=None, scale=None):  # 定义加载图像的函数
    img = Image.open(filename).convert('RGB')  # 打开图像并转换为RGB模式
    if size is not None:  # 如果指定了大小
        img = img.resize((size, size), Image.ANTIALIAS)  # 调整图像大小
    elif scale is not None:  # 如果指定了缩放因子
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)  # 根据缩放因子调整图像大小
    return img  # 返回加载的图像


def save_image(filename, data):  # 定义保存图像的函数
    img = data.clone().clamp(0, 255).numpy()  # 克隆数据并限制在0到255之间，然后转换为NumPy数组
    img = img.transpose(1, 2, 0).astype("uint8")  # 转换数组的维度并转换为无符号8位整数
    img = Image.fromarray(img)  # 从数组创建图像
    img.save(filename)  # 保存图像


def gram_matrix(y):  # 定义计算Gram矩阵的函数
    (b, ch, h, w) = y.size()  # 获取输入的批次大小、通道数、高度和宽度
    features = y.view(b, ch, w * h)  # 将输入展平为特征矩阵
    features_t = features.transpose(1, 2)  # 转置特征矩阵
    gram = features.bmm(features_t) / (ch * h * w)  # 计算Gram矩阵并进行归一化
    return gram  # 返回Gram矩阵


def normalize_batch(batch):  # 定义归一化批次的函数
    # normalize using imagenet mean and std  # 使用ImageNet的均值和标准差进行归一化
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)  # 创建均值张量
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)  # 创建标准差张量
    batch = batch.div_(255.0)  # 将批次数据除以255进行归一化
    return (batch - mean) / std  # 返回归一化后的批次数据