from __future__ import division  # 从 __future__ 导入 division，以确保除法行为与 Python 3 一致
from torchvision import models  # 从 torchvision 导入模型模块
from torchvision import transforms  # 从 torchvision 导入数据转换模块
from PIL import Image  # 从 PIL 导入图像处理模块
import argparse  # 导入 argparse 库，用于处理命令行参数
import torch  # 导入 PyTorch 库
import torchvision  # 导入 torchvision 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 配置设备，使用 GPU（如果可用）或 CPU

def load_image(image_path, transform=None, max_size=None, shape=None):  # 定义加载图像的函数
    """Load an image and convert it to a torch tensor."""  # 加载图像并将其转换为 PyTorch 张量
    image = Image.open(image_path)  # 打开图像文件
    
    if max_size:  # 如果提供了最大尺寸
        scale = max_size / max(image.size)  # 计算缩放比例
        size = np.array(image.size) * scale  # 根据缩放比例调整图像尺寸
        image = image.resize(size.astype(int), Image.ANTIALIAS)  # 调整图像大小并使用抗锯齿
    
    if shape:  # 如果提供了目标形状
        image = image.resize(shape, Image.LANCZOS)  # 调整图像到指定形状
    
    if transform:  # 如果提供了转换
        image = transform(image).unsqueeze(0)  # 应用转换并增加一个维度以适应模型输入
    
    return image.to(device)  # 返回处理后的图像并移动到指定设备


class VGGNet(nn.Module):  # 定义 VGG 网络类
    def __init__(self):  # 初始化方法
        """Select conv1_1 ~ conv5_1 activation maps."""  # 选择 conv1_1 到 conv5_1 的激活图
        super(VGGNet, self).__init__()  # 调用父类的初始化方法
        self.select = ['0', '5', '10', '19', '28']  # 选择的层索引
        self.vgg = models.vgg19(pretrained=True).features  # 加载预训练的 VGG19 模型的特征提取部分
        
    def forward(self, x):  # 前向传播方法
        """Extract multiple convolutional feature maps."""  # 提取多个卷积特征图
        features = []  # 初始化特征列表
        for name, layer in self.vgg._modules.items():  # 遍历 VGG 模型的每一层
            x = layer(x)  # 通过当前层处理输入
            if name in self.select:  # 如果当前层在选择的层中
                features.append(x)  # 将特征添加到列表中
        return features  # 返回提取的特征


def main(config):  # 主函数
    # Image preprocessing
    # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We use the same normalization statistics here.
    transform = transforms.Compose([  # 定义图像预处理步骤
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=(0.485, 0.456, 0.406),  # 进行归一化处理
                             std=(0.229, 0.224, 0.225))])  # 使用 ImageNet 数据集的均值和标准差
    
    # Load content and style images
    # Make the style image same size as the content image
    content = load_image(config.content, transform, max_size=config.max_size)  # 加载并处理内容图像
    style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])  # 加载并处理样式图像，使其与内容图像相同大小
    
    # Initialize a target image with the content image
    target = content.clone().requires_grad_(True)  # 使用内容图像初始化目标图像，并设置为需要梯度计算
    
    optimizer = torch.optim.Adam([target], lr=config.lr, betas=[0.5, 0.999])  # 定义优化器为 Adam
    vgg = VGGNet().to(device).eval()  # 创建 VGG 网络实例并设置为评估模式
    
    for step in range(config.total_step):  # 遍历总步数
        
        # Extract multiple(5) conv feature vectors
        target_features = vgg(target)  # 提取目标图像的特征
        content_features = vgg(content)  # 提取内容图像的特征
        style_features = vgg(style)  # 提取样式图像的特征

        style_loss = 0  # 初始化样式损失
        content_loss = 0  # 初始化内容损失
        for f1, f2, f3 in zip(target_features, content_features, style_features):  # 遍历特征
            # Compute content loss with target and content images
            content_loss += torch.mean((f1 - f2)**2)  # 计算内容损失

            # Reshape convolutional feature maps
            _, c, h, w = f1.size()  # 获取特征图的尺寸
            f1 = f1.view(c, h * w)  # 重塑特征图
            f3 = f3.view(c, h * w)  # 重塑样式特征图

            # Compute gram matrix
            f1 = torch.mm(f1, f1.t())  # 计算 Gram 矩阵
            f3 = torch.mm(f3, f3.t())  # 计算样式 Gram 矩阵

            # Compute style loss with target and style images
            style_loss += torch.mean((f1 - f3)**2) / (c * h * w)  # 计算样式损失 
        
        # Compute total loss, backprop and optimize
        loss = content_loss + config.style_weight * style_loss  # 计算总损失
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        if (step+1) % config.log_step == 0:  # 每 log_step 步打印一次信息
            print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'  # 打印当前步数、内容损失和样式损失
                   .format(step+1, config.total_step, content_loss.item(), style_loss.item()))

        if (step+1) % config.sample_step == 0:  # 每 sample_step 步保存一次生成的图像
            # Save the generated image
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))  # 定义反归一化
            img = target.clone().squeeze()  # 克隆目标图像并去掉多余的维度
            img = denorm(img).clamp_(0, 1)  # 反归一化并限制在 [0, 1] 范围内
            torchvision.utils.save_image(img, 'output-{}.png'.format(step+1))  # 保存生成的图像


if __name__ == "__main__":  # 如果是主程序
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--content', type=str, default='png/content.png')  # 添加内容图像路径参数
    parser.add_argument('--style', type=str, default='png/style.png')  # 添加样式图像路径参数
    parser.add_argument('--max_size', type=int, default=400)  # 添加最大图像大小参数
    parser.add_argument('--total_step', type=int, default=2000)  # 添加总步数参数
    parser.add_argument('--log_step', type=int, default=10)  # 添加日志打印步数参数
    parser.add_argument('--sample_step', type=int, default=500)  # 添加样本保存步数参数
    parser.add_argument('--style_weight', type=float, default=100)  # 添加样式权重参数
    parser.add_argument('--lr', type=float, default=0.003)  # 添加学习率参数
    config = parser.parse_args()  # 解析命令行参数
    print(config)  # 打印参数
    main(config)  # 调用主函数