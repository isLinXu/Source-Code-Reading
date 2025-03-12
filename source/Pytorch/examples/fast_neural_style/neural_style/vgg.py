from collections import namedtuple  # 从collections模块导入namedtuple，用于创建命名元组

import torch  # 导入PyTorch库
from torchvision import models  # 从torchvision导入模型模块


class Vgg16(torch.nn.Module):  # 定义Vgg16类，继承自torch.nn.Module
    def __init__(self, requires_grad=False):  # 初始化函数，接受是否需要梯度的参数
        super(Vgg16, self).__init__()  # 调用父类构造函数
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features  # 加载预训练的VGG16特征提取层
        self.slice1 = torch.nn.Sequential()  # 定义第一个序列容器
        self.slice2 = torch.nn.Sequential()  # 定义第二个序列容器
        self.slice3 = torch.nn.Sequential()  # 定义第三个序列容器
        self.slice4 = torch.nn.Sequential()  # 定义第四个序列容器
        for x in range(4):  # 遍历前4层
            self.slice1.add_module(str(x), vgg_pretrained_features[x])  # 将这些层添加到slice1
        for x in range(4, 9):  # 遍历第4到第8层
            self.slice2.add_module(str(x), vgg_pretrained_features[x])  # 将这些层添加到slice2
        for x in range(9, 16):  # 遍历第9到第15层
            self.slice3.add_module(str(x), vgg_pretrained_features[x])  # 将这些层添加到slice3
        for x in range(16, 23):  # 遍历第16到第22层
            self.slice4.add_module(str(x), vgg_pretrained_features[x])  # 将这些层添加到slice4
        if not requires_grad:  # 如果不需要梯度
            for param in self.parameters():  # 遍历模型的所有参数
                param.requires_grad = False  # 将参数的requires_grad属性设置为False

    def forward(self, X):  # 前向传播函数
        h = self.slice1(X)  # 通过第一个序列容器处理输入
        h_relu1_2 = h  # 保存第一个激活层的输出
        h = self.slice2(h)  # 通过第二个序列容器处理
        h_relu2_2 = h  # 保存第二个激活层的输出
        h = self.slice3(h)  # 通过第三个序列容器处理
        h_relu3_3 = h  # 保存第三个激活层的输出
        h = self.slice4(h)  # 通过第四个序列容器处理
        h_relu4_3 = h  # 保存第四个激活层的输出
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])  # 创建命名元组用于输出
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)  # 将激活层的输出打包
        return out  # 返回输出