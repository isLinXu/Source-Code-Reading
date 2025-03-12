from __future__ import print_function  # 为了兼容 Python 2 和 3 的 print 函数
import argparse  # 导入 argparse 库，用于处理命令行参数
import torch  # 导入 PyTorch 库
from PIL import Image  # 导入图像处理库 PIL
from torchvision.transforms import ToTensor  # 导入转换为张量的工具

import numpy as np  # 导入 NumPy 库

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')  # 创建参数解析器，描述为 PyTorch 超分辨率示例
parser.add_argument('--input_image', type=str, required=True, help='input image to use')  # 添加输入图像参数
parser.add_argument('--model', type=str, required=True, help='model file to use')  # 添加模型文件参数
parser.add_argument('--output_filename', type=str, help='where to save the output image')  # 添加输出图像文件名参数
parser.add_argument('--cuda', action='store_true', help='use cuda')  # 添加参数以使用 CUDA
opt = parser.parse_args()  # 解析命令行参数

print(opt)  # 打印解析后的参数
img = Image.open(opt.input_image).convert('YCbCr')  # 打开输入图像并转换为 YCbCr 格式
y, cb, cr = img.split()  # 分离 Y、Cb 和 Cr 通道

model = torch.load(opt.model)  # 加载指定的模型
img_to_tensor = ToTensor()  # 创建转换为张量的实例
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])  # 将 Y 通道转换为张量并调整形状

if opt.cuda:  # 如果指定使用 CUDA
    model = model.cuda()  # 将模型移动到 CUDA 设备
    input = input.cuda()  # 将输入张量移动到 CUDA 设备

out = model(input)  # 前向传播得到输出
out = out.cpu()  # 将输出移动到 CPU
out_img_y = out[0].detach().numpy()  # 将输出的 Y 通道转换为 NumPy 数组
out_img_y *= 255.0  # 将像素值缩放到 0-255 范围
out_img_y = out_img_y.clip(0, 255)  # 限制像素值在 0 到 255 之间
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')  # 将 Y 通道转换为图像

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)  # 使用双三次插值调整 Cb 通道大小
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)  # 使用双三次插值调整 Cr 通道大小
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')  # 合并 Y、Cb 和 Cr 通道并转换为 RGB 格式

out_img.save(opt.output_filename)  # 保存输出图像
print('output image saved to ', opt.output_filename)  # 打印保存路径