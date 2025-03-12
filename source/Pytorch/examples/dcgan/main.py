from __future__ import print_function  # 引入未来的 print 函数，确保兼容性
import argparse  # 导入 argparse 库，用于处理命令行参数
import os  # 导入 os 库，用于与操作系统交互
import random  # 导入 random 库，用于生成随机数
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.parallel  # 导入并行计算模块
import torch.backends.cudnn as cudnn  # 导入 cuDNN 后端支持
import torch.optim as optim  # 导入 PyTorch 的优化器模块
import torch.utils.data  # 导入数据处理工具
import torchvision.datasets as dset  # 导入 torchvision 数据集模块
import torchvision.transforms as transforms  # 导入数据转换模块
import torchvision.utils as vutils  # 导入 torchvision 工具模块


parser = argparse.ArgumentParser()  # 创建参数解析器
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist | imagenet | folder | lfw | fake')  # 数据集参数
parser.add_argument('--dataroot', required=False, help='path to dataset')  # 数据集路径
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)  # 数据加载工作线程数
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')  # 输入批量大小
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')  # 输入图像的高度/宽度
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')  # 潜在向量 z 的大小
parser.add_argument('--ngf', type=int, default=64)  # 生成器特征图的数量
parser.add_argument('--ndf', type=int, default=64)  # 判别器特征图的数量
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')  # 训练的轮数
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')  # 学习率
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')  # Adam 优化器的 beta1 参数
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')  # 是否启用 CUDA
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')  # 检查单个训练周期是否有效
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')  # 使用的 GPU 数量
parser.add_argument('--netG', default='', help="path to netG (to continue training)")  # 生成器模型路径
parser.add_argument('--netD', default='', help="path to netD (to continue training)")  # 判别器模型路径
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')  # 输出图像和模型检查点的文件夹
parser.add_argument('--manualSeed', type=int, help='manual seed')  # 手动设置随机种子
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')  # LSUN 数据集的类别
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')  # 是否启用 macOS GPU 训练

opt = parser.parse_args()  # 解析命令行参数
print(opt)  # 打印参数


try:
    os.makedirs(opt.outf)  # 创建输出文件夹
except OSError:
    pass  # 如果文件夹已经存在，则忽略错误

if opt.manualSeed is None:  # 如果没有手动设置随机种子
    opt.manualSeed = random.randint(1, 10000)  # 随机生成一个种子
print("Random Seed: ", opt.manualSeed)  # 打印随机种子
random.seed(opt.manualSeed)  # 设置 Python 随机种子
torch.manual_seed(opt.manualSeed)  # 设置 PyTorch 随机种子

cudnn.benchmark = True  # 启用 cuDNN 的基准模式，以优化性能

if torch.cuda.is_available() and not opt.cuda:  # 如果有 CUDA 设备但没有启用 CUDA
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")  # 提示用户启用 CUDA

if torch.backends.mps.is_available() and not opt.mps:  # 如果有 MPS 设备但没有启用 MPS
    print("WARNING: You have mps device, to enable macOS GPU run with --mps")  # 提示用户启用 MPS
  
if opt.dataroot is None and str(opt.dataset).lower() != 'fake':  # 如果没有数据集路径且数据集不是 fake
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)  # 抛出错误

if opt.dataset in ['imagenet', 'folder', 'lfw']:  # 如果数据集是 ImageNet、folder 或 LFW
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,  # 使用 ImageFolder 加载数据集
                               transform=transforms.Compose([  # 定义数据转换
                                   transforms.Resize(opt.imageSize),  # 调整图像大小
                                   transforms.CenterCrop(opt.imageSize),  # 中心裁剪图像
                                   transforms.ToTensor(),  # 转换为张量
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                               ]))
    nc=3  # 通道数为 3（RGB）
elif opt.dataset == 'lsun':  # 如果数据集是 LSUN
    classes = [ c + '_train' for c in opt.classes.split(',')]  # 获取训练类别
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,  # 使用 LSUN 加载数据集
                        transform=transforms.Compose([  # 定义数据转换
                            transforms.Resize(opt.imageSize),  # 调整图像大小
                            transforms.CenterCrop(opt.imageSize),  # 中心裁剪图像
                            transforms.ToTensor(),  # 转换为张量
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                        ]))
    nc=3  # 通道数为 3（RGB）
elif opt.dataset == 'cifar10':  # 如果数据集是 CIFAR10
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,  # 使用 CIFAR10 加载数据集
                           transform=transforms.Compose([  # 定义数据转换
                               transforms.Resize(opt.imageSize),  # 调整图像大小
                               transforms.ToTensor(),  # 转换为张量
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                           ]))
    nc=3  # 通道数为 3（RGB）

elif opt.dataset == 'mnist':  # 如果数据集是 MNIST
        dataset = dset.MNIST(root=opt.dataroot, download=True,  # 使用 MNIST 加载数据集
                           transform=transforms.Compose([  # 定义数据转换
                               transforms.Resize(opt.imageSize),  # 调整图像大小
                               transforms.ToTensor(),  # 转换为张量
                               transforms.Normalize((0.5,), (0.5,)),  # 归一化
                           ]))
        nc=1  # 通道数为 1（灰度）

elif opt.dataset == 'fake':  # 如果数据集是 fake
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),  # 使用 FakeData 生成假数据
                            transform=transforms.ToTensor())  # 转换为张量
    nc=3  # 通道数为 3（RGB）

assert dataset  # 确保数据集已加载
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,  # 创建数据加载器
                                         shuffle=True, num_workers=int(opt.workers))  # 随机打乱数据，设置工作线程数
use_mps = opt.mps and torch.backends.mps.is_available()  # 检查是否使用 MPS
if opt.cuda:  # 如果启用 CUDA
    device = torch.device("cuda:0")  # 设置设备为 CUDA
elif use_mps:  # 如果启用 MPS
    device = torch.device("mps")  # 设置设备为 MPS
else:  # 否则使用 CPU
    device = torch.device("cpu")  # 设置设备为 CPU

ngpu = int(opt.ngpu)  # 获取 GPU 数量
nz = int(opt.nz)  # 获取潜在向量 z 的大小
ngf = int(opt.ngf)  # 获取生成器特征图的数量
ndf = int(opt.ndf)  # 获取判别器特征图的数量


# custom weights initialization called on netG and netD
def weights_init(m):  # 自定义权重初始化函数
    classname = m.__class__.__name__  # 获取模块的类名
    if classname.find('Conv') != -1:  # 如果是卷积层
        torch.nn.init.normal_(m.weight, 0.0, 0.02)  # 正态分布初始化权重
    elif classname.find('BatchNorm') != -1:  # 如果是批归一化层
        torch.nn.init.normal_(m.weight, 1.0, 0.02)  # 正态分布初始化权重
        torch.nn.init.zeros_(m.bias)  # 将偏置初始化为 0


class Generator(nn.Module):  # 定义生成器类
    def __init__(self, ngpu):  # 初始化生成器
        super(Generator, self).__init__()  # 调用父类构造函数
        self.ngpu = ngpu  # 保存 GPU 数量
        self.main = nn.Sequential(  # 定义生成器的主网络
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # 转置卷积层
            nn.BatchNorm2d(ngf * 8),  # 批归一化层
            nn.ReLU(True),  # ReLU 激活函数
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 转置卷积层
            nn.BatchNorm2d(ngf * 4),  # 批归一化层
            nn.ReLU(True),  # ReLU 激活函数
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 转置卷积层
            nn.BatchNorm2d(ngf * 2),  # 批归一化层
            nn.ReLU(True),  # ReLU 激活函数
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 转置卷积层
            nn.BatchNorm2d(ngf),  # 批归一化层
            nn.ReLU(True),  # ReLU 激活函数
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # 转置卷积层
            nn.Tanh()  # Tanh 激活函数
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):  # 前向传播
        if input.is_cuda and self.ngpu > 1:  # 如果在 CUDA 上并且使用多个 GPU
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))  # 使用数据并行
        else:
            output = self.main(input)  # 直接通过主网络计算输出
        return output  # 返回输出


netG = Generator(ngpu).to(device)  # 实例化生成器并移动到设备
netG.apply(weights_init)  # 应用权重初始化
if opt.netG != '':  # 如果指定了生成器模型路径
    netG.load_state_dict(torch.load(opt.netG))  # 加载生成器模型
print(netG)  # 打印生成器结构


class Discriminator(nn.Module):  # 定义判别器类
    def __init__(self, ngpu):  # 初始化判别器
        super(Discriminator, self).__init__()  # 调用父类构造函数
        self.ngpu = ngpu  # 保存 GPU 数量
        self.main = nn.Sequential(  # 定义判别器的主网络
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 卷积层
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU 激活函数
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 卷积层
            nn.BatchNorm2d(ndf * 2),  # 批归一化层
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU 激活函数
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 卷积层
            nn.BatchNorm2d(ndf * 4),  # 批归一化层
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU 激活函数
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 卷积层
            nn.BatchNorm2d(ndf * 8),  # 批归一化层
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU 激活函数
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # 卷积层
            nn.Sigmoid()  # Sigmoid 激活函数
        )

    def forward(self, input):  # 前向传播
        if input.is_cuda and self.ngpu > 1:  # 如果在 CUDA 上并且使用多个 GPU
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))  # 使用数据并行
        else:
            output = self.main(input)  # 直接通过主网络计算输出

        return output.view(-1, 1).squeeze(1)  # 返回输出并调整形状


netD = Discriminator(ngpu).to(device)  # 实例化判别器并移动到设备
netD.apply(weights_init)  # 应用权重初始化
if opt.netD != '':  # 如果指定了判别器模型路径
    netD.load_state_dict(torch.load(opt.netD))  # 加载判别器模型
print(netD)  # 打印判别器结构

criterion = nn.BCELoss()  # 定义二元交叉熵损失函数

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)  # 生成固定的噪声用于生成图像
real_label = 1  # 真实标签
fake_label = 0  # 假标签

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 设置判别器优化器
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 设置生成器优化器

if opt.dry_run:  # 如果是干运行模式
    opt.niter = 1  # 将训练轮数设置为 1

for epoch in range(opt.niter):  # 进行指定轮数的训练
    for i, data in enumerate(dataloader, 0):  # 遍历数据加载器
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()  # 清除判别器的梯度
        real_cpu = data[0].to(device)  # 将真实数据移动到设备
        batch_size = real_cpu.size(0)  # 获取批量大小
        label = torch.full((batch_size,), real_label,  # 创建真实标签
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)  # 判别器对真实数据的输出
        errD_real = criterion(output, label)  # 计算真实数据的损失
        errD_real.backward()  # 反向传播
        D_x = output.mean().item()  # 计算真实数据的平均输出

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)  # 生成随机噪声
        fake = netG(noise)  # 通过生成器生成假数据
        label.fill_(fake_label)  # 填充假标签
        output = netD(fake.detach())  # 判别器对假数据的输出
        errD_fake = criterion(output, label)  # 计算假数据的损失
        errD_fake.backward()  # 反向传播
        D_G_z1 = output.mean().item()  # 计算假数据的平均输出
        errD = errD_real + errD_fake  # 总损失
        optimizerD.step()  # 更新判别器参数

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()  # 清除生成器的梯度
        label.fill_(real_label)  # 假标签在生成器损失中视为真实标签
        output = netD(fake)  # 判别器对假数据的输出
        errG = criterion(output, label)  # 计算生成器的损失
        errG.backward()  # 反向传播
        D_G_z2 = output.mean().item()  # 计算假数据的平均输出
        optimizerG.step()  # 更新生成器参数

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))  # 打印损失信息
        if i % 100 == 0:  # 每 100 次迭代保存图像
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,  # 保存真实样本
                    normalize=True)
            fake = netG(fixed_noise)  # 生成假样本
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),  # 保存假样本
                    normalize=True)

        if opt.dry_run:  # 如果是干运行模式
            break  # 结束训练

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))  # 保存生成器模型
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))  # 保存判别器模型