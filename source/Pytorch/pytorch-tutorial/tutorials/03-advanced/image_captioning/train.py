import argparse  # 导入 argparse 库，用于处理命令行参数
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库
import os  # 导入操作系统库
import pickle  # 导入 pickle 库，用于序列化和反序列化对象
from data_loader import get_loader  # 从 data_loader 导入获取数据加载器的函数
from build_vocab import Vocabulary  # 从 build_vocab 导入 Vocabulary 类
from model import EncoderCNN, DecoderRNN  # 从 model 导入编码器和解码器模型
from torch.nn.utils.rnn import pack_padded_sequence  # 从 PyTorch 导入用于处理填充序列的工具
from torchvision import transforms  # 从 torchvision 导入数据转换模块


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 配置设备，使用 GPU（如果可用）或 CPU

def main(args):  # 主函数
    # Create model directory
    if not os.path.exists(args.model_path):  # 如果模型目录不存在
        os.makedirs(args.model_path)  # 创建模型目录
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([  # 定义图像预处理步骤
        transforms.RandomCrop(args.crop_size),  # 随机裁剪图像
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.485, 0.456, 0.406),  # 进行归一化处理
                             (0.229, 0.224, 0.225))])  # 使用 ImageNet 数据集的均值和标准差
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:  # 以二进制读取模式打开词汇文件
        vocab = pickle.load(f)  # 反序列化加载词汇对象
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,  # 构建数据加载器
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)  # 创建编码器模型并移动到指定设备
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)  # 创建解码器模型并移动到指定设备
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())  # 获取模型参数
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)  # 定义优化器为 Adam
    
    # Train the models
    total_step = len(data_loader)  # 获取数据加载器的总步数
    for epoch in range(args.num_epochs):  # 遍历每个训练周期
        for i, (images, captions, lengths) in enumerate(data_loader):  # 遍历数据加载器中的每个批次
            
            # Set mini-batch dataset
            images = images.to(device)  # 将图像移动到指定设备
            captions = captions.to(device)  # 将注释移动到指定设备
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]  # 处理填充的注释
            
            # Forward, backward and optimize
            features = encoder(images)  # 前向传播编码器，获取特征
            outputs = decoder(features, captions, lengths)  # 前向传播解码器，获取输出
            loss = criterion(outputs, targets)  # 计算损失
            decoder.zero_grad()  # 清除解码器的梯度
            encoder.zero_grad()  # 清除编码器的梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数

            # Print log info
            if i % args.log_step == 0:  # 每 log_step 步打印一次信息
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))  # 打印当前周期、步数、损失和困惑度
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:  # 每 save_step 步保存一次模型
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))  # 保存解码器的状态字典
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))  # 保存编码器的状态字典


if __name__ == '__main__':  # 如果是主程序
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')  # 添加模型保存路径参数
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')  # 添加随机裁剪大小参数
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')  # 添加词汇路径参数
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')  # 添加图像目录参数
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')  # 添加注释文件路径参数
    parser.add_argument('--log_step', type=int , default=10, help='step size for printing log info')  # 添加日志打印步数参数
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')  # 添加模型保存步数参数
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')  # 添加嵌入层大小参数
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')  # 添加 LSTM 隐藏状态维度参数
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')  # 添加 LSTM 层数参数
    
    parser.add_argument('--num_epochs', type=int, default=5)  # 添加训练周期参数
    parser.add_argument('--batch_size', type=int, default=128)  # 添加批次大小参数
    parser.add_argument('--num_workers', type=int, default=2)  # 添加工作线程数量参数
    parser.add_argument('--learning_rate', type=float, default=0.001)  # 添加学习率参数
    args = parser.parse_args()  # 解析命令行参数
    print(args)  # 打印参数
    main(args)  # 调用主函数