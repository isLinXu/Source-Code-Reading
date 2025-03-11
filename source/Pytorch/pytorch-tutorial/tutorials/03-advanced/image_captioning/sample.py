import torch  # 导入 PyTorch 库
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 NumPy 库
import argparse  # 导入 argparse 库，用于处理命令行参数
import pickle  # 导入 pickle 库，用于序列化和反序列化对象
import os  # 导入操作系统库
from torchvision import transforms  # 从 torchvision 导入数据转换模块
from build_vocab import Vocabulary  # 从 build_vocab 导入 Vocabulary 类
from model import EncoderCNN, DecoderRNN  # 从 model 导入编码器和解码器模型
from PIL import Image  # 从 PIL 导入图像处理模块


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 配置设备，使用 GPU（如果可用）或 CPU

def load_image(image_path, transform=None):  # 定义加载图像的函数
    image = Image.open(image_path).convert('RGB')  # 打开图像并转换为 RGB 格式
    image = image.resize([224, 224], Image.LANCZOS)  # 调整图像大小为 224x224，使用 LANCZOS 算法
    
    if transform is not None:  # 如果提供了转换
        image = transform(image).unsqueeze(0)  # 应用转换并增加一个维度以适应模型输入
    
    return image  # 返回处理后的图像

def main(args):  # 主函数
    # Image preprocessing
    transform = transforms.Compose([  # 定义图像预处理步骤
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.485, 0.456, 0.406),  # 进行归一化处理
                             (0.229, 0.224, 0.225))])  # 使用 ImageNet 数据集的均值和标准差
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:  # 以二进制读取模式打开词汇文件
        vocab = pickle.load(f)  # 反序列化加载词汇对象

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # 创建编码器模型并设置为评估模式（批归一化使用移动均值/方差）
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)  # 创建解码器模型
    encoder = encoder.to(device)  # 将编码器模型移动到指定设备
    decoder = decoder.to(device)  # 将解码器模型移动到指定设备

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))  # 加载训练好的编码器参数
    decoder.load_state_dict(torch.load(args.decoder_path))  # 加载训练好的解码器参数

    # Prepare an image
    image = load_image(args.image, transform)  # 加载并处理输入图像
    image_tensor = image.to(device)  # 将图像张量移动到指定设备
    
    # Generate a caption from the image
    feature = encoder(image_tensor)  # 从图像生成特征向量
    sampled_ids = decoder.sample(feature)  # 使用解码器生成注释 ID
    sampled_ids = sampled_ids[0].cpu().numpy()  # 将 ID 转换为 NumPy 数组（从 GPU 转回 CPU）
    
    # Convert word_ids to words
    sampled_caption = []  # 初始化采样注释列表
    for word_id in sampled_ids:  # 遍历每个采样的单词 ID
        word = vocab.idx2word[word_id]  # 将单词 ID 转换为单词
        sampled_caption.append(word)  # 将单词添加到注释列表
        if word == '<end>':  # 如果遇到结束符
            break  # 终止循环
    sentence = ' '.join(sampled_caption)  # 将单词列表转换为句子
    
    # Print out the image and the generated caption
    print(sentence)  # 打印生成的注释
    image = Image.open(args.image)  # 打开原始输入图像
    plt.imshow(np.asarray(image))  # 显示图像

if __name__ == '__main__':  # 如果是主程序
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')  # 添加输入图像参数
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')  # 添加编码器路径参数
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')  # 添加解码器路径参数
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')  # 添加词汇路径参数
    
    # Model parameters (should be same as parameters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')  # 添加嵌入层大小参数
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')  # 添加 LSTM 隐藏状态维度参数
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')  # 添加 LSTM 层数参数
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 调用主函数