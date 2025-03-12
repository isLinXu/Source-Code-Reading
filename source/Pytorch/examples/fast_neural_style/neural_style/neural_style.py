import argparse  # 导入argparse模块，用于解析命令行参数
import os  # 导入os模块，用于与操作系统交互
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数
import time  # 导入time模块，用于时间相关的操作
import re  # 导入re模块，用于正则表达式操作

import numpy as np  # 导入NumPy库，用于数组和数值计算
import torch  # 导入PyTorch库
from torch.optim import Adam  # 从PyTorch导入Adam优化器
from torch.utils.data import DataLoader  # 从PyTorch导入数据加载器
from torchvision import datasets  # 从torchvision导入数据集
from torchvision import transforms  # 从torchvision导入图像转换函数
import torch.onnx  # 导入ONNX支持

import utils  # 导入自定义的utils模块
from transformer_net import TransformerNet  # 导入TransformerNet模型
from vgg import Vgg16  # 导入VGG16模型


def check_paths(args):  # 检查保存模型和检查点路径的函数
    try:
        if not os.path.exists(args.save_model_dir):  # 如果保存模型的目录不存在
            os.makedirs(args.save_model_dir)  # 创建目录
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):  # 如果检查点目录存在且不存在
            os.makedirs(args.checkpoint_model_dir)  # 创建检查点目录
    except OSError as e:  # 捕获操作系统错误
        print(e)  # 打印错误信息
        sys.exit(1)  # 退出程序


def train(args):  # 训练模型的函数
    if args.cuda:  # 如果使用CUDA
        device = torch.device("cuda")  # 设置设备为GPU
    elif args.mps:  # 如果使用MPS（macOS GPU）
        device = torch.device("mps")  # 设置设备为MPS
    else:
        device = torch.device("cpu")  # 设置设备为CPU

    np.random.seed(args.seed)  # 设置NumPy随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子

    transform = transforms.Compose([  # 定义图像转换操作
        transforms.Resize(args.image_size),  # 调整图像大小
        transforms.CenterCrop(args.image_size),  # 中心裁剪图像
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Lambda(lambda x: x.mul(255))  # 将张量值放大到255
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)  # 加载训练数据集
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)  # 创建数据加载器

    transformer = TransformerNet().to(device)  # 初始化Transformer模型并移动到设备
    optimizer = Adam(transformer.parameters(), args.lr)  # 创建Adam优化器
    mse_loss = torch.nn.MSELoss()  # 定义均方误差损失

    vgg = Vgg16(requires_grad=False).to(device)  # 初始化VGG16模型并移动到设备
    style_transform = transforms.Compose([  # 定义风格图像的转换操作
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Lambda(lambda x: x.mul(255))  # 将张量值放大到255
    ])
    style = utils.load_image(args.style_image, size=args.style_size)  # 加载风格图像
    style = style_transform(style)  # 应用转换
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)  # 重复风格图像以适应批量大小并移动到设备

    features_style = vgg(utils.normalize_batch(style))  # 提取风格图像的特征
    gram_style = [utils.gram_matrix(y) for y in features_style]  # 计算风格特征的Gram矩阵

    for e in range(args.epochs):  # 遍历每个训练周期
        transformer.train()  # 设置模型为训练模式
        agg_content_loss = 0.  # 初始化内容损失
        agg_style_loss = 0.  # 初始化风格损失
        count = 0  # 初始化计数器
        for batch_id, (x, _) in enumerate(train_loader):  # 遍历每个批次
            n_batch = len(x)  # 获取当前批次的大小
            count += n_batch  # 更新计数器
            optimizer.zero_grad()  # 清零梯度

            x = x.to(device)  # 将输入数据移动到设备
            y = transformer(x)  # 通过模型进行前向传播

            y = utils.normalize_batch(y)  # 归一化输出
            x = utils.normalize_batch(x)  # 归一化输入

            features_y = vgg(y)  # 提取输出图像的特征
            features_x = vgg(x)  # 提取输入图像的特征

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)  # 计算内容损失

            style_loss = 0.  # 初始化风格损失
            for ft_y, gm_s in zip(features_y, gram_style):  # 遍历输出特征和Gram矩阵
                gm_y = utils.gram_matrix(ft_y)  # 计算输出特征的Gram矩阵
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])  # 计算风格损失
            style_loss *= args.style_weight  # 加权风格损失

            total_loss = content_loss + style_loss  # 计算总损失
            total_loss.backward()  # 反向传播
            optimizer.step()  # 更新优化器

            agg_content_loss += content_loss.item()  # 累加内容损失
            agg_style_loss += style_loss.item()  # 累加风格损失

            if (batch_id + 1) % args.log_interval == 0:  # 每log_interval批次记录一次
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)  # 打印日志信息

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:  # 如果设置了检查点目录并且达到检查点间隔
                transformer.eval().cpu()  # 将模型设置为评估模式并移动到CPU
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"  # 检查点文件名
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)  # 检查点路径
                torch.save(transformer.state_dict(), ckpt_model_path)  # 保存模型状态
                transformer.to(device).train()  # 将模型移动回设备并设置为训练模式

    # save model
    transformer.eval().cpu()  # 将模型设置为评估模式并移动到CPU
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"  # 保存模型文件名
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)  # 保存模型路径
    torch.save(transformer.state_dict(), save_model_path)  # 保存模型状态

    print("\nDone, trained model saved at", save_model_path)  # 打印完成信息和模型保存路径


def stylize(args):  # 风格化图像的函数
    device = torch.device("cuda" if args.cuda else "cpu")  # 设置设备为CUDA或CPU

    content_image = utils.load_image(args.content_image, scale=args.content_scale)  # 加载内容图像
    content_transform = transforms.Compose([  # 定义内容图像的转换操作
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Lambda(lambda x: x.mul(255))  # 将张量值放大到255
    ])
    content_image = content_transform(content_image)  # 应用转换
    content_image = content_image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备

    if args.model.endswith(".onnx"):  # 如果模型是ONNX格式
        output = stylize_onnx(content_image, args)  # 使用ONNX模型进行风格化
    else:
        with torch.no_grad():  # 不计算梯度
            style_model = TransformerNet()  # 初始化风格模型
            state_dict = torch.load(args.model)  # 加载模型状态字典
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):  # 遍历状态字典中的键
                if re.search(r'in\d+\.running_(mean|var)$', k):  # 如果键是InstanceNorm的过时键
                    del state_dict[k]  # 删除过时键
            style_model.load_state_dict(state_dict)  # 加载模型状态
            style_model.to(device)  # 移动模型到设备
            style_model.eval()  # 设置模型为评估模式
            if args.export_onnx:  # 如果设置导出ONNX模型
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"  # 确保导出文件名正确
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11,
                ).cpu()  # 导出ONNX模型并移动到CPU
            else:
                output = style_model(content_image).cpu()  # 通过模型进行风格化并移动到CPU
    utils.save_image(args.output_image, output[0])  # 保存输出图像


def stylize_onnx(content_image, args):  # 使用ONNX模型进行风格化的函数
    """
    Read ONNX model and run it using onnxruntime
    """  # 读取ONNX模型并使用onnxruntime运行

    assert not args.export_onnx  # 确保不导出ONNX模型

    import onnxruntime  # 导入ONNX运行时

    ort_session = onnxruntime.InferenceSession(args.model)  # 创建ONNX推理会话

    def to_numpy(tensor):  # 将张量转换为NumPy数组的辅助函数
        return (
            tensor.detach().cpu().numpy()  # 如果张量需要梯度，则分离并移动到CPU
            if tensor.requires_grad
            else tensor.cpu().numpy()  # 否则直接移动到CPU
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}  # 准备ONNX输入
    ort_outs = ort_session.run(None, ort_inputs)  # 运行推理
    img_out_y = ort_outs[0]  # 获取输出图像

    return torch.from_numpy(img_out_y)  # 将输出转换为张量


def main():  # 主函数
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")  # 创建主参数解析器
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")  # 添加子命令解析器

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")  # 添加训练参数解析器
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")  # 添加训练周期参数
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")  # 添加批次大小参数
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")  # 添加数据集路径参数
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")  # 添加风格图像路径参数
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")  # 添加保存模型目录参数
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")  # 添加检查点目录参数
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")  # 添加图像大小参数
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")  # 添加风格图像大小参数
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")  # 添加CUDA参数
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")  # 添加随机种子参数
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")  # 添加内容损失权重参数
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")  # 添加风格损失权重参数
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")  # 添加学习率参数
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")  # 添加日志间隔参数
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")  # 添加检查点间隔参数

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")  # 添加评估参数解析器
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")  # 添加内容图像路径参数
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")  # 添加内容图像缩放因子参数
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")  # 添加输出图像路径参数
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")  # 添加模型路径参数
    eval_arg_parser.add_argument("--cuda", type=int, default=False,
                                 help="set it to 1 for running on cuda, 0 for CPU")  # 添加CUDA参数
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")  # 添加导出ONNX模型参数
    eval_arg_parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU training')  # 添加MPS参数

    args = main_arg_parser.parse_args()  # 解析命令行参数

    if args.subcommand is None:  # 如果没有指定子命令
        print("ERROR: specify either train or eval")  # 打印错误信息
        sys.exit(1)  # 退出程序
    if args.cuda and not torch.cuda.is_available():  # 如果指定使用CUDA但CUDA不可用
        print("ERROR: cuda is not available, try running on CPU")  # 打印错误信息
        sys.exit(1)  # 退出程序
    if not args.mps and torch.backends.mps.is_available():  # 如果没有指定使用MPS但MPS可用
        print("WARNING: mps is available, run with --mps to enable macOS GPU")  # 打印警告信息

    if args.subcommand == "train":  # 如果子命令是训练
        check_paths(args)  # 检查路径
        train(args)  # 训练模型
    else:  # 否则
        stylize(args)  # 风格化图像


if __name__ == "__main__":  # 如果是主模块
    main()  # 调用主函数