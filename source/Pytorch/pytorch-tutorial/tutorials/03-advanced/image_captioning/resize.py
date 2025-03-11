import argparse  # 导入 argparse 库，用于处理命令行参数
import os  # 导入操作系统库
from PIL import Image  # 从 PIL 导入图像处理模块


def resize_image(image, size):  # 定义调整图像大小的函数
    """Resize an image to the given size."""  # 将图像调整为给定的大小
    return image.resize(size, Image.ANTIALIAS)  # 调整图像大小并使用抗锯齿


def resize_images(image_dir, output_dir, size):  # 定义调整多个图像大小的函数
    """Resize the images in 'image_dir' and save into 'output_dir'."""  # 调整 'image_dir' 中的图像大小并保存到 'output_dir'
    if not os.path.exists(output_dir):  # 如果输出目录不存在
        os.makedirs(output_dir)  # 创建输出目录

    images = os.listdir(image_dir)  # 列出图像目录中的所有文件
    num_images = len(images)  # 获取图像数量
    for i, image in enumerate(images):  # 遍历每个图像
        with open(os.path.join(image_dir, image), 'r+b') as f:  # 以读写模式打开图像文件
            with Image.open(f) as img:  # 打开图像文件
                img = resize_image(img, size)  # 调整图像大小
                img.save(os.path.join(output_dir, image), img.format)  # 保存调整后的图像到输出目录
        if (i+1) % 100 == 0:  # 每 100 个图像打印一次信息
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))  # 打印已处理的图像数量和输出目录

def main(args):  # 主函数
    image_dir = args.image_dir  # 获取图像目录
    output_dir = args.output_dir  # 获取输出目录
    image_size = [args.image_size, args.image_size]  # 设置图像大小
    resize_images(image_dir, output_dir, image_size)  # 调用调整图像大小的函数


if __name__ == '__main__':  # 如果是主程序
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--image_dir', type=str, default='./data/train2014/',  # 添加图像目录参数
                        help='directory for train images')  # 参数帮助信息
    parser.add_argument('--output_dir', type=str, default='./data/resized2014/',  # 添加输出目录参数
                        help='directory for saving resized images')  # 参数帮助信息
    parser.add_argument('--image_size', type=int, default=256,  # 添加图像大小参数
                        help='size for image after processing')  # 参数帮助信息
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 调用主函数