# coding=utf-8
# Description:  visualize yolo label image.
# 描述：可视化YOLO标签图像

# 导入所需的库
import argparse  # 用于解析命令行参数
import os       # 用于文件和目录操作
import cv2      # 用于图像处理
import numpy as np  # 用于数值计算

# 定义支持的图片格式列表
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
# 将所有格式添加对应的大写版本
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

def main(args):
    # 从参数中解析输入路径和类别名称
    img_dir, label_dir, class_names = args.img_dir, args.label_dir, args.class_names

    # 创建类别ID到类别名称的映射字典
    label_map = dict()
    for class_id, classname in enumerate(class_names):
        label_map[class_id] = classname

    # 遍历图片目录中的所有文件
    for file in os.listdir(img_dir):
        # 检查文件格式是否支持
        if file.split('.')[-1] not in IMG_FORMATS:
            print(f'[Warning]: Non-image file {file}')
            continue
        
        # 构建图片和标签文件的完整路径
        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(label_dir, file[: file.rindex('.')] + '.txt')

        try:
            # 读取图片并获取尺寸信息
            img_data = cv2.imread(img_path)
            height, width, _ = img_data.shape
            # 为每个类别随机生成一个RGB颜色
            color = [tuple(np.random.choice(range(256), size=3)) for i in class_names]
            thickness = 2  # 设置绘制线条的粗细

            # 读取标签文件
            with open(label_path, 'r') as f:
                for bbox in f:
                    # 解析每行的边界框信息：类别、中心点坐标(x,y)、宽度、高度
                    cls, x_c, y_c, w, h = [float(v) if i > 0 else int(v) for i, v in enumerate(bbox.split('\n')[0].split(' '))]

                    # 将YOLO格式的坐标转换为像素坐标
                    x_tl = int((x_c - w / 2) * width)  # 左上角x坐标
                    y_tl = int((y_c - h / 2) * height) # 左上角y坐标
                    
                    # 绘制矩形框
                    cv2.rectangle(img_data, (x_tl, y_tl), 
                                (x_tl + int(w * width), y_tl + int(h * height)), 
                                tuple([int(x) for x in color[cls]]), thickness)
                    # 添加类别标签文本
                    cv2.putText(img_data, label_map[cls], (x_tl, y_tl - 10), 
                              cv2.FONT_HERSHEY_COMPLEX, 1, 
                              tuple([int(x) for x in color[cls]]), thickness)

            # 显示标注后的图片
            cv2.imshow('image', img_data)
            cv2.waitKey(0)  # 等待按键继续
        except Exception as e:
            print(f'[Error]: {e} {img_path}')
    print('======All Done!======')

# 程序入口点
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('--img_dir', default='VOCdevkit/voc_07_12/images')  # 图片目录
    parser.add_argument('--label_dir', default='VOCdevkit/voc_07_12/labels')  # 标签目录
    parser.add_argument('--class_names', default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
                        'sofa', 'train', 'tvmonitor'])  # VOC数据集的20个类别

    # 解析命令行参数
    args = parser.parse_args()
    print(args)

    # 调用主函数
    main(args)