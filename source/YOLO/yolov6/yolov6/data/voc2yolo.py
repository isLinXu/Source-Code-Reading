import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import shutil
import argparse

# VOC dataset (refer https://github.com/ultralytics/yolov5/blob/master/data/VOC.yaml)
# VOC2007 trainval: 446MB, 5012 images
# VOC2007 test:     438MB, 4953 images
# VOC2012 trainval: 1.95GB, 17126 images

VOC_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
             'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]  # 计算宽高的归一化因子
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]  # 计算中心点和宽高
        return x * dw, y * dh, w * dw, h * dh  # 返回归一化后的坐标和尺寸

    in_file = open(os.path.join(path, f'VOC{year}/Annotations/{image_id}.xml'))  # 打开对应图像的XML注释文件
    out_file = open(lb_path, 'w')  # 打开输出标签文件
    tree = ET.parse(in_file)  # 解析XML文件
    root = tree.getroot()  # 获取XML根节点
    size = root.find('size')  # 找到图像尺寸节点
    w = int(size.find('width').text)  # 获取图像宽度
    h = int(size.find('height').text)  # 获取图像高度
    for obj in root.iter('object'):  # 遍历所有对象
        cls = obj.find('name').text  # 获取对象类别
        if cls in VOC_NAMES and not int(obj.find('difficult').text) == 1:  # 如果类别在VOC_NAMES中且不为困难样本
            xmlbox = obj.find('bndbox')  # 获取边界框信息
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])  # 转换边界框坐标
            cls_id = VOC_NAMES.index(cls)  # class id  # 获取类别ID
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')  # 将类别ID和边界框信息写入标签文件



def gen_voc07_12(voc_path):
    '''
    Generate voc07+12 setting dataset:
    train: # train images 16551 images
        - images/train2012
        - images/train2007
        - images/val2012
        - images/val2007
    val: # val images (relative to 'path')  4952 images
        - images/test2007
    '''
    dataset_root = os.path.join(voc_path, 'voc_07_12')  # 设置数据集根目录
    if not os.path.exists(dataset_root):  # 如果数据集目录不存在
        os.makedirs(dataset_root)  # 创建数据集目录

    dataset_settings = {'train': ['train2007', 'val2007', 'train2012', 'val2012'], 'val':['test2007']}  # 数据集设置
    for item in ['images', 'labels']:  # 遍历图像和标签
        for data_type, data_list in dataset_settings.items():  # 遍历数据集设置
            for data_name in data_list:  # 遍历每种数据类型
                ori_path = os.path.join(voc_path, item, data_name)  # 原始路径
                new_path = os.path.join(dataset_root, item, data_type)  # 新路径
                if not os.path.exists(new_path):  # 如果新路径不存在
                    os.makedirs(new_path)  # 创建新路径

                print(f'[INFO]: Copying {ori_path} to {new_path}')  # 打印复制信息
                for file in os.listdir(ori_path):  # 遍历原始路径下的文件
                    shutil.copy(os.path.join(ori_path, file), new_path)  # 复制文件到新路径


def main(args):
    voc_path = args.voc_path  # 获取VOC数据集的路径
    for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):  # 遍历不同年份和数据集类型
        imgs_path = os.path.join(voc_path, 'images', f'{image_set}')  # 设置图像存储路径
        lbs_path = os.path.join(voc_path, 'labels', f'{image_set}')  # 设置标签存储路径

        try:
            with open(os.path.join(voc_path, f'VOC{year}/ImageSets/Main/{image_set}.txt'), 'r') as f:  # 打开对应的图像集文件
                image_ids = f.read().strip().split()  # 读取图像ID
            if not os.path.exists(imgs_path):  # 如果图像路径不存在
                os.makedirs(imgs_path)  # 创建图像路径
            if not os.path.exists(lbs_path):  # 如果标签路径不存在
                os.makedirs(lbs_path)  # 创建标签路径

            for id in tqdm(image_ids, desc=f'{image_set}{year}'):  # 遍历图像ID并显示进度条
                f = os.path.join(voc_path, f'VOC{year}/JPEGImages/{id}.jpg')  # old img path  # 获取旧的图像路径
                lb_path = os.path.join(lbs_path, f'{id}.txt')  # new label path  # 设置新的标签路径
                convert_label(voc_path, lb_path, year, id)  # convert labels to YOLO format  # 将标签转换为YOLO格式
                if os.path.exists(f):  # 如果旧图像存在
                    shutil.move(f, imgs_path)  # move image  # 将图像移动到新的图像路径
        except Exception as e:  # 捕获异常
            print(f'[Warning]: {e} {year}{image_set} convert fail!')  # 打印警告信息，说明转换失败

    gen_voc07_12(voc_path)  # 生成VOC 07+12数据集设置



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_path', default='VOCdevkit')

    args = parser.parse_args()
    print(args)

    main(args)
