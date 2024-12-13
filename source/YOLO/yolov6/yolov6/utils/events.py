#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os  # 导入os模块，用于处理操作系统相关的功能
import yaml  # 导入yaml模块，用于处理YAML文件
import logging  # 导入logging模块，用于日志记录
import shutil  # 导入shutil模块，用于文件操作

def set_logging(name=None):
    # 设置日志记录
    rank = int(os.getenv('RANK', -1))  # 获取环境变量RANK，默认为-1
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    # 配置日志格式和级别，如果rank为-1或0，则使用INFO级别，否则使用WARNING级别
    return logging.getLogger(name)  # 返回指定名称的日志记录器

LOGGER = set_logging(__name__)  # 创建一个模块级的日志记录器
NCOLS = min(100, shutil.get_terminal_size().columns)  # 获取终端的列数，最多为100

def load_yaml(file_path):
    """Load data from yaml file."""
    # 从YAML文件加载数据
    if isinstance(file_path, str):  # 检查file_path是否为字符串
        with open(file_path, errors='ignore') as f:  # 打开YAML文件，忽略错误
            data_dict = yaml.safe_load(f)  # 安全加载YAML文件内容
    return data_dict  # 返回加载的数据字典

def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    # 将数据保存到YAML文件
    with open(save_path, 'w') as f:  # 打开保存路径的文件，写入模式
        yaml.safe_dump(data_dict, f, sort_keys=False)  # 安全地将数据字典写入文件，不排序键

def write_tblog(tblogger, epoch, results, lrs, losses):
    """Display mAP and loss information to log."""
    # 将mAP和损失信息写入日志
    tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)  # 添加验证集mAP@0.5
    tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)  # 添加验证集mAP@0.50:0.95

    tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)  # 添加训练集IOU损失
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)  # 添加训练集焦点损失
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)  # 添加训练集分类损失

    tblogger.add_scalar("x/lr0", lrs[0], epoch + 1)  # 添加学习率0
    tblogger.add_scalar("x/lr1", lrs[1], epoch + 1)  # 添加学习率1
    tblogger.add_scalar("x/lr2", lrs[2], epoch + 1)  # 添加学习率2

def write_tbimg(tblogger, imgs, step, type='train'):
    """Display train_batch and validation predictions to tensorboard."""
    # 将训练批次和验证预测图像写入TensorBoard
    if type == 'train':  # 如果类型为训练
        tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')  # 添加训练批次图像
    elif type == 'val':  # 如果类型为验证
        for idx, img in enumerate(imgs):  # 遍历验证图像
            tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')  # 添加验证图像
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')  # 记录警告信息，未知的图像类型