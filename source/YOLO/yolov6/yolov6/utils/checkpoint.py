#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os  # 导入os模块，用于处理文件和目录
import shutil  # 导入shutil模块，用于文件操作
import torch  # 导入PyTorch库
import os.path as osp  # 导入os.path模块，并重命名为osp
from yolov6.utils.events import LOGGER  # 从yolov6.utils.events导入日志记录器
from yolov6.utils.torch_utils import fuse_model  # 从yolov6.utils.torch_utils导入模型融合函数

def load_state_dict(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    # 从检查点文件加载权重，只分配名称和形状匹配的层的权重
    ckpt = torch.load(weights, map_location=map_location)  # 加载权重文件
    state_dict = ckpt['model'].float().state_dict()  # 获取模型的状态字典并转换为浮点型
    model_state_dict = model.state_dict()  # 获取当前模型的状态字典
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    # 只保留名称和形状与当前模型匹配的权重
    model.load_state_dict(state_dict, strict=False)  # 加载权重到模型，允许不严格匹配
    del ckpt, state_dict, model_state_dict  # 删除临时变量以释放内存
    return model  # 返回更新后的模型

def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    # 从检查点文件加载模型
    LOGGER.info("Loading checkpoint from {}".format(weights))  # 记录加载检查点的信息
    ckpt = torch.load(weights, map_location=map_location)  # load  # 加载检查点
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()  # 根据是否存在ema选择模型
    if fuse:  # 如果需要融合模型
        LOGGER.info("\nFusing model...")  # 记录模型融合的信息
        model = fuse_model(model).eval()  # 融合模型并设置为评估模式
    else:
        model = model.eval()  # 如果不需要融合，直接设置为评估模式
    return model  # 返回加载的模型

def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """ Save checkpoint to the disk."""
    # 将检查点保存到磁盘
    if not osp.exists(save_dir):  # 如果保存目录不存在
        os.makedirs(save_dir)  # 创建保存目录
    filename = osp.join(save_dir, model_name + '.pt')  # 设置文件名
    torch.save(ckpt, filename)  # 保存检查点
    if is_best:  # 如果是最佳模型
        best_filename = osp.join(save_dir, 'best_ckpt.pt')  # 设置最佳检查点文件名
        shutil.copyfile(filename, best_filename)  # 复制文件为最佳检查点

def strip_optimizer(ckpt_dir, epoch):
    """Delete optimizer from saved checkpoint file"""
    # 从保存的检查点文件中删除优化器
    for s in ['best', 'last']:  # 遍历最佳和最后的检查点
        ckpt_path = osp.join(ckpt_dir, '{}_ckpt.pt'.format(s))  # 获取检查点路径
        if not osp.exists(ckpt_path):  # 如果检查点文件不存在
            continue  # 跳过
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))  # 加载检查点
        if ckpt.get('ema'):  # 如果存在ema
            ckpt['model'] = ckpt['ema']  # 将模型替换为ema模型
        for k in ['optimizer', 'ema', 'updates']:  # 遍历需要删除的键
            ckpt[k] = None  # 删除优化器、ema和更新信息
        ckpt['epoch'] = epoch  # 更新当前epoch
        ckpt['model'].half()  # 转换模型为FP16
        for p in ckpt['model'].parameters():  # 遍历模型参数
            p.requires_grad = False  # 将参数的梯度计算设置为False
        torch.save(ckpt, ckpt_path)  # 保存修改后的检查点