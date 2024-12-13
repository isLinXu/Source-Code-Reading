import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.common import RealVGGBlock, LinearAddBlock
from torch.optim.sgd import SGD
from yolov6.utils.events import LOGGER


def extract_blocks_into_list(model, blocks):
    # 遍历模型的所有子模块
    for module in model.children():
        # 如果模块是 LinearAddBlock 或 RealVGGBlock 的实例
        if isinstance(module, LinearAddBlock) or isinstance(module, RealVGGBlock):
            # 将该模块添加到 blocks 列表中
            blocks.append(module)
        else:
            # 递归调用，继续提取子模块中的块
            extract_blocks_into_list(module, blocks)

def extract_scales(model):
    # 初始化一个空列表，用于存储提取的块
    blocks = []
    # 调用 extract_blocks_into_list 函数，提取模型中的块
    extract_blocks_into_list(model['model'], blocks)
    # 初始化一个空列表，用于存储提取的尺度
    scales = []
    # 遍历提取到的块
    for b in blocks:
        # 断言 b 是 LinearAddBlock 的实例
        assert isinstance(b, LinearAddBlock)
        # 如果块具有 scale_identity 属性
        if hasattr(b, 'scale_identity'):
            # 将权重添加到 scales 列表中，使用 detach() 方法防止梯度计算
            scales.append((b.scale_identity.weight.detach(), b.scale_1x1.weight.detach(), b.scale_conv.weight.detach()))
        else:
            # 如果没有 scale_identity 属性，则只添加 scale_1x1 和 scale_conv 的权重
            scales.append((b.scale_1x1.weight.detach(), b.scale_conv.weight.detach()))
        # 打印当前提取的尺度的平均值
        print('extract scales: ', scales[-1][-2].mean(), scales[-1][-1].mean())
    # 返回提取的尺度列表
    return scales


def check_keywords_in_name(name, keywords=()):
    # 检查给定名称中是否包含任何关键字
    isin = False
    for keyword in keywords:
        # 如果关键字在名称中
        if keyword in name:
            isin = True
    # 返回是否找到关键字
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=(), echo=False):
    # 设置权重衰减的函数
    has_decay = []  # 存储需要权重衰减的参数
    no_decay = []   # 存储不需要权重衰减的参数

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过被冻结的权重
        if 'identity.weight' in name:
            # 如果参数名称中包含 'identity.weight'
            has_decay.append(param)
            if echo:
                print(f"{name} USE weight decay")  # 打印使用权重衰减的信息
        elif len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
            check_keywords_in_name(name, skip_keywords):
            # 如果参数是偏置或在跳过列表中，或名称包含跳过关键字
            no_decay.append(param)
            if echo:
                print(f"{name} has no weight decay")  # 打印不使用权重衰减的信息
        else:
            # 其他情况，添加到需要权重衰减的列表
            has_decay.append(param)
            if echo:
                print(f"{name} USE weight decay")  # 打印使用权重衰减的信息

    # 返回一个包含参数组的列表
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]  # 对不需要衰减的参数设置 weight_decay 为 0


def get_optimizer_param(args, cfg, model):
    """ Build optimizer from cfg file. """
    # 从配置文件构建优化器
    accumulate = max(1, round(64 / args.batch_size))  # 计算累积步数
    cfg.solver.weight_decay *= args.batch_size * accumulate / 64  # 调整权重衰减

    g_bnw, g_w, g_b = [], [], []  # 初始化三个列表，分别存储不同类型的参数
    # 遍历模型的所有模块
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            # 如果模块具有偏置参数
            g_b.append(v.bias)  # 将偏置参数添加到 g_b 列表中
        if isinstance(v, nn.BatchNorm2d):
            # 如果模块是 BatchNorm2d 类型
            g_bnw.append(v.weight)  # 将权重参数添加到 g_bnw 列表中
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            # 如果模块具有权重参数
            g_w.append(v.weight)  # 将权重参数添加到 g_w 列表中
    # 返回包含参数组的列表
    return [{'params': g_bnw},
            {'params': g_w, 'weight_decay': cfg.solver.weight_decay},
            {'params': g_b}]

class RepVGGOptimizer(SGD):
    '''scales is a list, scales[i] is a triple (scale_identity.weight, scale_1x1.weight, scale_conv.weight) or a two-tuple (scale_1x1.weight, scale_conv.weight) (if the block has no scale_identity)'''
    # 构造函数，初始化优化器
    def __init__(self, model, scales,
                 args, cfg, momentum=0, dampening=0,
                 weight_decay=0, nesterov=True,
                 reinit=True, use_identity_scales_for_reinit=True,
                 cpu_mode=False):

        # 设置优化器的默认参数
        defaults = dict(lr=cfg.solver.lr0, momentum=cfg.solver.momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        # 检查 Nesterov 动量的条件
        if nesterov and (cfg.solver.momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        # parameters = set_weight_decay(model)
        # 获取优化器参数
        parameters = get_optimizer_param(args, cfg, model)
        # 调用父类构造函数
        super(SGD, self).__init__(parameters, defaults)
        # 记录尺度的数量
        self.num_layers = len(scales)

        # 提取模型中的块
        blocks = []
        extract_blocks_into_list(model, blocks)
        # 获取卷积层
        convs = [b.conv for b in blocks]
        # 确保尺度和卷积层数量一致
        assert len(scales) == len(convs)

        # 如果需要重新初始化
        if reinit:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    # 检查 BatchNorm2d 模块的权重均值
                    gamma_init = m.weight.mean()
                    if gamma_init == 1.0:
                        LOGGER.info('Checked. This is training from scratch.')
                    else:
                        LOGGER.warning('========================== Warning! Is this really training from scratch ? =================')
            LOGGER.info('##################### Re-initialize #############')
            # 重新初始化卷积层
            self.reinitialize(scales, convs, use_identity_scales_for_reinit)

        # 生成梯度掩码
        self.generate_gradient_masks(scales, convs, cpu_mode)

    def reinitialize(self, scales_by_idx, conv3x3_by_idx, use_identity_scales):
        # 重新初始化卷积层权重
        for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
            in_channels = conv3x3.in_channels
            out_channels = conv3x3.out_channels
            # 创建 1x1 卷积层
            kernel_1x1 = nn.Conv2d(in_channels, out_channels, 1, device=conv3x3.weight.device)
            if len(scales) == 2:
                # 如果有两个尺度，更新卷积层权重
                conv3x3.weight.data = conv3x3.weight * scales[1].view(-1, 1, 1, 1) \
                                      + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[0].view(-1, 1, 1, 1)
            else:
                assert len(scales) == 3
                assert in_channels == out_channels
                # 创建单位矩阵
                identity = torch.from_numpy(np.eye(out_channels, dtype=np.float32).reshape(out_channels, out_channels, 1, 1)).to(conv3x3.weight.device)
                # 更新卷积层权重
                conv3x3.weight.data = conv3x3.weight * scales[2].view(-1, 1, 1, 1) + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[1].view(-1, 1, 1, 1)
                if use_identity_scales:     # 可以使用训练好的 identity_scale 值初始化虚拟的 CSLA 块，几乎没有差别
                    identity_scale_weight = scales[0]
                    conv3x3.weight.data += F.pad(identity * identity_scale_weight.view(-1, 1, 1, 1), [1, 1, 1, 1])
                else:
                    conv3x3.weight.data += F.pad(identity, [1, 1, 1, 1])

    def generate_gradient_masks(self, scales_by_idx, conv3x3_by_idx, cpu_mode=False):
        # 生成梯度掩码
        self.grad_mask_map = {}
        for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
            para = conv3x3.weight
            if len(scales) == 2:
                # 创建掩码
                mask = torch.ones_like(para, device=scales[0].device) * (scales[1] ** 2).view(-1, 1, 1, 1)
                mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1, device=scales[0].device) * (scales[0] ** 2).view(-1, 1, 1, 1)
            else:
                mask = torch.ones_like(para, device=scales[0].device) * (scales[2] ** 2).view(-1, 1, 1, 1)
                mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1, device=scales[0].device) * (scales[1] ** 2).view(-1, 1, 1, 1)
                ids = np.arange(para.shape[1])
                assert para.shape[1] == para.shape[0]
                mask[ids, ids, 1:2, 1:2] += 1.0
            # 根据 cpu_mode 决定掩码的存储方式
            if cpu_mode:
                self.grad_mask_map[para] = mask
            else:
                self.grad_mask_map[para] = mask.cuda()

    def __setstate__(self, state):
        # 设置优化器的状态
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        # 执行一步优化
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue  # 跳过没有梯度的参数

                # 应用梯度掩码
                if p in self.grad_mask_map:
                    d_p = p.grad.data * self.grad_mask_map[p]  # 注意：在这里乘以掩码
                else:
                    d_p = p.grad.data

                # 应用权重衰减
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # 更新参数
                p.data.add_(-group['lr'], d_p)

        return loss