#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from ast import Pass
import os
import time
from copy import deepcopy
import os.path as osp

from tqdm import tqdm

import cv2
import numpy as np
import math
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import tools.eval as eval
from yolov6.data.data_load import create_dataloader
from yolov6.models.yolo import build_model
from yolov6.models.yolo_lite import build_model as build_lite_model

from yolov6.models.losses.loss import ComputeLoss as ComputeLoss
from yolov6.models.losses.loss_fuseab import ComputeLoss as ComputeLoss_ab
from yolov6.models.losses.loss_distill import ComputeLoss as ComputeLoss_distill
from yolov6.models.losses.loss_distill_ns import ComputeLoss as ComputeLoss_distill_ns

from yolov6.utils.events import LOGGER, NCOLS, load_yaml, write_tblog, write_tbimg
from yolov6.utils.ema import ModelEMA, de_parallel
from yolov6.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
from yolov6.solver.build import build_optimizer, build_lr_scheduler
from yolov6.utils.RepOptimizer import extract_scales, RepVGGOptimizer
from yolov6.utils.nms import xywh2xyxy
from yolov6.utils.general import download_ckpt


class Trainer:
    """
    Trainer 类负责管理训练过程。
    它初始化训练相关的参数，加载数据，创建模型，并配置优化器和学习率调度器。
    """
    def __init__(self, args, cfg, device):
        """
        初始化 Trainer 实例。

        参数:
            args: 命令行参数，包含用户指定的参数。
            cfg: 从配置文件中加载的配置参数。
            device: 指定模型运行的硬件设备，如 CPU 或 GPU。
        """
        # 初始化成员变量
        self.args = args
        self.cfg = cfg
        self.device = device
        self.max_epoch = args.epochs

        # 如果指定了恢复训练，加载检查点
        if args.resume:
            self.ckpt = torch.load(args.resume, map_location='cpu')

        # 初始化分布式训练参数
        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir

        # 加载数据字典并获取类别数量
        # get data loader
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict['nc']

        # 判断是否启用蒸馏模式，如果是，则创建教师模型
        # get model and optimizer
        self.distill_ns = True if self.args.distill and self.cfg.model.type in ['YOLOv6n','YOLOv6s'] else False

        # 获取模型和优化器
        model = self.get_model(args, cfg, self.num_classes, device)

        # 如果启用了蒸馏模式，获取教师模型
        if self.args.distill:
            if self.args.fuse_ab:
                LOGGER.error('ERROR in: Distill models should turn off the fuse_ab.\n')
                exit()
            self.teacher_model = self.get_teacher_model(args, cfg, self.num_classes, device)

        # 如果启用了量化，设置量化参数
        if self.args.quant:
            self.quant_setup(model, cfg, device)

        # 根据训练模式选择优化器
        if cfg.training_mode == 'repopt':
            scales = self.load_scale_from_pretrained_models(cfg, device)
            reinit = False if cfg.model.pretrained is not None else True
            self.optimizer = RepVGGOptimizer(model, scales, args, cfg, reinit=reinit)
        else:
            self.optimizer = self.get_optimizer(args, cfg, model)

        # 获取学习率调度器
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)

        # 初始化 EMA 模型
        self.ema = ModelEMA(model) if self.main_process else None

        # 初始化 TensorBoard 记录器
        # tensorboard
        self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None

        # 初始化起始轮数
        self.start_epoch = 0

        # 如果有检查点，恢复状态
        #resume
        if hasattr(self, "ckpt"):
            resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(resume_state_dict, strict=True)  # load
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.scheduler.load_state_dict(self.ckpt['scheduler'])
            if self.main_process:
                self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']
            if self.start_epoch > (self.max_epoch - self.args.stop_aug_last_n_epoch):
                self.cfg.data_aug.mosaic = 0.0
                self.cfg.data_aug.mixup = 0.0

        # 获取数据加载器
        self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)

        # 并行化模型
        self.model = self.parallel_model(args, model, device)
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']

        # 设置最大步数、批量大小、图像大小等参数
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.rect = args.rect
        self.vis_imgs_list = []
        self.write_trainbatch_tb = args.write_trainbatch_tb

        # 为类别名称设置颜色
        # set color for classnames
        self.color = [tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)]

        # 设置特定形状、高度和宽度
        self.specific_shape = args.specific_shape
        self.height = args.height
        self.width = args.width

        # 设置损失项数量和损失信息
        self.loss_num = 3
        self.loss_info = ['Epoch', 'lr', 'iou_loss', 'dfl_loss', 'cls_loss']
        if self.args.distill:
            self.loss_num += 1
            self.loss_info += ['cwd_loss']


    # Training Process
    def train(self):
        """
        训练模型的主要方法，包含训练前的准备、 epoch 循环训练、以及训练结束后的处理。
        """
        try:
            # 执行训练前的准备工作
            self.before_train_loop()

            # 进入训练的 epoch 循环
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # 每个 epoch 训练前的准备工作
                self.before_epoch()

                # 执行一个 epoch 的训练
                self.train_one_epoch(self.epoch)

                # 每个 epoch 训练结束后的处理工作
                self.after_epoch()

            # 训练结束后，清理模型，例如移除不必要的权重层
            self.strip_model()

        except Exception as _:
            # 捕获训练过程中可能发生的异常，并记录错误日志
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            # 无论是否发生异常，最终执行训练后的收尾工作
            self.train_after_loop()

    # Training loop for each epoch
    def train_one_epoch(self, epoch_num):
        """
        训练一个epoch的数据。

        该函数迭代地处理每个训练步骤，并尝试打印训练详情。
        如果在训练步骤中发生异常，则记录错误并重新抛出。

        参数:
        - epoch_num (int): 当前训练的epoch编号。

        返回:
        无返回值。
        """
        try:
            # 迭代处理每个训练步骤，pbar是进度条对象，提供当前步骤号和批次数据
            for self.step, self.batch_data in self.pbar:
                # 在指定的epoch和步骤中进行训练
                self.train_in_steps(epoch_num, self.step)
                # 打印训练详情
                self.print_details()
        except Exception as _:
            # 如果在训练步骤中发生异常，记录错误并重新抛出
            LOGGER.error('ERROR in training steps.')
            raise

    # Training one batch data.
    def train_in_steps(self, epoch_num, step_num):
        """
        按步骤训练模型。

        该函数负责预处理数据，每轮训练的第一个步骤绘制训练批次并保存到TensorBoard，
        执行前向传播和反向传播，计算损失，并更新优化器。

        参数:
        epoch_num (int): 当前轮数。
        step_num (int): 当前步骤号。
        """
        # 对训练数据进行预处理，包括数据预处理和将数据移动到指定设备
        images, targets = self.prepro_data(self.batch_data, self.device)

        # 每个epoch开始时绘制训练批次并保存到TensorBoard
        # 根据特定条件决定是否绘制和记录训练批次图像
        # plot train_batch and save to tensorboard once an epoch
        if self.write_trainbatch_tb and self.main_process and self.step == 0:
            # 调用绘图函数处理并生成训练批次图像
            self.plot_train_batch(images, targets)
            # 将处理后的图像写入TensorBoard以可视化训练批次数据
            write_tbimg(self.tblogger, self.vis_train_batch, self.step + self.max_stepnum * self.epoch, type='train')

        # forward
        # 使用自动混合精度训练，如果设备不是CPU则启用,否则不启用
        with amp.autocast(enabled=self.device != 'cpu'):
            # 获取输入图像的维度信息，包括批次大小、通道数、高度和宽度
            _, _, batch_height, batch_width = images.shape
            # 将图像输入模型进行前向传播，获取预测结果和学生模型的特征图
            preds, s_featmaps = self.model(images)
            # 如果启用知识蒸馏，使用教师模型进行前向传播，获取教师模型的预测结果和特征图
            if self.args.distill:
                with torch.no_grad():
                    t_preds, t_featmaps = self.teacher_model(images)
                # 获取温度参数，用于知识蒸馏中的软化概率分布
                temperature = self.args.temperature
                # 计算知识蒸馏损失，包括学生模型和教师模型的预测结果和特征图、目标、当前epoch、最大epoch、温度、步数和图像尺寸
                total_loss, loss_items = self.compute_loss_distill(preds, t_preds, s_featmaps, t_featmaps, targets, \
                                                                  epoch_num, self.max_epoch, temperature, step_num,
                                                                  batch_height, batch_width)

            # 如果设置了fuse_ab参数，则同时计算两种不同模型架构的损失并合并到总损失中
            elif self.args.fuse_ab:
                # 计算YOLOv6_af模型的损失,包括预测结果、目标、当前epoch、步数和图像尺寸
                total_loss, loss_items = self.compute_loss((preds[0],preds[3],preds[4]), targets, epoch_num,
                                                            step_num, batch_height, batch_width) # YOLOv6_af

                # 计算YOLOv6_ab模型的损失，并将其加到之前的损失中，即total_loss += total_loss_ab
                total_loss_ab, loss_items_ab = self.compute_loss_ab(preds[:3], targets, epoch_num, step_num,
                                                                     batch_height, batch_width) # YOLOv6_ab

                # 合并两种模型的总损失
                total_loss += total_loss_ab

                # 合并两种模型的损失项
                loss_items += loss_items_ab
            else:
                # 如果不满足某些特定条件，则正常计算损失，包括预测结果、目标、当前epoch、步数和图像尺寸
                total_loss, loss_items = self.compute_loss(preds, targets, epoch_num, step_num,
                                                            batch_height, batch_width) # YOLOv6_af

            # 如果使用分布式训练，根据世界大小缩放总损失
            if self.rank != -1:
                total_loss *= self.world_size

        # 反向传播
        # 执行反向传播，计算模型参数的梯度
        # backward
        self.scaler.scale(total_loss).backward()

        # 存储损失项，以便可能的监控或日志记录
        self.loss_items = loss_items

        # 更新优化器，可能调整学习率或执行梯度下降
        self.update_optimizer()



    def after_epoch(self):
        """
        在每个训练epoch结束后调用此函数，以执行学习率更新、模型评估和检查点保存等任务。
        """
        # 记录当前epoch的学习率
        lrs_of_this_epoch = [x['lr'] for x in self.optimizer.param_groups]
        # 更新学习率调度器
        self.scheduler.step()  # update lr
        # 主进程执行以下操作
        if self.main_process:
            # 更新模型的EMA属性
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'])  # update attributes for ema model

            # 计算剩余的训练epoch数
            remaining_epochs = self.max_epoch - 1 - self.epoch  # self.epoch is start from 0
            # 根据剩余的epoch数和配置，确定评估模型的间隔
            eval_interval = self.args.eval_interval if remaining_epochs >= self.args.heavy_eval_range else min(3,
                                                                                                               self.args.eval_interval)
            # 判断当前epoch是否需要进行模型评估
            is_val_epoch = (remaining_epochs == 0) or (
                        (not self.args.eval_final_only) and ((self.epoch + 1) % eval_interval == 0))
            # 如果是评估epoch，则执行模型评估
            if is_val_epoch:
                self.eval_model()
                # 更新当前的平均精度和最佳平均精度
                self.ap = self.evaluate_results[1]
                self.best_ap = max(self.ap, self.best_ap)

            # 保存检查点
            ckpt = {
                'model': deepcopy(de_parallel(self.model)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'results': self.evaluate_results,
            }

            # 定义保存检查点的目录
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            # 保存最后一个检查点，如果是评估epoch且当前平均精度为最佳，则保存为最佳模型
            save_checkpoint(ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt')
            # 在训练的最后n个epoch中，每个epoch都保存检查点
            if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
                save_checkpoint(ckpt, False, save_ckpt_dir, model_name=f'{self.epoch}_ckpt')

            # 在停止强增强的最后n个epoch中，如果当前平均精度为最佳，则保存为最佳模型
            if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
                if self.best_stop_strong_aug_ap < self.ap:
                    self.best_stop_strong_aug_ap = max(self.ap, self.best_stop_strong_aug_ap)
                    save_checkpoint(ckpt, False, save_ckpt_dir, model_name='best_stop_aug_ckpt')

            # 释放检查点占用的内存
            del ckpt

            # 将评估结果转换为列表格式
            self.evaluate_results = list(self.evaluate_results)

            # 记录TensorBoard日志
            write_tblog(self.tblogger, self.epoch, self.evaluate_results, lrs_of_this_epoch, self.mean_loss)
            # 将验证预测结果保存到TensorBoard
            write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')

    def eval_model(self):
        """
        评估模型在验证数据集上的表现。此函数会根据配置中是否存在 `eval_params` 来选择不同的评估参数。
        如果没有定义 `eval_params`，则使用默认参数进行评估；否则，使用 `eval_params` 中指定的参数进行评估。
        """
        if not hasattr(self.cfg, "eval_params"):
            # 没有自定义评估参数，使用默认设置进行评估
            results, vis_outputs, vis_paths = eval.run(self.data_dict,
                            batch_size=self.batch_size // self.world_size * 2,
                            img_size=self.img_size,
                            model=self.ema.ema if self.args.calib is False else self.model,
                            conf_thres=0.03,
                            dataloader=self.val_loader,
                            save_dir=self.save_dir,
                            task='train',
                            specific_shape=self.specific_shape,
                            height=self.height,
                            width=self.width
                            )
        else:
            # 定义一个辅助函数来获取配置值
            def get_cfg_value(cfg_dict, value_str, default_value):
                # 获取指定的配置值，如果未找到则返回默认值
                if value_str in cfg_dict:
                    if isinstance(cfg_dict[value_str], list):
                        return cfg_dict[value_str][0] if cfg_dict[value_str][0] is not None else default_value
                    else:
                        return cfg_dict[value_str] if cfg_dict[value_str] is not None else default_value
                else:
                    return default_value

            # 使用自定义评估参数进行评估
            eval_img_size = get_cfg_value(self.cfg.eval_params, "img_size", self.img_size)
            results, vis_outputs, vis_paths = eval.run(self.data_dict,
                            batch_size=get_cfg_value(self.cfg.eval_params, "batch_size", self.batch_size // self.world_size * 2),
                            img_size=eval_img_size,
                            model=self.ema.ema if self.args.calib is False else self.model,
                            conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres", 0.03),
                            dataloader=self.val_loader,
                            save_dir=self.save_dir,
                            task='train',
                            shrink_size=get_cfg_value(self.cfg.eval_params, "shrink_size", eval_img_size),
                            infer_on_rect=get_cfg_value(self.cfg.eval_params, "infer_on_rect", False),
                            verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                            do_coco_metric=get_cfg_value(self.cfg.eval_params, "do_coco_metric", True),
                            do_pr_metric=get_cfg_value(self.cfg.eval_params, "do_pr_metric", False),
                            plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve", False),
                            plot_confusion_matrix=get_cfg_value(self.cfg.eval_params, "plot_confusion_matrix", False),
                            specific_shape=self.specific_shape,
                            height=self.height,
                            width=self.width
                            )
        # 记录评估结果
        LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
        self.evaluate_results = results[:2]
        # 绘制验证预测结果
        # plot validation predictions
        self.plot_val_pred(vis_outputs, vis_paths)


    def before_train_loop(self):
        """
        准备训练前的必要操作和设置。
        该函数初始化和配置训练所需的各个参数和对象，包括学习率预热设置、训练监控变量的初始化、
        模型权重的加载以及损失计算方法的初始化。
        """
        # 记录训练开始时间
        LOGGER.info('Training start...')

        # 记录训练开始时间，用于后续计算训练持续时间
        self.start_time = time.time()

        # 根据配置计算预热步数，如果启用量化训练则设置为0
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum), 1000) if self.args.quant is False else 0

        # 设置学习率调度器的最后一个epoch，确保学习率从正确的epoch开始
        self.scheduler.last_epoch = self.start_epoch - 1

        # 初始化最后优化步骤的步数为-1，表示尚未进行优化
        self.last_opt_step = -1

        # 初始化混合精度训练的梯度缩放器，仅在设备不是CPU时启用
        self.scaler = amp.GradScaler(enabled=self.device != 'cpu')

        # 初始化最佳平均精度（AP）和当前AP值为0
        self.best_ap, self.ap = 0.0, 0.0

        # 初始化评估结果元组为(0, 0)，分别表示(AP50, AP50_95)
        self.best_stop_strong_aug_ap = 0.0
        self.evaluate_results = (0, 0) # AP50, AP50_95

        # 如果存在检查点，从检查点加载评估结果和最佳AP值
        # resume results
        if hasattr(self, "ckpt"):
            self.evaluate_results = self.ckpt['results']
            self.best_ap = self.evaluate_results[1]
            self.best_stop_strong_aug_ap = self.evaluate_results[1]

        # 根据模型配置和数据配置初始化损失计算对象
        self.compute_loss = ComputeLoss(num_classes=self.data_dict['nc'],
                                        ori_img_size=self.img_size,
                                        warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                        use_dfl=self.cfg.model.head.use_dfl,
                                        reg_max=self.cfg.model.head.reg_max,
                                        iou_type=self.cfg.model.head.iou_type,
					                    fpn_strides=self.cfg.model.head.strides)

        # 如果使用A和B的特征融合，也初始化相应的损失计算对象
        if self.args.fuse_ab:
            self.compute_loss_ab = ComputeLoss_ab(num_classes=self.data_dict['nc'],
                                        ori_img_size=self.img_size,
                                        warmup_epoch=0,
                                        use_dfl=False,
                                        reg_max=0,
                                        iou_type=self.cfg.model.head.iou_type,
                                        fpn_strides=self.cfg.model.head.strides,
                                        )

        # 如果启用了知识蒸馏训练，根据模型类型初始化相应的损失计算对象
        if self.args.distill :
            if self.cfg.model.type in ['YOLOv6n','YOLOv6s']:
                Loss_distill_func = ComputeLoss_distill_ns
            else:
                Loss_distill_func = ComputeLoss_distill

            self.compute_loss_distill = Loss_distill_func(num_classes=self.data_dict['nc'],
                                                        ori_img_size=self.img_size,
                                                        fpn_strides=self.cfg.model.head.strides,
                                                        warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                                        use_dfl=self.cfg.model.head.use_dfl,
                                                        reg_max=self.cfg.model.head.reg_max,
                                                        iou_type=self.cfg.model.head.iou_type,
                                                        distill_weight = self.cfg.model.head.distill_weight,
                                                        distill_feat = self.args.distill_feat,
                                                        )

    def before_epoch(self):
        """
        在每个训练周期开始前执行必要的操作。
        该方法主要关注是否根据当前周期停止强数据增强方法，必要时更新数据加载器，准备模型和优化器进行训练，并初始化相关变量。
        """
        # 当达到由 stop_aug_last_n_epoch 指定的周期时，禁用强数据增强方法。
        #stop strong aug like mosaic and mixup from last n epoch by recreate dataloader
        if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
            # 禁用 RAM 缓存以节省内存使用
            self.args.cache_ram = False # disable cache ram when stop strong augmentation.
            # 重新创建数据加载器以应用更新的数据增强设置。
            self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)

        # 将模型设置为训练模式。这将启用模型中的所有可训练参数，如权重、偏置等。
        self.model.train()

        # 如果使用分布式训练，同步训练加载器的采样器。
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)

        # 初始化均值损失张量，用于存储本周期的累积损失。
        self.mean_loss = torch.zeros(self.loss_num, device=self.device)

        # 清除优化器中的梯度，为计算本周期的梯度做准备
        self.optimizer.zero_grad()

        # 记录损失信息的表头
        LOGGER.info(('\n' + '%10s' * (self.loss_num + 2)) % (*self.loss_info,))

        # 初始化本周期的进度条，用于显示训练进度
        self.pbar = enumerate(self.train_loader)

        # 如果当前进程是主进程，配置进度条显示
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    # Print loss after each steps

    def print_details(self):
        """
        更新并打印训练的详细信息。

        该方法主要用于更新和打印与训练过程相关的详细信息，包括平均损失和学习率等。
        如果当前进程是主进程，则执行以下操作：
        1. 计算平均损失，将当前损失项加入平均损失计算中。
        2. 更新进度条的描述信息，包括当前的训练周期、学习率和平均损失。
        """
        if self.main_process:
            # 计算新的平均损失值
            self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)
            # 更新进度条描述信息，包括训练周期、学习率和平均损失
            self.pbar.set_description(
                ('%10s' + ' %10.4g' + '%10.4g' * self.loss_num) % (f'{self.epoch}/{self.max_epoch - 1}', \
                                                                   self.scheduler.get_last_lr()[0], *(self.mean_loss)))

    def strip_model(self):
        """
            当主进程完成训练后，记录训练时间，并优化保存的模型。

            本函数在主进程(self.main_process为True)完成训练后调用。首先，它计算并记录自训练开始
            (self.start_time)到当前时间的总小时数。然后，确定保存模型的目录，并调用strip_optimizer函数
            来优化该目录下保存的模型文件，以减少模型文件的大小或提高模型的加载速度。
            """
        if self.main_process:
            # 如果是主进程，记录训练完成所需的时间
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')
            # 定义保存模型权重的目录
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            # 调用函数优化保存的pt模型的大小
            strip_optimizer(save_ckpt_dir, self.epoch)  # strip optimizers for saved pt model

    # Empty cache if training finished
    def train_after_loop(self):
        """
        在训练循环结束后执行的操作。

        此方法旨在清理GPU设备上的未使用缓存。当self.device表明设备不是CPU时，
        调用torch.cuda.empty_cache()来释放缓存，以优化内存使用，确保后续操作
        有足够的内存资源可用。
        """
        if self.device != 'cpu':
            torch.cuda.empty_cache()

    def update_optimizer(self):
        """
        更新优化器状态，主要包括调整累积梯度的次数、学习率和动量的暖启动，以及在适当的时候进行模型的参数更新和EMA更新。
        该方法主要用于训练过程中的优化器状态调整，以提高模型的训练效果。
        """
        # 计算当前训练步数
        curr_step = self.step + self.max_stepnum * self.epoch
        # 计算梯度累积次数，确保在小批量大小时不会过小
        self.accumulate = max(1, round(64 / self.batch_size))

        # 如果当前步数小于等于暖启动步数，则应用暖启动策略
        if curr_step <= self.warmup_stepnum:
            # 调整累积梯度的次数，以适应暖启动阶段
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                # 根据配置，设置偏置项的学习率
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                # 线性插值计算当前学习率
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                        [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                # 如果存在动量参数，则进行动量的暖启动
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                                  [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])

        # 如果从上次优化到现在的步数差大于等于累积次数，则进行优化器更新
        if curr_step - self.last_opt_step >= self.accumulate:
            # 使用梯度缩放来帮助优化器更新
            self.scaler.step(self.optimizer)
            # 更新梯度缩放因子
            self.scaler.update()
            # 清零梯度
            self.optimizer.zero_grad()
            # 如果启用了EMA，则更新EMA
            if self.ema:
                self.ema.update(self.model)
            # 更新最后一步优化的步数
            self.last_opt_step = curr_step

    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        """
        创建并返回训练和验证数据加载器。

        参数:
        - args: 命令行参数对象，包含运行配置。
        - cfg: 配置对象，包含模型和数据增强配置。
        - data_dict: 数据集信息字典，包含训练和验证数据路径、类别数和类别名称。

        返回:
        - train_loader: 训练数据加载器。
        - val_loader: 验证数据加载器，仅在需要时创建。
        """
        # 提取训练和验证数据路径
        train_path, val_path = data_dict['train'], data_dict['val']
        # 检查数据集的类别数与类别名称是否匹配
        nc = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
        # 计算网格大小，用于数据加载器的创建
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # 创建训练数据加载器
        train_loader = create_dataloader(train_path, args.img_size, args.batch_size // args.world_size, grid_size,
                                         hyp=dict(cfg.data_aug), augment=True, rect=args.rect, rank=args.local_rank,
                                         workers=args.workers, shuffle=True, check_images=args.check_images,
                                         check_labels=args.check_labels, data_dict=data_dict, task='train',
                                         specific_shape=args.specific_shape, height=args.height, width=args.width,
                                         cache_ram=args.cache_ram)[0]
        # 初始化验证数据加载器为None
        val_loader = None
        # 在主进程或未使用分布式训练时，创建验证数据加载器
        if args.rank in [-1, 0]:
            # TODO: check whether to set rect to self.rect?
            val_loader = create_dataloader(val_path, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                                           hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                                           workers=args.workers, check_images=args.check_images,
                                           check_labels=args.check_labels, data_dict=data_dict, task='val',
                                           specific_shape=args.specific_shape, height=args.height, width=args.width,
                                           cache_ram=args.cache_ram)[0]

        # 返回训练和验证数据加载器
        return train_loader, val_loader

    @staticmethod
    def prepro_data(batch_data, device):
        """
        预处理数据函数。

        该函数将一批数据转换到指定设备上，并对图像数据进行归一化处理。

        参数:
        batch_data: 一批数据，包含图像和目标。
        device: 设备信息，用于指定数据转换的目标设备。

        返回:
        images: 转换并归一化后的图像数据。
        targets: 转换后的目标数据。
        """
        # 将图像数据转换到指定设备，并进行归一化处理
        images = batch_data[0].to(device, non_blocking=True).float() / 255
        # 将目标数据转换到指定设备
        targets = batch_data[1].to(device)
        # 返回处理后的图像和目标数据
        return images, targets

    def get_model(self, args, cfg, nc, device):
        """
        根据配置和设备信息构建模型。

        参数:
        - args: 命令行参数，包含一些模型配置选项。
        - cfg: 模型配置对象，包含模型的详细配置信息。
        - nc: 类别数量，用于模型输出层的配置。
        - device: 设备信息，指示使用CPU还是GPU。

        返回:
        - model: 构建好的模型对象。
        """
        # 根据配置信息构建相应的模型
        if 'YOLOv6-lite' in cfg.model.type:
            # 确保轻量级模型不使用不支持的模式
            assert not self.args.fuse_ab, 'ERROR in: YOLOv6-lite models not support fuse_ab mode.'
            assert not self.args.distill, 'ERROR in: YOLOv6-lite models not support distill mode.'
            # 构建轻量级模型
            model = build_lite_model(cfg, nc, device)
        else:
            # 构建标准模型
            model = build_model(cfg, nc, device, fuse_ab=self.args.fuse_ab, distill_ns=self.distill_ns)

        # 获取预训练权重路径
        weights = cfg.model.pretrained
        # 如果设置了预训练模型权重，则进行微调
        if weights:
            # 如果权重文件不存在，则下载
            if not os.path.exists(weights):
                download_ckpt(weights)
            # 加载预训练模型的权重
            LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
            model = load_state_dict(weights, model, map_location=device)

        # 打印模型信息
        LOGGER.info('Model: {}'.format(model))
        return model



    def get_teacher_model(self, args, cfg, nc, device):
        """
        加载并配置教师模型。

        Args:
            args: 命令行参数或配置，包含预训练模型的路径。
            cfg: 模型配置，包括模型架构参数。
            nc: 类别数量。
            device: 运行模型的设备，如CPU或GPU。

        Returns:
            model: 加载并配置好的教师模型。
        """
        # 根据模型头部的层数确定 teacher_fuse_ab 的值
        teacher_fuse_ab = False if cfg.model.head.num_layers != 3 else True

        # 根据配置、类别数量、设备和确定的 teacher_fuse_ab 值构建模型
        model = build_model(cfg, nc, device, fuse_ab=teacher_fuse_ab)

        # 如果提供了预训练模型权重，则加载这些权重
        weights = args.teacher_model_path
        if weights:  # 如果设置了预训练模型，则微调
            LOGGER.info(f'从 {weights} 加载 state_dict 以用于教师模型')
            model = load_state_dict(weights, model, map_location=device)

        # 记录模型结构
        LOGGER.info('模型: {}'.format(model))

        # 不更新运行均值和方差
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False

        # 返回配置好的教师模型
        return model

    @staticmethod
    def load_scale_from_pretrained_models(cfg, device):
        """
        从预训练模型中加载尺度参数。

        该函数旨在从提供的模型权重中提取尺度参数，以便初始化RepOptimizer。
        如果没有提供尺度参数，则会记录错误信息。

        参数:
        - cfg: 包含模型配置的配置对象，包括尺度参数的路径。
        - device: 用于加载模型权重的设备，如CPU或GPU。

        返回:
        - scales: 提取的尺度参数，如果没有提供权重，则为None。
        """
        # 获取配置中的尺度参数权重路径
        weights = cfg.model.scales
        # 初始化尺度参数为None
        scales = None

        # 检查是否提供了尺度参数权重路径
        if not weights:
            # 如果没有提供，记录错误信息
            LOGGER.error("ERROR: No scales provided to init RepOptimizer!")
        else:
            # 如果提供了，加载模型权重
            ckpt = torch.load(weights, map_location=device)
            # 从加载的权重中提取尺度参数
            scales = extract_scales(ckpt)

        # 返回提取的尺度参数
        return scales


    @staticmethod
    def parallel_model(args, model, device):
        """
        配置模型以在多个GPU上并行执行。

        该函数根据设备类型和命令行参数决定使用DataParallel (DP) 或 DistributedDataParallel (DDP) 模式，
        并相应地包装模型以启用并行执行。

        参数:
        - args: 命令行参数或配置对象，期望包含 'rank' 和 'local_rank' 属性，用于确定模式和设备设置。
        - model: 要配置为并行执行的神经网络模型。
        - device: 表示设备类型（例如，'cpu'，'cuda'）的设备对象，模型将在其上执行。

        返回:
        - model: 根据指定条件，被 DataParallel 或 DistributedDataParallel 包装的模型。
        """

        # 判断是否为DP模式
        # 检查设备不是CPU且rank为-1，表示应使用DP模式
        # If DP mode
        dp_mode = device.type != 'cpu' and args.rank == -1
        # 在DP模式下，如果有多于一个GPU可用，将模型包装为DataParallel
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)

        # 判断是否为DDP模式
        # 检查设备不是CPU且rank不为-1，表示应使用DDP模式
        # If DDP mode
        ddp_mode = device.type != 'cpu' and args.rank != -1
        # 在DDP模式下，将模型包装为DistributedDataParallel
        if ddp_mode:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

        return model

    def get_optimizer(self, args, cfg, model):
        """
        根据配置和参数构建并返回优化器。

        本函数旨在根据给定的配置文件（cfg）、命令行参数（args）和模型（model）来构建一个优化器。
        它会根据全局批量大小调整学习率和权重衰减率，以确保在不同硬件配置下保持一致的优化行为。

        参数:
        - args: 命令行参数，包含训练设置如批量大小等。
        - cfg: 配置文件，包含训练的详细配置信息，如优化器设置等。
        - model: 训练所用的模型。

        返回:
        - optimizer: 构建好的优化器实例。
        """
        # 计算梯度累积步数，确保在小批量大小下能有效更新权重
        accumulate = max(1, round(64 / args.batch_size))

        # 根据当前批量大小调整权重衰减，以保持在不同批量大小下的一致性
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64

        # 根据当前批量大小调整基础学习率，以适应不同规模的训练环境
        cfg.solver.lr0 *= args.batch_size / (self.world_size * args.bs_per_gpu) # rescale lr0 related to batchsize

        # 使用更新后的配置构建优化器
        optimizer = build_optimizer(cfg, model)

        # 返回构建好的优化器实例
        return optimizer

    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        """
        根据配置和优化器构建学习率调度器。

        参数:
        args: 包含训练参数的对象，如训练的总轮数（epochs）。
        cfg: 包含学习率调度配置信息的对象。
        optimizer: 训练过程中使用的优化器。

        返回:
        lr_scheduler: 学习率调度器对象，用于在训练过程中调整学习率。
        lf: 学习率变化因子，可以是函数或查找表，定义了学习率如何随训练轮数变化。
        """
        # 获取训练的总轮数
        epochs = args.epochs

        # 构建学习率调度器和学习率变化因子
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)

        # 返回学习率调度器和学习率变化因子
        return lr_scheduler, lf

    def plot_train_batch(self, images, targets, max_size=1920, max_subplots=16):
        """
        绘制带有标签的训练批次图像。

        参数:
            images: 一批训练图像。
            targets: 对应于这批图像的标签信息。
            max_size: 生成图像的最大尺寸。
            max_subplots: 最大子图数量。
        """
        # 如果输入是张量，则转换为 numpy 数组
        # Plot train_batch with labels
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        # 如果图像的最大值不超过 1，则可选地反归一
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)
        # 从图像形状中提取批量大小、高度和宽度
        bs, _, h, w = images.shape  # batch size, _, height, width
        # 限制绘制的图像数量，以避免过多的子图
        bs = min(bs, max_subplots)  # limit plot images
        # 计算子图数量（方形布局）
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)
        # 获取图像路径
        paths = self.batch_data[2]  # image paths

        # 初始化一个空白的马赛克图像，用于存储所有图像的拼接结果
        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        # 将每个图像填充到马赛克图像中
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y:y + h, x:x + w, :] = im

        # 可选地调整马赛克图像的大小，以适应指定的最大尺寸
        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

        # 在每个图像子图上添加边框和文件名称
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)  # borders
            cv2.putText(mosaic, f"{os.path.basename(paths[i])[:40]}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(220, 220, 220), thickness=1)  # filename

            # 如果有目标信息，则在图像上添加边界框和标签
            if len(targets) > 0:
                ti = targets[targets[:, 0] == i]  # image targets
                boxes = xywh2xyxy(ti[:, 2:6]).T
                classes = ti[:, 1].astype('int')
                labels = ti.shape[1] == 6  # labels if no conf column
                if boxes.shape[1]:
                    if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                        boxes[[0, 2]] *= w  # scale to pixels
                        boxes[[1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes *= scale
                boxes[[0, 2]] += x
                boxes[[1, 3]] += y
                for j, box in enumerate(boxes.T.tolist()):
                    box = [int(k) for k in box]
                    cls = classes[j]
                    color = tuple([int(x) for x in self.color[cls]])
                    cls = self.data_dict['names'][cls] if self.data_dict['names'] else cls
                    if labels:
                        label = f'{cls}'
                        cv2.rectangle(mosaic, (box[0], box[1]), (box[2], box[3]), color, thickness=1)
                        cv2.putText(mosaic, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, thickness=1)
        # 存储生成的马赛克图像
        self.vis_train_batch = mosaic.copy()

    def plot_val_pred(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=5):
        """
        绘制验证集预测结果。

        该函数用于绘制验证集的预测结果。它从指定路径读取图像，并根据预测结果绘制边界框、置信度和类别标签。

        参数:
        - vis_outputs: 模型的预测输出，包含边界框信息。
        - vis_paths: 预测输出对应的图像路径列表。
        - vis_conf: 置信度阈值，只有置信度高于此值的边界框才会被绘制。
        - vis_max_box_num: 每张图像上绘制的最大边界框数量。
        """
        # 初始化存储可视化图像的列表
        # plot validation predictions
        self.vis_imgs_list = []
        # 遍历预测输出和对应的图像路径
        for (vis_output, vis_path) in zip(vis_outputs, vis_paths):
            # 将预测输出转换为 numpy 数组以便后续处理
            vis_output_array = vis_output.cpu().numpy()     # xyxy
            # 读取原始图像
            ori_img = cv2.imread(vis_path)
            # 遍历预测输出中的边界
            for bbox_idx, vis_bbox in enumerate(vis_output_array):
                # 提取边界框的坐标和得分
                x_tl = int(vis_bbox[0])
                y_tl = int(vis_bbox[1])
                x_br = int(vis_bbox[2])
                y_br = int(vis_bbox[3])
                box_score = vis_bbox[4]
                cls_id = int(vis_bbox[5])
                # 只绘制置信度高于阈值且数量不超过最大限制的前 n 个边界框
                # draw top n bbox
                if box_score < vis_conf or bbox_idx > vis_max_box_num:
                    break
                # 在图像上绘制边界框
                cv2.rectangle(ori_img, (x_tl, y_tl), (x_br, y_br), tuple([int(x) for x in self.color[cls_id]]), thickness=1)
                # 在图像上绘制边界框的类别和得分
                cv2.putText(ori_img, f"{self.data_dict['names'][cls_id]}: {box_score:.2f}", (x_tl, y_tl - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple([int(x) for x in self.color[cls_id]]), thickness=1)
            # 将处理后的图像添加到可视化图像列表中
            self.vis_imgs_list.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))


    # PTQ
    def calibrate(self, cfg):
        """
        使用提供的配置校准模型。

        参数:
            cfg: 包含校准设置的配置对象。
        """
        def save_calib_model(model, cfg):
            """
            保存校准后的模型检查点。

            参数:
                model: 要保存的模型。
                cfg: 包含输出路径和校准方法的配置对象。
            """
            # 生成校准后模型的保存路径
            # Save calibrated checkpoint
            output_model_path = os.path.join(cfg.ptq.calib_output_path, '{}_calib_{}.pt'.
                                             format(os.path.splitext(os.path.basename(cfg.model.pretrained))[0], cfg.ptq.calib_method))
            # 如果跳过了敏感层，修改输出模型路径名称
            if cfg.ptq.sensitive_layers_skip is True:
                output_model_path = output_model_path.replace('.pt', '_partial.pt')
            # 记录校准后模型的保存路径
            LOGGER.info('Saving calibrated model to {}... '.format(output_model_path))
            # 如果输出目录不存在，则创建它
            if not os.path.exists(cfg.ptq.calib_output_path):
                os.mkdir(cfg.ptq.calib_output_path)
            # 保存模型状态
            torch.save({'model': deepcopy(de_parallel(model)).half()}, output_model_path)

        # 确保量化和校准已启用
        assert self.args.quant is True and self.args.calib is True
        # 在主进程中执行校准
        if self.main_process:
            # 导入校准工具
            from tools.qat.qat_utils import ptq_calibrate
            # 使用训练数据加载器和配置校准模型
            ptq_calibrate(self.model, self.train_loader, cfg)
            # 重置轮次计数器以进行评估
            self.epoch = 0
            # 评估校准后的模型
            self.eval_model()
            # 保存校准后的模型
            save_calib_model(self.model, cfg)
    # QAT
    def quant_setup(self, model, cfg, device):
        """
        准备模型进行量化。

        对模型进行量化设置，包括初始化量化模型、跳过敏感层、加载校准模型等。

        参数:
        - model: 待量化的模型。
        - cfg: 配置信息，包含量化和校准的配置。
        - device: 设备信息，指定模型运行的硬件设备。
        """
        # 如果开启了量化选项，则执行量化相关操作
        if self.args.quant:
            # 导入量化所需的工具函数
            from tools.qat.qat_utils import qat_init_model_manu, skip_sensitive_layers

            # 初始化模型的量化参数
            qat_init_model_manu(model, cfg, self.args)

            # 对模型的上采样部分进行量化使能
            # workaround
            model.neck.upsample_enable_quant(cfg.ptq.num_bits, cfg.ptq.calib_method)

            # 如果是主进程，打印模型结构
            # if self.main_process:
            #     print(model)

            # 如果不处于校准模式，则执行QAT（Quantization-Aware Training）流程
            # QAT
            if self.args.calib is False:
                # 跳过敏感层的量化
                if cfg.qat.sensitive_layers_skip:
                    skip_sensitive_layers(model, cfg.qat.sensitive_layers_list)

                # 加载已校准模型的权重
                # QAT flow load calibrated model
                assert cfg.qat.calib_pt is not None, 'Please provide calibrated model'
                model.load_state_dict(torch.load(cfg.qat.calib_pt)['model'].float().state_dict())
            # 将模型移动到指定设备上
            model.to(device)
