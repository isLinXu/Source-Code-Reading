#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp  # 保存实验配置
        self.args = args  # 保存命令行参数

        # training related attr
        self.max_epoch = exp.max_epoch  # 最大训练轮数
        self.amp_training = args.fp16  # 是否使用半精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)  # 创建梯度缩放器
        self.is_distributed = get_world_size() > 1  # 判断是否为分布式训练
        self.rank = get_rank()  # 获取当前进程的排名
        self.local_rank = get_local_rank()  # 获取当前进程的本地排名
        self.device = "cuda:{}".format(self.local_rank)  # 设置设备为对应的 CUDA 设备
        self.use_model_ema = exp.ema  # 是否使用模型的 EMA（指数移动平均）
        self.save_history_ckpt = exp.save_history_ckpt  # 是否保存历史检查点

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32  # 设置数据类型
        self.input_size = exp.input_size  # 输入尺寸
        self.best_ap = 0  # 最佳平均精度初始化为 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)  # 创建度量记录器
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)  # 创建输出文件夹路径

        if self.rank == 0:  # 只有主进程创建输出目录
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(  # 设置日志记录
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()  # 执行训练前的准备工作
        try:
            self.train_in_epoch()  # 开始训练每个轮次
        except Exception:  # 捕获异常
            raise
        finally:
            self.after_train()  # 执行训练后的清理工作

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):  # 遍历每个训练轮次
            self.before_epoch()  # 执行每个轮次前的准备工作
            self.train_in_iter()  # 开始训练每个迭代
            self.after_epoch()  # 执行每个轮次后的清理工作

    def train_in_iter(self):
        for self.iter in range(self.max_iter):  # 遍历每个迭代
            self.before_iter()  # 执行每个迭代前的准备工作
            self.train_one_iter()  # 执行单次迭代的训练
            self.after_iter()  # 执行每个迭代后的清理工作

    def train_one_iter(self):
            iter_start_time = time.time()  # 记录迭代开始时间

            inps, targets = self.prefetcher.next()  # 从预取器中获取输入和目标
            inps = inps.to(self.data_type)  # 将输入转换为指定的数据类型
            targets = targets.to(self.data_type)  # 将目标转换为指定的数据类型
            targets.requires_grad = False  # 不计算目标的梯度
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)  # 对输入和目标进行预处理
            data_end_time = time.time()  # 记录数据处理结束时间

            with torch.cuda.amp.autocast(enabled=self.amp_training):  # 启用自动混合精度
                outputs = self.model(inps, targets)  # 通过模型进行前向传播

            loss = outputs["total_loss"]  # 获取总损失

            self.optimizer.zero_grad()  # 清除优化器的梯度
            self.scaler.scale(loss).backward()  # 计算损失的反向传播
            self.scaler.step(self.optimizer)  # 更新优化器
            self.scaler.update()  # 更新缩放器

            if self.use_model_ema:  # 如果使用模型的 EMA
                self.ema_model.update(self.model)  # 更新 EMA 模型

            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)  # 更新学习率
            for param_group in self.optimizer.param_groups:  # 遍历优化器的参数组
                param_group["lr"] = lr  # 设置每个参数组的学习率

            iter_end_time = time.time()  # 记录迭代结束时间
            self.meter.update(  # 更新度量记录器
                iter_time=iter_end_time - iter_start_time,  # 迭代时间
                data_time=data_end_time - iter_start_time,  # 数据时间
                lr=lr,  # 当前学习率
                **outputs,  # 其他输出
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))  # 记录命令行参数
        logger.info("exp value:\n{}".format(self.exp))  # 记录实验配置

        # model related init
        torch.cuda.set_device(self.local_rank)  # 设置当前设备为本地排名的 CUDA 设备
        model = self.exp.get_model()  # 获取模型
        logger.info(  # 记录模型摘要
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)  # 将模型移动到指定设备

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)  # 获取优化器

        # value of epoch will be set in [resume_train](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolox/yolox/core/trainer.py:291:4-324:20)
        model = self.resume_train(model)  # 恢复训练

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs  # 判断是否不进行数据增强
        self.train_loader = self.exp.get_data_loader(  # 获取数据加载器
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")  # 记录预取器初始化信息
        self.prefetcher = DataPrefetcher(self.train_loader)  # 初始化预取器
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)  # 设置每个轮次的最大迭代次数

        self.lr_scheduler = self.exp.get_lr_scheduler(  # 获取学习率调度器
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:  # 如果需要占用内存
            occupy_mem(self.local_rank)  # 占用内存

        if self.is_distributed:  # 如果是分布式训练
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)  # 使用分布式数据并行

        if self.use_model_ema:  # 如果使用模型的 EMA
            self.ema_model = ModelEMA(model, 0.9998)  # 初始化 EMA 模型
            self.ema_model.updates = self.max_iter * self.start_epoch  # 设置 EMA 更新次数

        self.model = model  # 设置当前模型

        self.evaluator = self.exp.get_evaluator(  # 获取评估器
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:  # 只有主进程初始化日志记录器
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))  # 初始化 TensorBoard 日志记录器
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(  # 初始化 Wandb 日志记录器
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")  # 抛出异常

        logger.info("Training start...")  # 记录训练开始信息
        logger.info("\n{}".format(model))  # 记录模型信息

    def after_train(self):
            logger.info(  # 记录训练结束信息
                "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)  # 输出最佳平均精度
            )
            if self.rank == 0:  # 只有主进程执行以下操作
                if self.args.logger == "wandb":  # 如果使用 Wandb 记录器
                    self.wandb_logger.finish()  # 结束 Wandb 记录

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))  # 记录当前训练轮次开始

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:  # 如果达到不进行增强的轮次
            logger.info("--->No mosaic aug now!")  # 记录当前不进行马赛克增强
            self.train_loader.close_mosaic()  # 关闭马赛克增强
            logger.info("--->Add additional L1 loss now!")  # 记录将添加额外的 L1 损失
            if self.is_distributed:  # 如果是分布式训练
                self.model.module.head.use_l1 = True  # 使用 L1 损失
            else:
                self.model.head.use_l1 = True  # 使用 L1 损失
            self.exp.eval_interval = 1  # 设置评估间隔为 1
            if not self.no_aug:  # 如果没有禁用增强
                self.save_ckpt(ckpt_name="last_mosaic_epoch")  # 保存最后的马赛克轮次检查点

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")  # 保存最新的检查点

        if (self.epoch + 1) % self.exp.eval_interval == 0:  # 如果当前轮次是评估间隔的倍数
            all_reduce_norm(self.model)  # 进行模型的全归约操作
            self.evaluate_and_save_model()  # 评估并保存模型

    def before_iter(self):
        pass  # 在每个迭代前执行的操作（目前为空）

    # def after_iter(self):
    #     """
    #     `after_iter` contains two parts of logic:
    #         * log information
    #         * reset setting of resize
    #     """
    #     # log needed information
    #     if (self.iter + 1) % self.exp.print_interval == 0:
    #         # TODO check ETA logic
    #         left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
    #         eta_seconds = self.meter["iter_time"].global_avg * left_iters
    #         eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

    #         progress_str = "epoch: {}/{}, iter: {}/{}".format(
    #             self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
    #         )
    #         loss_meter = self.meter.get_filtered_meter("loss")
    #         loss_str = ", ".join(
    #             ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
    #         )

    #         time_meter = self.meter.get_filtered_meter("time")
    #         time_str = ", ".join(
    #             ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
    #         )

    #         mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

    #         logger.info(
    #             "{}, {}, {}, {}, lr: {:.3e}".format(
    #                 progress_str,
    #                 mem_str,
    #                 time_str,
    #                 loss_str,
    #                 self.meter["lr"].latest,
    #             )
    #             + (", size: {:d}, {}".format(self.input_size[0], eta_str))
    #         )

    #         if self.rank == 0:
    #             if self.args.logger == "tensorboard":
    #                 self.tblogger.add_scalar(
    #                     "train/lr", self.meter["lr"].latest, self.progress_in_iter)
    #                 for k, v in loss_meter.items():
    #                     self.tblogger.add_scalar(
    #                         f"train/{k}", v.latest, self.progress_in_iter)
    #             if self.args.logger == "wandb":
    #                 metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
    #                 metrics.update({
    #                     "train/lr": self.meter["lr"].latest
    #                 })
    #                 self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

    #         self.meter.clear_meters()

    #     # random resizing
    #     if (self.progress_in_iter + 1) % 10 == 0:
    #         self.input_size = self.exp.random_resize(
    #             self.train_loader, self.epoch, self.rank, self.is_distributed
    #         )

    # @property
    # def progress_in_iter(self):
    #     return self.epoch * self.max_iter + self.iter

    # def resume_train(self, model):
    #     if self.args.resume:
    #         logger.info("resume training")
    #         if self.args.ckpt is None:
    #             ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
    #         else:
    #             ckpt_file = self.args.ckpt

    #         ckpt = torch.load(ckpt_file, map_location=self.device)
    #         # resume the model/optimizer state dict
    #         model.load_state_dict(ckpt["model"])
    #         self.optimizer.load_state_dict(ckpt["optimizer"])
    #         self.best_ap = ckpt.pop("best_ap", 0)
    #         # resume the training states variables
    #         start_epoch = (
    #             self.args.start_epoch - 1
    #             if self.args.start_epoch is not None
    #             else ckpt["start_epoch"]
    #         )
    #         self.start_epoch = start_epoch
    #         logger.info(
    #             "loaded checkpoint '{}' (epoch {})".format(
    #                 self.args.resume, self.start_epoch
    #             )
    #         )  # noqa
    #     else:
    #         if self.args.ckpt is not None:
    #             logger.info("loading checkpoint for fine tuning")
    #             ckpt_file = self.args.ckpt
    #             ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
    #             model = load_ckpt(model, ckpt)
    #         self.start_epoch = 0

    #     return model

    def after_iter(self):
            """
            [after_iter](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/YOLO/yolox/yolox/core/trainer.py:226:4-285:13) contains two parts of logic:
                * log information
                * reset setting of resize
            """
            # log needed information
            if (self.iter + 1) % self.exp.print_interval == 0:  # 每隔一定的迭代次数记录信息
                # TODO check ETA logic
                left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)  # 计算剩余迭代次数
                eta_seconds = self.meter["iter_time"].global_avg * left_iters  # 估算剩余时间
                eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))  # 格式化剩余时间字符串
    
                progress_str = "epoch: {}/{}, iter: {}/{}".format(  # 记录当前进度
                    self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
                )
                loss_meter = self.meter.get_filtered_meter("loss")  # 获取损失度量
                loss_str = ", ".join(  # 格式化损失信息
                    ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
                )
    
                time_meter = self.meter.get_filtered_meter("time")  # 获取时间度量
                time_str = ", ".join(  # 格式化时间信息
                    ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
                )
    
                mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())  # 获取内存使用情况
    
                logger.info(  # 记录日志
                    "{}, {}, {}, {}, lr: {:.3e}".format(
                        progress_str,
                        mem_str,
                        time_str,
                        loss_str,
                        self.meter["lr"].latest,
                    )
                    + (", size: {:d}, {}".format(self.input_size[0], eta_str))  # 记录输入大小和剩余时间
                )
    
                if self.rank == 0:  # 只有主进程执行以下操作
                    if self.args.logger == "tensorboard":  # 如果使用 TensorBoard 记录器
                        self.tblogger.add_scalar(  # 记录学习率
                            "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                        for k, v in loss_meter.items():  # 记录损失信息
                            self.tblogger.add_scalar(
                                f"train/{k}", v.latest, self.progress_in_iter)
                    if self.args.logger == "wandb":  # 如果使用 Wandb 记录器
                        metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}  # 记录损失信息
                        metrics.update({
                            "train/lr": self.meter["lr"].latest  # 更新学习率
                        })
                        self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)  # 记录 Wandb 信息
    
                self.meter.clear_meters()  # 清除度量记录器
    
            # random resizing
            if (self.progress_in_iter + 1) % 10 == 0:  # 每 10 次迭代随机调整输入大小
                self.input_size = self.exp.random_resize(  # 随机调整输入大小
                    self.train_loader, self.epoch, self.rank, self.is_distributed
                )
    
    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter  # 计算当前进度
    
    def resume_train(self, model):
        if self.args.resume:  # 如果需要恢复训练
            logger.info("resume training")  # 记录恢复训练信息
            if self.args.ckpt is None:  # 如果没有指定检查点
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")  # 使用默认检查点文件名
            else:
                ckpt_file = self.args.ckpt  # 使用指定的检查点文件名

            ckpt = torch.load(ckpt_file, map_location=self.device)  # 加载检查点
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])  # 恢复模型状态字典
            self.optimizer.load_state_dict(ckpt["optimizer"])  # 恢复优化器状态字典
            self.best_ap = ckpt.pop("best_ap", 0)  # 恢复最佳平均精度
            # resume the training states variables
            start_epoch = (  # 设置开始轮次
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch  # 更新开始轮次
            logger.info(  # 记录加载的检查点信息
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:  # 如果指定了检查点
                logger.info("loading checkpoint for fine tuning")  # 记录加载微调检查点信息
                ckpt_file = self.args.ckpt  # 获取检查点文件名
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]  # 加载模型部分的检查点
                model = load_ckpt(model, ckpt)  # 加载检查点到模型
            self.start_epoch = 0  # 如果没有恢复训练，设置开始轮次为 0

        return model  # 返回模型


    def evaluate_and_save_model(self):
        # 使用模型的 EMA（Exponential Moving Average）进行评估
        if self.use_model_ema:
            evalmodel = self.ema_model.ema  # 如果使用EMA模型，则评估模型为EMA模型
        else:
            evalmodel = self.model  # 否则，评估模型为常规模型
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module  # 如果模型是并行的，则获取其模块

        with adjust_status(evalmodel, training=False):  # 调整模型状态为评估模式
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )  # 评估模型并获取AP值和预测结果

        update_best_ckpt = ap50_95 > self.best_ap  # 检查是否需要更新最佳检查点
        self.best_ap = max(self.best_ap, ap50_95)  # 更新最佳AP值

        if self.rank == 0:  # 如果是主进程
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)  # 记录AP50值
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)  # 记录AP50_95值
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,  # 记录AP50值
                    "val/COCOAP50_95": ap50_95,  # 记录AP50_95值
                    "train/epoch": self.epoch + 1,  # 记录当前训练的epoch
                })
                self.wandb_logger.log_images(predictions)  # 记录预测结果的图像
            logger.info("\n" + summary)  # 记录评估摘要信息
        synchronize()  # 同步所有进程

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)  # 保存最后一个epoch的检查点
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)  # 保存历史检查点


    
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        # 如果是主进程
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model  # 决定保存的模型是EMA还是常规模型
            logger.info("Save weights to {}".format(self.file_name))  # 记录保存模型权重的文件名
            ckpt_state = {
                "start_epoch": self.epoch + 1,  # 当前epoch
                "model": save_model.state_dict(),  # 模型的状态字典
                "optimizer": self.optimizer.state_dict(),  # 优化器的状态字典
                "best_ap": self.best_ap,  # 最佳AP值
                "curr_ap": ap,  # 当前AP值
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )  # 保存检查点

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,  # 当前epoch
                        "optimizer": self.optimizer.state_dict(),  # 优化器的状态字典
                        "best_ap": self.best_ap,  # 最佳AP值
                        "curr_ap": ap  # 当前AP值
                    }
                )  # 使用wandb记录检查点
