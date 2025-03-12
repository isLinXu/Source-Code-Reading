import os  # 导入操作系统模块
import argparse  # 导入命令行参数解析模块
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from transformers import AutoTokenizer, GPT2TokenizerFast  # 从transformers库导入自动分词器和GPT2快速分词器
from transformers import T5Tokenizer, T5ForConditionalGeneration  # 从transformers库导入T5分词器和条件生成模型
import functools  # 导入functools模块
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器
import torch.nn.functional as F  # 再次导入功能性神经网络模块（可能是多余的）
import torch.distributed as dist  # 导入分布式训练模块
import torch.multiprocessing as mp  # 导入多进程模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块
from torch.utils.data.distributed import DistributedSampler  # 导入分布式采样器
from transformers.models.t5.modeling_t5 import T5Block  # 从transformers库导入T5模型块
from nlp import load_dataset  # 从nlp库导入加载数据集的函数

from torch.distributed.fsdp import (  # 从PyTorch的分布式库中导入全分布式数据并行相关类
    FullyShardedDataParallel as FSDP,  # 完全分片的数据并行
    CPUOffload,  # CPU卸载
    MixedPrecision,  # 混合精度
    BackwardPrefetch,  # 后向预取
    ShardingStrategy,  # 分片策略
    FullStateDictConfig,  # 完整状态字典配置
    StateDictType,  # 状态字典类型
)

from functools import partial  # 从functools模块导入部分函数
from torch.utils.data import DataLoader  # 导入数据加载器
from pathlib import Path  # 导入路径处理模块
from summarization_dataset import *  # 导入摘要数据集模块中的所有内容
import policies  # 导入策略模块
import model_checkpointing  # 导入模型检查点模块
from configs import fsdp_config, train_config  # 从配置文件导入FSDP和训练配置
from utils import (bfloat_support, setup,  # 从工具模块导入所需函数
                   cleanup, get_date_of_run,
                   format_metrics_to_gb,
                   train, validation, setup_model)
from transformers.models.t5.modeling_t5 import T5Block  # 再次导入T5模型块（可能是多余的）
from typing import Type  # 导入类型提示
import time  # 导入时间模块
import tqdm  # 导入进度条模块
from datetime import datetime  # 导入日期时间模块

def get_policies(cfg, rank):  # 定义获取策略的函数

    """establish current policies for mixed precision and fsdp wrapping"""  # 确定当前的混合精度和FSDP包装策略

    mixed_precision_policy = None  # 初始化混合精度策略为None
    wrapping_policy = None  # 初始化包装策略为None

    # mixed precision -----
    if cfg.mixed_precision:  # 如果配置中启用了混合精度
        bfloat_available = bfloat_support()  # 检查bfloat16支持
        if bfloat_available and not cfg.use_fp16:  # 如果支持bfloat16且未使用FP16
            mixed_precision_policy = policies.bfSixteen  # 设置混合精度策略为bfloat16
            if rank == 0:  # 如果是主进程
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")  # 打印启用的策略
        elif cfg.use_fp16:  # 如果使用FP16
            mixed_precision_policy = policies.fpSixteen  # 设置混合精度策略为FP16
            if rank == 0:  # 如果是主进程
                print(f"FP16 enabled. ")  # 打印启用的策略
        else:  # 如果不支持bfloat16且未使用FP16
            # mixed_precision_policy = policies.fpSixteen
            print(  # 打印不支持混合精度的信息
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    wrapping_policy = policies.get_t5_wrapper()  # 获取T5包装策略

    return mixed_precision_policy, wrapping_policy  # 返回混合精度和包装策略

def fsdp_main(args):  # 定义主函数

    model, tokenizer = setup_model(train_config.model_name)  # 设置模型和分词器

    local_rank = int(os.environ['LOCAL_RANK'])  # 获取本地进程的排名
    rank = int(os.environ['RANK'])  # 获取全局进程的排名
    world_size = int(os.environ['WORLD_SIZE'])  # 获取总进程数

    dataset = load_dataset('wikihow', 'all', data_dir='data/')  # 加载wikihow数据集
    print(dataset.keys())  # 打印数据集的键
    print("Size of train dataset: ", dataset['train'].shape)  # 打印训练数据集的大小
    print("Size of Validation dataset: ", dataset['validation'].shape)  # 打印验证数据集的大小

    #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
    train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False)  # 创建训练数据集
    val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)  # 创建验证数据集

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)  # 创建训练数据的分布式采样器
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)  # 创建验证数据的分布式采样器

    setup()  # 设置环境

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}  # 设置训练参数
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}  # 设置测试参数
    cuda_kwargs = {'num_workers': 2,  # 设置CUDA参数
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)  # 更新训练参数
    test_kwargs.update(cuda_kwargs)  # 更新测试参数

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)  # 创建训练数据加载器
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)  # 创建验证数据加载器

    torch.cuda.set_device(local_rank)  # 设置CUDA设备

    # Set up FSDP parameters
    mixed_precision_policy, t5_auto_wrap_policy = get_policies(train_config, rank)  # 获取FSDP参数

    # Apply FSDP wrapping to the model
    model = FSDP(model,  # 应用FSDP包装到模型
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=fsdp_config.sharding_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=fsdp_config.limit_all_gathers)

    # Enabling this causes https://github.com/pytorch/examples/issues/1210
    if fsdp_config.fsdp_activation_checkpointing:  # 如果启用了FSDP激活检查点
        policies.apply_fsdp_checkpointing(model)  # 应用FSDP检查点

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)  # 设置优化器

    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)  # 设置学习率调度器
    best_val_loss = float("inf")  # 初始化最佳验证损失为无穷大
    curr_val_loss = float("inf")  # 初始化当前验证损失为无穷大
    file_save_name = "T5-model-"  # 设置保存模型的文件名

    if rank == 0:  # 如果是主进程
        time_of_run = get_date_of_run()  # 获取运行时间
        dur = []  # 初始化持续时间列表
        train_acc_tracking = []  # 初始化训练准确率追踪列表
        val_acc_tracking = []  # 初始化验证准确率追踪列表
        training_start_time = time.time()  # 记录训练开始时间

    if rank == 0 and args.track_memory:  # 如果是主进程且需要跟踪内存
        mem_alloc_tracker = []  # 初始化内存分配跟踪列表
        mem_reserved_tracker = []  # 初始化保留内存跟踪列表

    for epoch in range(1, args.epochs + 1):  # 遍历每个训练周期
        t0 = time.time()  # 记录当前时间
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)  # 训练模型
        if args.run_validation:  # 如果需要运行验证
            curr_val_loss = validation(model, rank, world_size, val_loader)  # 验证模型
        scheduler.step()  # 更新学习率

        if rank == 0:  # 如果是主进程
            print(f"--> epoch {epoch} completed...entering save and stats zone")  # 打印训练周期完成信息

            dur.append(time.time() - t0)  # 记录持续时间
            train_acc_tracking.append(train_accuracy.item())  # 记录训练准确率

            if args.run_validation:  # 如果需要运行验证
                val_acc_tracking.append(curr_val_loss.item())  # 记录验证准确率

            if args.track_memory:  # 如果需要跟踪内存
                mem_alloc_tracker.append(  # 记录分配的内存
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(  # 记录保留的内存
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )

        if train_config.save_model and curr_val_loss < best_val_loss:  # 如果需要保存模型且当前验证损失更好
            if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:  # 如果检查点类型为完整状态字典
                model_checkpointing.save_model_checkpoint(  # 保存模型检查点
                    model, optimizer, rank, fsdp_config, epoch=1
                )
            elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:  # 如果检查点类型为分片状态字典
                model_checkpointing.save_model_and_optimizer_sharded(model, rank, fsdp_config)  # 保存模型和优化器
                if fsdp_config.save_optimizer:  # 如果需要保存优化器
                    model_checkpointing.save_model_and_optimizer_sharded(model, rank, fsdp_config, optim=optimizer)  # 保存优化器检查点

            if fsdp_config.save_optimizer:  # 如果需要保存优化器
                model_checkpointing.save_optimizer_checkpoint(  # 保存优化器检查点
                    model, optimizer, rank, fsdp_config, epoch=1
                )
        if curr_val_loss < best_val_loss:  # 如果当前验证损失更好
            best_val_loss = curr_val_loss  # 更新最佳验证损失
            if rank == 0:  # 如果是主进程
                print(f"-->>>> New Val Loss Record: {best_val_loss}")  # 打印新的最佳验证损失

    dist.barrier()  # 同步所有进程
    cleanup()  # 清理资源

if __name__ == '__main__':  # 如果是主程序
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')  # 创建参数解析器
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',  # 添加批量大小参数
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',  # 添加测试批量大小参数
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',  # 添加训练周期参数
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',  # 添加随机种子参数
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,  # 添加跟踪内存参数
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,  # 添加运行验证参数
                        help='running the validation')
    args = parser.parse_args()  # 解析参数

    torch.manual_seed(args.seed)  # 设置随机种子

    fsdp_main(args)  # 调用主函数