import logging
import os
import sys

import torch.distributed as dist


root_logger = None

# 定义print_rank0函数，用于在分布式训练中只有rank为0的进程打印信息
def print_rank0(*args):
    """
    在分布式训练环境中，只有当进程的local_rank为0时，才会打印传入的信息。

    :param args: 需要打印的信息
    """
    local_rank = dist.get_rank()                # 获取当前进程的local_rank
    if local_rank == 0:                         # 如果local_rank为0
        print(*args)                            # 打印信息

# 定义logger_setting函数，用于设置日志记录器
def logger_setting(save_dir=None):
    """
    设置日志记录器，可以指定日志保存的目录。

    :param save_dir: 日志保存的目录路径，默认为None，表示不保存到文件
    :return: 配置好的日志记录器
    """
    global root_logger                          # 声明root_logger为全局变量
    if root_logger is not None:                 # 如果root_logger已经配置过
        return root_logger                      # 直接返回已配置的root_logger
    else:
        root_logger = logging.getLogger()       # 获取根日志记录器
        root_logger.setLevel(logging.INFO)      # 设置日志级别为INFO

        # 创建一个StreamHandler，用于将日志输出到控制台
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)               # 设置日志级别为INFO
        formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s") # 定义日志格式
        ch.setFormatter(formatter)              # 设置日志格式
        root_logger.addHandler(ch)              # 将StreamHandler添加到根日志记录器

        # 如果指定了保存目录
        if save_dir:
            if not os.path.exists(save_dir):              # 如果目录不存在
                os.makedirs(save_dir, exist_ok=True)      # 创建目录
            save_file = os.path.join(save_dir, 'log.txt') # 定义日志文件路径
            if not os.path.exists(save_file):             # 如果日志文件不存在
                os.system(f"touch {save_file}")           # 创建日志文件

            # 创建一个FileHandler，用于将日志保存到文件
            fh = logging.FileHandler(save_file, mode='a')
            fh.setLevel(logging.INFO)                     # 设置日志级别为INFO
            fh.setFormatter(formatter)                    # 设置日志格式
            root_logger.addHandler(fh)                    # 将FileHandler添加到根日志记录器
            return root_logger                            # 返回配置好的根日志记录器

# 定义日志函数，用于记录信息
def log(*args):
    global root_logger
    local_rank = dist.get_rank()  # 获取当前进程的rank
    if local_rank == 0:           # 只有rank为0的进程记录日志
        root_logger.info(*args)   # 记录信息



        
def log_trainable_params(model):
    """
    记录模型的总参数数量和可训练参数数量。

    :param model: 需要记录参数的模型
    """
    total_params = sum(p.numel() for p in model.parameters())                                           # 计算模型的总参数数量
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)              # 计算模型的可训练参数数量
    log(f'Total Parameters: {total_params}, Total Trainable Parameters: {total_trainable_params}')      # 记录总参数和可训练参数数量
    log(f'Trainable Parameters:')                                                                       # 记录可训练参数信息
    for name, param in model.named_parameters():                                                        # 遍历模型的所有参数
        if param.requires_grad:                                                                         # 如果参数需要梯度更新
            print_rank0(f"{name}: {param.numel()} parameters")                                          # 记录参数名称和数量
