from pathlib import Path  # 从pathlib模块导入Path类，用于处理文件路径
from datetime import datetime  # 从datetime模块导入datetime类，用于处理日期和时间
import torch  # 导入PyTorch库
import time  # 导入时间模块

from torch.distributed.fsdp import (  # 从PyTorch的分布式库中导入全分片数据并行相关类
    FullyShardedDataParallel as FSDP,  # 完全分片的数据并行
    StateDictType,  # 状态字典类型
    FullStateDictConfig,  # 一般模型非分片、非扁平化参数
    LocalStateDictConfig,  # 扁平化参数，仅可由FSDP使用
    # ShardedStateDictConfig, # 非扁平化参数但分片，可用于其他并行方案
)

from torch.distributed.checkpoint import (  # 从PyTorch的分布式检查点库中导入相关类
    FileSystemReader,  # 文件系统读取器
    FileSystemWriter,  # 文件系统写入器
    save_state_dict,  # 保存状态字典的函数
    load_state_dict,  # 加载状态字典的函数
)
from torch.distributed.checkpoint.default_planner import (  # 从默认计划器中导入相关类
    DefaultSavePlanner,  # 默认保存计划器
    DefaultLoadPlanner,  # 默认加载计划器
)

from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType  # 从全分片数据并行库中导入状态字典类型
import torch.distributed.checkpoint as dist_cp  # 导入分布式检查点模块并重命名为dist_cp
import torch.distributed as dist  # 导入分布式模块并重命名为dist


def get_date_of_run():  # 定义获取运行日期和时间的函数
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """  # 创建文件保存唯一性的日期和时间，示例：2022-05-07-08:31:12_PM
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")  # 获取当前日期和时间并格式化
    print(f"--> current date and time of run = {date_of_run}")  # 打印当前日期和时间
    return date_of_run  # 返回当前日期和时间


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)  # 创建单例保存策略以避免重复创建


def load_model_sharded(model, rank, cfg, verbose=True):  # 定义加载分片模型的函数
    # torch.manual_seed(103)  # 设置随机种子（注释掉的代码）
    folder_name = (  # 创建文件夹名称
        cfg.dist_checkpoint_root_folder  # 分布式检查点根文件夹
        + "/"  # 添加分隔符
        + cfg.dist_checkpoint_folder  # 分布式检查点文件夹
        + "-"  # 添加分隔符
        + cfg.model_name  # 模型名称
    )

    load_dir = Path.cwd() / folder_name  # 获取当前工作目录并与文件夹名称组合

    if not load_dir.exists():  # 检查加载目录是否存在
        if rank == 0:  # 如果是主进程
            print(f"No sharded_state_dict checkpoint directory found...skipping")  # 打印未找到分片状态字典检查点目录的信息
        return  # 返回

    reader = FileSystemReader(load_dir)  # 创建文件系统读取器

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):  # 设置模型的状态字典类型为分片状态字典
        checkpoint = model.state_dict()  # 获取模型的状态字典
        if rank == 0:  # 如果是主进程
            ck = checkpoint.keys()  # 获取检查点的键
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")  # 打印检查点键的长度和内容

        dist_cp.load_state_dict(  # 加载状态字典
            state_dict=checkpoint,  # 状态字典
            storage_reader=reader,  # 存储读取器
        )
        if rank == 0:  # 如果是主进程
            print(f"checkpoint after load_state_dict()")  # 打印加载状态字典后的信息
            ck = checkpoint.keys()  # 获取检查点的键
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")  # 打印检查点键的长度和内容
        model.load_state_dict(checkpoint)  # 将状态字典加载到模型中
    if rank == 0:  # 如果是主进程
        print(f"Sharded state checkpoint loaded from {load_dir}")  # 打印从加载目录加载分片状态检查点的信息


def save_model_and_optimizer_sharded(model, rank, cfg, optim=None, verbose=True):  # 定义保存模型和优化器的函数
    """save model and optimizer via sharded_state_dict to save_dir"""  # 通过分片状态字典保存模型和优化器到保存目录
    folder_name = (  # 创建文件夹名称
        cfg.dist_checkpoint_root_folder  # 分布式检查点根文件夹
        + "/"  # 添加分隔符
        + cfg.dist_checkpoint_folder  # 分布式检查点文件夹
        + "-"  # 添加分隔符
        + cfg.model_name  # 模型名称
    )

    save_dir = Path.cwd() / folder_name  # 获取当前工作目录并与文件夹名称组合
    if rank == 0:  # 如果是主进程
        print(f"Saving model to {save_dir}")  # 打印保存模型的目录

    distributed_writer = dist_cp.FileSystemWriter(  # 创建文件系统写入器
        save_dir,  # 保存目录
    )
    t0 = time.perf_counter()  # 记录开始时间

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):  # 设置模型的状态字典类型为分片状态字典

        state_dict = {"model": model.state_dict()}  # 创建状态字典，包含模型的状态字典
        if optim is not None:  # 如果优化器不为None
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)  # 将优化器的状态字典添加到状态字典中

        dist_cp.save_state_dict(  # 保存状态字典
            state_dict=state_dict,  # 状态字典
            storage_writer=distributed_writer,  # 存储写入器
            planner=DefaultSavePlanner(),  # 默认保存计划器
        )
    dist.barrier()  # 同步所有进程
    t1 = time.perf_counter()  # 记录结束时间
    if rank == 0:  # 如果是主进程
        print(f"Sharded state checkpoint saved to {save_dir}")  # 打印保存分片状态检查点的目录
        print(  # 打印检查点保存时间和使用的线程数
            f"Checkpoint Time = {t1-t0:.4f}\n using {cfg.save_using_num_threads=} total threads"
        )

def save_model_checkpoint(  # 定义保存模型检查点的函数
    model,  # 模型
    optimizer,  # 优化器
    rank,  # 进程排名
    cfg,  # 配置
    epoch=1,  # 训练周期，默认为1
):
    """saving model via rank0 cpu streaming and full_state_dict"""  # 通过rank0 CPU流式传输和完整状态字典保存模型

    # saving with rank0 cpu
    if not cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:  # 检查点类型必须为完整状态字典
        print(f" unable to handle checkpoint type {cfg.checkpoint_type}, aborting")  # 打印无法处理的检查点类型信息

    with FSDP.state_dict_type(  # 设置模型的状态字典类型为完整状态字典
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()  # 获取模型的状态字典

    if cfg.verbose:  # 如果配置中设置了详细模式
        print(f"saving process: rank {rank}  done w model state_dict\n")  # 打印保存过程的信息


    if rank == 0:  # 如果是主进程
        print(f"--> saving model ...")  # 打印保存模型的信息
        # create save path
        save_dir = Path.cwd() / cfg.checkpoint_folder  # 创建保存目录
        save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
        save_name = cfg.model_save_name + "-" + str(epoch) + ".pt"  # 创建保存文件名
        save_full_path = str(save_dir) + "/" + save_name  # 创建保存的完整路径

        # save model
        torch.save(cpu_state, save_full_path)  # 保存模型状态字典

        if cfg.verbose:  # 如果配置中设置了详细模式
            print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")  # 打印模型检查点保存的信息


def load_model_checkpoint(model, rank, cfg, verbose=True):  # 定义加载模型检查点的函数
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""  # 加载本地检查点到rank0 CPU，必须在传递给FSDP之前调用

    if rank != 0:  # 如果不是主进程
        return  # 返回

    # where is the checkpoint at...
    full_state_dict_model_path = (  # 创建完整状态字典模型路径
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename  # 当前工作目录与检查点文件夹和模型文件名组合
    )
    # is it present...
    if not full_state_dict_model_path.is_file():  # 检查文件是否存在
        print(  # 打印检查点文件未找到的信息
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return  # 返回


    model_checkpoint = torch.load(full_state_dict_model_path)  # 加载模型检查点
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)  # 将检查点加载到模型中

    if cfg.verbose:  # 如果配置中设置了详细模式
        print(f"model checkpoint loaded to rank0 cpu")  # 打印模型检查点加载的信息


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):  # 定义保存优化器检查点的函数
    """save optimizer state via full state dict"""  # 通过完整状态字典保存优化器状态

    if cfg.verbose:  # 如果配置中设置了详细模式
        print(f"--> optim state call on rank {rank}\n")  # 打印优化器状态调用的信息

    # pull all sharded optimizer states to rank0 cpu...
    optim_state = FSDP.full_optim_state_dict(model, optimizer)  # 获取完整的优化器状态字典

    if cfg.verbose:  # 如果配置中设置了详细模式
        print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")  # 打印优化器状态字典的长度信息

    if rank == 0:  # 如果是主进程
        save_dir = Path.cwd() / cfg.checkpoint_folder  # 创建保存目录
        save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）

        opt_save_name = (  # 创建优化器保存文件名
            cfg.optimizer_name + "-" + cfg.model_save_name + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name  # 创建优化器保存的完整路径

        print(f"--> saving optimizer state...")  # 打印保存优化器状态的信息

        torch.save(optim_state, opt_save_full_path)  # 保存优化器状态字典

        print(f"--> saved {opt_save_full_path} to disk")  # 打印保存成功的信息


def load_optimizer_checkpoint(model, optimizer, rank, cfg):  # 定义加载优化器检查点的函数
    """load an fdsp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """  # 使用散播方法加载FSDP优化器完整状态检查点，确保只有rank 0加载优化器状态字典并分发给其他进程

    opt_file_path = Path.cwd() / cfg.checkpoint_folder / cfg.optimizer_checkpoint_file  # 创建优化器检查点文件路径

    if not opt_file_path.is_file():  # 检查文件是否存在
        print(  # 打印优化器检查点未找到的信息
            f"warning - optimizer checkpoint not present {opt_file_path}. Returning. "
        )
        return  # 返回

    full_osd = None  # 初始化完整优化器状态字典为None

    if rank == 0:  # 如果是主进程
        full_osd = torch.load(opt_file_path)  # 加载完整优化器状态字典

        if cfg.verbose:  # 如果配置中设置了详细模式
            print(f"loaded full osd on rank 0")  # 打印加载完整优化器状态字典的信息

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)  # 将完整优化器状态字典分发给其他进程

    if cfg.verbose:  # 如果配置中设置了详细模式
        print(f"optimizer shard loaded on rank {rank}")  # 打印优化器分片加载的信息


def load_distributed_model_checkpoint(model, rank, cfg):  # 定义加载分布式模型检查点的函数
    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:  # 检查点类型为本地状态字典
        print(f"loading distributed checkpoint, rank {rank}...")  # 打印加载分布式检查点的信息
        folder_name = (  # 创建文件夹名称
            cfg.dist_checkpoint_root_folder  # 分布式检查点根文件夹
            + "/"  # 添加分隔符
            + cfg.dist_checkpoint_folder  # 分布式检查点文件夹
            + "-"  # 添加分隔符
            + cfg.model_name  # 模型名称
        )

        checkdir = Path.cwd() / folder_name  # 获取当前工作目录并与文件夹名称组合

        if not checkdir.exists():  # 检查检查点目录是否存在
            if rank == 0:  # 如果是主进程
                print(f"No checkpoint directory found...skipping")  # 打印未找到检查点目录的信息
            return  # 返回


        reader = FileSystemReader(checkdir)  # 创建文件系统读取器

        with FSDP.state_dict_type(  # 设置模型的状态字典类型为本地状态字典
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()  # 获取模型的状态字典
            load_state_dict(state_dict, reader)  # 加载状态字典
            model.load_state_dict(state_dict)  # 将状态字典加载到模型中

        print(f"--> local state loaded on rank {rank}")  # 打印本地状态加载的信息

        return  # 返回


def save_distributed_model_checkpoint(model, rank, cfg, epoch=1):  # 定义保存分布式模型检查点的函数
    # distributed checkpoint saving

    # confirm type of checkpoint and save
    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:  # 检查点类型为本地状态字典
        # create writer to current path
        folder_name = (  # 创建文件夹名称
            cfg.dist_checkpoint_root_folder  # 分布式检查点根文件夹
            + "/"  # 添加分隔符
            + cfg.dist_checkpoint_folder  # 分布式检查点文件夹
            + "-"  # 添加分隔符
            + cfg.model_name  # 模型名称
        )
        save_dir = Path.cwd() / folder_name  # 获取当前工作目录并与文件夹名称组合

        writer = FileSystemWriter(  # 创建文件系统写入器
            save_dir,  # 保存目录
        )

        with FSDP.state_dict_type(  # 设置模型的状态字典类型为本地状态字典
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()  # 获取模型的状态字典

        # write out distributed checkpoint
        save_state_dict(state_dict, writer)  # 写入分布式检查点

        return  # 返回