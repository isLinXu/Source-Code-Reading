import logging  # 导入logging模块，用于记录日志
import torch  # 导入PyTorch库

logging.basicConfig(  # 配置日志的基本设置
    format="%(asctime)s %(message)s",  # 设置日志消息的格式
    datefmt="%m/%d/%Y %I:%M:%S %p",  # 设置日期格式
    level=logging.INFO  # 设置日志级别为INFO
)

def get_logger():  # 获取日志记录器的函数
    return logging.getLogger(__name__)  # 返回当前模块的日志记录器


def rank_log(_rank, logger, msg):  # 记录日志的辅助函数，仅在全局排名为0时记录
    """helper function to log only on global rank 0"""  # 辅助函数，仅在全局排名为0时记录日志
    if _rank == 0:  # 如果当前进程的排名是0
        logger.info(f" {msg}")  # 记录信息日志


def verify_min_gpu_count(min_gpus: int = 2) -> bool:  # 验证是否有至少min_gpus个GPU可用
    """ verification that we have at least 2 gpus to run dist examples """  # 验证是否至少有2个GPU可用于分布式示例
    has_cuda = torch.cuda.is_available()  # 检查CUDA是否可用
    gpu_count = torch.cuda.device_count()  # 获取可用GPU的数量
    return has_cuda and gpu_count >= min_gpus  # 返回是否有足够的GPU可用