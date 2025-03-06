# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os  # 导入os模块，用于与操作系统进行交互
import shutil  # 导入shutil模块，用于文件和目录的操作
import socket  # 导入socket模块，用于网络通信
import sys  # 导入sys模块，用于与Python解释器进行交互
import tempfile  # 导入tempfile模块，用于创建临时文件

from . import USER_CONFIG_DIR  # 从当前包导入USER_CONFIG_DIR
from .torch_utils import TORCH_1_9  # 从torch_utils模块导入TORCH_1_9

def find_free_network_port() -> int:
    """
    Finds a free port on localhost.
    在本地主机上查找一个空闲端口。
    
    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    当我们不想连接到真实的主节点，但必须设置`MASTER_PORT`环境变量时，这在单节点训练中非常有用。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 创建一个TCP/IP套接字
        s.bind(("127.0.0.1", 0))  # 绑定到本地主机的一个空闲端口
        return s.getsockname()[1]  # 返回端口号


def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name.
    生成一个DDP文件并返回其文件名。"""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)  # 获取训练器的模块名和类名

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
# Ultralytics多GPU训练临时文件（使用后应自动删除）
overrides = {vars(trainer.args)}  # 获取训练器参数的变量字典

if __name__ == "__main__":  # 如果该文件是主程序运行
    from {module} import {name}  # 从模块中导入训练器类
    from ultralytics.utils import DEFAULT_CFG_DICT  # 从utils模块导入默认配置字典

    cfg = DEFAULT_CFG_DICT.copy()  # 复制默认配置字典
    cfg.update(save_dir='')  # 处理额外的键'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)  # 创建训练器实例
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"  # 设置模型参数
    results = trainer.train()  # 开始训练并获取结果
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)  # 创建DDP目录（如果不存在）
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",  # 临时文件前缀
        suffix=f"{id(trainer)}.py",  # 临时文件后缀，包含训练器的ID
        mode="w+",  # 读写模式
        encoding="utf-8",  # 文件编码
        dir=USER_CONFIG_DIR / "DDP",  # 临时文件目录
        delete=False,  # 不删除临时文件
    ) as file:
        file.write(content)  # 写入内容到临时文件
    return file.name  # 返回临时文件名


def generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training.
    生成并返回分布式训练的命令。"""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    # 本地导入以避免特定问题

    if not trainer.resume:  # 如果训练器没有恢复
        shutil.rmtree(trainer.save_dir)  # 删除保存目录
    file = generate_ddp_file(trainer)  # 生成DDP文件
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"  # 选择分布式命令
    port = find_free_network_port()  # 查找空闲网络端口
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]  # 生成命令
    return cmd, file  # 返回命令和文件


def ddp_cleanup(trainer, file):
    """Delete temp file if created.
    如果创建了临时文件，则删除它。"""
    if f"{id(trainer)}.py" in file:  # 如果临时文件后缀在文件名中
        os.remove(file)  # 删除临时文件