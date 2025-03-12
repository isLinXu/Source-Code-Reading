import os  # 导入操作系统模块
import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch的分布式模块
from datetime import datetime  # 从datetime模块导入datetime类，用于处理日期和时间
import tqdm  # 导入进度条模块
from transformers import AutoTokenizer, GPT2TokenizerFast  # 从transformers库导入自动分词器和GPT2快速分词器
from transformers import T5Tokenizer, T5ForConditionalGeneration  # 从transformers库导入T5分词器和条件生成模型

g_gigabyte = 1024**3  # 定义1GB的字节数

def setup():  # 定义初始化函数
    # initialize the process group  # 初始化进程组
    dist.init_process_group("nccl")  # 使用NCCL初始化分布式进程组

def cleanup():  # 定义清理函数
    dist.destroy_process_group()  # 销毁分布式进程组

def get_date_of_run():  # 定义获取运行日期和时间的函数
    """create date and time for file save uniqueness  # 创建文件保存唯一性的日期和时间
    example: 2022-05-07-08:31:12_PM'  # 示例：2022-05-07-08:31:12_PM
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")  # 获取当前日期和时间并格式化
    print(f"--> current date and time of run = {date_of_run}")  # 打印当前日期和时间
    return date_of_run  # 返回当前日期和时间

def format_metrics_to_gb(item):  # 定义将数字格式化为GB的函数
    """quick function to format numbers to gigabyte and round to 4 digit precision"""  # 快速将数字格式化为GB并四舍五入到4位精度
    metric_num = item / g_gigabyte  # 将字节数转换为GB
    metric_num = round(metric_num, ndigits=4)  # 四舍五入到4位精度
    return metric_num  # 返回格式化后的数字

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):  # 定义训练函数
    model.train()  # 设置模型为训练模式
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取本地进程的排名
    fsdp_loss = torch.zeros(2).to(local_rank)  # 初始化FSDP损失为零

    if sampler:  # 如果存在采样器
        sampler.set_epoch(epoch)  # 设置采样器的周期
    if rank == 0:  # 如果是主进程
        inner_pbar = tqdm.tqdm(  # 创建进度条
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"  # 设置进度条颜色和描述
        )
    for batch in train_loader:  # 遍历训练数据加载器
        for key in batch.keys():  # 遍历批次中的每个键
            batch[key] = batch[key].to(local_rank)  # 将数据移动到本地设备
        optimizer.zero_grad()  # 清零优化器的梯度
        output = model(input_ids=batch["source_ids"], attention_mask=batch["source_mask"], labels=batch["target_ids"])  # 前向传播
        loss = output["loss"]  # 获取损失值
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器
        fsdp_loss[0] += loss.item()  # 累加损失
        fsdp_loss[1] += len(batch)  # 累加批次大小
        if rank == 0:  # 如果是主进程
            inner_pbar.update(1)  # 更新进度条

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)  # 在所有进程中汇总损失
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]  # 计算训练准确率

    if rank == 0:  # 如果是主进程
        inner_pbar.close()  # 关闭进度条
        print(  # 打印训练周期和损失
            f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
        )
    return train_accuracy  # 返回训练准确率

def validation(model, rank, world_size, val_loader):  # 定义验证函数
    model.eval()  # 设置模型为评估模式
    correct = 0  # 初始化正确预测数量
    local_rank = int(os.environ['LOCAL_RANK'])  # 获取本地进程的排名
    fsdp_loss = torch.zeros(2).to(local_rank)  # 初始化FSDP损失为零
    if rank == 0:  # 如果是主进程
        inner_pbar = tqdm.tqdm(  # 创建进度条
            range(len(val_loader)), colour="green", desc="Validation Epoch"  # 设置进度条颜色和描述
        )
    with torch.no_grad():  # 在不计算梯度的情况下进行验证
        for batch in val_loader:  # 遍历验证数据加载器
            for key in batch.keys():  # 遍历批次中的每个键
                batch[key] = batch[key].to(local_rank)  # 将数据移动到本地设备
            output = model(input_ids=batch["source_ids"], attention_mask=batch["source_mask"], labels=batch["target_ids"])  # 前向传播
            fsdp_loss[0] += output["loss"].item()  # 累加批次损失
            fsdp_loss[1] += len(batch)  # 累加批次大小

            if rank == 0:  # 如果是主进程
                inner_pbar.update(1)  # 更新进度条

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)  # 在所有进程中汇总损失
    val_loss = fsdp_loss[0] / fsdp_loss[1]  # 计算验证损失
    if rank == 0:  # 如果是主进程
        inner_pbar.close()  # 关闭进度条
        print(f"Validation Loss: {val_loss:.4f}")  # 打印验证损失
    return val_loss  # 返回验证损失

def setup_model(model_name):  # 定义设置模型的函数
    model = T5ForConditionalGeneration.from_pretrained(model_name)  # 从预训练模型中加载T5模型
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  # 从预训练模型中加载T5分词器
    return model, tokenizer  # 返回模型和分词器