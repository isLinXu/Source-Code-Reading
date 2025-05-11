import time  # 导入时间模块
import torch  # 导入PyTorch库
import wandb  # 导入wandb库，用于实验跟踪和可视化
import argparse  # 导入argparse库，用于解析命令行参数
import torch.optim as optim  # 导入PyTorch优化器模块
from dataclasses import asdict  # 从dataclasses导入asdict函数，用于将dataclass转换为字典
from datasets import load_dataset, concatenate_datasets  # 从datasets库导入load_dataset和concatenate_datasets函数
from torch.utils.data import DataLoader  # 从torch.utils.data导入DataLoader，用于数据加载

from data.collators import VAQCollator, MMStarCollator  # 从本地模块导入数据整理器
from data.datasets import MMStarDataset, VQADataset  # 从本地模块导入数据集类
from data.processors import get_image_processor, get_tokenizer  # 从本地模块导入获取图像处理器和tokenizer的函数
from models.vision_language_model import VisionLanguageModel  # 从本地模块导入VisionLanguageModel
import models.config as config  # 导入模型配置模块
import models.utils as utils  # 导入模型工具模块

#Otherwise, the tokenizer will through a warning
# 否则，tokenizer会抛出警告
import os  # 导入os模块
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 设置环境变量，禁用tokenizer的并行化，避免警告

torch.manual_seed(0)  # 设置PyTorch的CPU随机种子
torch.cuda.manual_seed_all(0)  # 设置PyTorch的GPU随机种子，确保可复现性

def get_run_name(train_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"  # 根据数据截断索引生成数据集大小字符串
    batch_size = f"bs{train_cfg.batch_size}"  # 生成批次大小字符串
    epochs = f"ep{train_cfg.epochs}"  # 生成epoch数量字符串
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"  # 生成学习率字符串（backbones-modality_projector）
    date = time.strftime("%m%d")  # 获取当前日期字符串

    return f"nanoVLM_{dataset_size}_{batch_size}_{epochs}_{learning_rate}_{date}"  # 组合生成运行名称

def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    # 创建数据集
    image_processor = get_image_processor(vlm_cfg.vit_img_size)  # 获取图像处理器
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)  # 获取tokenizer

    # Load and combine all training datasets
    # 加载并合并所有训练数据集
    combined_train_data = []  # 初始化一个列表来存储训练数据集
    for dataset_name in train_cfg.train_dataset_name:  # 遍历训练数据集名称列表
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)  # 加载指定名称的训练数据集
        combined_train_data.append(train_ds['train'])  # 将加载的数据集的'train'部分添加到列表中
    train_ds = concatenate_datasets(combined_train_data)  # 将列表中的所有数据集拼接起来
    
    test_ds = load_dataset(train_cfg.test_dataset_path)  # 加载测试数据集

    # Apply cutoff if specified
    # 如果指定了数据截断，则应用
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset # 使用整个数据集
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)  # 使用指定数量或数据集总数中的较小值

    train_dataset = VQADataset(train_ds.select(range(total_samples)), tokenizer, image_processor)  # 创建VQA训练数据集实例，应用数据截断
    test_dataset = MMStarDataset(test_ds['val'], tokenizer, image_processor)  # 创建MMStar测试数据集实例，使用'val'部分

    # Create collators
    # 创建数据整理器
    vqa_collator = VAQCollator(tokenizer, vlm_cfg.lm_max_length)  # 创建VQA数据整理器
    mmstar_collator = MMStarCollator(tokenizer)  # 创建MMStar数据整理器

    # Create dataloaders
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,  # 训练数据集
        batch_size=train_cfg.batch_size,  # 批次大小
        shuffle=True,  # 每个epoch打乱数据
        collate_fn=vqa_collator,  # 使用VQA数据整理器
        num_workers=8,  # 数据加载的工作进程数
        pin_memory=True,  # 将数据加载到CUDA固定内存
        drop_last=True,  # 如果最后一个批次小于批次大小，则丢弃
    )

    test_loader = DataLoader(
        test_dataset,  # 测试数据集
        batch_size=train_cfg.mmstar_batch_size,  # MMStar测试的批次大小
        shuffle=False,  # 不打乱测试数据
        collate_fn=mmstar_collator,  # 使用MMStar数据整理器
        pin_memory=True  # 将数据加载到CUDA固定内存
        )

    return train_loader, test_loader  # 返回训练和测试数据加载器

def test_mmstar(model, tokenizer, test_loader, device):
    model.eval()  # 将模型设置为评估模式
    total_examples = 0  # 初始化总样本数计数器
    correct_predictions = 0  # 初始化正确预测数计数器
    with torch.no_grad():  # 在此块中禁用梯度计算
        for batch in test_loader:  # 遍历测试数据加载器中的批次
            image = batch['images'].to(device)  # 将图像数据移动到指定设备
            input_ids = batch['input_ids'].to(device)  # 将输入token ID移动到指定设备
            labels = batch['labels'].to(device)  # 将标签移动到指定设备
            attention_mask = batch['attention_mask'].to(device)  # 将注意力掩码移动到指定设备
            
            correct_answer = tokenizer.batch_decode(labels, skip_special_tokens=True)  # 将标签token ID解码为文本形式的正确答案
            
            gen = model.generate(input_ids, image, attention_mask)  # 调用模型生成答案token
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)  # 将生成的token ID解码为文本形式的模型输出
            
            is_correct = utils.check_multiple_choice_with_regex(model_output, correct_answer)  # 使用正则表达式检查模型输出是否包含正确答案
            
            total_examples += len(is_correct)  # 累加当前批次的样本数
            if is_correct:  # 如果is_correct列表非空
                correct_predictions += sum(is_correct)  # 累加正确预测的数量（True会被视为1）

    accuracy = correct_predictions / total_examples if total_examples > 0 else 0  # 计算准确率，避免除以零

    return accuracy  # 返回准确率

def train(train_cfg, vlm_cfg):
    train_loader, test_loader = get_dataloaders(train_cfg, vlm_cfg)  # 获取训练和测试数据加载器
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)  # 获取tokenizer

    total_dataset_size = len(train_loader.dataset)  # 获取训练数据集的总样本数
    if train_cfg.log_wandb:  # 如果配置中启用了wandb日志
        run_name = get_run_name(train_cfg)  # 获取运行名称
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")  # 如果没有数据截断，更新运行名称中的数据集大小
        run = wandb.init(
            entity="huggingface",  # wandb实体名称
            project="nanoVLM",  # wandb项目名称
            config={
                "VLMConfig": asdict(vlm_cfg),  # 记录VLM配置
                "TrainConfig": asdict(train_cfg)  # 记录训练配置
            },
            name=run_name,  # 设置wandb运行名称
        )

    # Initialize model
    # 初始化模型
    if train_cfg.resume_from_vlm_checkpoint:  # 如果配置中指定从VLM检查点恢复训练
        model = VisionLanguageModel(vlm_cfg)  # 创建模型实例
        model.load_checkpoint(vlm_cfg.vlm_checkpoint_path)  # 加载VLM检查点权重
    elif vlm_cfg.vlm_load_backbone_weights:  # 如果配置中指定加载骨干网络权重
        model = VisionLanguageModel.from_pretrained(vlm_cfg)  # 从预训练加载骨干网络权重创建模型
    else:  # 否则，从头开始初始化模型
        model = VisionLanguageModel(vlm_cfg)
    
    print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")  # 打印模型参数数量
    print(f"Training summary: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch, batch size {train_cfg.batch_size}")  # 打印训练摘要信息

    # Define optimizer groups
    # 定义优化器参数组
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train the with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    # 由于我们有预训练的视觉和语言骨干网络，但模态投影层是新初始化的，因此使用相同的学习率进行训练没有意义。
    # 你可以选择完全冻结骨干网络，只训练MP层，但使用较低的学习率对它们进行微调可以使整个训练过程更容易。
    param_groups = [{'params': model.MP.parameters(), 'lr': train_cfg.lr_mp},  # 模态投影层的参数组，使用lr_mp学习率
                    {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_backbones}]  # 解码器和视觉编码器的参数组，使用lr_backbones学习率
    optimizer = optim.AdamW(param_groups)  # 使用AdamW优化器，传入参数组

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 再次确定设备
    model.to(device)  # 将模型移动到指定设备
    if train_cfg.compile:  # 如果配置中启用了torch.compile
        model = torch.compile(model)  # 编译模型以提高性能

    epoch_times = []  # 初始化列表存储每个epoch的训练时间
    best_accuracy = 0  # 初始化最佳准确率
    global_step = 0  # 初始化全局步数计数器
    for epoch in range(train_cfg.epochs):  # 遍历每个epoch
        epoch_start_time = time.time()  # 记录epoch开始时间
        model.train()  # 将模型设置为训练模式
        total_train_loss = 0  # 初始化总训练损失
        total_tokens_processed = 0  # 初始化总处理token数

        for batch in train_loader:  # 遍历训练数据加载器中的批次
            batch_start_time = time.time()  # 记录批次开始时间
            images = batch["image"].to(device)  # 将图像数据移动到设备
            input_ids = batch["input_ids"].to(device)  # 将输入token ID移动到设备
            labels = batch["labels"].to(device)  # 将标签移动到设备
            attention_mask = batch["attention_mask"].to(device)  # 将注意力掩码移动到设备

            optimizer.zero_grad()  # 清零优化器的梯度

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # Set to float16 if your hardware doesn't support bfloat16ß # 使用自动混合精度训练（如果硬件支持bfloat16，否则设置为float16）
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)  # 前向传播，计算损失

            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            batch_loss = loss.item()  # 获取当前批次的损失值
            total_train_loss += batch_loss  # 累加总训练损失

            num_tokens = torch.sum(attention_mask).item() # Sum of attention mask gives number of tokens # 计算当前批次的token数量（通过注意力掩码求和）
            num_tokens += images.shape[0] * ((images.shape[2] / vlm_cfg.vit_patch_size) ** 2) / (vlm_cfg.mp_pixel_shuffle_factor ** 2) # Add image tokens = batch_size * (((img_size / patch_size) ** 2) / (pixel_shuffle_factor ** 2)) # 加上图像token的数量（批次大小 * (图像块数量 / 像素重排因子平方)）
            total_tokens_processed += num_tokens  # 累加总处理token数

            batch_end_time = time.time()  # 记录批次结束时间
            batch_duration = batch_end_time - batch_start_time  # 计算批次处理时间
            tokens_per_second = num_tokens / batch_duration  # 计算每秒处理的token数

            if train_cfg.eval_in_epochs and global_step % 100 == 0:  # 如果配置中启用了epoch内评估且达到评估步数间隔
                epoch_accuracy = test_mmstar(model, tokenizer, test_loader, device)  # 在测试集上评估模型准确率
                if epoch_accuracy > best_accuracy:  # 如果当前准确率优于最佳准确率
                    best_accuracy = epoch_accuracy  # 更新最佳准确率
                    torch.save(getattr(model, '_orig_mod', model).state_dict(), vlm_cfg.vlm_checkpoint_path)  # 保存模型检查点（处理了torch.compile的情况）
                    print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}, Accuracy: {epoch_accuracy:.4f} | Saving checkpoint to {vlm_cfg.vlm_checkpoint_path}")  # 打印评估结果和保存信息
                else:
                    print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}, Accuracy: {epoch_accuracy:.4f}")  # 打印评估结果
                if train_cfg.log_wandb:  # 如果启用了wandb日志
                    run.log({"accuracy": epoch_accuracy}, step=global_step)  # 记录准确率到wandb

            if train_cfg.log_wandb:  # 如果启用了wandb日志
                run.log({"batch_loss": batch_loss,  # 记录批次损失
                         "tokens_per_second": tokens_per_second}, step=global_step)  # 记录每秒处理token数

            global_step += 1  # 全局步数加一

        avg_train_loss = total_train_loss / len(train_loader)  # 计算平均训练损失

        epoch_end_time = time.time()  # 记录epoch结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算epoch持续时间
        epoch_times.append(epoch_duration)  # 将epoch持续时间添加到列表中

        epoch_tokens_per_second = total_tokens_processed / epoch_duration  # 计算每个epoch每秒处理的token数

        if train_cfg.log_wandb:  # 如果启用了wandb日志
            run.log({"epoch_loss": avg_train_loss,  # 记录平均epoch损失
                     "epoch_duration": epoch_duration,  # 记录epoch持续时间
                     "epoch_tokens_per_second": epoch_tokens_per_second})  # 记录每个epoch每秒处理token数

        print(f"Epoch {epoch+1}/{train_cfg.epochs}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")  # 打印epoch训练结果

    # Summary Statistics
    # 总结统计信息
    avg_epoch_time = sum(epoch_times) / len(epoch_times)  # 计算平均epoch时间
    total_training_time = sum(epoch_times)  # 计算总训练时间
    total_samples_processed = len(train_loader.dataset) * train_cfg.epochs  # 计算总处理样本数
    avg_time_per_sample = total_training_time / total_samples_processed  # 计算平均每个样本处理时间
    print(f"Average time per epoch: {avg_epoch_time:.2f}s")  # 打印平均epoch时间
    print(f"Average time per sample: {avg_time_per_sample:.4f}s")  # 打印平均每个样本处理时间

    accuracy = test_mmstar(model, tokenizer, test_loader, device)  # 在测试集上进行最终评估
    print(f"MMStar Accuracy: {accuracy:.4f}")  # 打印最终准确率

    if train_cfg.log_wandb:  # 如果启用了wandb日志
        run.summary["avg_epoch_time"] = avg_epoch_time  # 记录平均epoch时间到wandb summary
        run.summary["avg_time_per_sample"] = avg_time_per_sample  # 记录平均每个样本处理时间到wandb summary
        run.summary["mmstar_acc"] = accuracy  # 记录最终准确率到wandb summary
        run.finish()  # 结束wandb运行

def main():
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象，用于解析命令行参数
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')  # 添加一个命令行参数，用于设置模态投影网络的学习率
    parser.add_argument('--lr_backbones', type=float, help='Learning rate for the backbones')  # 添加一个命令行参数，用于设置骨干网络（视觉和语言）的学习率
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint')  # 添加一个命令行参数，用于指定VLM检查点路径
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint')  # 添加一个命令行参数，用于指定是否从VLM检查点恢复训练，默认为False

    args = parser.parse_args()  # 解析命令行参数

    vlm_cfg = config.VLMConfig()  # 创建VLMConfig配置对象，加载默认或文件中的配置
    train_cfg = config.TrainConfig()  # 创建TrainConfig配置对象，加载默认或文件中的配置

    if args.lr_mp is not None:  # 如果命令行参数中指定了lr_mp
        train_cfg.lr_mp = args.lr_mp  # 使用命令行参数的值覆盖训练配置中的lr_mp
    if args.lr_backbones is not None:  # 如果命令行参数中指定了lr_backbones
        train_cfg.lr_backbones = args.lr_backbones  # 使用命令行参数的值覆盖训练配置中的lr_backbones
    if args.vlm_checkpoint_path is not None:  # 如果命令行参数中指定了vlm_checkpoint_path
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path  # 使用命令行参数的值覆盖VLM配置中的vlm_checkpoint_path
    # Override resume flag based on whether a checkpoint path was provided or explicitly set
    # 根据是否提供了检查点路径或显式设置了恢复标志来覆盖恢复标志
    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:  # 如果命令行参数中显式设置了resume_from_vlm_checkpoint且提供了vlm_checkpoint_path
         train_cfg.resume_from_vlm_checkpoint = True  # 设置训练配置中的resume_from_vlm_checkpoint为True
         # Ensure loading flags are set correctly if resuming
         # 如果恢复训练，确保加载标志设置正确
         vlm_cfg.vlm_load_backbone_weights = False  # 如果从VLM检查点恢复，则不加载骨干网络权重

    print("--- VLM Config ---")  # 打印VLM配置的标题
    print(vlm_cfg)  # 打印VLM配置的详细信息
    print("--- Train Config ---")  # 打印训练配置的标题
    print(train_cfg)  # 打印训练配置的详细信息

    train(train_cfg, vlm_cfg)  # 调用train函数，开始训练过程，传入训练和VLM配置


if __name__ == "__main__":  # 检查当前脚本是否作为主程序运行
    main()  # 如果是主程序，则调用main函数