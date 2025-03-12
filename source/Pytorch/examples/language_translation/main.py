from time import time  # Track how long an epoch takes  # 导入time模块，用于跟踪每个周期的时间
import os  # Creating and finding files/directories  # 导入os模块，用于创建和查找文件/目录
import logging  # Logging tools  # 导入logging模块，用于日志记录
from datetime import date  # Logging the date for model versioning  # 导入date模块，用于记录模型版本的日期

import torch  # For ML  # 导入PyTorch库，用于机器学习
from tqdm import tqdm  # For fancy progress bars  # 从tqdm导入，用于美观的进度条

from src.model import Translator  # Our model  # 从src.model导入Translator类，作为我们的模型
from src.data import get_data, create_mask, generate_square_subsequent_mask  # Loading data and data preprocessing  # 从src.data导入数据加载和预处理相关函数
from argparse import ArgumentParser  # For args  # 从argparse导入ArgumentParser，用于命令行参数解析

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果可用，则在GPU上训练，否则使用CPU

# Function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):  # 定义贪婪解码函数

    # Move to device
    src = src.to(DEVICE)  # 将源输入移动到设备
    src_mask = src_mask.to(DEVICE)  # 将源掩码移动到设备

    # Encode input
    memory = model.encode(src, src_mask)  # 编码输入并获取内存

    # Output will be stored here
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)  # 初始化输出序列，填充开始符号

    # For each element in our translation (which could range up to the maximum translation length)
    for _ in range(max_len - 1):  # 对于翻译中的每个元素（最多到最大翻译长度）

        # Decode the encoded representation of the input
        memory = memory.to(DEVICE)  # 将内存移动到设备
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), DEVICE).type(torch.bool)).to(DEVICE)  # 生成目标掩码
        out = model.decode(ys, memory, tgt_mask)  # 解码输入

        # Reshape
        out = out.transpose(0, 1)  # 转置输出

        # Convert to probabilities and take the max of these probabilities
        prob = model.ff(out[:, -1])  # 获取最后一个时间步的输出概率
        _, next_word = torch.max(prob, dim=1)  # 获取最大概率的下一个单词
        next_word = next_word.item()  # 转换为Python标量

        # Now we have an output which is the vector representation of the translation
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)  # 将下一个单词添加到输出序列
        if next_word == end_symbol:  # 如果下一个单词是结束符号
            break  # 结束循环

    return ys  # 返回生成的输出序列

# Opens a user interface where users can translate an arbitrary sentence
def inference(opts):  # 定义推理函数

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    _, _, src_vocab, tgt_vocab, src_transform, _, special_symbols = get_data(opts)  # 获取训练数据、分词器和词汇对象

    src_vocab_size = len(src_vocab)  # 获取源语言词汇表大小
    tgt_vocab_size = len(tgt_vocab)  # 获取目标语言词汇表大小

    # Create model
    model = Translator(  # 创建翻译器模型
        num_encoder_layers=opts.enc_layers,  # 编码器层数
        num_decoder_layers=opts.dec_layers,  # 解码器层数
        embed_size=opts.embed_size,  # 嵌入维度
        num_heads=opts.attn_heads,  # 注意力头数
        src_vocab_size=src_vocab_size,  # 源语言词汇表大小
        tgt_vocab_size=tgt_vocab_size,  # 目标语言词汇表大小
        dim_feedforward=opts.dim_feedforward,  # 前馈网络维度
        dropout=opts.dropout  # 丢弃率
    ).to(DEVICE)  # 移动模型到设备

    # Load in weights
    model.load_state_dict(torch.load(opts.model_path))  # 加载模型权重

    # Set to inference
    model.eval()  # 设置模型为评估模式

    # Accept input and keep translating until they quit
    while True:  # 循环接受输入直到用户退出
        print("> ", end="")  # 打印提示符
        sentence = input()  # 获取用户输入的句子

        # Convert to tokens
        src = src_transform(sentence).view(-1, 1)  # 将输入句子转换为标记并调整形状
        num_tokens = src.shape[0]  # 获取标记数量

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)  # 创建源掩码

        # Decode
        tgt_tokens = greedy_decode(  # 解码生成目标标记
            model, src, src_mask, max_len=num_tokens + 5, start_symbol=special_symbols["<bos>"], end_symbol=special_symbols["<eos>"]  # 设置最大长度、开始符号和结束符号
        ).flatten()  # 展平目标标记

        # Convert to list of tokens
        output_as_list = list(tgt_tokens.cpu().numpy())  # 将目标标记转换为列表

        # Convert tokens to words
        output_list_words = tgt_vocab.lookup_tokens(output_as_list)  # 将标记转换为单词

        # Remove special tokens and convert to string
        translation = " ".join(output_list_words).replace("<bos>", "").replace("<eos>", "")  # 移除特殊标记并转换为字符串

        print(translation)  # 打印翻译结果

# Train the model for 1 epoch
def train(model, train_dl, loss_fn, optim, special_symbols, opts):  # 定义训练函数

    # Object for accumulating losses
    losses = 0  # 初始化损失累积器

    # Put model into training mode
    model.train()  # 设置模型为训练模式
    for src, tgt in tqdm(train_dl, ascii=True):  # 遍历训练数据加载器

        src = src.to(DEVICE)  # 将源数据移动到设备
        tgt = tgt.to(DEVICE)  # 将目标数据移动到设备

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]  # 获取目标输入（去掉最后一个标记）

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)  # 创建掩码

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)  # 通过模型前向传播获取输出概率

        # Reset gradients before we try to compute the gradients over the loss
        optim.zero_grad()  # 清零梯度

        # Get original shape back
        tgt_out = tgt[1:, :]  # 获取目标输出（去掉第一个标记）

        # Compute loss and gradient over that loss
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))  # 计算损失
        loss.backward()  # 反向传播计算梯度

        # Step weights
        optim.step()  # 更新模型参数

        # Accumulate a running loss for reporting
        losses += loss.item()  # 累加损失

        if opts.dry_run:  # 如果是干运行模式
            break  # 退出循环

    # Return the average loss
    return losses / len(list(train_dl))  # 返回平均损失

# Check the model accuracy on the validation dataset
def validate(model, valid_dl, loss_fn, special_symbols):  # 定义验证函数
    
    # Object for accumulating losses
    losses = 0  # 初始化损失累积器

    # Turn off gradients a moment
    model.eval()  # 设置模型为评估模式

    for src, tgt in tqdm(valid_dl):  # 遍历验证数据加载器

        src = src.to(DEVICE)  # 将源数据移动到设备
        tgt = tgt.to(DEVICE)  # 将目标数据移动到设备

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]  # 获取目标输入（去掉最后一个标记）

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)  # 创建掩码

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)  # 通过模型前向传播获取输出概率

        # Get original shape back, compute loss, accumulate that loss
        tgt_out = tgt[1:, :]  # 获取目标输出（去掉第一个标记）
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))  # 计算损失
        losses += loss.item()  # 累加损失

    # Return the average loss
    return losses / len(list(valid_dl))  # 返回平均损失

# Train the model
def main(opts):  # 定义主函数

    # Set up logging
    os.makedirs(opts.logging_dir, exist_ok=True)  # 创建日志目录
    logger = logging.getLogger(__name__)  # 获取日志记录器
    logging.basicConfig(filename=opts.logging_dir + "log.txt", level=logging.INFO)  # 设置日志记录配置

    # This prints it to the screen as well
    console = logging.StreamHandler()  # 创建控制台日志处理器
    console.setLevel(logging.INFO)  # 设置控制台日志级别
    logging.getLogger().addHandler(console)  # 将控制台处理器添加到日志记录器

    logging.info(f"Translation task: {opts.src} -> {opts.tgt}")  # 记录翻译任务信息
    logging.info(f"Using device: {DEVICE}")  # 记录使用的设备信息

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    train_dl, valid_dl, src_vocab, tgt_vocab, _, _, special_symbols = get_data(opts)  # 获取训练和验证数据

    logging.info("Loaded data")  # 记录数据加载信息

    src_vocab_size = len(src_vocab)  # 获取源语言词汇表大小
    tgt_vocab_size = len(tgt_vocab)  # 获取目标语言词汇表大小

    logging.info(f"{opts.src} vocab size: {src_vocab_size}")  # 记录源语言词汇表大小
    logging.info(f"{opts.tgt} vocab size: {tgt_vocab_size}")  # 记录目标语言词汇表大小

    # Create model
    model = Translator(  # 创建翻译器模型
        num_encoder_layers=opts.enc_layers,  # 编码器层数
        num_decoder_layers=opts.dec_layers,  # 解码器层数
        embed_size=opts.embed_size,  # 嵌入维度
        num_heads=opts.attn_heads,  # 注意力头数
        src_vocab_size=src_vocab_size,  # 源语言词汇表大小
        tgt_vocab_size=tgt_vocab_size,  # 目标语言词汇表大小
        dim_feedforward=opts.dim_feedforward,  # 前馈网络维度
        dropout=opts.dropout  # 丢弃率
    ).to(DEVICE)  # 移动模型到设备

    logging.info("Model created... starting training!")  # 记录模型创建信息

    # Set up our learning tools
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])  # 定义损失函数

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)  # 设置优化器

    best_val_loss = 1e6  # 初始化最佳验证损失
    
    for idx, epoch in enumerate(range(1, opts.epochs + 1)):  # 遍历每个周期

        start_time = time()  # 记录开始时间
        train_loss = train(model, train_dl, loss_fn, optim, special_symbols, opts)  # 训练模型
        epoch_time = time() - start_time  # 计算周期时间
        val_loss = validate(model, valid_dl, loss_fn, special_symbols)  # 验证模型

        # Once training is done, we want to save out the model
        if val_loss < best_val_loss:  # 如果当前验证损失优于最佳验证损失
            best_val_loss = val_loss  # 更新最佳验证损失
            logging.info("New best model, saving...")  # 记录新的最佳模型保存信息
            torch.save(model.state_dict(), opts.logging_dir + "best.pt")  # 保存最佳模型权重

        torch.save(model.state_dict(), opts.logging_dir + "last.pt")  # 保存最后一次训练的模型权重

        logger.info(f"Epoch: {epoch}\n\tTrain loss: {train_loss:.3f}\n\tVal loss: {val_loss:.3f}\n\tEpoch time = {epoch_time:.1f} seconds\n\tETA = {epoch_time * (opts.epochs - idx - 1):.1f} seconds")  # 记录周期信息

if __name__ == "__main__":  # 主程序入口

    parser = ArgumentParser(  # 创建参数解析器
        prog="Machine Translator training and inference",  # 程序名称
    )

    # Inference mode
    parser.add_argument("--inference", action="store_true",  # 推理模式
                        help="Set true to run inference")  # 帮助信息
    parser.add_argument("--model_path", type=str,  # 模型路径
                        help="Path to the model to run inference on")  # 帮助信息

    # Translation settings
    parser.add_argument("--src", type=str, default="de",  # 源语言
                        help="Source language (translating FROM this language)")  # 帮助信息
    parser.add_argument("--tgt", type=str, default="en",  # 目标语言
                        help="Target language (translating TO this language)")  # 帮助信息

    # Training settings 
    parser.add_argument("-e", "--epochs", type=int, default=30,  # 训练周期数
                        help="Epochs")  # 帮助信息
    parser.add_argument("--lr", type=float, default=1e-4,  # 学习率
                        help="Default learning rate")  # 帮助信息
    parser.add_argument("--batch", type=int, default=128,  # 批量大小
                        help="Batch size")  # 帮助信息
    parser.add_argument("--backend", type=str, default="cpu",  # 后端设备
                        help="Batch size")  # 帮助信息
    
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=8,  # 注意力头数
                        help="Number of attention heads")  # 帮助信息
    parser.add_argument("--enc_layers", type=int, default=5,  # 编码器层数
                        help="Number of encoder layers")  # 帮助信息
    parser.add_argument("--dec_layers", type=int, default=5,  # 解码器层数
                        help="Number of decoder layers")  # 帮助信息
    parser.add_argument("--embed_size", type=int, default=512,  # 嵌入维度
                        help="Size of the language embedding")  # 帮助信息
    parser.add_argument("--dim_feedforward", type=int, default=512,  # 前馈网络维度
                        help="Feedforward dimensionality")  # 帮助信息
    parser.add_argument("--dropout", type=float, default=0.1,  # 丢弃率
                        help="Transformer dropout")  # 帮助信息

    # Logging settings
    parser.add_argument("--logging_dir", type=str, default="./" + str(date.today()) + "/",  # 日志目录
                        help="Where the output of this program should be placed")  # 帮助信息

    # Just for continuous integration
    parser.add_argument("--dry_run", action="store_true")  # 干运行模式

    args = parser.parse_args()  # 解析命令行参数

    DEVICE = torch.device("cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu")  # 设置设备

    if args.inference:  # 如果是推理模式
        inference(args)  # 调用推理函数
    else:  # 否则
        main(args)  # 调用主函数