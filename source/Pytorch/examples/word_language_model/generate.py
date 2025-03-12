###############################################################################
# Language Modeling on Wikitext-2  # 在 Wikitext-2 上进行语言建模
#
# This file generates new sentences sampled from the language model.  # 该文件生成从语言模型中采样的新句子
#
###############################################################################
import argparse  # 导入 argparse 库，用于处理命令行参数
import torch  # 导入 PyTorch 库

import data  # 导入自定义数据模块

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')  # 创建参数解析器，描述为 PyTorch Wikitext-2 语言模型
# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',  # 添加数据参数，指定数据集位置
                    help='location of the data corpus')  # 帮助信息：数据集位置
parser.add_argument('--checkpoint', type=str, default='./model.pt',  # 添加检查点参数，指定模型检查点文件
                    help='model checkpoint to use')  # 帮助信息：使用的模型检查点
parser.add_argument('--outf', type=str, default='generated.txt',  # 添加输出文件参数，指定生成文本的输出文件
                    help='output file for generated text')  # 帮助信息：生成文本的输出文件
parser.add_argument('--words', type=int, default='1000',  # 添加单词数量参数，指定生成的单词数量
                    help='number of words to generate')  # 帮助信息：要生成的单词数量
parser.add_argument('--seed', type=int, default=1111,  # 添加随机种子参数，指定随机种子
                    help='random seed')  # 帮助信息：随机种子
parser.add_argument('--cuda', action='store_true',  # 添加 CUDA 参数，指定是否使用 CUDA
                    help='use CUDA')  # 帮助信息：使用 CUDA
parser.add_argument('--mps', action='store_true', default=False,  # 添加 MPS 参数，指定是否启用 macOS GPU 训练
                        help='enables macOS GPU training')  # 帮助信息：启用 macOS GPU 训练
parser.add_argument('--temperature', type=float, default=1.0,  # 添加温度参数，指定生成文本的温度
                    help='temperature - higher will increase diversity')  # 帮助信息：温度 - 较高的值将增加多样性
parser.add_argument('--log-interval', type=int, default=100,  # 添加日志间隔参数，指定报告间隔
                    help='reporting interval')  # 帮助信息：报告间隔
args = parser.parse_args()  # 解析命令行参数

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)  # 手动设置随机种子以确保可重复性
if torch.cuda.is_available():  # 如果 CUDA 可用
    if not args.cuda:  # 如果未指定使用 CUDA
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")  # 警告：您有 CUDA 设备，因此应该使用 --cuda 运行
if torch.backends.mps.is_available():  # 如果 MPS 可用
    if not args.mps:  # 如果未指定使用 MPS
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")  # 警告：您有 MPS 设备，要启用 macOS GPU，请使用 --mps 运行
        
use_mps = args.mps and torch.backends.mps.is_available()  # 确定是否使用 MPS
if args.cuda:  # 如果指定使用 CUDA
    device = torch.device("cuda")  # 设置设备为 CUDA
elif use_mps:  # 如果使用 MPS
    device = torch.device("mps")  # 设置设备为 MPS
else:  # 否则
    device = torch.device("cpu")  # 设置设备为 CPU

if args.temperature < 1e-3:  # 如果温度小于 1e-3
    parser.error("--temperature has to be greater or equal 1e-3.")  # 报错：温度必须大于或等于 1e-3

with open(args.checkpoint, 'rb') as f:  # 以二进制模式打开检查点文件
    model = torch.load(f, map_location=device)  # 加载模型，指定设备
model.eval()  # 设置模型为评估模式

corpus = data.Corpus(args.data)  # 创建 Corpus 实例，加载数据
ntokens = len(corpus.dictionary)  # 获取词典中的单词数量

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'  # 检查模型是否为 Transformer 模型
if not is_transformer_model:  # 如果不是 Transformer 模型
    hidden = model.init_hidden(1)  # 初始化隐藏状态
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)  # 随机生成输入张量

with open(args.outf, 'w') as outf:  # 以写模式打开输出文件
    with torch.no_grad():  # 在不跟踪历史的情况下执行
        for i in range(args.words):  # 遍历要生成的单词数量
            if is_transformer_model:  # 如果是 Transformer 模型
                output = model(input, False)  # 获取模型输出
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()  # 计算单词权重
                word_idx = torch.multinomial(word_weights, 1)[0]  # 根据权重采样单词索引
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)  # 创建单词张量
                input = torch.cat([input, word_tensor], 0)  # 将新单词添加到输入中
            else:  # 如果不是 Transformer 模型
                output, hidden = model(input, hidden)  # 获取模型输出和隐藏状态
                word_weights = output.squeeze().div(args.temperature).exp().cpu()  # 计算单词权重
                word_idx = torch.multinomial(word_weights, 1)[0]  # 根据权重采样单词索引
                input.fill_(word_idx)  # 用新单词索引填充输入

            word = corpus.dictionary.idx2word[word_idx]  # 获取单词

            outf.write(word + ('\n' if i % 20 == 19 else ' '))  # 写入单词到输出文件，每 20 个单词换行

            if i % args.log_interval == 0:  # 每 log_interval 步打印一次信息
                print('| Generated {}/{} words'.format(i, args.words))  # 打印生成的单词数量