import torch  # 导入PyTorch库
from torch.nn.utils.rnn import pad_sequence  # 从PyTorch导入pad_sequence，用于填充序列
from torch.utils.data import DataLoader  # 从PyTorch导入DataLoader，用于数据加载
from torchtext.data.utils import get_tokenizer  # 从torchtext导入get_tokenizer，用于获取分词器
from torchtext.vocab import build_vocab_from_iterator  # 从torchtext导入build_vocab_from_iterator，用于构建词汇表
from torchtext.datasets import Multi30k, multi30k  # 从torchtext导入Multi30k数据集

# Turns an iterable into a generator
def _yield_tokens(iterable_data, tokenizer, src):  # 将可迭代数据转换为生成器

    # Iterable data stores the samples as (src, tgt) so this will help us select just one language or the other
    index = 0 if src else 1  # 根据src选择索引，0表示源语言，1表示目标语言

    for data in iterable_data:  # 遍历可迭代数据
        yield tokenizer(data[index])  # 使用分词器处理数据并生成标记

# Get data, tokenizer, text transform, vocab objs, etc. Everything we
# need to start training the model
def get_data(opts):  # 获取数据、分词器、文本转换和词汇对象等

    src_lang = opts.src  # 源语言
    tgt_lang = opts.tgt  # 目标语言

    # 设置Multi30k数据集的URL
    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

    # Define a token "unkown", "padding", "beginning of sentence", and "end of sentence"
    special_symbols = {  # 定义特殊符号
        "<unk>": 0,  # 未知标记
        "<pad>": 1,  # 填充标记
        "<bos>": 2,  # 句子开始标记
        "<eos>": 3   # 句子结束标记
    }

    # Get training examples from torchtext (the multi30k dataset)
    train_iterator = Multi30k(split="train", language_pair=(src_lang, tgt_lang))  # 获取训练数据迭代器
    valid_iterator = Multi30k(split="valid", language_pair=(src_lang, tgt_lang))  # 获取验证数据迭代器

    # Grab a tokenizer for these languages
    src_tokenizer = get_tokenizer("spacy", src_lang)  # 获取源语言的分词器
    tgt_tokenizer = get_tokenizer("spacy", tgt_lang)  # 获取目标语言的分词器

    # Build a vocabulary object for these languages
    src_vocab = build_vocab_from_iterator(  # 为源语言构建词汇表
        _yield_tokens(train_iterator, src_tokenizer, src_lang),  # 从训练迭代器生成标记
        min_freq=1,  # 最小词频
        specials=list(special_symbols.keys()),  # 特殊符号列表
        special_first=True  # 将特殊符号放在词汇表的前面
    )

    tgt_vocab = build_vocab_from_iterator(  # 为目标语言构建词汇表
        _yield_tokens(train_iterator, tgt_tokenizer, tgt_lang),  # 从训练迭代器生成标记
        min_freq=1,  # 最小词频
        specials=list(special_symbols.keys()),  # 特殊符号列表
        special_first=True  # 将特殊符号放在词汇表的前面
    )

    src_vocab.set_default_index(special_symbols["<unk>"])  # 设置源语言词汇表的默认索引
    tgt_vocab.set_default_index(special_symbols["<unk>"])  # 设置目标语言词汇表的默认索引

    # Helper function to sequentially apply transformations
    def _seq_transform(*transforms):  # 定义一个帮助函数，依次应用转换
        def func(txt_input):  # 定义内部函数
            for transform in transforms:  # 遍历所有转换
                txt_input = transform(txt_input)  # 应用转换
            return txt_input  # 返回转换后的文本
        return func  # 返回内部函数

    # Function to add BOS/EOS and create tensor for input sequence indices
    def _tensor_transform(token_ids):  # 定义函数，添加BOS/EOS并创建张量
        return torch.cat(  # 拼接张量
            (torch.tensor([special_symbols["<bos>"]]),  # 添加BOS标记
             torch.tensor(token_ids),  # 添加标记ID
             torch.tensor([special_symbols["<eos>"]]))  # 添加EOS标记
        )

    src_lang_transform = _seq_transform(src_tokenizer, src_vocab, _tensor_transform)  # 定义源语言的转换
    tgt_lang_transform = _seq_transform(tgt_tokenizer, tgt_vocab, _tensor_transform)  # 定义目标语言的转换

    # Now we want to convert the torchtext data pipeline to a dataloader. We
    # will need to collate batches
    def _collate_fn(batch):  # 定义合并函数
        src_batch, tgt_batch = [], []  # 初始化源和目标批次
        for src_sample, tgt_sample in batch:  # 遍历批次中的样本
            src_batch.append(src_lang_transform(src_sample.rstrip("\n")))  # 处理源样本并添加到源批次
            tgt_batch.append(tgt_lang_transform(tgt_sample.rstrip("\n")))  # 处理目标样本并添加到目标批次

        src_batch = pad_sequence(src_batch, padding_value=special_symbols["<pad>"])  # 填充源批次
        tgt_batch = pad_sequence(tgt_batch, padding_value=special_symbols["<pad>"])  # 填充目标批次
        return src_batch, tgt_batch  # 返回填充后的批次

    # Create the dataloader
    train_dataloader = DataLoader(train_iterator, batch_size=opts.batch, collate_fn=_collate_fn)  # 创建训练数据加载器
    valid_dataloader = DataLoader(valid_iterator, batch_size=opts.batch, collate_fn=_collate_fn)  # 创建验证数据加载器

    return train_dataloader, valid_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols  # 返回所有对象

def generate_square_subsequent_mask(size, device):  # 生成方形后续掩码
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)  # 创建上三角矩阵并转置
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # 填充掩码
    return mask  # 返回掩码

# Create masks for input into model
def create_mask(src, tgt, pad_idx, device):  # 创建输入模型的掩码

    # Get sequence length
    src_seq_len = src.shape[0]  # 获取源序列长度
    tgt_seq_len = tgt.shape[0]  # 获取目标序列长度

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)  # 生成目标掩码
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  # 创建源掩码

    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx).transpose(0, 1)  # 创建源填充掩码
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)  # 创建目标填充掩码
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask  # 返回所有掩码

# A small test to make sure our data loads in correctly
if __name__ == "__main__":  # 主程序入口

    class Opts:  # 定义选项类
        def __init__(self):  # 初始化
            self.src = "en",  # 源语言为英语
            self.tgt = "de"  # 目标语言为德语
            self.batch = 128  # 批量大小

    opts = Opts()  # 创建选项对象
    
    train_dl, valid_dl, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols = get_data(opts)  # 获取数据

    print(f"{opts.src} vocab size: {len(src_vocab)}")  # 打印源语言词汇表大小
    print(f"{opts.src} vocab size: {len(tgt_vocab)}")  # 打印目标语言词汇表大小