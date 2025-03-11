import gzip  # 导入gzip库，用于处理gzip压缩文件
import html  # 导入html库，用于处理HTML实体
import os  # 导入os库，用于处理文件和目录
from functools import lru_cache  # 从functools导入lru_cache装饰器，用于缓存函数结果

import ftfy  # 导入ftfy库，用于修复文本中的编码问题
import regex as re  # 导入regex库，作为正则表达式的增强版


@lru_cache()  # 使用lru_cache装饰器缓存函数结果
def default_bpe():  # 定义默认的BPE（Byte Pair Encoding）文件路径函数
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")  # 返回BPE文件的绝对路径


@lru_cache()  # 使用lru_cache装饰器缓存函数结果
def bytes_to_unicode():  # 定义将字节转换为Unicode的函数
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    返回utf-8字节的列表和相应的Unicode字符串列表。
    The reversible bpe codes work on unicode strings.
    可逆的BPE编码在Unicode字符串上工作。
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    这意味着如果你想避免UNK（未知字符），你需要在词汇中包含大量Unicode字符。
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    当你处理大约10B的令牌数据集时，你最终需要大约5K的字符以获得良好的覆盖率。
    This is a significant percentage of your normal, say, 32K bpe vocab.
    这在你的正常词汇表中占有相当大的比例，比如32K的BPE词汇表。
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    为了避免这种情况，我们需要在utf-8字节和Unicode字符串之间建立查找表。
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    并避免映射到BPE编码无法处理的空白/控制字符。
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))  # 定义有效的字节范围
    cs = bs[:]  # 复制字节列表
    n = 0  # 初始化计数器
    for b in range(2**8):  # 遍历所有可能的字节值
        if b not in bs:  # 如果字节值不在有效字节列表中
            bs.append(b)  # 将字节值添加到有效字节列表中
            cs.append(2**8+n)  # 添加对应的Unicode字符
            n += 1  # 增加计数器
    cs = [chr(n) for n in cs]  # 将Unicode代码点转换为字符
    return dict(zip(bs, cs))  # 返回字节和Unicode字符的映射字典


def get_pairs(word):  # 返回单词中符号对的集合
    """Return set of symbol pairs in a word.
    返回单词中的符号对集合。
    Word is represented as tuple of symbols (symbols being variable-length strings).
    单词表示为符号的元组（符号是可变长度的字符串）。
    """
    pairs = set()  # 初始化符号对集合
    prev_char = word[0]  # 获取单词的第一个字符
    for char in word[1:]:  # 遍历单词中的其他字符
        pairs.add((prev_char, char))  # 将前一个字符和当前字符组成的对添加到集合中
        prev_char = char  # 更新前一个字符
    return pairs  # 返回符号对集合


def basic_clean(text):  # 基本清理文本的函数
    text = ftfy.fix_text(text)  # 修复文本中的编码问题
    text = html.unescape(html.unescape(text))  # 反转义HTML实体
    return text.strip()  # 去除文本首尾空白并返回


def whitespace_clean(text):  # 清理文本中的空白字符
    text = re.sub(r'\s+', ' ', text)  # 将多个空白字符替换为一个空格
    text = text.strip()  # 去除文本首尾空白
    return text  # 返回清理后的文本


class SimpleTokenizer(object):  # 定义简单的分词器类
    def __init__(self, bpe_path: str = default_bpe()):  # 初始化方法，默认BPE文件路径
        self.byte_encoder = bytes_to_unicode()  # 获取字节到Unicode的映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # 创建Unicode到字节的映射
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')  # 读取BPE合并规则
        merges = merges[1:49152-256-2+1]  # 选择合并规则的有效部分
        merges = [tuple(merge.split()) for merge in merges]  # 将合并规则转换为元组
        vocab = list(bytes_to_unicode().values())  # 获取字节到Unicode的所有值
        vocab = vocab + [v+'</w>' for v in vocab]  # 为每个词汇添加结束标记
        for merge in merges:  # 遍历合并规则
            vocab.append(''.join(merge))  # 将合并的符号添加到词汇中
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # 添加特殊标记
        self.encoder = dict(zip(vocab, range(len(vocab))))  # 创建词汇到索引的映射
        self.decoder = {v: k for k, v in self.encoder.items()}  # 创建索引到词汇的映射
        self.bpe_ranks = dict(zip(merges, range(len(merges))))  # 创建合并规则的排名
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}  # 初始化缓存
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)  # 编译正则表达式模式

    def bpe(self, token):  # 定义BPE编码方法
        if token in self.cache:  # 如果token在缓存中
            return self.cache[token]  # 返回缓存中的结果
        word = tuple(token[:-1]) + (token[-1] + '</w>',)  # 将token转换为元组并添加结束标记
        pairs = get_pairs(word)  # 获取符号对

        if not pairs:  # 如果没有符号对
            return token + '</w>'  # 返回带结束标记的token

        while True:  # 循环直到没有更多的合并
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))  # 找到最小排名的符号对
            if bigram not in self.bpe_ranks:  # 如果符号对不在排名中
                break  # 退出循环
            first, second = bigram  # 解包符号对
            new_word = []  # 初始化新的单词列表
            i = 0  # 初始化索引
            while i < len(word):  # 遍历单词中的字符
                try:
                    j = word.index(first, i)  # 找到第一个字符的索引
                    new_word.extend(word[i:j])  # 将前面的字符添加到新单词
                    i = j  # 更新索引
                except:  # 如果没有找到
                    new_word.extend(word[i:])  # 添加剩余的字符
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:  # 如果当前字符和下一个字符是符号对
                    new_word.append(first + second)  # 添加合并后的字符
                    i += 2  # 更新索引
                else:
                    new_word.append(word[i])  # 添加当前字符
                    i += 1  # 更新索引
            new_word = tuple(new_word)  # 将新单词转换为元组
            word = new_word  # 更新单词
            if len(word) == 1:  # 如果单词长度为1
                break  # 退出循环
            else:
                pairs = get_pairs(word)  # 获取新的符号对
        word = ' '.join(word)  # 将单词转换为字符串
        self.cache[token] = word  # 将结果添加到缓存
        return word  # 返回编码后的单词

    def encode(self, text):  # 定义编码方法
        bpe_tokens = []  # 初始化BPE令牌列表
        text = whitespace_clean(basic_clean(text)).lower()  # 清理文本并转换为小写
        for token in re.findall(self.pat, text):  # 遍历匹配的token
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))  # 将token编码为字节
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))  # 添加BPE编码后的令牌
        return bpe_tokens  # 返回BPE令牌列表

    def decode(self, tokens):  # 定义解码方法
        text = ''.join([self.decoder[token] for token in tokens])  # 将令牌解码为文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')  # 将字节转换为字符串并替换结束标记
        return text  # 返回解码后的文本