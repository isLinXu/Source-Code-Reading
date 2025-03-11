import nltk  # 导入 nltk 库
import pickle  # 导入 pickle 库，用于序列化对象
import argparse  # 导入 argparse 库，用于处理命令行参数
from collections import Counter  # 从 collections 导入 Counter 类，用于计数
from pycocotools.coco import COCO  # 从 pycocotools 导入 COCO 类，用于处理 COCO 数据集


class Vocabulary(object):  # 定义词汇类
    """Simple vocabulary wrapper."""  # 简单的词汇封装器
    def __init__(self):  # 初始化方法
        self.word2idx = {}  # 创建一个空字典，用于存储单词到索引的映射
        self.idx2word = {}  # 创建一个空字典，用于存储索引到单词的映射
        self.idx = 0  # 初始化索引为 0

    def add_word(self, word):  # 添加单词的方法
        if not word in self.word2idx:  # 如果单词不在字典中
            self.word2idx[word] = self.idx  # 将单词和当前索引添加到字典
            self.idx2word[self.idx] = word  # 将当前索引和单词添加到反向字典
            self.idx += 1  # 索引加 1

    def __call__(self, word):  # 使类可调用的方法
        if not word in self.word2idx:  # 如果单词不在字典中
            return self.word2idx['<unk>']  # 返回未知单词的索引
        return self.word2idx[word]  # 返回单词的索引

    def __len__(self):  # 获取字典长度的方法
        return len(self.word2idx)  # 返回单词到索引映射的长度

def build_vocab(json, threshold):  # 构建词汇的方法
    """Build a simple vocabulary wrapper."""  # 构建一个简单的词汇封装器
    coco = COCO(json)  # 加载 COCO 数据集
    counter = Counter()  # 创建计数器
    ids = coco.anns.keys()  # 获取所有注释的 ID
    for i, id in enumerate(ids):  # 遍历每个注释 ID
        caption = str(coco.anns[id]['caption'])  # 获取注释的文本
        tokens = nltk.tokenize.word_tokenize(caption.lower())  # 将注释文本分词并转换为小写
        counter.update(tokens)  # 更新计数器

        if (i+1) % 1000 == 0:  # 每 1000 个注释打印一次信息
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))  # 打印已处理的注释数量

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]  # 过滤频率低于阈值的单词

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()  # 创建词汇实例
    vocab.add_word('<pad>')  # 添加填充符
    vocab.add_word('<start>')  # 添加开始符
    vocab.add_word('<end>')  # 添加结束符
    vocab.add_word('<unk>')  # 添加未知符

    # Add the words to the vocabulary.
    for i, word in enumerate(words):  # 遍历所有有效单词
        vocab.add_word(word)  # 将单词添加到词汇中
    return vocab  # 返回构建的词汇

def main(args):  # 主函数
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)  # 构建词汇
    vocab_path = args.vocab_path  # 获取词汇保存路径
    with open(vocab_path, 'wb') as f:  # 以二进制写入模式打开文件
        pickle.dump(vocab, f)  # 将词汇对象序列化并保存到文件
    print("Total vocabulary size: {}".format(len(vocab)))  # 打印词汇的总大小
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))  # 打印保存路径

if __name__ == '__main__':  # 如果是主程序
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--caption_path', type=str,  # 添加注释文件路径参数
                        default='data/annotations/captions_train2014.json',  # 默认路径
                        help='path for train annotation file')  # 参数帮助信息
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',  # 添加词汇保存路径参数
                        help='path for saving vocabulary wrapper')  # 参数帮助信息
    parser.add_argument('--threshold', type=int, default=4,  # 添加词频阈值参数
                        help='minimum word count threshold')  # 参数帮助信息
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 调用主函数