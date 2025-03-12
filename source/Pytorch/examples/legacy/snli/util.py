import os  # 导入os模块，用于处理文件和目录
from argparse import ArgumentParser  # 从argparse导入ArgumentParser，用于处理命令行参数

def makedirs(name):  # 定义makedirs函数，创建目录
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""  # 辅助函数，用于创建目录，避免已存在目录时报错

    import os, errno  # 导入os和errno模块

    try:
        os.makedirs(name)  # 尝试创建目录
    except OSError as ex:  # 捕获OSError异常
        if ex.errno == errno.EEXIST and os.path.isdir(name):  # 如果目录已存在且是一个目录
            # ignore existing directory  # 忽略已存在的目录
            pass  # 不做任何操作
        else:  # 如果发生其他错误
            # a different error happened  # 发生了其他错误
            raise  # 抛出异常


def get_args():  # 定义get_args函数，获取命令行参数
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')  # 创建参数解析器
    parser.add_argument('--epochs', type=int, default=50,  # 训练周期数
                        help='the number of total epochs to run.')  # 帮助信息
    parser.add_argument('--batch_size', type=int, default=128,  # 批量大小
                        help='batch size. (default: 128)')  # 帮助信息
    parser.add_argument('--d_embed', type=int, default=100,  # 嵌入向量大小
                        help='the size of each embedding vector.')  # 帮助信息
    parser.add_argument('--d_proj', type=int, default=300,  # 投影层大小
                        help='the size of each projection layer.')  # 帮助信息
    parser.add_argument('--d_hidden', type=int, default=300,  # 隐藏状态特征数量
                        help='the number of features in the hidden state.')  # 帮助信息
    parser.add_argument('--n_layers', type=int, default=1,  # 循环层数
                        help='the number of recurrent layers. (default: 50)')  # 帮助信息
    parser.add_argument('--log_every', type=int, default=50,  # 日志输出间隔
                        help='iteration period to output log.')  # 帮助信息
    parser.add_argument('--lr', type=float, default=.001,  # 初始学习率
                        help='initial learning rate.')  # 帮助信息
    parser.add_argument('--dev_every', type=int, default=1000,  # 验证结果日志输出间隔
                        help='log period of validation results.')  # 帮助信息
    parser.add_argument('--save_every', type=int, default=1000,  # 模型快照保存间隔
                        help='model checkpoint period.')  # 帮助信息
    parser.add_argument('--dp_ratio', type=int, default=0.2,  # 丢弃率
                        help='probability of an element to be zeroed.')  # 帮助信息
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn',  # 禁用双向LSTM
                        help='disable bidirectional LSTM.')  # 帮助信息
    parser.add_argument('--preserve-case', action='store_false', dest='lower',  # 保持大小写
                        help='case-sensitivity.')  # 帮助信息
    parser.add_argument('--no-projection', action='store_false', dest='projection',  # 禁用投影层
                        help='disable projection layer.')  # 帮助信息
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb',  # 启用嵌入词训练
                        help='enable embedding word training.')  # 帮助信息
    parser.add_argument('--gpu', type=int, default=0,  # 使用的GPU ID
                        help='gpu id to use. (default: 0)')  # 帮助信息
    parser.add_argument('--save_path', type=str, default='results',  # 结果保存路径
                        help='save path of results.')  # 帮助信息
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'),  # 词向量缓存路径
                        help='name of vector cache directory, which saved input word-vectors.')  # 帮助信息
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d',  # 词向量类型
                        help='one of or a list containing instantiations of the GloVe, CharNGram, or Vectors classes.'
                        'Alternatively, one of or a list of available pretrained vectors: '
                        'charngram.100d fasttext.en.300d fasttext.simple.300d'
                        'glove.42B.300d glove.840B.300d glove.twitter.27B.25d'
                        'glove.twitter.27B.50d glove.twitter.27B.100d glove.twitter.27B.200d'
                        'glove.6B.50d glove.6B.100d glove.6B.200d glove.6B.300d')  # 帮助信息
    parser.add_argument('--resume_snapshot', type=str, default='',  # 恢复模型快照
                        help='model snapshot to resume.')  # 帮助信息
    parser.add_argument('--dry-run', action='store_true',  # 干运行模式
                        help='run only a few iterations')  # 帮助信息
    args = parser.parse_args()  # 解析命令行参数
    return args  # 返回参数