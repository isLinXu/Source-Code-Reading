# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514  # 代码引用自指定的 GitHub Gist
import tensorflow as tf  # 导入 TensorFlow 库
import numpy as np  # 导入 NumPy 库
import scipy.misc  # 导入 SciPy 的杂项模块
try:
    from StringIO import StringIO  # Python 2.7 中的 StringIO
except ImportError:
    from io import BytesIO  # Python 3.x 中的 BytesIO


class Logger(object):  # 定义 Logger 类
    
    def __init__(self, log_dir):  # 初始化方法，接受日志目录作为参数
        """Create a summary writer logging to log_dir."""  # 创建一个摘要写入器，记录到指定的日志目录
        self.writer = tf.summary.FileWriter(log_dir)  # 创建 TensorFlow 的文件写入器，用于写入日志

    def scalar_summary(self, tag, value, step):  # 记录标量变量的方法
        """Log a scalar variable."""  # 记录一个标量变量
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])  # 创建一个摘要对象
        self.writer.add_summary(summary, step)  # 将摘要写入日志

    def image_summary(self, tag, images, step):  # 记录图像列表的方法
        """Log a list of images."""  # 记录一组图像
        img_summaries = []  # 初始化图像摘要列表
        for i, img in enumerate(images):  # 遍历每个图像
            # Write the image to a string
            try:
                s = StringIO()  # 尝试在 Python 2.7 中创建 StringIO 对象
            except:
                s = BytesIO()  # 在 Python 3.x 中创建 BytesIO 对象
            scipy.misc.toimage(img).save(s, format="png")  # 将图像保存到字符串中，格式为 PNG

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),  # 创建图像对象
                                       height=img.shape[0],  # 设置图像高度
                                       width=img.shape[1])  # 设置图像宽度
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))  # 将图像摘要添加到列表中

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)  # 创建摘要
        self.writer.add_summary(summary, step)  # 将摘要写入日志
        
    def histo_summary(self, tag, values, step, bins=1000):  # 记录张量值的直方图的方法
        """Log a histogram of the tensor of values."""  # 记录值的直方图

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)  # 使用 NumPy 创建直方图

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()  # 创建直方图原型
        hist.min = float(np.min(values))  # 设置直方图的最小值
        hist.max = float(np.max(values))  # 设置直方图的最大值
        hist.num = int(np.prod(values.shape))  # 设置直方图的样本数量
        hist.sum = float(np.sum(values))  # 设置直方图的总和
        hist.sum_squares = float(np.sum(values**2))  # 设置直方图的平方和

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]  # 丢弃第一个箱的起始边缘

        # Add bin edges and counts
        for edge in bin_edges:  # 遍历箱的边缘
            hist.bucket_limit.append(edge)  # 将边缘添加到直方图中
        for c in counts:  # 遍历每个箱的计数
            hist.bucket.append(c)  # 将计数添加到直方图中

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])  # 创建摘要
        self.writer.add_summary(summary, step)  # 将摘要写入日志
        self.writer.flush()  # 刷新写入器