from __future__ import print_function  # 引入未来的 print 函数，确保兼容性
import argparse  # 导入 argparse 库，用于处理命令行参数
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
import numpy as np  # 导入 NumPy 库
import matplotlib  # 导入 Matplotlib 库
matplotlib.use('Agg')  # 使用 Agg 后端，适合在没有显示界面的环境中绘图
import matplotlib.pyplot as plt  # 导入 Matplotlib 的绘图模块


class Sequence(nn.Module):
    def __init__(self):
        """Initialize the sequence model.
        初始化序列模型。
        """
        super(Sequence, self).__init__()  # 调用父类构造函数
        self.lstm1 = nn.LSTMCell(1, 51)  # 第一个 LSTM 单元，输入特征为 1，隐藏层特征为 51
        self.lstm2 = nn.LSTMCell(51, 51)  # 第二个 LSTM 单元，输入特征为 51，隐藏层特征为 51
        self.linear = nn.Linear(51, 1)  # 线性层，将 51 个特征映射到 1 个输出

    def forward(self, input, future=0):
        """Forward pass of the sequence model.
        序列模型的前向传播。
        
        Args:
            input: 输入张量
            future: 预测未来的步数
        Returns:
            outputs: 模型的输出
        """
        outputs = []  # 用于存储输出的列表
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)  # 初始化第一个 LSTM 的隐藏状态
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)  # 初始化第一个 LSTM 的细胞状态
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)  # 初始化第二个 LSTM 的隐藏状态
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)  # 初始化第二个 LSTM 的细胞状态

        for input_t in input.split(1, dim=1):  # 按时间步分割输入
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))  # 通过第一个 LSTM 计算隐藏状态和细胞状态
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))  # 通过第二个 LSTM 计算隐藏状态和细胞状态
            output = self.linear(h_t2)  # 通过线性层得到输出
            outputs += [output]  # 将输出添加到列表中

        for i in range(future):  # 如果需要预测未来
            h_t, c_t = self.lstm1(output, (h_t, c_t))  # 通过第一个 LSTM 计算新的隐藏状态和细胞状态
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))  # 通过第二个 LSTM 计算新的隐藏状态和细胞状态
            output = self.linear(h_t2)  # 通过线性层得到输出
            outputs += [output]  # 将输出添加到列表中

        outputs = torch.cat(outputs, dim=1)  # 将所有输出沿时间维度连接
        return outputs  # 返回所有输出


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--steps', type=int, default=15, help='steps to run')  # 添加步骤参数，默认值为 15
    opt = parser.parse_args()  # 解析命令行参数
    # set random seed to 0
    np.random.seed(0)  # 设置 NumPy 随机种子为 0
    torch.manual_seed(0)  # 设置 PyTorch 随机种子为 0
    # load data and make training set
    data = torch.load('traindata.pt')  # 加载训练数据
    input = torch.from_numpy(data[3:, :-1])  # 生成输入张量，去掉最后一列
    target = torch.from_numpy(data[3:, 1:])  # 生成目标张量，去掉第一列
    test_input = torch.from_numpy(data[:3, :-1])  # 生成测试输入张量
    test_target = torch.from_numpy(data[:3, 1:])  # 生成测试目标张量
    # build the model
    seq = Sequence()  # 实例化序列模型
    seq.double()  # 将模型参数转换为双精度
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)  # 使用 LBFGS 优化器，学习率为 0.8
    # begin to train
    for i in range(opt.steps):  # 进行指定步数的训练
        print('STEP: ', i)  # 打印当前步骤
        def closure():  # 定义闭包函数
            optimizer.zero_grad()  # 清除梯度
            out = seq(input)  # 通过模型进行前向传播
            loss = criterion(out, target)  # 计算损失
            print('loss:', loss.item())  # 打印损失值
            loss.backward()  # 反向传播
            return loss  # 返回损失

        optimizer.step(closure)  # 执行优化步骤

        # begin to predict, no need to track gradient here
        with torch.no_grad():  # 不需要计算梯度
            future = 1000  # 设置未来预测步数为 1000
            pred = seq(test_input, future=future)  # 进行未来预测
            loss = criterion(pred[:, :-future], test_target)  # 计算测试损失
            print('test loss:', loss.item())  # 打印测试损失值
            y = pred.detach().numpy()  # 将预测结果转换为 NumPy 数组

        # draw the result
        plt.figure(figsize=(30, 10))  # 创建绘图窗口，设置图形大小
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)  # 设置标题
        plt.xlabel('x', fontsize=20)  # 设置 x 轴标签
        plt.ylabel('y', fontsize=20)  # 设置 y 轴标签
        plt.xticks(fontsize=20)  # 设置 x 轴刻度字体大小
        plt.yticks(fontsize=20)  # 设置 y 轴刻度字体大小

        def draw(yi, color):  # 定义绘图函数
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)  # 绘制实际值
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)  # 绘制预测值

        draw(y[0], 'r')  # 绘制第一条曲线
        draw(y[1], 'g')  # 绘制第二条曲线
        draw(y[2], 'b')  # 绘制第三条曲线
        plt.savefig('predict%d.pdf' % i)  # 保存绘图结果为 PDF 文件
        plt.close()  # 关闭绘图窗口