import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库
import matplotlib.pyplot as plt  # 导入 Matplotlib 库用于绘图

# Hyper-parameters
input_size = 1  # 输入大小
output_size = 1  # 输出大小
num_epochs = 60  # 训练的轮数
learning_rate = 0.001  # 学习率

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],  # 训练数据的输入特征
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)  # 数据类型为 float32

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],  # 训练数据的目标值
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)  # 数据类型为 float32

# Linear regression model
model = nn.Linear(input_size, output_size)  # 创建线性回归模型

# Loss and optimizer
criterion = nn.MSELoss()  # 定义损失函数为均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 定义优化器为随机梯度下降（SGD）

# Train the model
for epoch in range(num_epochs):  # 遍历每个训练轮
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)  # 将 NumPy 数组转换为 PyTorch 张量
    targets = torch.from_numpy(y_train)  # 将 NumPy 数组转换为 PyTorch 张量

    # Forward pass
    outputs = model(inputs)  # 前向传播，获取输出
    loss = criterion(outputs, targets)  # 计算损失
    
    # Backward and optimize
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    
    if (epoch+1) % 5 == 0:  # 每 5 个轮次打印一次信息
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))  # 打印当前轮次和损失值

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()  # 预测输出并转换为 NumPy 数组
plt.plot(x_train, y_train, 'ro', label='Original data')  # 绘制原始数据点
plt.plot(x_train, predicted, label='Fitted line')  # 绘制拟合线
plt.legend()  # 显示图例
plt.show()  # 显示图形

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')  # 保存模型的状态字典