import argparse  # 导入 argparse 库，用于处理命令行参数
import gym  # 导入 OpenAI Gym 库，用于环境模拟
import numpy as np  # 导入 NumPy 库
from itertools import count  # 导入计数器，用于生成无限序列
from collections import deque  # 导入双端队列，用于存储返回值
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
from torch.distributions import Categorical  # 导入分类分布，用于动作选择


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')  # 创建参数解析器，描述为 PyTorch REINFORCE 示例
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')  # 添加折扣因子参数
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')  # 添加随机种子参数
parser.add_argument('--render', action='store_true',
                    help='render the environment')  # 添加渲染环境的参数
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')  # 添加日志间隔参数
args = parser.parse_args()  # 解析命令行参数


env = gym.make('CartPole-v1')  # 创建 CartPole 环境
env.reset(seed=args.seed)  # 重置环境并设置随机种子
torch.manual_seed(args.seed)  # 设置 PyTorch 随机种子


class Policy(nn.Module):
    def __init__(self):
        """Initialize the policy network.
        初始化策略网络。
        """
        super(Policy, self).__init__()  # 调用父类构造函数
        self.affine1 = nn.Linear(4, 128)  # 输入层，接受 4 个特征并输出 128 个特征
        self.dropout = nn.Dropout(p=0.6)  # Dropout 层，防止过拟合
        self.affine2 = nn.Linear(128, 2)  # 输出层，输出两个动作的概率

        self.saved_log_probs = []  # 保存动作的对数概率的列表
        self.rewards = []  # 保存奖励的列表

    def forward(self, x):
        """Forward pass of the policy network.
        策略网络的前向传播。
        
        Args:
            x: 输入张量
        Returns:
            action_prob: 动作概率
        """
        x = self.affine1(x)  # 通过输入层
        x = self.dropout(x)  # 应用 Dropout
        x = F.relu(x)  # 应用 ReLU 激活
        action_scores = self.affine2(x)  # 通过输出层得到动作分数
        return F.softmax(action_scores, dim=1)  # 返回动作概率


policy = Policy()  # 实例化策略模型
optimizer = optim.Adam(policy.parameters(), lr=1e-2)  # 创建 Adam 优化器
eps = np.finfo(np.float32).eps.item()  # 获取浮点数的最小值


def select_action(state):
    """Select an action based on the current state.
    根据当前状态选择一个动作。
    
    Args:
        state: 当前状态
    Returns:
        action: 选择的动作
    """
    state = torch.from_numpy(state).float().unsqueeze(0)  # 将状态转换为张量并增加维度
    probs = policy(state)  # 通过模型获取动作概率
    m = Categorical(probs)  # 创建一个分类分布
    action = m.sample()  # 根据分布抽样一个动作
    policy.saved_log_probs.append(m.log_prob(action))  # 保存动作的对数概率
    return action.item()  # 返回选择的动作


def finish_episode():
    """Calculate the loss and perform backpropagation.
    计算损失并执行反向传播。
    """
    R = 0  # 初始化奖励
    policy_loss = []  # 保存策略损失的列表
    returns = deque()  # 保存返回值的双端队列
    for r in policy.rewards[::-1]:  # 反向遍历奖励
        R = r + args.gamma * R  # 计算折扣值
        returns.appendleft(R)  # 将计算的值插入到返回列表的开头
    returns = torch.tensor(returns)  # 将返回值转换为张量
    returns = (returns - returns.mean()) / (returns.std() + eps)  # 标准化返回值
    for log_prob, R in zip(policy.saved_log_probs, returns):  # 遍历保存的动作和返回值
        policy_loss.append(-log_prob * R)  # 计算策略损失
    optimizer.zero_grad()  # 清除梯度
    policy_loss = torch.cat(policy_loss).sum()  # 计算总损失
    policy_loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数
    del policy.rewards[:]  # 清空奖励列表
    del policy.saved_log_probs[:]  # 清空动作列表


def main():
    running_reward = 10  # 初始化运行奖励
    for i_episode in count(1):  # 无限循环
        state, _ = env.reset()  # 重置环境并获取初始状态
        ep_reward = 0  # 初始化每个回合的奖励
        for t in range(1, 10000):  # 每个回合最多运行 9999 步
            action = select_action(state)  # 从策略中选择动作
            state, reward, done, _, _ = env.step(action)  # 执行动作并获取下一个状态和奖励
            if args.render:  # 如果需要渲染
                env.render()  # 渲染环境
            policy.rewards.append(reward)  # 保存奖励
            ep_reward += reward  # 累加奖励
            if done:  # 如果回合结束
                break  # 结束回合

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward  # 更新运行奖励
        finish_episode()  # 完成回合并进行反向传播
        if i_episode % args.log_interval == 0:  # 每 log_interval 回合打印一次信息
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))  # 打印回合信息
        if running_reward > env.spec.reward_threshold:  # 检查是否解决了平衡杆问题
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))  # 打印解决信息
            break  # 结束循环


if __name__ == '__main__':
    main()  # 运行主函数