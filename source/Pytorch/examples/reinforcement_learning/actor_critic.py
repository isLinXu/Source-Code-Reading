import argparse  # 导入 argparse 库，用于处理命令行参数
import gym  # 导入 OpenAI Gym 库，用于环境模拟
import numpy as np  # 导入 NumPy 库
from itertools import count  # 导入计数器，用于生成无限序列
from collections import namedtuple  # 导入命名元组，用于简单的数据结构

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
from torch.distributions import Categorical  # 导入分类分布，用于动作选择

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')  # 创建参数解析器，描述为 PyTorch actor-critic 示例
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


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])  # 定义命名元组，用于保存动作的对数概率和价值


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    实现一个模型中的演员和评论家
    """
    def __init__(self):
        super(Policy, self).__init__()  # 调用父类构造函数
        # get resnet model
        self.affine1 = nn.Linear(4, 128)  # 输入层，接受 4 个特征并输出 128 个特征

        # actor's layer
        self.action_head = nn.Linear(128, 2)  # 演员层，输出两个动作的概率

        # critic's layer
        self.value_head = nn.Linear(128, 1)  # 评论家层，输出状态值

        # action & reward buffer
        self.saved_actions = []  # 保存动作的列表
        self.rewards = []  # 保存奖励的列表

    def forward(self, x):
        """
        forward of both actor and critic
        演员和评论家的前向传播
        
        Args:
            x: 输入张量
        Returns:
            action_prob: 动作概率
            state_values: 状态值
        """
        x = F.relu(self.affine1(x))  # 通过输入层并应用 ReLU 激活

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)  # 通过演员层得到动作概率

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)  # 通过评论家层得到状态值

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values  # 返回动作概率和状态值


model = Policy()  # 实例化策略模型
optimizer = optim.Adam(model.parameters(), lr=3e-2)  # 创建 Adam 优化器
eps = np.finfo(np.float32).eps.item()  # 获取浮点数的最小值


def select_action(state):
    """Select an action based on the current state.
    根据当前状态选择一个动作。
    
    Args:
        state: 当前状态
    Returns:
        action: 选择的动作
    """
    state = torch.from_numpy(state).float()  # 将状态转换为张量
    probs, state_value = model(state)  # 通过模型获取动作概率和状态值

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)  # 创建一个分类分布

    # and sample an action using the distribution
    action = m.sample()  # 根据分布抽样一个动作

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))  # 保存动作的对数概率和状态值

    # the action to take (left or right)
    return action.item()  # 返回选择的动作


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    训练代码。计算演员和评论家的损失并执行反向传播。
    """
    R = 0  # 初始化奖励
    saved_actions = model.saved_actions  # 获取保存的动作
    policy_losses = []  # 保存演员（策略）损失的列表
    value_losses = []  # 保存评论家（价值）损失的列表
    returns = []  # 保存真实值的列表

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:  # 反向遍历奖励
        # calculate the discounted value
        R = r + args.gamma * R  # 计算折扣值
        returns.insert(0, R)  # 将计算的值插入到返回列表的开头

    returns = torch.tensor(returns)  # 将返回值转换为张量
    returns = (returns - returns.mean()) / (returns.std() + eps)  # 标准化返回值

    for (log_prob, value), R in zip(saved_actions, returns):  # 遍历保存的动作和返回值
        advantage = R - value.item()  # 计算优势

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)  # 计算演员（策略）损失

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))  # 计算评论家（价值）损失

    # reset gradients
    optimizer.zero_grad()  # 清除梯度

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()  # 计算总损失

    # perform backprop
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数

    # reset rewards and action buffer
    del model.rewards[:]  # 清空奖励列表
    del model.saved_actions[:]  # 清空动作列表


def main():
    running_reward = 10  # 初始化运行奖励

    # run infinitely many episodes
    for i_episode in count(1):  # 无限循环

        # reset environment and episode reward
        state, _ = env.reset()  # 重置环境并获取初始状态
        ep_reward = 0  # 初始化每个回合的奖励

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):  # 每个回合最多运行 9999 步

            # select action from policy
            action = select_action(state)  # 从策略中选择动作

            # take the action
            state, reward, done, _, _ = env.step(action)  # 执行动作并获取下一个状态和奖励

            if args.render:  # 如果需要渲染
                env.render()  # 渲染环境

            model.rewards.append(reward)  # 保存奖励
            ep_reward += reward  # 累加奖励
            if done:  # 如果回合结束
                break  # 结束回合

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward  # 更新运行奖励

        # perform backprop
        finish_episode()  # 完成回合并进行反向传播

        # log results
        if i_episode % args.log_interval == 0:  # 每 log_interval 回合打印一次信息
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))  # 打印回合信息

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:  # 检查是否解决了平衡杆问题
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))  # 打印解决信息
            break  # 结束循环


if __name__ == '__main__':
    main()  # 运行主函数