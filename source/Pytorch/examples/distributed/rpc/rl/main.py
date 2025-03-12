import argparse  # 导入argparse模块，用于解析命令行参数
import gymnasium as gym  # 导入gymnasium库，用于创建和管理环境
import numpy as np  # 导入NumPy库，用于数值计算
import os  # 导入os模块，用于与操作系统交互
from itertools import count  # 从itertools模块导入count，用于生成计数器

import torch  # 导入PyTorch库
import torch.distributed.rpc as rpc  # 导入分布式RPC模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块
import torch.optim as optim  # 导入优化器模块
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote  # 导入远程引用和RPC相关函数
from torch.distributions import Categorical  # 导入Categorical分布

TOTAL_EPISODE_STEP = 5000  # 总步数
AGENT_NAME = "agent"  # 代理名称
OBSERVER_NAME = "observer{}"  # 观察者名称模板

parser = argparse.ArgumentParser(description='PyTorch RPC RL example')  # 创建命令行参数解析器
parser.add_argument('--world-size', type=int, default=2, metavar='W',  # 添加世界规模参数
                    help='world size for RPC, rank 0 is the agent, others are observers')  # 世界规模的帮助信息
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',  # 添加折扣因子参数
                    help='discount factor (default: 0.99)')  # 折扣因子的帮助信息
parser.add_argument('--seed', type=int, default=543, metavar='N',  # 添加随机种子参数
                    help='random seed (default: 543)')  # 随机种子的帮助信息
parser.add_argument('--log-interval', type=int, default=10, metavar='N',  # 添加日志间隔参数
                    help='interval between training status logs (default: 10)')  # 日志间隔的帮助信息
args = parser.parse_args()  # 解析命令行参数

torch.manual_seed(args.seed)  # 设置随机种子


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """  # 辅助函数，用于在给定的RRef上调用方法
    return method(rref.local_value(), *args, **kwargs)  # 调用方法并返回结果


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """  # 辅助函数，在RRef的拥有者上运行方法并通过RPC获取结果
    args = [method, rref] + list(args)  # 将方法和RRef添加到参数列表中
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)  # 在拥有RRef的远程节点上同步调用方法


class Policy(nn.Module):
    r"""
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/main/reinforcement_learning
    """  # 从强化学习示例中借用“Policy”类。复制代码使这两个示例独立。
    def __init__(self):
        super(Policy, self).__init__()  # 调用父类构造函数
        self.affine1 = nn.Linear(4, 128)  # 创建第一个全连接层
        self.dropout = nn.Dropout(p=0.6)  # 创建Dropout层
        self.affine2 = nn.Linear(128, 2)  # 创建第二个全连接层

        self.saved_log_probs = []  # 保存的对数概率
        self.rewards = []  # 保存的奖励

    def forward(self, x):
        x = self.affine1(x)  # 通过第一个全连接层
        x = self.dropout(x)  # 应用Dropout
        x = F.relu(x)  # 应用ReLU激活函数
        action_scores = self.affine2(x)  # 通过第二个全连接层
        return F.softmax(action_scores, dim=1)  # 返回动作的概率分布


class Observer:
    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.

    It is true that CartPole-v1 is a relatively inexpensive environment, and it
    might be an overkill to use RPC to connect observers and trainers in this
    specific use case. However, the main goal of this tutorial to how to build
    an application using the RPC API. Developers can extend the similar idea to
    other applications with much more expensive environment.
    """  # 观察者独占访问自己的环境。每个观察者从环境中捕获状态，并将状态发送给代理以选择动作。
    def __init__(self):
        self.id = rpc.get_worker_info().id  # 获取当前工作者的ID
        self.env = gym.make('CartPole-v1')  # 创建CartPole-v1环境
        self.env.reset(seed=args.seed)  # 重置环境并设置随机种子

    def run_episode(self, agent_rref, n_steps):
        r"""
        Run one episode of n_steps.

        Args:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """  # 运行n_steps的一个回合
        state, ep_reward = self.env.reset()[0], 0  # 重置环境并获取初始状态和奖励
        for step in range(n_steps):  # 遍历每一步
            # send the state to the agent to get an action
            action = _remote_method(Agent.select_action, agent_rref, self.id, state)  # 发送状态给代理以获取动作

            # apply the action to the environment, and get the reward
            state, reward, terminated, truncated, _ = self.env.step(action)  # 将动作应用于环境并获取奖励

            # report the reward to the agent for training purpose
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)  # 向代理报告奖励

            if terminated or truncated:  # 如果回合结束
                break  # 退出循环


class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []  # 初始化观察者的远程引用列表
        self.agent_rref = RRef(self)  # 创建代理的远程引用
        self.rewards = {}  # 初始化奖励字典
        self.saved_log_probs = {}  # 初始化保存的对数概率字典
        self.policy = Policy()  # 创建策略对象
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)  # 创建Adam优化器
        self.eps = np.finfo(np.float32).eps.item()  # 获取浮点数的最小值
        self.running_reward = 0  # 初始化运行奖励
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold  # 获取奖励阈值
        for ob_rank in range(1, world_size):  # 遍历所有观察者
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))  # 获取观察者的信息
            self.ob_rrefs.append(remote(ob_info, Observer))  # 添加观察者的远程引用
            self.rewards[ob_info.id] = []  # 初始化观察者的奖励列表
            self.saved_log_probs[ob_info.id] = []  # 初始化观察者的保存对数概率列表

    def select_action(self, ob_id, state):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/main/reinforcement_learning
        The main difference is that instead of keeping all probs in one list,
        the agent keeps probs in a dictionary, one key per observer.

        NB: no need to enforce thread-safety here as GIL will serialize
        executions.
        """  # 选择动作的函数
        state = torch.from_numpy(state).float().unsqueeze(0)  # 将状态转换为张量并添加维度
        probs = self.policy(state)  # 获取动作的概率分布
        m = Categorical(probs)  # 创建Categorical分布对象
        action = m.sample()  # 从分布中采样动作
        self.saved_log_probs[ob_id].append(m.log_prob(action))  # 保存对数概率
        return action.item()  # 返回动作

    def report_reward(self, ob_id, reward):
        r"""
        Observers call this function to report rewards.
        """  # 观察者调用此函数报告奖励
        self.rewards[ob_id].append(reward)  # 将奖励添加到对应观察者的奖励列表中

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each observer to run n_steps.
        """  # 运行一个回合，代理会告诉每个观察者运行n_steps
        futs = []  # 初始化未来对象列表
        for ob_rref in self.ob_rrefs:  # 遍历所有观察者的远程引用
            # make async RPC to kick off an episode on all observers
            futs.append(  # 添加未来对象到列表
                rpc_async(
                    ob_rref.owner(),  # 获取观察者的拥有者
                    _call_method,  # 调用方法
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps)  # 传递参数
                )
            )

        # wait until all observers have finished this episode
        for fut in futs:  # 遍历所有未来对象
            fut.wait()  # 等待每个未来对象完成

    def finish_episode(self):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/main/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """  # 结束回合的函数
        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []  # 初始化总奖励、概率和奖励列表
        for ob_id in self.rewards:  # 遍历所有观察者的奖励
            probs.extend(self.saved_log_probs[ob_id])  # 将保存的对数概率添加到列表
            rewards.extend(self.rewards[ob_id])  # 将奖励添加到列表

        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])  # 获取最小奖励
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward  # 更新运行奖励

        # clear saved probs and rewards
        for ob_id in self.rewards:  # 遍历所有观察者
            self.rewards[ob_id] = []  # 清空奖励列表
            self.saved_log_probs[ob_id] = []  # 清空保存的对数概率列表

        policy_loss, returns = [], []  # 初始化策略损失和返回值列表
        for r in rewards[::-1]:  # 反向遍历奖励
            R = r + args.gamma * R  # 计算折扣后的奖励
            returns.insert(0, R)  # 将返回值插入到列表的开头
        returns = torch.tensor(returns)  # 转换为张量
        returns = (returns - returns.mean()) / (returns.std() + self.eps)  # 标准化返回值
        for log_prob, R in zip(probs, returns):  # 遍历对数概率和返回值
            policy_loss.append(-log_prob * R)  # 计算策略损失
        self.optimizer.zero_grad()  # 清零梯度
        policy_loss = torch.cat(policy_loss).sum()  # 计算总损失
        policy_loss.backward()  # 反向传播
        self.optimizer.step()  # 更新优化器
        return min_reward  # 返回最小奖励


def run_worker(rank, world_size):
    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """  # 所有进程的入口点，rank 0是代理，其他是观察者
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '29500'  # 设置主节点端口
    if rank == 0:  # 如果是代理
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)  # 初始化RPC

        agent = Agent(world_size)  # 创建代理实例
        for i_episode in count(1):  # 遍历每个回合
            n_steps = int(TOTAL_EPISODE_STEP / (args.world_size - 1))  # 计算每个观察者的步数
            agent.run_episode(n_steps=n_steps)  # 运行回合
            last_reward = agent.finish_episode()  # 结束回合并获取最后奖励

            if i_episode % args.log_interval == 0:  # 每隔一定回合打印日志
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, last_reward, agent.running_reward))  # 打印回合信息

            if agent.running_reward > agent.reward_threshold:  # 如果运行奖励超过阈值
                print("Solved! Running reward is now {}!".format(agent.running_reward))  # 打印解决信息
                break  # 退出循环
    else:  # 如果是观察者
        # other ranks are the observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)  # 初始化观察者的RPC
        # observers passively waiting for instructions from agents
    rpc.shutdown()  # 关闭RPC


def main():
    mp.spawn(  # 启动多个进程
        run_worker,  # 目标函数
        args=(args.world_size, ),  # 传递参数
        nprocs=args.world_size,  # 进程数量
        join=True  # 等待所有进程完成
    )

if __name__ == '__main__':
    main()  # 运行主函数