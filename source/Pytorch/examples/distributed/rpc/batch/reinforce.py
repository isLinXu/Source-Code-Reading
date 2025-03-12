import argparse  # 导入argparse模块，用于解析命令行参数
import gym  # 导入gym库，用于创建和使用强化学习环境
import os  # 导入os模块，用于与操作系统交互
import threading  # 导入threading模块，用于多线程操作
import time  # 导入time模块，用于时间相关的操作

import torch  # 导入PyTorch库
import torch.distributed.rpc as rpc  # 导入PyTorch的分布式RPC模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote  # 导入RPC相关的类和函数
from torch.distributions import Categorical  # 导入分类分布模块

# demonstrating using rpc.functions.async_execution to speed up training
# 演示如何使用rpc.functions.async_execution来加速训练

NUM_STEPS = 500  # 每个episode的步数
AGENT_NAME = "agent"  # 代理的名称
OBSERVER_NAME = "observer{}"  # 观察者的名称模板

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch RPC Batch RL example')
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor (default: 1.0)')  # 折扣因子
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')  # 随机种子
parser.add_argument('--num-episode', type=int, default=10, metavar='E',
                    help='number of episodes (default: 10)')  # 训练的episode数量
args = parser.parse_args()  # 解析命令行参数

torch.manual_seed(args.seed)  # 设置随机种子

class Policy(nn.Module):
    r"""
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/main/reinforcement_learning
    """
    # 从强化学习示例中借用``Policy``类。复制代码使这两个示例独立。
    # 参见：https://github.com/pytorch/examples/tree/main/reinforcement_learning

    def __init__(self, batch=True):
        super(Policy, self).__init__()  # 调用父类构造函数
        self.affine1 = nn.Linear(4, 128)  # 第一层全连接层，将输入从4维映射到128维
        self.dropout = nn.Dropout(p=0.6)  # Dropout层，丢弃概率为0.6
        self.affine2 = nn.Linear(128, 2)  # 第二层全连接层，将输入从128维映射到2维
        self.dim = 2 if batch else 1  # 根据是否批处理设置维度

    def forward(self, x):
        x = self.affine1(x)  # 通过第一层全连接层
        x = self.dropout(x)  # 应用Dropout
        x = F.relu(x)  # 使用ReLU激活函数
        action_scores = self.affine2(x)  # 通过第二层全连接层
        return F.softmax(action_scores, dim=self.dim)  # 返回动作的概率分布

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
    """
    # 观察者独占其环境的访问权。每个观察者捕获其环境的状态，并将状态发送给代理以选择动作。
    # 然后，观察者将动作应用于其环境并向代理报告奖励。
    #
    # 确实，CartPole-v1是一个相对便宜的环境，在这种特定用例中使用RPC连接观察者和训练者可能是过度的。
    # 然而，本教程的主要目标是如何使用RPC API构建应用程序。开发人员可以将类似的思想扩展到
    # 其他具有更高成本的环境的应用程序。

    def __init__(self, batch=True):
        self.id = rpc.get_worker_info().id - 1  # 获取观察者的ID
        self.env = gym.make('CartPole-v1')  # 创建CartPole-v1环境
        self.env.seed(args.seed)  # 设置环境的随机种子
        self.select_action = Agent.select_action_batch if batch else Agent.select_action  # 根据是否批处理选择动作函数

    def run_episode(self, agent_rref, n_steps):
        r"""
        Run one episode of n_steps.

        Args:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        # 运行n_steps的一个episode。
        #
        # 参数：
        #     agent_rref (RRef): 参考代理对象的RRef。
        #     n_steps (int): 此episode中的步数

        state, ep_reward = self.env.reset(), NUM_STEPS  # 重置环境并获取初始状态和episode奖励
        rewards = torch.zeros(n_steps)  # 初始化奖励张量
        start_step = 0  # 初始化起始步数
        for step in range(n_steps):  # 遍历每一步
            state = torch.from_numpy(state).float().unsqueeze(0)  # 将状态转换为张量并增加维度
            # send the state to the agent to get an action
            # 将状态发送给代理以获取动作
            action = rpc.rpc_sync(
                agent_rref.owner(),  # 获取代理的拥有者
                self.select_action,  # 调用选择动作的方法
                args=(agent_rref, self.id, state)  # 传递参数
            )

            # apply the action to the environment, and get the reward
            # 将动作应用于环境，并获取奖励
            state, reward, done, _ = self.env.step(action)  # 执行动作并获取下一个状态和奖励
            rewards[step] = reward  # 记录奖励

            if done or step + 1 >= n_steps:  # 如果episode结束或达到最大步数
                curr_rewards = rewards[start_step:(step + 1)]  # 获取当前奖励
                R = 0  # 初始化回报
                for i in range(curr_rewards.numel() - 1, -1, -1):  # 从后向前计算回报
                    R = curr_rewards[i] + args.gamma * R  # 计算折扣回报
                    curr_rewards[i] = R  # 更新当前奖励
                state = self.env.reset()  # 重置环境
                if start_step == 0:  # 如果是第一步
                    ep_reward = min(ep_reward, step - start_step + 1)  # 更新episode奖励
                start_step = step + 1  # 更新起始步数

        return [rewards, ep_reward]  # 返回奖励和episode奖励

class Agent:
    def __init__(self, world_size, batch=True):
        self.ob_rrefs = []  # 观察者的RRef列表
        self.agent_rref = RRef(self)  # 代理的RRef
        self.rewards = {}  # 奖励字典
        self.policy = Policy(batch).cuda()  # 创建策略并移动到GPU
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)  # 创建Adam优化器
        self.running_reward = 0  # 初始化运行奖励

        for ob_rank in range(1, world_size):  # 遍历所有观察者
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))  # 获取观察者信息
            self.ob_rrefs.append(remote(ob_info, Observer, args=(batch,)))  # 创建观察者的远程对象
            self.rewards[ob_info.id] = []  # 初始化奖励列表

        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)  # 初始化状态张量
        self.batch = batch  # 保存批处理标志
        # With batching, saved_log_probs contains a list of tensors, where each
        # tensor contains probs from all observers in one step.
        # Without batching, saved_log_probs is a dictionary where the key is the
        # observer id and the value is a list of probs for that observer.
        # 使用批处理时，saved_log_probs包含一个张量列表，其中每个张量包含来自所有观察者的概率。
        # 不使用批处理时，saved_log_probs是一个字典，键是观察者ID，值是该观察者的概率列表。
        self.saved_log_probs = [] if self.batch else {k: [] for k in range(len(self.ob_rrefs))}
        self.future_actions = torch.futures.Future()  # 创建未来动作的占位符
        self.lock = threading.Lock()  # 创建线程锁
        self.pending_states = len(self.ob_rrefs)  # 初始化待处理状态数

    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, ob_id, state):
        r"""
        Batching select_action: In each step, the agent waits for states from
        all observers, and process them together. This helps to reduce the
        number of CUDA kernels launched and hence speed up amortized inference
        speed.
        """
        # 批量选择动作：在每一步中，代理等待来自所有观察者的状态，并一起处理它们。
        # 这有助于减少启动的CUDA内核数量，从而加速摊销推理速度。

        self = agent_rref.local_value()  # 获取代理的本地值
        self.states[ob_id].copy_(state)  # 复制状态到对应的观察者
        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[ob_id].item()  # 获取未来动作
        )

        with self.lock:  # 使用锁确保线程安全
            self.pending_states -= 1  # 减少待处理状态数
            if self.pending_states == 0:  # 如果所有状态都已处理
                self.pending_states = len(self.ob_rrefs)  # 重置待处理状态数
                probs = self.policy(self.states.cuda())  # 计算策略
                m = Categorical(probs)  # 创建分类分布
                actions = m.sample()  # 采样动作
                self.saved_log_probs.append(m.log_prob(actions).t()[0])  # 记录动作的对数概率
                future_actions = self.future_actions  # 保存未来动作
                self.future_actions = torch.futures.Future()  # 创建新的未来动作占位符
                future_actions.set_result(actions.cpu())  # 设置未来动作的结果
        return future_action  # 返回未来动作

    @staticmethod
    def select_action(agent_rref, ob_id, state):
        r"""
        Non-batching select_action, return the action right away.
        """
        # 非批量选择动作，立即返回动作。

        self = agent_rref.local_value()  # 获取代理的本地值
        probs = self.policy(state.cuda())  # 计算策略
        m = Categorical(probs)  # 创建分类分布
        action = m.sample()  # 采样动作
        self.saved_log_probs[ob_id].append(m.log_prob(action))  # 记录动作的对数概率
        return action.item()  # 返回动作

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each observer to run one episode
        with n_steps. Then it collects all actions and rewards, and use those to
        train the policy.
        """
        # 运行一个episode。代理将告诉每个观察者运行一个包含n_steps的episode。
        # 然后它收集所有动作和奖励，并用这些来训练策略。

        futs = []  # 初始化未来对象列表
        for ob_rref in self.ob_rrefs:  # 遍历所有观察者的RRef
            # make async RPC to kick off an episode on all observers
            # 异步RPC以启动所有观察者的episode
            futs.append(ob_rref.rpc_async().run_episode(self.agent_rref, n_steps))  # 启动观察者的episode

        # wait until all observers have finished this episode
        # 等待所有观察者完成这个episode
        rets = torch.futures.wait_all(futs)  # 等待所有未来对象完成
        rewards = torch.stack([ret[0] for ret in rets]).cuda().t()  # 收集所有奖励并转置
        ep_rewards = sum([ret[1] for ret in rets]) / len(rets)  # 计算平均episode奖励

        if self.batch:  # 如果使用批处理
            probs = torch.stack(self.saved_log_probs)  # 堆叠所有对数概率
        else:  # 如果不使用批处理
            probs = [torch.stack(self.saved_log_probs[i]) for i in range(len(rets))]  # 堆叠每个观察者的对数概率
            probs = torch.stack(probs)  # 堆叠所有观察者的对数概率

        policy_loss = -probs * rewards / len(rets)  # 计算策略损失
        policy_loss.sum().backward()  # 反向传播
        self.optimizer.step()  # 更新优化器
        self.optimizer.zero_grad()  # 清零梯度

        # reset variables
        # 重置变量
        self.saved_log_probs = [] if self.batch else {k: [] for k in range(len(self.ob_rrefs))}  # 重置对数概率
        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)  # 重置状态张量

        # calculate running rewards
        # 计算运行奖励
        self.running_reward = 0.5 * ep_rewards + 0.5 * self.running_reward  # 更新运行奖励
        return ep_rewards, self.running_reward  # 返回episode奖励和运行奖励

def run_worker(rank, world_size, n_episode, batch, print_log=True):
    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """
    # 这是所有进程的入口点。rank 0是代理，所有其他rank是观察者。

    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '29500'  # 设置主节点端口
    if rank == 0:  # 如果是代理
        # rank0 is the agent
        # rank0是代理
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)  # 初始化RPC

        agent = Agent(world_size, batch)  # 创建代理实例
        for i_episode in range(n_episode):  # 遍历每个episode
            last_reward, running_reward = agent.run_episode(n_steps=NUM_STEPS)  # 运行episode

            if print_log:  # 如果需要打印日志
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, last_reward, running_reward))  # 打印episode信息
    else:  # 如果是观察者
        # other ranks are the observer
        # 其他rank是观察者
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)  # 初始化观察者的RPC
        # observers passively waiting for instructions from agents
        # 观察者被动等待代理的指令
    rpc.shutdown()  # 关闭RPC

def main():
    for world_size in range(2, 12):  # 遍历世界规模
        delays = []  # 初始化延迟列表
        for batch in [True, False]:  # 遍历批处理标志
            tik = time.time()  # 记录开始时间
            mp.spawn(  # 启动多个进程
                run_worker,  # 目标函数
                args=(world_size, args.num_episode, batch),  # 传递参数
                nprocs=world_size,  # 进程数量
                join=True  # 等待所有进程完成
            )
            tok = time.time()  # 记录结束时间
            delays.append(tok - tik)  # 计算延迟并添加到列表

        print(f"{world_size}, {delays[0]}, {delays[1]}")  # 打印世界规模和延迟

if __name__ == '__main__':
    main()  # 运行主函数