import numpy as np
import random
from collections import namedtuple, deque

from model import Net
import torch
import torch.nn.functional as F
import torch.optim as optim

CAPACITY = int(1e5)  #经验池的容量
LR = 5e-4       #学习率
GAMMA = 0.99    #折扣率
TAU = 0.001     #软更新折损率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory():
    def __init__(self,batch_size, capacity = CAPACITY):
        '''
        :param batch_size: 训练批次大小
        :param capacity: 经验池容量
        :param information: 经验池每条数据的组成
        :param memory：经验池
        '''
        self.batch_size = batch_size
        self.capacity = capacity
        self.information = namedtuple('Memory', field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = deque(maxlen=capacity)
        self.loss = 0.0

    def add(self, state, action, reward, next_state, done):
        i = self.information(state,action,reward,next_state,done)
        self.memory.append(i)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self,n_states,n_actions):
        '''

        :param n_states:状态空间（状态的组成）
        :param n_actions:动作空间K（动作的个数）
        '''
        self.n_states = n_states
        self.n_actions = n_actions

        self.eval_net = Net(n_states,n_actions).to(device)
        self.target_net = Net(n_states,n_actions).to(device)
        #只需要对eval_net进行优化，target_net更新只依赖于eval_net
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)

        self.length = 0
        self.exp_pool = Memory(batch_size=64, capacity=CAPACITY)

    def choose_action(self,state, eps = 0):
        # 先将state转换为tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #非训练过程，只取当前训练过的值
        self.eval_net.eval()
        with torch.no_grad():
            Qsa = self.eval_net(state)
        #结束取值，恢复训练过程
        self.eval_net.train()

        #随机游走过程
        if random.random() > eps:
            return np.argmax(Qsa.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_actions))

    def store_information(self,state,action,reward,next_state,done):
        self.exp_pool.add(state,action,reward,next_state,done)

    def learn(self, gamma = GAMMA):
        samples = self.exp_pool.sample()
        states,actions,rewards,next_states,dones = samples

        #论文中的公式计算
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma*Q_targets_next*(1-dones))

        Q_evaluations = self.eval_net(states).gather(1,actions)

        #对eval_net参数进行反向传播
        self.loss = F.mse_loss(Q_evaluations,Q_targets)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        #采用软更新的方式来更新target_net网络，要使用的话，取消下列注释，并注释掉main中的判断语句
        # self.soft_update(self.eval_net, self.target_net, TAU)

    def soft_update(self, local_model, target_model, tau = TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def copy_params(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())