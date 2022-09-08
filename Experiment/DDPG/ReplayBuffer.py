import numpy as np
import random
from collections import namedtuple, deque


class ReplayBuffer():
    def __init__(self, max_size, batch_size):
        """
        :param max_size:经验池的规模
        :param batch_size: 需要采样用于训练的批次大小
        """

        self.mem_size = max_size
        self.batch_size = batch_size
        self.information = namedtuple('Memory', field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = deque(maxlen=max_size)

    # 添加经验
    def add(self, state, action, reward, next_state, done):
        i = self.information(state, action, reward, next_state, done)
        self.memory.append(i)

    # 随机采样
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    E_Pool = ReplayBuffer(max_size=10, batch_size=3)

    state = np.array([[1., 2, 3, 4], ])
    next_state = np.array([[5, 6, 7, 8], ])
    action = np.array([[9, 10], ])
    reward = 0
    done = False

    for i in range(5):
        E_Pool.add(state, action, reward, next_state, done)

    # print(E_Pool.memory[1].state.shape)
    # print(np.array([e.state for e in E_Pool.memory]).shape)

    print(E_Pool.sample())
