import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 存在Actor网络与Critic网络，每个网络均有两个网络（eval和target）

def weight_init(m):
    if isinstance(m, nn.Linear):
        # 初始化该层weight，保证均值方差基本一致
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            # 初始化bias为0
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        # 产生一个动作
        self.action = nn.Linear(fc2_dim, action_dim)

        # Optimize
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # 初始化所有参数
        self.apply(weight_init)
        # 交给对应Gpu或cpu进行计算
        self.to(device)

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        # 激活函数
        action = torch.tanh(self.action(x))

        return action

    def save_checkpoint(self, file):
        torch.save(self.state_dict(), file)

    def load_checkpoint(self, file):
        self.load_state_dict(torch.load(file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)

        self.fc3 = nn.Linear(action_dim, fc2_dim)

        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.001)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state, action):
        x_s = torch.relu(self.ln1(self.fc1(state)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(action)

        x = torch.relu(x_s + x_a)
        q = self.q(x)

        return q

    def save_checkpoint(self, file):
        torch.save(self.state_dict(), file)

    def load_checkpoint(self, file):
        self.load_state_dict(torch.load(file))
