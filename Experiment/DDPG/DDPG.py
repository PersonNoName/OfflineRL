import os
import torch
import torch.nn.functional as F
import numpy as np
from TargetNetwork import ActorNetwork, CriticNetwork
from ReplayBuffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:
    def __init__(self,
                 alpha,
                 beta,
                 state_dim,
                 action_dim,
                 actor_fc1_dim,
                 actor_fc2_dim,
                 critic_fc1_dim,
                 critic_fc2_dim,
                 checkpoint_dir,
                 gamma=0.99,
                 tau=0.005,
                 action_noise=0.1,
                 max_size=1000000,
                 batch_size=256):
        # 参数
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.batch_size = batch_size
        self.actor_loss = 0
        self.critic_loss = 0
        # checkpoint保存目录
        self.checkpoint_dir = './Checkpoint/' + checkpoint_dir
        # 四个网络
        self.actor = ActorNetwork(alpha=alpha,
                                  state_dim=state_dim,
                                  action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim,
                                  fc2_dim=actor_fc2_dim)
        self.target_actor = ActorNetwork(alpha=alpha,
                                         state_dim=state_dim,
                                         action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim,
                                         fc2_dim=actor_fc2_dim)

        self.critic = CriticNetwork(beta=beta,
                                    state_dim=state_dim,
                                    action_dim=action_dim,
                                    fc1_dim=critic_fc1_dim,
                                    fc2_dim=critic_fc2_dim)
        self.target_critic = CriticNetwork(beta=beta,
                                           state_dim=state_dim,
                                           action_dim=action_dim,
                                           fc1_dim=critic_fc1_dim,
                                           fc2_dim=critic_fc2_dim)

        # 经验池
        self.memory = ReplayBuffer(max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(),
                                                       self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def choose_action(self, observation, train=True):
        self.actor.eval()
        # 交给网络时要保证第一维为传入的个数
        state = torch.tensor([observation], dtype=torch.float).to(device)
        action = self.actor.forward(state).squeeze()

        # 测试size是否符合预期
        # print('choose_action-state：', state.shape)
        # print('choose-action-action: ', action.shape)

        # noice用来在训练时提供更多的未知性
        if train:
            # 从正态（高斯）分布中抽取随机样本, https://blog.csdn.net/wzy628810/article/details/103807829
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise), dtype=torch.float).to(device)
            # 限制tensor的每个元素的范围在-1,1之间
            action = torch.clamp(action + noise, -1, 1)
            # print('noise：', noise)
            # print('train：', action)
        self.actor.train()

        return action.detach().cpu().numpy()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        # 转换为tensor进行计算
        t_states = torch.from_numpy(states).float().to(device)
        t_actions = torch.from_numpy(actions).float().to(device)
        t_rewards = torch.from_numpy(rewards).float().to(device)
        t_next_states = torch.from_numpy(next_states).float().to(device)
        t_dones = torch.from_numpy(dones).to(device)

        # 测试sample
        # print('states: ', states.shape)
        # print('t_states: ', t_states)
        # print(t_dones.shape)
        # 测试是否在gpu上
        # print('t_state is in cuda: ', t_states.is_cuda)

        with torch.no_grad():
            # 测试一个batch是否能在网络中进行计算
            # print(t_next_states.shape)
            # self.target_actor(t_next_states)

            t_next_actions = self.target_actor.forward(t_next_states)
            next_q = self.target_critic.forward(t_next_states, t_next_actions)

            # 测试当done为Ture时给其赋予0值是否正常
            # next_q[t_dones] = 0
            # print(next_q)
            # print(next_q.shape)

            # 如果下一state为终止状态，其q值赋0
            next_q[t_dones] = 0.0
            # y为真值，其后需要将evalue的Q值与其算td-error
            y = t_rewards + self.gamma * next_q.view(-1)

        q = self.critic.forward(t_states, t_actions).view(-1)
        # 测试y与q的size是否匹配
        # print(t_rewards.shape)
        # print(next_q.shape)
        # print(y.shape)
        # print(q.shape)
        #
        # print(y)
        # print(t_rewards + self.gamma * next_q)

        # 更新Critic网络，而不是target_critic
        critic_loss = F.mse_loss(q, y.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # 更新Actor网络，而不是target_actor
        new_t_actions = self.actor.forward(t_states)
        actor_loss = -torch.mean(self.critic(t_states, new_t_actions))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # 获取actor和critic的loss
        self.actor_loss = actor_loss.data
        self.critic_loss = critic_loss.data

        self.update_network_parameters()

    def save_models(self, episode):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir + '/Actor')
            os.makedirs(self.checkpoint_dir + '/TargetActor')
            os.makedirs(self.checkpoint_dir + '/Critic')
            os.makedirs(self.checkpoint_dir + '/TargetCritic')
        # 保存actor网络
        self.actor.save_checkpoint(self.checkpoint_dir + '/Actor/actor_{}.pth'.format(episode))
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          '/TargetActor/target_actor_{}.pth'.format(episode))

        # 保存critic网络
        self.critic.save_checkpoint(self.checkpoint_dir + '/Critic/critic_{}.pth'.format(episode))
        self.target_critic.save_checkpoint(self.checkpoint_dir +
                                           '/TargetCritic/target_critic_{}.pth'.format(episode))

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + '/Actor/actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          '/TargetActor/target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')

        self.critic.load_checkpoint(self.checkpoint_dir + '/Critic/critic_{}.pth'.format(episode))
        print('Loading critic network successfully!')
        self.target_critic.load_checkpoint(self.checkpoint_dir +
                                           '/TargetCritic/target_critic_{}.pth'.format(episode))
        print('Loading target critic network successfully!')


if __name__ == '__main__':
    ddpg = DDPG(0.1, 0.1, 4, 2, 8, 8, 8, 8, 'test', batch_size=3)

    state = np.array([[1., 2, 3, 4], ])
    next_state = np.array([[5, 6, 7, 8], ])
    action = np.array([[9, 10], ])
    reward = 0.0
    done = False
    # 测试choose_action
    # data = ddpg.choose_action(state)
    # print(data)

    # 测试从buffer取出数据
    for i in range(10):
        ddpg.store_transition(state, action, reward, next_state, done)
    ddpg.learn()
    ddpg.save_models(3)
    ddpg.load_models(3)
    # print(ddpg.choose_action(state))
