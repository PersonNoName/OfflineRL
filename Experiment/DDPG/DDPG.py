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
        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        # checkpoint保存目录
        self.checkpoint_dir = checkpoint_dir
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
            target_actor_params.data.copy_(tau * actor_params + (1-tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(),
                                                       self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1-tau) * target_critic_params)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def choose_action(self, observation, train=True):
        self.actor.eval()
        # 交给网络时要保证增加一维，第一维为传入的个数
        state = torch.tensor([observation], dtype=torch.float).to(device)
        action = self.actor.forward(state).squeeze()

        print('choose_action-state：', state.shape)
        print('choose-action-action: ', action.shape)

        # noice用来在训练时提供更多的未知性
        if train:
            # 从正态（高斯）分布中抽取随机样本, https://blog.csdn.net/wzy628810/article/details/103807829
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise), dtype=torch.float).to(device)
            action = torch.clamp(action+noise, -1,1)
            print('train：', action)
        self.actor.train()

        return action.detach().cpu().numpy()


if __name__ == '__main__':
    ddpg = DDPG(0.1,0.1,4,2,8,8,8,8,None)
    observation = np.array([[1.,2,3,4],])
    data = ddpg.choose_action(observation)
    print(data)