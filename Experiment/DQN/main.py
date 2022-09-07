import gym
import random
import torch
import numpy as np
from collections import deque
from agent import Agent
import matplotlib.pyplot as plt

EPISODES = 1000
INTERVAL_LEARN = 1  #每N步学习一次
INTERVAL_REPLACE = 1000 #每N步更新一次target_net
#设置随机种子
def setup_seed(env,seed):
    env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic =True

def dqn(
        env,
        agent,
        interval_learn=INTERVAL_LEARN,
        interval_replace=INTERVAL_REPLACE,
        n_episode=EPISODES,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995 ):
    '''
    :param n_episode:  玩多少局游戏
    :param max_t: 每局游戏最多可以走多少步（执行多少次动作）  
    :param eps_start: eps用来控制随机游走的概率，即越大说明其采取的动作越随机，利于发现新的路径
    :param eps_end: 
    # :param eps_decay: eps每次训练后对其进行修改即eps*eps_decay
    :return: 
    '''
    #记录每局的分数用来画图
    scores = []
    #记录最近的100局分数来求平均值判断更具有稳定性
    scores_window = deque(maxlen=100)
    #step用来更新间隔
    step = 0

    eps = eps_start

    for i_episode in range(1,n_episode+1):
        state = env.reset()
        score = 0

        for t in range(max_t):
            action = agent.choose_action(state,eps)
            next_state, reward, done,_ = env.step(action)
            #存储每一步的信息
            agent.store_information(state,action,reward,next_state,done)
            #每m步学习一次
            if step%interval_learn==0 and len(agent.exp_pool)>100:
                agent.learn()
            #每n步更新一次参数，使用软更新时注释下列判断语句即可
            if step%interval_replace==0:
                agent.copy_params()

            state = next_state
            score += reward
            step += 1

            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end,eps*eps_decay)

        print('\rEpisode {}\tAverage Score: {:.2f}\tEps {:.2f}\tloss {:.2f}'.format(i_episode, np.mean(scores_window),eps,agent.loss), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0 or i_episode >= n_episode:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.target_net.state_dict(), 'checkpoint.pth')
            break
    return scores

def draw_plot(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)),scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # #设置随机种子
    setup_seed(env,seed=0)
    #
    agent = Agent(n_states=env.observation_space.shape[0], n_actions=env.action_space.n)
    #
    scores = dqn(env,agent)
    draw_plot(scores)

    # load the weights from file
    agent.eval_net.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(3):
        state = env.reset()
        for j in range(2000):
            action = agent.choose_action(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break

    env.close()