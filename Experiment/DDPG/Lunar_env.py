import gym
import numpy as np
import argparse
from DDPG import DDPG
from collections import deque
import utils


def scale_action(action, low, high):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2

    return action * weight + bias


def main():
    game = 'LunarLanderContinuous-v2'
    env = gym.make(game)
    agent = DDPG(alpha=0.0003,
                 beta=0.0003,
                 state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0],
                 actor_fc1_dim=400,
                 actor_fc2_dim=300,
                 critic_fc1_dim=400,
                 critic_fc2_dim=300,
                 checkpoint_dir=game, )
    # 100局游戏
    max_episodes = 1000
    # 每局游戏最多走多少步，以免不动
    max_t = 1000
    scores = []
    scores_window = deque(maxlen=100)

    for episode in range(max_episodes):
        done = False
        score = 0
        observation = env.reset()

        for t in range(max_t):
            action = agent.choose_action(observation, train=True)
            # 对action进行缩放
            action_ = scale_action(action.copy(), env.action_space.low, env.action_space.high)
            next_observation, reward, done, info = env.step(action_)
            agent.store_transition(observation, action, reward, next_observation, done)
            agent.learn()
            score += reward
            observation = next_observation

            if done:
                break
        scores_window.append(score)
        scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}\tactor_loss {:.2f}\tcritic_loss {:.2f}'.format(
            episode + 1, np.mean(scores_window), agent.actor_loss, agent.critic_loss
        ), end='')
        if (episode + 1) % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tactor_loss{:.2f}\tcritic_loss {:.2f}'.format(
                episode + 1, np.mean(scores_window), agent.actor_loss, agent.critic_loss))

        if (episode + 1) % 100 == 0:
            agent.save_models(episode + 1)


def show_result(episode):
    game = 'LunarLanderContinuous-v2'
    env = gym.make(game)
    agent = DDPG(alpha=0.0003,
                 beta=0.0003,
                 state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0],
                 actor_fc1_dim=400,
                 actor_fc2_dim=300,
                 critic_fc1_dim=400,
                 critic_fc2_dim=300,
                 checkpoint_dir=game, )
    agent.load_models(episode)

    for i in range(10):
        done = False
        observation = env.reset()
        while not done:
            env.render()
            action = agent.choose_action(observation, train=True)
            # 对action进行缩放
            action_ = scale_action(action.copy(), env.action_space.low, env.action_space.high)
            next_observation, reward, done, info = env.step(action_)
            agent.store_transition(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation



if __name__ == '__main__':
    # main()
    show_result(500)
