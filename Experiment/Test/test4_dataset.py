import gym
# import d4rl # Import required to register environments

# Create the environment
env = gym.make('maze2d-umaze-v1')

# d4rl abides by the OpenAI gym interface
env.reset()

for i in range(500):
    obs = env.step(env.action_space.sample())
    print(obs.shape)

