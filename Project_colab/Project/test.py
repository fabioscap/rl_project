import gym
from agent._sac import SacAgent
from utils import UniformReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("Pendulum-v1")
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

agent = SacAgent(s_dim, a_dim, "cpu")

replay_buffer = UniformReplayBuffer(10000, (s_dim,), (a_dim,))

from statistics import mean

max_frames  = 40000
max_steps   = 200
frame_idx   = 0
episode = 0
rewards     = []
losses = []
batch_size  = 128
desired_rew = -100
window = 50

while frame_idx < max_frames:
    
    state, _ = env.reset()
    episode_reward = 0
    losses_ep = []
    
    for step in range(max_steps):
        action = agent.sample_action(state)
        next_state, reward, done, _ , _= env.step(action)
        
        replay_buffer.store(state, action, reward, done, next_state)
        if replay_buffer.size() > batch_size:
            loss = agent.update(replay_buffer, frame_idx, batch_size)
            losses_ep.append(loss)
        
        state = next_state
        episode_reward += reward
        frame_idx += 1
      
        if done:
            break
    episode += 1

    rewards.append(episode_reward)
    mean_rewards = mean(rewards[-window:])
    print("\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}".format(
                            episode, mean_rewards, episode_reward), end="")

plt.plot(rewards)
plt.show()