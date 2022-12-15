import gym 

import torch

env = NormalizedActions(gym.make("Pendulum-v1"))

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]

agent = SAC(s_dim = state_dim, 
            a_dim = action_dim,
            Q_hidden_dims=(256,),
            policy_hidden_dims=(256,),
            gamma = 0.99,
            tau = 0.01, 
            log_std_bounds=(-10,2),
            alpha= 1e-2, 
            actor_lr = 1e-2,
            Q1_lr = 1e-2,
            Q2_lr = 1e-2, 
            epsilon = 1e-6,
            actor_betas = (0.9, 0.999),
            critic_betas = (0.9, 0.999),
            alpha_lr = 1e-4,
            init_temperature = 0.1,
            learnable_temperature = True,
            alpha_betas = (0.9, 0.999),
            critic_tau = 0.005,
            train_alpha = True
            )


from statistics import mean
import random
import math
import numpy as np

import matplotlib.pyplot as plt

def plot(frame_idx, rewards):

    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)

    plt.show()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

max_frames  = 80000
max_steps   = 500
frame_idx   = 0
episode = 0
rewards     = []
losses = []
batch_size  = 128
desired_rew = -100
window = 50
replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)
        
from statistics import mean

while frame_idx < max_frames:
    state, _ = env.reset()
    episode_reward = 0
    losses_ep = []
    for step in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, done, _ , _= env.step(action)
        real_done = done
        replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward
        if len(replay_buffer) > batch_size:
            state1, action1, reward1, next_state1, done1 = replay_buffer.sample(batch_size)

            loss = agent.update_SAC(state1, reward1, action1, next_state1, done1)
            losses_ep.append(loss)
            if not agent.learnable_temperature: # alpha not learnable
              agent.alpha_decay()
        state = next_state
        
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            plot(frame_idx, rewards)
        
        if real_done:
            break
    episode += 1

    rewards.append(episode_reward)
    losses.append(mean(losses_ep))
    mean_rewards = mean(rewards[-window:])
    mean_loss = mean(losses[-window:])
    print("\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}   mean loss = {:.2f}\t\t".format(
                            episode, mean_rewards, episode_reward, mean_loss), end="")
    if mean_rewards >= desired_rew:
        break