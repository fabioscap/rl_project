import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy

import utils
from logger import Logger
from video import VideoRecorder

from agent.agent import Agent


seed = 1
domain_name = "ball_in_cup"
task_name = "catch"
image_size = 84
action_repeat = 1
frame_stack = 3
work_dir = '.'
save_video = True

replay_buffer_capacity = 100000
batch_size = 32

s_dim = 128
a_dim = 50

num_train_steps = 1000000

init_steps = 1000

save_model = True
save_buffer = True

num_eval_episodes = 10
eval_frequency = 1000

def evaluate(env, agent, video, num_episodes, L, step):
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done: 
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
                action = action.astype(np.float32)
            
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)

def main():
    utils.set_seed_everywhere(seed)

    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat
    )
    env.seed(seed)

    # stack several consecutive frames together

    env = utils.FrameStack(env, k=frame_stack)

    utils.make_dir(work_dir)
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(work_dir, 'buffer'))

    video = VideoRecorder(video_dir if save_video else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=replay_buffer_capacity,
        batch_size=batch_size,
        device=device
    )

    agent = Agent(
        obs_shape=env.observation_space.shape,
        a_shape=env.action_space.shape,
        s_dim = s_dim,
        a_dim = a_dim,
        device=device
    )

    L = Logger(work_dir, use_tb=False)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(num_train_steps):
        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if step % eval_frequency == 0:
                L.log('eval/episode', episode, step)
                evaluate(env, agent, video, num_eval_episodes, L, step)
                if save_model:
                    agent.save(model_dir, step)
                if save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent): 
                action = agent.sample_action(obs)

        # run training update
        if step >= init_steps:
            num_updates = init_steps if step == init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1

      

if __name__ == '__main__':
    main()
