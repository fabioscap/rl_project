from pickle import FALSE
import numpy as np
import torch
import os
import time
from dm_control import suite
import dmc2gym
import utils
from logger import Logger
from video import VideoRecorder

from agent.sac import SAC

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='reacher')
    parser.add_argument('--task_name', default='hard')

    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--frame_skip', default=3, type=int)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=1, type=int)
    # misc
    parser.add_argument('--seed', default=1834913, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)

    args = parser.parse_args()
    return args

args = parse_args()

seed = args.seed
domain_name = args.domain_name
task_name = args.task_name
frame_stack = args.frame_stack
frame_skip = args.frame_skip
work_dir = args.work_dir
save_video = args.save_video

replay_buffer_capacity = args.replay_buffer_capacity
batch_size = args.batch_size


num_train_steps = args.num_train_steps
max_episode_steps = 1000

init_steps = args.init_steps

save_model = args.save_model
save_buffer = args.save_buffer

num_eval_episodes = args.num_eval_episodes
eval_frequency = args.eval_freq

def evaluate(env, agent, video, num_episodes, L, step):
    for i in range(num_episodes):
        obs = env.reset()

        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done: 
            with utils.eval_mode(agent):
                action = agent.select_action(obs)

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
        from_pixels=False,
        frame_skip=frame_skip
    )

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape 

    utils.make_dir(work_dir)
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(work_dir, 'buffer'))

    video = VideoRecorder(video_dir if save_video else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    replay_buffer = utils.ReplayBuffer(
        obs_shape=observation_shape,
        action_shape=action_shape,
        capacity=replay_buffer_capacity,
        batch_size=batch_size,
        device=device
    )

    agent = SAC(s_dim = observation_shape[0], 
                a_dim = action_shape, # pass to SAC the actual action, not embedded
                Q_hidden_dims=(512,),
                policy_hidden_dims=(512,),
                gamma = 0.99,
                tau = 0.01, 
                log_std_bounds=(-10,2),
                alpha= 1e-2, 
                epsilon = 1e-6,
                critic_tau = 0.005,
                init_temperature = 0.1,
                learnable_temperature = True,
                actor_lr = 1e-3,
                Q1_lr = 1e-3,
                Q2_lr = 1e-3, 
                actor_betas = (0.9, 0.999),
                critic_betas = (0.9, 0.999),
                alpha_lr = 1e-4,
                alpha_betas = (0.9, 0.999),
                device=device
                ).to(device)

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
            if step > 0 and step % eval_frequency == 0:
                L.log('eval/episode', episode, step)
                evaluate(env, agent, video, num_eval_episodes, L, step)
                if save_model:
                    pass # agent.save(model_dir, step)
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
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                lc1, lc2, la = agent.update(replay_buffer, step)
                L.log("train/critic_loss", lc1+lc2, step)
                L.log("train/actor_loss", la, step)

        next_obs, reward, done, _ = env.step(action)
        done = episode_step + 1 == max_episode_steps

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


      
if __name__ == '__main__':
    main()
