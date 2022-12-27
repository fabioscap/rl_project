import numpy as np
import torch
import os
import time
from dm_control import suite
import dmc2gym
import utils
from logger import Logger
from video import VideoRecorder

from agent.agent import Agent



seed = 1815675
domain_name = 'ball_in_cup'
task_name = 'catch'
image_size = 100
image_cropped_size = 84
frame_stack = 3
frame_skip = 2
work_dir = '/content/drive/MyDrive/RL/Project_CURL1'
save_video = True

replay_buffer_capacity = 100000
batch_size = 512

s_dim = 50
a_dim = 2

num_train_steps = 1000000
max_episode_steps = 1000

init_steps = 1000

save_model = True
save_buffer = False

num_eval_episodes = 5
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
        frame_skip=frame_skip
    )

    observation_shape = env.observation_space.shape
    observation_cropped_shape = (observation_shape[0],) + (image_cropped_size, image_cropped_size)

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

    agent = Agent(
        obs_cropped_shape=observation_cropped_shape,
        a_shape=action_shape,
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
        if step < init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                lc, la, lcont = agent.update(replay_buffer, step, L)
                L.log("train/critic_loss", lc, step)
                L.log("train/actor_loss", la, step)
                L.log("train/ae_loss", lcont, step)

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