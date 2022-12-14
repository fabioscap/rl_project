from dm_control import suite
from dm_control.rl.control import Environment
from dm_control.suite.wrappers import pixels
from utils import UniformReplayBuffer, FrameStack
import time
import numpy as np

# train parameters
# TODO add command line arguments
n_train_steps = 1000000
eval_frequency = 10
update_frequency = 10
log_interval = 10
save_model = True
save_buffer = True
init_steps = 1000
n_frames = 4

img_shape = (240,320)
env: Environment = pixels.Wrapper(suite.load(domain_name="cartpole", task_name="swingup"),
                                             render_kwargs={"camera_id": 0, # front, fixed
                                                            "height": img_shape[0],
                                                            "width":  img_shape[1]})
action_spec = env.action_spec()


agent = ...
frame_stack = FrameStack(4)

replay_buffer = UniformReplayBuffer(s_shape=..., a_shape=...)

episode = 0
episode_reward = 0
done = True
for step in range(n_train_steps):
    
    if done:
        # sart a new episode
        time_step = env.reset()
        observation = time_step.observation['pixels']
        state = frame_stack.reset(observation)
        episode_reward = 0

    if step < init_steps:
        # at the start choose random actions
        action = np.random.uniform(action_spec.minimum,
                            action_spec.maximum,
                            size=action_spec.shape)
    else: # follow the policy
        # no grad
        action = agent.sample_action(state)
    
    time_step = env.step(action)

    next_observation = time_step.observation['pixels']

    reward = time_step.reward
    episode_reward += reward
    
    done = time_step.last()

    next_state = frame_stack.append_frame(next_observation)

    replay_buffer.store(state,action,reward,done,next_state)

    # remember to override state
    state = next_state
    
    # if update...
    if step > 0 and step % update_frequency:
        agent.update(replay_buffer, step)

    # if log...    