import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.dones[start:end] = payload[4]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        s = env.reset()
        shp = s.observation['pixels'].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(shp),
            dtype=s.observation['pixels'].dtype
        )
        self._max_episode_steps = 1000

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs.observation['pixels'])
        return self._get_obs()

    def step(self, action):
        obs = self.env.step(action)
        
        s = obs.observation['pixels'] 
      
        self._frames.append(s)
  
        reward = obs.reward   
        done = obs.last()
        
        return self._get_obs(), reward, done, done

    def _get_obs(self):
        assert len(self._frames) == self._k

        return np.concatenate(list(self._frames), axis=0)

def make_MLP(in_dim: int, 
             out_dim: int, 
             hidden_dims: tuple, 
             hidden_act = nn.Tanh, 
             out_act = nn.Tanh) -> nn.Module:
    layers = []
    if len(hidden_dims) == 0:
        layers.append(nn.Linear(in_dim,out_dim))
        if out_act != None:
            layers.append(out_act())
        return nn.Sequential(*layers)
    else:
        d_in = in_dim
        for d_out in hidden_dims:
            layers.append(nn.Linear(d_in,d_out))
            layers.append(hidden_act())
            d_in = d_out
        layers.append(nn.Linear(d_out,out_dim))
        if out_act != None:     
            layers.append(out_act())
        return nn.Sequential(*layers)

# implement the contrastive loss used in CURL, CCFDM
def infoNCE(queries: torch.Tensor, keys: torch.Tensor, similarity):
    # the positive key is at the same index as the query
    # the other indexes are the negative keys
    
    # q: (b,n)
    # k: (b,n)
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    # compute the similarities
    sims = similarity(queries,keys) # (b,b)
    # the diagonal elements can be interpreted as positive keys
    # the off diagonal as negative keys

    # compute the fake labels
    labels = torch.arange(sims.shape[0], dtype=torch.long).to(device)

    return nn.functional.cross_entropy(sims,labels)

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

def copy_params(copy_from: nn.Module, copy_to: nn.Module):
    copy_to.load_state_dict(copy_from.state_dict())