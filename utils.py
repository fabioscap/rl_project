import numpy as np
import random
import math
import torch.nn as nn
import torch 
from collections import deque

class UniformReplayBuffer():

    def __init__(self, capacity: int,   # the capacity of the buffer
                       s_shape:  tuple, # the shape of a state
                       a_shape:  tuple):   # the shape of an action
        
        self.s_shape = s_shape
        self.a_shape = a_shape

        self.capacity = capacity
        self.idx = -1

        self.states =      np.empty(shape=(capacity,*s_shape), dtype=np.float32)
        self.actions =     np.empty(shape=(capacity,*a_shape), dtype=np.float32)
        self.rewards =     np.empty(shape=capacity,            dtype=np.float32)
        self.dones =       np.empty(shape=capacity,            dtype=np.bool8)
        self.next_states = np.empty(shape=(capacity,*s_shape), dtype=np.float32)

        self.full = False # wether the buffer is full or it contains empty spots

    def size(self):
        if self.full:
            return self.capacity
        else:
            return self.idx+1

    def store(self, s: np.ndarray, 
                    a: np.ndarray, # or int if discrete actions 
                    r: float, 
                    d: bool, 
                    sp: np.ndarray) -> int:
        
        self.idx += 1
        if (self.idx == self.capacity):
            self.full = True # the buffer is full
            self.idx = 0     # reset the index (start to overwrite old experiences)

        np.copyto(self.states[self.idx], s)
        np.copyto(self.actions[self.idx], a)
        np.copyto(self.rewards[self.idx], r)
        np.copyto(self.dones[self.idx],   d)
        np.copyto(self.next_states[self.idx], sp)

        return self.idx

    def sample_idxes_weights(self, n):
        high = self.size()
        return random.choices(population=range(high), k=n), None     

    def sample(self, n: int):
        # random.sample performs sampling without replacement
        idxes, w = self.sample_idxes_weights(n)

        states =     self.states[idxes]
        actions =    self.actions[idxes]
        rewards =    self.rewards[idxes]
        dones =      self.dones[idxes]
        next_states = self.next_states[idxes]

        return (states,actions,rewards,dones,next_states,idxes,w)

    def save(self, path):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

class PrioritizedReplayBuffer(UniformReplayBuffer):

    def __init__(self, capacity: int, s_shape: tuple, a_shape=(1, ), alpha=0.6, beta_0=0.4, beta_inc=1.001):
        super().__init__(capacity, s_shape, a_shape)
        if math.ceil(math.log2(capacity)) != math.floor(math.log2(capacity)):
            capacity = 2**math.ceil(math.log2(capacity))
            print(f"rescaling buffer to the next power of two: {capacity}.")
        
        # store the priorities in a tree
        self.priorities = SumTree(capacity)
        self.max_priority = 1.0

        self.alpha = alpha
        self.beta = beta_0
        self.beta_inc = beta_inc

    def sample_idxes_weights(self, n):
        high = self.size()

        (idxes, Ps) = self.priorities.sample_batch(n)

        w = (high*Ps)**-self.beta

        w /= w.max()
        if self.beta < 1: # beta annealing
            self.beta*= self.beta_inc 

        return idxes, w

    def store(self, s: np.ndarray, 
                    a: np.ndarray, # or int if discrete actions 
                    r: float, 
                    d: bool, 
                    sp: np.ndarray):
        super().store(s,a,r,d,sp)
        self.priorities.set_priority(self.idx,self.max_priority)

    def update_priorities(self, idxes, td_errors, eps=1e-6):
        updated_priorities = np.abs(td_errors)**self.alpha + eps

        _m = updated_priorities.max()
        if _m > self.max_priority: # update the maximum priority
            self.max_priority = _m

        for i in range(len(idxes)):
            self.priorities.set_priority(idxes[i],updated_priorities[i])

class SumTree(): 

    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.size = 2*n_bins - 1
        self.data = np.zeros(self.size)
        self.height = math.log2(n_bins)

    def _left(self, i):
        return 2*i+1

    def _right(self, i):
        return 2*i+2

    def _parent(self, i):
        return (i-1) // 2

    def _update_cumulative(self, i):
        value_left = self.data[self._left(i)]
        value_right = self.data[self._right(i)]
        self.data[i] = value_left + value_right

        if i == 0: # the root of the tree
            return
        else: # update the parent
            self._update_cumulative(self._parent(i)) 

    def _is_leaf(self, i):
        # it is a leaf if it's stored in the last self.n_bins positions
        return i >= self.size - self.n_bins 

    def _importance_sampling(self, priority, i=0):
        # https://adventuresinmachinelearning.com/sumtree-introduction-python/
        if self._is_leaf(i):
            # return transition to which i corresponds
            return i - (self.size - self.n_bins), self.data[i] 
        else:
            value_left = self.data[self._left(i)]
            # value_right = self.data[self._right(i)]
            
            if priority < value_left:
                return self._importance_sampling(priority, self._left(i))
            else: # priority >= value_left
                return self._importance_sampling(priority-value_left, self._right(i))

    def get_sum(self):
        return self.data[0]        

    def set_priority(self, idx, priority):
        # where is the leaf stored on the array
        pos = self.size - self.n_bins + idx

        self.data[pos] = priority
        self._update_cumulative(self._parent(pos))

    def sample_batch(self, k):
        rng = self.get_sum() / k
        # low variance sampling like in particle filter
        unif = np.random.uniform() * rng
        
        idxes = np.zeros(k, dtype=np.uint32)
        Ps = np.zeros(k)

        for i in range(k):
            idxes[i], Ps[i]  = self._importance_sampling(unif)
            unif += rng
        return idxes, Ps

class FrameStack():

    # stack n observation together to build the new state
    def __init__(self, n_frames: int):
        self.n_frames = n_frames

        self.frame_stack = deque(maxlen=n_frames)

    def append_frame(self, obs: np.ndarray):
        self.frame_stack.append(obs)

    def reset(self, obs: np.ndarray):
        self.frame_stack = deque(self.n_frames * [obs], maxlen=self.n_frames)

    def get_state(self)-> np.ndarray:
        frames = np.array(self.frame_stack,dtype=np.float32)

        # stack along channels
        state = frames.reshape(-1, *frames.shape[2:]) # shape (N*C,W,H)

        return state 

def make_MLP(in_dim: int, 
             out_dim: int, 
             hidden_dims: tuple, 
             hidden_act = nn.ReLU, 
             out_act = None) -> nn.Module:
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

    # compute the similarities
    sims = similarity(queries,keys) # (b,b)
    # the diagonal elements can be interpreted as positive keys
    # the off diagonal as negative keys

    # compute the fake labels
    labels = torch.arange(sims.shape[0], dtype=torch.long)

    return nn.functional.cross_entropy(sims,labels)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )
def copy_params(copy_from: nn.Module, copy_to: nn.Module):
    copy_to.load_state_dict(copy_from.state_dict())

# TODO 
# pre processing observations (as they are generated)
# - gray scale (?)
# - [0,255] -> [0,1]
# - np -> torch (?)

# center crop observations (as they are sampled)
# - why do they center crop? (is it related to contrastive learning?)
