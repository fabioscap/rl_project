from sac import SacAgent
from encoder import FeatureEncoder
import torch
import numpy as np
from utils import soft_update_params
class Agent():
    def __init__(self,
                 obs_shape: tuple,
                 a_shape: tuple,
                 n_frames: int, # how many frames to stack
                 s_dim: int, # state representation dimension
                 a_dim: int, # action representation dimension
                 device = "cpu"
                ):
        self.obs_shape = obs_shape
        self.n_frames = n_frames
        
        self.s_shape = (self.obs_shape[0]*n_frames, *self.obs_shape[1:])
        self.feature_encoder = FeatureEncoder(self.s_shape, a_shape, s_dim, a_dim)
        self.sac = SacAgent(s_dim = s_dim, 
                            a_dim = a_dim,
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
                            )


        # optimizers ...

        # move the SAC update methods here, we want to backprop also through the encoder

    def update_Q_networks(self, state, reward, action, done, new_state):
        state      = torch.FloatTensor(state)
        new_state  = torch.FloatTensor(new_state)
        action     = torch.FloatTensor(action)
        reward     = torch.FloatTensor(reward).unsqueeze(1)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        log_prob, new_action = SacAgent.return_log(new_state)

        Q_net1, Q_net2 = SacAgent.Qnet_forward(state, action)
        
        Q_targ1, Q_targ2 = SacAgent.Qtarg_forward(new_state, new_action)

        loss1, loss2 = self.update_Qnetworks(reward, done, Q_targ1, Q_targ2, Q_net1, Q_net2, log_prob)
        
        soft_update_params(self.Q_network1, self.Q_target1,self.critic_tau)
        soft_update_params(self.Q_network2, self.Q_target2,self.critic_tau)       

    def update_policy_network(self, state, Q1, Q2):
        state      = torch.FloatTensor(state)
        dist = self.actor(state)
        dist = dist.unsqueeze(-1)

        Q1, Q2 = self.Qnet_forward(state, dist)
        loss3 = self.update_policy(state, Q1, Q2)        

    def sample_action(self, state):
        raise NotImplementedError()

