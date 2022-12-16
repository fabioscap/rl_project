from sac import SAC
from encoder import FeatureEncoder
import torch
import numpy as np
from utils import soft_update_params, UniformReplayBuffer
class Agent():
    def __init__(self,
                 obs_shape: tuple,
                 a_shape: tuple,
                 n_frames: int, # how many frames to stack
                 s_dim: int, # state representation dimension
                 a_dim: int, # action representation dimension
                 encoder_lr = 1e-3,
                 encoder_betas = (0.9, 0.999),
                 device = "cpu"
                ):
        self.obs_shape = obs_shape
        self.n_frames = n_frames
        
        self.s_shape = (self.obs_shape[0]*n_frames, *self.obs_shape[1:])
        self.feature_encoder = FeatureEncoder(self.s_shape, a_shape, s_dim, a_dim, device=device)
        self.sac = SAC(s_dim = s_dim, 
                            a_dim = a_shape, # pass to SAC the actual action, not embedded
                            Q_hidden_dims=(256,),
                            policy_hidden_dims=(256,),
                            gamma = 0.99,
                            tau = 0.01, 
                            log_std_bounds=(-10,2),
                            alpha= 1e-2, 
                            epsilon = 1e-6,
                            critic_tau = 0.005,
                            init_temperature = 0.1,
                            learnable_temperature = True,
                            actor_lr = 1e-2,
                            Q1_lr = 1e-2,
                            Q2_lr = 1e-2, 
                            actor_betas = (0.9, 0.999),
                            critic_betas = (0.9, 0.999),
                            alpha_lr = 1e-4,
                            alpha_betas = (0.9, 0.999),
                            )

        # optimizer
        self.encoder_optimizer = torch.optim.Adam(params=self.feature_encoder.parameters(),
                                                  lr=encoder_lr, betas=encoder_betas)

    def update(self, replay_buffer: UniformReplayBuffer, batch_size: int, step: int):
        # do sample
        state, action, reward, done, new_state, *_ = replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state)
        new_state  = torch.FloatTensor(new_state)
        action     = torch.FloatTensor(action)
        re         = torch.FloatTensor(reward).unsqueeze(1)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        q, ri, contrastive_loss = self.feature_encoder.encode_reward_loss(state,action,new_state, step)
        reward = re + ri
        qp = self.feature_encoder.encode(new_state, target=False, grad=False)

        self.sac.update_SAC(q, reward, action, qp, done)
        # the encoder will also receive gradients due to the backward passes
        # in update_SAC

        self.encoder_optimizer.zero_grad()
        contrastive_loss.backward()
        self.encoder_optimizer.step()

        # update the targets
        self.feature_encoder.update_key_network()

    def sample_action(self, state):
        raise NotImplementedError()

