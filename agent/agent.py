from .sac import SAC
from .encoder import FeatureEncoder
import torch
import numpy as np
class Agent():
    def __init__(self,
                 obs_cropped_shape: tuple,
                 a_shape: tuple,
                 s_dim: int, # state representation dimension
                 a_dim: int, # action representation dimension
                 encoder_lr = 1e-3,
                 encoder_betas = (0.9, 0.999),
                 device = "cpu"
                ):
        self.s_shape = obs_cropped_shape
        
        self.feature_encoder = FeatureEncoder(self.s_shape, a_shape, s_dim, a_dim, device=device)
        self.sac = SAC(s_dim = s_dim, 
                            a_dim = a_shape, # pass to SAC the actual action, not embedded
                            Q_hidden_dims=(1024,),
                            policy_hidden_dims=(1024,),
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
                            alpha_betas = (0.5, 0.999),
                            device=device
                            ).to(device)
        self.device = device
        self.training = True
        # optimizer
        self.encoder_optimizer = torch.optim.Adam(params=self.feature_encoder.parameters(),
                                                  lr=encoder_lr, betas=encoder_betas)

        self.max_extrinsic = 1

    def update(self, replay_buffer, step: int, L):
        # do sample
        state, action, re, new_state, dones, *_ = replay_buffer.sample()
        state/= 255
        new_state/=255
        
        state = state.to(self.device)
        action = action.to(self.device)
        re = re.to(self.device)
        new_state = new_state.to(self.device)
        done = dones.to(self.device)

        max_extrinsic = max(re)
        if max_extrinsic > self.max_extrinsic:
            self.max_extrinsic = max_extrinsic
            
        q, ri, contrastive_loss = self.feature_encoder.encode_reward_loss(state,action,new_state, step, self.max_extrinsic)

        reward = re + ri
        
        qp = self.feature_encoder.encode(new_state, target=False, grad=False)
        lc1, lc2, la = self.sac.update_SAC(q.detach(), reward, action, qp, done)
        
        self.encoder_optimizer.zero_grad()
        contrastive_loss.backward()
        self.encoder_optimizer.step()

        # update the targets
        self.feature_encoder.update_key_network()

        return lc1+lc2, la, contrastive_loss

    def sample_action(self, obs):
        obs = torch.from_numpy(obs).to(self.device) / 255.0
        q = self.feature_encoder.encode(obs.unsqueeze(0),grad=False, center_crop=True)
        return self.sac.sample_action(q)

    def select_action(self,obs):
        obs = torch.from_numpy(obs).to(self.device) / 255.0
        q = self.feature_encoder.encode(obs.unsqueeze(0),grad=False, center_crop=True)
        return self.sac.select_action(q)


    def train(self, training=True):
        self.sac.train()

    def save(self, model_dir, step):
        torch.save(
            self.feature_encoder.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.sac.state_dict(), '%s/sac_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.feature_encoder.load_state_dict(
            torch.load('%s/encoder_%s.pt' % (model_dir, step))
        )
        self.sac.load_state_dict(
            torch.load('%s/sac_%s.pt' % (model_dir, step))
        )