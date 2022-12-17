from .sac import SAC
from .encoder import FeatureEncoder
import torch
import numpy as np
class Agent():
    def __init__(self,
                 obs_shape: tuple,
                 a_shape: tuple,
                 s_dim: int, # state representation dimension
                 a_dim: int, # action representation dimension
                 encoder_lr = 1e-3,
                 encoder_betas = (0.9, 0.999),
                 device = "cpu"
                ):
        self.s_shape = obs_shape
        
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
                            ).to(device)
        self.device = device
        self.training = True
        # optimizer
        self.encoder_optimizer = torch.optim.Adam(params=self.feature_encoder.parameters(),
                                                  lr=encoder_lr, betas=encoder_betas)
    
    def update(self, replay_buffer, step: int):
        # do sample
        state, action, reward, new_state, dones, *_ = replay_buffer.sample()
        state/= 255
        
        state      = torch.FloatTensor(state)
        new_state  = torch.FloatTensor(new_state)
        action     = torch.FloatTensor(action)
        re         = torch.FloatTensor(reward)
        done       = torch.FloatTensor(np.float32(dones))
        
        max_reward = max(re)
        
        q, ri, contrastive_loss = self.feature_encoder.encode_reward_loss(state,action,new_state, step, max_reward)
        ri = ri.unsqueeze(1)
        
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

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            q = self.feature_encoder.encode(obs)
            pi = self.sac.actor(q)
            return pi.cpu().data.numpy().flatten()

    def select_action(self,obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            q = self.feature_encoder.encode(obs)
            #pi = self.sac.actor(q)
            mu, _ = self.sac.policy_forward(q)
            return mu.cpu().data.numpy().flatten()


    def train(self, training=True):
        self.training = training
        self.sac.policy_network.train(training)
        self.sac.Q_network1.train(training)
        self.sac.Q_network2.train(training)
        if self.feature_encoder is not None:
            self.feature_encoder.train(training)

    def save(self, model_dir, step):
        torch.save(
            self.feature_encoder.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.sac.state_dict(), '%s/sac_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/encoder_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/sac_%s.pt' % (model_dir, step))
        )
