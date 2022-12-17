import torch.nn as nn
import torch
from utils import make_MLP, copy_params, soft_update_params
from torch.distributions import Normal
import numpy as np
# soft actor critic for vector states
class SAC(nn.Module):
    def __init__(self,
                s_dim,
                a_dim,
                Q_hidden_dims: tuple,
                policy_hidden_dims: tuple,
                gamma: float,
                tau: float,
                log_std_bounds: tuple,
                alpha: float,
                actor_lr: int,
                Q1_lr: int,
                Q2_lr: int,
                epsilon: float,
                actor_betas: tuple,
                critic_betas: tuple,
                alpha_lr: int,
                init_temperature: int,
                learnable_temperature: bool,
                alpha_betas: tuple,
                critic_tau: int,

                ):
        super().__init__()

        self.num_actions = a_dim[0]
        
        self.Q_network1 = make_MLP(s_dim + self.num_actions, 1, Q_hidden_dims)
        self.Q_network2 = make_MLP(s_dim + self.num_actions, 1, Q_hidden_dims)
      
        self.Q_target1 = make_MLP(s_dim + self.num_actions, 1, Q_hidden_dims)
        self.Q_target2 = make_MLP(s_dim + self.num_actions, 1, Q_hidden_dims)

        for param1, param2 in zip(self.Q_target1.parameters(),self.Q_target2.parameters()): 

            param1.requires_grad = False # disable gradient computation for target network
            param2.requires_grad = False
        # copy params at start ? Yes as written in OpenAI pseudocode https://spinningup.openai.com/en/latest/algorithms/sac.html (M) 
        copy_params(self.Q_network1, self.Q_target1)
        copy_params(self.Q_network2, self.Q_target2)

        self.policy_network = make_MLP(s_dim, 2* self.num_actions, policy_hidden_dims) 
                                              # half for the mean
                                              # and half for the (log) std

        self.gamma = gamma
        self.tau = tau
    
        self.log_std_bounds = log_std_bounds
        self.epsilon = epsilon
        
        self.target_entropy = -self.num_actions
        self.critic_tau = critic_tau
        self.critic1_loss = nn.MSELoss()
        self.critic2_loss = nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                                lr=actor_lr,
                                                betas = actor_betas)
        self.Q_network1_optimizer = torch.optim.Adam(self.Q_network1.parameters(),
                                                lr=Q1_lr,
                                                betas = critic_betas
                                                )
        self.Q_network2_optimizer = torch.optim.Adam(self.Q_network2.parameters(),
                                                lr=Q2_lr,
                                                betas = critic_betas
                                                )
        self.learnable_temperature = learnable_temperature
        
        self.log_alpha = torch.tensor(np.log(init_temperature))
        if self.learnable_temperature:
          self.log_alpha.requires_grad = True
        else: 
          self.alpha = alpha


        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)                                                                    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def rep_trick(self, mu, std):
        normal = Normal(0, 1)
        z = normal.sample()
        return torch.tanh(mu + std*z)

    def check_tensor(self, x):
        if type(x) != torch.Tensor:
            x = np.array(x)
            x = torch.from_numpy(x)
        return x


    def policy_forward(self, state):
        state = self.check_tensor(state)
        
        out = self.policy_network(state)


        mu = out[:, 0 : self.num_actions]
        log_std = out[:,self.num_actions:]

        
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds

        # bound the log_std between min and max
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
                                                                             
        std = log_std.exp()
       
        return mu, std

    def Qnet_forward(self, state, action):
        state = self.check_tensor(state)
        action = self.check_tensor(action)
       
        obs_action = torch.cat([state, action], dim = -1)
        Q1 = self.Q_network1(obs_action)
        Q2 = self.Q_network2(obs_action)
        return Q1, Q2

    def Qtarg_forward(self, state, action):
        state = self.check_tensor(state)
        action = self.check_tensor(action)
        
        obs_action = torch.cat([state, action], dim = -1 )

        Q1 = self.Q_target1(obs_action)
        Q2 = self.Q_target2(obs_action)
        return Q1, Q2

    def actor(self, state):
        mu, std = self.policy_forward(state)
        dist = self.rep_trick(mu, std)

        return dist
    def gaussian_logprob(self,noise, log_std):
        """Compute Gaussian log probability."""
        residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
        return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

    def return_log(self, state):
        mu, std = self.policy_forward(state)
        
        z = torch.randn_like(mu)
        action = self.rep_trick(mu, std)
        
        log_prob = self.gaussian_logprob(z,std.log())
  
        return log_prob, action

    def get_action(self,state):
        state = self.check_tensor(state)
        dist = self.actor(state)
        action  = dist.cpu().detach().numpy()
        return action


    def alpha_decay(self):
        self.alpha*=0.8

    def update_policy(self, state, Q_net1, Q_net2):

        log_prob, _ = self.return_log(state)
        actor_Q = torch.min(Q_net1, Q_net2)

        # minus because we perform a gradient descent
        actor_loss = -(actor_Q - self.alpha*log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
 
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return actor_loss

    def update_Qnetworks(self, reward, done, Q_targ1, Q_targ2, Q_net1, Q_net2, log_prob):
        
        Q_critic = torch.min(Q_targ1, Q_targ2)
       
        target = reward + self.gamma*(1-done)*(Q_critic - self.alpha*log_prob)


        critic1_loss = self.critic1_loss(Q_net1, target.detach())
        self.Q_network1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph = True)
        self.Q_network1_optimizer.step()

        critic2_loss = self.critic1_loss(Q_net2, target.detach())
        self.Q_network2_optimizer.zero_grad()
        critic2_loss.backward(retain_graph = True)
        self.Q_network2_optimizer.step()

        return critic1_loss, critic2_loss
    def update_SAC(self, state, reward, action, new_state, done):
        log_prob, new_action = self.return_log(new_state)
        
        Q_net1, Q_net2 = self.Qnet_forward(state, action)
        
        Q_targ1, Q_targ2 = self.Qtarg_forward(new_state, new_action)
        
        loss1, loss2 = self.update_Qnetworks(reward, done, Q_targ1, Q_targ2, Q_net1, Q_net2, log_prob)

        dist = self.actor(state)
        if len(dist) == 1:
            dist.unsqueeze(-1)
        

        Q1, Q2 = self.Qnet_forward(state, dist)
        loss3 = self.update_policy(state, Q1, Q2)

        soft_update_params(self.Q_network1, self.Q_target1,self.critic_tau)
        soft_update_params(self.Q_network2, self.Q_target2,self.critic_tau)       

        return loss1.item() + loss2.item() + loss3.item()   
