# Denis Yarat implementation adapted to our case (removed the encoder)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import soft_update_params

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, s_dim, a_dim, hidden_dim,
        log_std_min, log_std_max
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * a_dim)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, state, compute_pi=True, compute_log_pi=True
    ):
        mu, log_std = self.trunk(state).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, s_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(s_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        assert state.size(0) == action.size(0)

        state_action = torch.cat([state, action], dim=1)
        return self.trunk(state_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, s_dim, a_dim, hidden_dim
    ):
        super().__init__()

        self.Q1 = QFunction(
            s_dim, a_dim, hidden_dim
        )
        self.Q2 = QFunction(
            s_dim, a_dim, hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, state, action):
        
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class SacAgent(object):
    def __init__(
        self,
        s_dim,
        a_dim,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        self.actor = Actor(
            s_dim, a_dim, hidden_dim,
            actor_log_std_min, actor_log_std_max
        ).to(device)

        self.critic = Critic(
            s_dim, a_dim, hidden_dim
        ).to(device)

        self.critic_target = Critic(
            s_dim, a_dim, hidden_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -a_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.device = device
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0)
            mu, _, _, _ = self.actor(
                state, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0)
            mu, pi, _, _ = self.actor(state, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, state, action, reward, next_state, not_done, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward.reshape(target_V.shape) + (not_done.reshape(target_V.shape) * (self.discount * target_V))


        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, state, step):
        _, pi, log_pi, log_std = self.actor(state)
        actor_Q1, actor_Q2 = self.critic(state, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()

        alpha_loss.backward()
        self.log_alpha_optimizer.step()


    def update(self, replay_buffer, step, batch_size):
        state, action, reward, done, next_state, *_ = replay_buffer.sample(batch_size)
        not_done = ~done

        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        not_done = torch.from_numpy(not_done).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)

        self.update_critic(state, action, reward, next_state, not_done, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(state, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )


    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )


