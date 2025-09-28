# dra/ppo_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value


class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_dim, act_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy_old(obs)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        values = torch.stack(values)
        return returns, values

    def update(self, memory):
        old_states = torch.FloatTensor(np.array(memory["states"]))
        old_actions = torch.LongTensor(np.array(memory["actions"]))
        old_logprobs = torch.FloatTensor(np.array(memory["logprobs"]))
        rewards = memory["rewards"]
        dones = memory["dones"]

        with torch.no_grad():
            _, next_value = self.policy(old_states[-1].unsqueeze(0))
        returns, values = self.compute_returns(rewards, dones, memory["values"], next_value)

        for _ in range(self.k_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(action_probs)
            entropy = dist.entropy().mean()

            logprobs = dist.log_prob(old_actions)
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = returns - state_values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())