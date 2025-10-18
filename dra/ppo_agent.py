import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))  # learnable std dev

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std)
        state_value = self.critic(x)
        return mean, std, state_value

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(obs_dim, act_dim)
        self.policy_old = ActorCritic(obs_dim, act_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

        # Memory buffer
        self.memory = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "dones": [],
            "values": []
        }

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.policy_old(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)  # sum across dims

        return action.squeeze().numpy(), logprob.item()

    def store_transition(self, state, action, logprob, reward, done):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mean, std, value = self.policy_old(state_tensor.unsqueeze(0))

        self.memory["states"].append(state_tensor)
        self.memory["actions"].append(torch.tensor(action, dtype=torch.float32))
        self.memory["logprobs"].append(logprob)
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)
        self.memory["values"].append(value.squeeze())

    def clear_memory(self):
        for key in self.memory:
            self.memory[key] = []

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        values = torch.stack(values)
        return returns, values

    def update(self, memory=None):
        if memory is None:
            memory = self.memory

        old_states = torch.stack(memory["states"])
        old_actions = torch.stack(memory["actions"])
        old_logprobs = torch.FloatTensor(np.array(memory["logprobs"]))
        rewards = memory["rewards"]
        dones = memory["dones"]

        with torch.no_grad():
            mean, std, next_value = self.policy(old_states[-1].unsqueeze(0))
        returns, values = self.compute_returns(rewards, dones, memory["values"], next_value)

        # Compute normalized advantage
        advantages = returns - values.detach().squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.k_epochs):
            mean, std, state_values = self.policy(old_states)
            dist = torch.distributions.Normal(mean, std)
            entropy = dist.entropy().sum(dim=-1).mean()

            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"[PPO Update] Epoch {epoch+1}/{self.k_epochs} "
                  f"actor_loss={actor_loss.item():.4f}, "
                  f"critic_loss={critic_loss.item():.4f}, "
                  f"entropy={entropy.item():.4f}")

        self.policy_old.load_state_dict(self.policy.state_dict())

    def train(self):
        self.update(self.memory)
        self.clear_memory()