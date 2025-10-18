# train/train_dra.py
import sys, os
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dra.ppo_agent import PPOAgent
from env.dr_env import DemandResponseEnv

# build absolute path to synthetic data
data_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # go to project root
    "data",
    "synthetic"
)

env = DemandResponseEnv(data_path=data_path)

# Get observation and action dimensions
obs_dim = len(env._get_observation())
act_dim = env.n_houses

agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim)

num_episodes = 1000
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    memory = {"states": [], "actions": [], "logprobs": [], "rewards": [], "dones": [], "values": []}
    total_reward = 0
    while not done:
        action, logprob = agent.select_action(obs)
        value = agent.policy.critic(torch.FloatTensor(obs).unsqueeze(0))
        next_obs, reward, done, info = env.step(action)

        memory["states"].append(obs)
        memory["actions"].append(action)
        memory["logprobs"].append(logprob)
        memory["rewards"].append(reward)
        memory["dones"].append(done)
        memory["values"].append(value.squeeze())

        obs = next_obs
        total_reward += reward

    agent.update(memory)
    print(f"[Episode {episode+1}/{num_episodes}] Total Reward: {total_reward:.2f}")