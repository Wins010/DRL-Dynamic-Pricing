# evaluate/run_simulation.py

from dra.ppo_agent import PPOAgent

data_path = "data/synthetic"
agent = PPOAgent(data_path=data_path)
agent.load()
agent.evaluate(n_episodes=5)