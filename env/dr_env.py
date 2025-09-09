# env/dr_env.py

import numpy as np
import pandas as pd
import os

from hems.residential_agent import ResidentialAgent
from models.thermal_model import ThermalModel

class DemandResponseEnv:
    def __init__(self, data_path, episode_length=96):
        self.data_path = data_path
        self.episode_length = episode_length  # 96 = 1 day of 15-min steps
        self.agents = []
        self.current_step = 0
        self.n_houses = 0

        self._load_data()

    def _load_data(self):
        """
        Load synthetic CSVs and create agents.
        Assumes files: house_1.csv, house_2.csv, ...
        """
        files = sorted([f for f in os.listdir(self.data_path) if f.endswith(".csv")])
        self.n_houses = len(files)
        self.outdoor_temps = []
        self.fixed_loads = []

        for f in files:
            df = pd.read_csv(os.path.join(self.data_path, f))
            setpoint = df["setpoint_temp"].iloc[0]
            min_temp = df["min_temp"].iloc[0]
            outdoor_temp = df["outdoor_temp"].values
            fixed_load = df["fixed_load"].values

            agent = ResidentialAgent(setpoint_temp=setpoint, min_temp=min_temp)
            self.agents.append(agent)
            self.outdoor_temps.append(outdoor_temp)
            self.fixed_loads.append(fixed_load)

        self.outdoor_temps = np.array(self.outdoor_temps)  # shape: [n_houses, T]
        self.fixed_loads = np.array(self.fixed_loads)

    def reset(self):
        """
        Reset the environment at the beginning of an episode.
        """
        self.current_step = 0

        # Reset all agents
        for agent in self.agents:
            agent.reset()

        return self._get_observation()

    def _get_observation(self):
        """
        Returns a flat observation vector.
        For now: [indoor_temps | outdoor_temps | fixed_loads]
        """
        indoor = np.array([agent.indoor_temp for agent in self.agents])
        outdoor = self.outdoor_temps[:, self.current_step]
        fixed = self.fixed_loads[:, self.current_step]
        return np.concatenate([indoor, outdoor, fixed])

    def step(self, price_signal):
        """
        Apply price signal, simulate one time step.
        price_signal: float or array of shape [n_houses]
        """
        total_load = []
        rewards = []

        for i, agent in enumerate(self.agents):
            price = price_signal if np.isscalar(price_signal) else price_signal[i]
            power = agent.decide_heating(price)
            agent.update_temperature(
                outdoor_temp=self.outdoor_temps[i, self.current_step],
                power_used=power
            )
            total = power + self.fixed_loads[i, self.current_step]
            total_load.append(total)

            # Reward: simple comfort metric (optional: add price penalty)
            comfort_penalty = max(0, agent.setpoint_temp - agent.indoor_temp)
            rewards.append(-comfort_penalty)  # More negative = worse comfort

        self.current_step += 1
        done = self.current_step >= self.episode_length
        obs = self._get_observation()

        return obs, np.mean(rewards), done, {"total_load": np.sum(total_load)}

    def render(self):
        temps = [round(agent.indoor_temp, 1) for agent in self.agents]
        print(f"Step {self.current_step}: Indoor temps = {temps}")