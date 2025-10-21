import numpy as np
import pandas as pd
import os

from hems.residential_agent import ResidentialAgent
from models.thermal_model import ThermalModel

class DemandResponseEnv:
    def __init__(self, data_path, episode_length=96, M=80, pi_min=0.05, pi_max=0.25):
        self.data_path = data_path
        self.episode_length = episode_length
        self.M = M  # System capacity constraint (can be changed per episode)
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.agents = []
        self._load_data()

    def _load_data(self):
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
        self.outdoor_temps = np.array(self.outdoor_temps)
        self.fixed_loads = np.array(self.fixed_loads)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def price_generator(self, aggregate_demand, eta):
        """Sigmoid price function as in the paper."""
        eta = max(1e-4, abs(eta))
        price_profile = self.pi_min + (self.pi_max - self.pi_min) / (1 + np.exp(-(aggregate_demand - self.M) / eta))
        return price_profile

    def run_episode(self, eta, max_iter=10, tol=1e-3):
        """Coordination loop for a full episode. Returns reward and episode stats."""
        self.reset()
        n_steps = self.episode_length
        # Start with an initial guess (flat or from previous episode if available)
        agg_demand = np.zeros(n_steps)
        prev_agg_demand = np.ones(n_steps) * np.inf

        for it in range(max_iter):
            price_profile = self.price_generator(agg_demand, eta)
            all_schedules = []
            for i, agent in enumerate(self.agents):
                # Agent optimizes its entire schedule given price_profile and exogenous variables
                schedule = agent.optimize_schedule(price_profile, self.outdoor_temps[i], self.fixed_loads[i])
                all_schedules.append(schedule)
            agg_demand = np.sum(all_schedules, axis=0)
            if np.linalg.norm(agg_demand - prev_agg_demand) < tol:
                break
            prev_agg_demand = agg_demand.copy()

        reward, diagnostics = self.compute_episode_reward(agg_demand, price_profile)
        episode_stats = {
            "aggregate_demand": agg_demand,
            "price_profile": price_profile,
            **diagnostics
        }
        return reward, episode_stats

    def compute_episode_reward(self, agg_demand, price_profile, u1=1.0, u2=1.0):
        """
        Compute episode-level reward according to the paper:
        reward = u1 * total generation cost - u2 * max_overrun
        """
        # Example: quadratic generation cost
        total_cost = np.sum(price_profile * agg_demand)
        max_overrun = np.maximum(agg_demand - self.M, 0).max()
        reward = -u1 * total_cost - u2 * max_overrun
        diagnostics = {
            "total_cost": total_cost,
            "max_overrun": max_overrun
        }
        return reward, diagnostics