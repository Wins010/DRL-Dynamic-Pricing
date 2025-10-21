import numpy as np
import cvxpy as cp

class ResidentialAgent:
    def __init__(self, setpoint_temp, min_temp, initial_temp=None, max_power=3.5):
        self.setpoint_temp = setpoint_temp
        self.min_temp = min_temp
        self.max_power = max_power
        self.initial_temp = initial_temp if initial_temp is not None else setpoint_temp
        self.indoor_temp = self.initial_temp
        self.last_power = 0.0

    def reset(self):
        self.indoor_temp = self.initial_temp
        self.last_power = 0.0

    def optimize_schedule(self, price_profile, outdoor_temp, fixed_load,
                          lambda_cost=1.0, lambda_comfort=10.0,
                          alpha=0.1, heating_effect=0.5):
        """
        Full-episode optimization as in the research paper:
        - price_profile: (T,) array of prices
        - outdoor_temp: (T,) array of outdoor temperatures
        - fixed_load: (T,) array of fixed loads (can be ignored/added to agg. demand in env)
        - lambda_cost: weight for cost term in objective
        - lambda_comfort: weight for discomfort term
        - alpha: building thermal loss coefficient
        - heating_effect: temp increase per unit power
        Returns: heating_power: (T,) array
        """
        T = len(price_profile)
        u = cp.Variable(T)  # heating power
        x = cp.Variable(T+1)  # indoor temp trajectory

        constraints = [x[0] == self.initial_temp]
        for t in range(T):
            # Paper's thermal model: x_{t+1} = x_t + alpha*(outdoor_temp[t] - x_t) + heating_effect*u[t]
            constraints += [x[t+1] == x[t] + alpha * (outdoor_temp[t] - x[t]) + heating_effect * u[t]]
            constraints += [u[t] >= 0, u[t] <= self.max_power]
            constraints += [x[t+1] >= self.min_temp]
        # Paper's objective: cost + discomfort (only penalize if below setpoint)
        discomfort = cp.sum(cp.pos(self.setpoint_temp - x[1:]))
        elec_cost = cp.sum(cp.multiply(price_profile, u))
        obj = lambda_cost * elec_cost + lambda_comfort * discomfort

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)

        if u.value is None:
            # If solver fails, fallback to no heating
            return np.zeros(T)
        return np.clip(u.value, 0, self.max_power)