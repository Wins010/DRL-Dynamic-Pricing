# hems/residential_agent.py

import numpy as np
from models.thermal_model import ThermalModel

class ResidentialAgent:
    def __init__(self, setpoint_temp, min_temp, initial_temp=20.0, max_power=3.0, efficiency=0.9):
        self.setpoint_temp = setpoint_temp
        self.min_temp = min_temp
        self.indoor_temp = initial_temp
        self.max_power = max_power
        self.efficiency = efficiency

        self.thermal_model = ThermalModel(efficiency=efficiency)
        self.history = []

    def decide_heating(self, price):
        if self.indoor_temp < self.min_temp:
            power = self.max_power
        else:
            power = 0.0
        self.history.append(power)
        return power

    def update_temperature(self, outdoor_temp, power_used):
        self.indoor_temp = self.thermal_model.compute_next_temperature(
            self.indoor_temp, outdoor_temp, power_used
        )

    def reset(self, initial_temp=20.0):
        self.indoor_temp = initial_temp
        self.history = []