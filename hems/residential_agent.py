import numpy as np

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

    def decide_heating(self, price):
        # Simple rule-based decision: turn on heating if below min_temp
        if self.indoor_temp < self.min_temp:
            power = self.max_power
        else:
            power = 0.0
        self.last_power = power
        return power

    def update_temperature(self, outdoor_temp, power_used):
        # Very basic thermal model: indoor temp moves toward outdoor temp unless heated
        alpha = 0.1  # how fast temp equalizes
        heating_effect = 0.5 * power_used  # temp increase per unit power
        temp_change = alpha * (outdoor_temp - self.indoor_temp) + heating_effect
        self.indoor_temp += temp_change