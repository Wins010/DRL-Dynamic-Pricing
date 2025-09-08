# hems/residential_agent.py

import numpy as np

class ResidentialAgent:
    def __init__(self, setpoint_temp, min_temp, initial_temp=20.0, max_power=3.0, efficiency=0.9):
        self.setpoint_temp = setpoint_temp  # comfort target
        self.min_temp = min_temp            # minimum comfort threshold
        self.indoor_temp = initial_temp     # current indoor temp
        self.max_power = max_power          # heating power cap (kW)
        self.efficiency = efficiency        # efficiency of heating system

        self.history = []  # stores power usage per time step


    def decide_heating(self, price):
        """
        Simple rule-based heating strategy:
        - If indoor temp < min comfort → use full power
        - Else → no heating (or later: optimize for price)
        """
        if self.indoor_temp < self.min_temp:
            power = self.max_power
        else:
            power = 0.0

        self.history.append(power)
        return power


    def update_temperature(self, outdoor_temp, power_used, timestep_hours=0.25):
        """
        Updates indoor temperature:
        - power_used adds heat
        - outdoor temp pulls heat out
        - simplified thermal model for now
        """
        heat_gain = power_used * self.efficiency * timestep_hours * 10
        heat_loss = (self.indoor_temp - outdoor_temp) * 0.1 * timestep_hours
        self.indoor_temp += heat_gain - heat_loss

    def reset(self, initial_temp=20.0):
        self.indoor_temp = initial_temp
        self.history = []