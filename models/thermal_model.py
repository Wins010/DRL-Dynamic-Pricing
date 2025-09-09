# models/thermal_model.py

class ThermalModel:
    def __init__(self, efficiency=0.9, heat_transfer_coeff=0.1, timestep_hours=0.25):
        self.efficiency = efficiency                  # heater efficiency
        self.heat_transfer_coeff = heat_transfer_coeff  # house insulation quality
        self.timestep_hours = timestep_hours          # each step = 15 minutes

    def compute_next_temperature(self, indoor_temp, outdoor_temp, power_used):
        """
        Compute the next indoor temperature based on:
        - heating power input
        - temperature difference with outdoor
        """
        # Heat gained from heater
        heat_gain = power_used * self.efficiency * self.timestep_hours * 10

        # Heat lost to outdoors (basic linear loss model)
        heat_loss = (indoor_temp - outdoor_temp) * self.heat_transfer_coeff * self.timestep_hours

        # Net effect
        next_temp = indoor_temp + heat_gain - heat_loss

        return next_temp
