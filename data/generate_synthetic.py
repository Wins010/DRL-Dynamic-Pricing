#import necessary libraries

import os
import numpy as np
import pandas as pd

# setting a random seed for reproducibility
np.random.seed(1)

# Initializing constants
N_HOUSES = 11
N_STEPS = 24 * 4

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "synthetic"))

# Create folder if it doesn't exist
os.makedirs(DATA_PATH, exist_ok=True)

"""Winter temperature profile with daily pattern."""
def generate_outdoor_temperature(n_steps=N_STEPS):
    base_temp = -5 + 5 * np.sin(np.linspace(0, 2 * np.pi, n_steps))
    noise = np.random.normal(0, 1.5, n_steps)
    return base_temp + noise

def generate_comfort_preference():
    """Comfort setpoints and setbacks for all houses."""
    setpoints = np.random.choice([20, 21, 22, 23], size=N_HOUSES, p=[0.1, 0.3, 0.5, 0.1])
    setbacks = np.random.choice([1, 2, 3, 4], size=N_HOUSES, p=[0.1, 0.3, 0.4, 0.2])
    x_min = setpoints - setbacks
    return setpoints, x_min

def generate_fixed_load(n_steps=N_STEPS):
    """Simulate non-flexible appliance loads with morning/evening peak."""
    time = np.linspace(0, 2 * np.pi, n_steps)
    base_profile = 1.5 + 0.5 * np.sin(time - np.pi / 2)
    profiles = []
    for _ in range(N_HOUSES):
        noise = np.random.normal(0, 0.3, n_steps)
        load = np.clip(base_profile + noise, 0.5, 3.0)
        profiles.append(load)
    return np.array(profiles)

def main():
    outdoor_temp = generate_outdoor_temperature()
    setpoints, min_temps = generate_comfort_preference()
    fixed_loads = generate_fixed_load()

    for i in range(N_HOUSES):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=N_STEPS, freq="15min"),
            "outdoor_temp": outdoor_temp,
            "fixed_load": fixed_loads[i],
            "setpoint_temp": setpoints[i],
            "min_temp": min_temps[i]
        })
        filename = os.path.join(DATA_PATH, f'house_{i + 1}.csv')
        df.to_csv(filename, index=False)

    print(f" Generated synthetic data for {N_HOUSES} houses in → {DATA_PATH}")

if __name__ == "__main__":
    main()