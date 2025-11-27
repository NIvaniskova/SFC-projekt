import numpy as np
import matplotlib.pyplot as plt
import argparse
import ast

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from anfis import Room, ANFISThermostat
from plotting import plot_one_run_static
from generate_data import generate_outdoor_temps_30mins, generate_window_events


def anfis_static(half_hourly_temps, window_events, lr, n_rules, T_target, T_inside, initial_heater_power, heater_power_when_window_open):

    # GET DATA
    half_hourly_temps = half_hourly_temps
    window_events = window_events

    # INITIALIZE SYSTEM
    

    room = Room(T_target, T_inside, window_state = False)
    thermostat = ANFISThermostat(n_rules=n_rules, learning_rate=lr, initial_heater_power=initial_heater_power)

    n_minutes = 1440
    history_time = []
    history_T_inside = []
    history_T_outside = []
    history_heater_power = []
    total_energy_used = 0.0

    # SIMULATION
    print("Starting simulation...")

    for t in range(n_minutes):

        # --- A.1 Update Outside Temperature ---
        current_hour = t // 30 
        current_T_outside = half_hourly_temps[current_hour]

        # --- A.2 Update window state ---
        for start, end in window_events:
            if start <= t < end:
                room.set_window(True)
                break
        else:
            room.set_window(False)

        # --- B. Meassure ---
        error = T_target - room.T_inside

        # --- C. ANFIS Control ---
        # Inputs: Error, T_outside
        if room.window_open:
            heater_power = heater_power_when_window_open
        else:
            heater_power = thermostat.forward(error, current_T_outside)
            heater_power = max(0, min(5, heater_power))
            # --- D. Learn ---
            # Only in case window is closed, otherwise the system would go crazy
            thermostat.adapt(error)

        # Log energy output for system performance review
        total_energy_used += heater_power 

        # --- E. Actuate ---
        new_T_inside = room.update_temperature(current_T_outside, heater_power)
        
        # --- F. Log Data ---
        history_time.append(t / 60.0)  # Convert to hours for plotting
        history_T_inside.append(new_T_inside)
        history_T_outside.append(current_T_outside)
        history_heater_power.append(heater_power)

    print("Simulation complete.")

    # SYSTEM PERFORMANCE REVIEW
    arr_target = np.array([T_target] * n_minutes)
    arr_actual = np.array(history_T_inside)
    arr_valve = np.array(history_heater_power)

    # 1. RMSE (Overall Accuracy)
    rmse = np.sqrt(np.mean((arr_target - arr_actual)**2))

    # 2. Valve Efficiency (Chattering)
    # Standard deviation of the CHANGE in valve position
    valve_changes = np.diff(arr_valve)
    chattering = np.std(valve_changes)

    

    # --- PLOT RESULTS ---
    return history_time, history_T_inside, history_T_outside, history_heater_power, rmse, chattering, total_energy_used


def main(mode):

    T_target = 22.0
    T_inside = 18.0
    initial_heater_power = 3.0 
    heater_power_when_window_open = 2.0
    lr = 0.01
    n_rules = 5

    if mode == "test":
        
        file_path_outside_temperatures = "../data/outside_temperatures.txt"
        half_hourly_temps = []
        with open(file_path_outside_temperatures, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()  # remove whitespace and newline
                if line:  # skip empty lines
                    # Remove brackets and split by comma
                    numbers = line.strip("[]").split(",")
                    # Convert each number to float
                    numbers = [float(num.strip()) for num in numbers]
                    half_hourly_temps.append(numbers)

        file_path_window_events = "../data/window_events.txt"
        window_events = []
        with open(file_path_window_events, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()  # remove whitespace/newlines
                if line:
                    # Remove outer brackets
                    line = line.strip("[]")
                    tuples_list = []
                    for t in line.split("),"):  # split each tuple
                        t = t.strip(" ()")  # remove spaces and parentheses
                        if t:
                            # Convert numbers to int
                            numbers = tuple(int(x) for x in t.split(","))
                            tuples_list.append(numbers)
                    window_events.append(tuples_list)

        
            
        for i in range (100):
            history_time, history_T_inside, history_T_outside, history_heater_power, rmse, chattering, total_energy_used = anfis_static(half_hourly_temps[i], window_events[i], lr, n_rules, T_target, T_inside, initial_heater_power, heater_power_when_window_open)
    
    
    elif mode == "static":

        half_hourly_temps = generate_outdoor_temps_30mins()

        window_events = generate_window_events()

        history_time, history_T_inside, history_T_outside, history_heater_power, rmse, chattering, total_energy_used = anfis_static(half_hourly_temps, window_events, lr, n_rules, T_target, T_inside, initial_heater_power, heater_power_when_window_open)
    
        print(f"--- PERFORMANCE REPORT ---")
        print(f"RMSE:              {rmse:.4f} °C")
        print(f"Total Energy Used: {total_energy_used:.2f} units")
        print(f"Valve Smoothness:  {chattering:.4f}")
        print(f"--------------------------")

        plot_one_run_static(T_target, history_time, history_T_inside, history_T_outside, history_heater_power, window_events)

    elif mode == "dynamic":   
        pass

    else:
        print("Invalid mode selected. Choose 'static' or 'dynamic'.")


if __name__ == "__main__":
    
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description="ANFIS Thermostat Simulation")
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'dynamic', 'test'],
                        help="Select simulation mode: 'static' for pre-defined scenario, 'dynamic' for real-time control, 'test' to run algorithm multiple times and meassure performance.")
    args = parser.parse_args()  

    mode = args.mode

    main(mode)