import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import ast

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from anfis import Room, ANFISThermostat
from plotting import plot_one_run_static, plot_heatmap_gridsearch, print_results_gridsearch
from generate_data import generate_outdoor_temps_30mins, generate_window_events, parse_outside_temperatures, parse_window_events


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
    #print("Starting simulation...")
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

    #print("Simulation complete.")

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


class InteractiveANFIS:
    def __init__(self, outdoor_temps, T_target, T_start, anfis_model, room_model):
        self.outdoor_temps = outdoor_temps
        self.T_target = T_target
        self.thermostat = anfis_model
        self.room = room_model
        
        # Simulation State
        self.current_minute = 0
        self.window_is_open = False
        self.history_time = []
        self.history_Tin = []
        self.history_Tout = []
        self.history_power = []
        self.history_window = [] # To visualize when it was open later
        
        # Setup Plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        plt.subplots_adjust(bottom=0.2) # Make room for button
        
        self.setup_plots()
        self.setup_interaction()
        
    def setup_plots(self):
        # --- Axis 1: Temperatures ---
        self.ax1.set_xlim(0, 24)
        self.ax1.set_ylim(0, 30)
        self.ax1.set_ylabel('Temperature (°C)')
        self.ax1.set_title('Interactive ANFIS Control (Click Button to Open Window)')
        self.ax1.grid(True)
        
        # Static Target Line
        self.ax1.axhline(self.T_target, color='k', linestyle='--', label='Target')
        
        # dynamic lines (empty for now)
        self.line_Tin, = self.ax1.plot([], [], 'r-', linewidth=2, label='Inside')
        self.line_Tout, = self.ax1.plot([], [], 'b-', alpha=0.5, label='Outside')
        self.ax1.legend(loc='upper left')

        # Visual indicator for window
        self.window_text = self.ax1.text(12, 28, "", color='red', fontsize=12, fontweight='bold', ha='center')
        
        # --- Axis 2: Heater ---
        self.ax2.set_xlim(0, 24)
        self.ax2.set_ylim(0, 6)
        self.ax2.set_ylabel('Valve (0-5)')
        self.ax2.set_xlabel('Time (Hours)')
        self.ax2.grid(True)
        
        self.line_power, = self.ax2.plot([], [], 'g-', linewidth=2, label='Valve')
        self.fill_power = None # Placeholder for fill_between
        self.ax2.legend(loc='upper left')

    def setup_interaction(self):
        # Create the button area
        ax_btn = plt.axes([0.4, 0.05, 0.2, 0.075]) # [left, bottom, width, height]
        self.btn = Button(ax_btn, 'Toggle Window', color='lightgray', hovercolor='0.97')
        self.btn.on_clicked(self.toggle_window_state)

    def toggle_window_state(self, event):
        self.window_is_open = not self.window_is_open
        
        # Visual feedback on the button itself
        if self.window_is_open:
            self.btn.label.set_text("CLOSE WINDOW")
            self.btn.color = '#ffcccc' # Red-ish
        else:
            self.btn.label.set_text("OPEN WINDOW")
            self.btn.color = 'lightgray'

    def update(self, frame):
        # Stop if 24 hours reached
        if self.current_minute >= 1440:
            self.anim.event_source.stop()
            print("Simulation Finished.")
            return
        
        # A. Update Inputs
        idx = self.current_minute // 30
        if idx >= len(self.outdoor_temps): idx = len(self.outdoor_temps) - 1
        current_T_outside = self.outdoor_temps[idx]
        
        # B. Set Window State in Room
        self.room.set_window(self.window_is_open)
        
        # C. Measure Error
        error = self.T_target - self.room.T_inside
        
        # D. ANFIS Control
        if self.room.window_open:
            heater_power = 0.0 # Force OFF logic
            # Visual feedback
            self.window_text.set_text("⚠️ WINDOW OPEN ⚠️")
            self.ax1.set_facecolor('#f2f2f2') # Grey out background slightly
        else:
            heater_power = self.thermostat.forward(error, current_T_outside)
            heater_power = max(0, min(5, heater_power)) # Clip 0-5
            self.thermostat.adapt(error) # Learn only when closed
            
            # clear visual feedback
            self.window_text.set_text("")
            self.ax1.set_facecolor('white')

        # E. Actuate
        new_T_inside = self.room.update_temperature(current_T_outside, heater_power)
        
        # ---------------------------------------------------------
        # 2. LOGGING
        # ---------------------------------------------------------
        time_hour = self.current_minute / 60.0
        self.history_time.append(time_hour)
        self.history_Tin.append(new_T_inside)
        self.history_Tout.append(current_T_outside)
        self.history_power.append(heater_power)
        
        self.current_minute += 1
        
        # ---------------------------------------------------------
        # 3. UPDATE PLOT (Efficiently)
        # ---------------------------------------------------------
        self.line_Tin.set_data(self.history_time, self.history_Tin)
        self.line_Tout.set_data(self.history_time, self.history_Tout)
        self.line_power.set_data(self.history_time, self.history_power)
               
        return self.line_Tin, self.line_Tout, self.line_power, self.window_text

    def start(self):
        self.anim = FuncAnimation(self.fig, self.update, frames=range(1440), interval=10, blit=False, repeat=False)
        plt.show()


def main(mode):

    T_target = 22.0
    T_inside = 18.0
    initial_heater_power = 3.0 
    heater_power_when_window_open = 2.0
    lr = 0.01
    n_rules = 5

    if mode == "test":
        
        file_path_outside_temperatures = "../data/outside_temperatures.txt"
        half_hourly_temps = parse_outside_temperatures(file_path_outside_temperatures)

        file_path_window_events = "../data/window_events.txt"
        window_events = parse_window_events(file_path_window_events)
            
        # 1. DEFINE GRID
        rule_options = [3, 5, 7]
        lr_options = [0.001, 0.005, 0.010, 0.050] 
        results_grid = []

        print(f"Starting Grid Search ({len(rule_options) * len(lr_options)} combinations)...")

        # 2. NESTED LOOPS
        for n_rules in rule_options:
            for lr in lr_options:
                print(f"  Testing: Rules={n_rules}, LR={lr}")
                
                current_rmse_scores = []
                current_chatter_scores = []
                
                for i in range(50): 
                    ht, t_in, t_out, heater, rmse, chat, energy = anfis_static(
                        half_hourly_temps[i], 
                        window_events[i], 
                        lr, 
                        n_rules, 
                        T_target, 
                        T_inside, 
                        initial_heater_power, 
                        heater_power_when_window_open
                    )
                    current_rmse_scores.append(rmse)
                    current_chatter_scores.append(chat)
                    
                # Store Aggregates
                results_grid.append({
                    "Rules": n_rules,
                    "Learning_Rate": lr,
                    "RMSE": np.mean(current_rmse_scores),
                    "Chattering": np.mean(current_chatter_scores)
                })
        
        # 3. CONVERT TO PIVOT TABLES
        df = pd.DataFrame(results_grid)
        rmse_pivot = df.pivot(index="Rules", columns="Learning_Rate", values="RMSE")
        chat_pivot = df.pivot(index="Rules", columns="Learning_Rate", values="Chattering")

        print("Grid Search Complete.")

        # 4.A PLOTTING THE HEATMAPS
        plot_heatmap_gridsearch(rmse_pivot, chat_pivot)

        # 4.B RESULTS
        print_results_gridsearch(df)
            
    
    elif mode == "static":

        half_hourly_temps = generate_outdoor_temps_30mins()
        window_events = generate_window_events()

        lr = 0.01
        n_rules = 5

        history_time, history_T_inside, history_T_outside, history_heater_power, rmse, chattering, total_energy_used = anfis_static(half_hourly_temps, window_events, lr, n_rules, T_target, T_inside, initial_heater_power, heater_power_when_window_open)
    
        print(f"--- PERFORMANCE REPORT ---")
        print(f"RMSE:              {rmse:.4f} °C")
        print(f"Total Energy Used: {total_energy_used:.2f} units")
        print(f"Valve Smoothness:  {chattering:.4f}")
        print(f"--------------------------")

        plot_one_run_static(T_target, history_time, history_T_inside, history_T_outside, history_heater_power, window_events)

    elif mode == "dynamic":   

        # 1. Prepare Data
        half_hourly_temps = generate_outdoor_temps_30mins() # Your function

        # 2. Initialize system
        room = Room(T_target, T_inside, False)
        thermostat = ANFISThermostat(n_rules=5, learning_rate=0.01)

        # 3. Launch App
        app = InteractiveANFIS(half_hourly_temps, T_target, T_inside, thermostat, room)
        app.start()

    else:
        print("Invalid mode selected. Choose 'static' or 'dynamic'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ANFIS Thermostat Simulation")
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'dynamic', 'test'],
                        help="Select simulation mode: 'static' for pre-defined scenario, 'dynamic' for real-time control, 'test' to run algorithm multiple times and meassure performance.")
    args = parser.parse_args()  

    mode = args.mode

    main(mode)