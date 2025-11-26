import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class Room:
    def __init__(self, T_target, T_inside=20.0, window_state=False):
        self.T_inside = T_inside  # Initial inside temperature
        self.T_target = T_target
        self.insulation_loss = 0.1
        self.window_open = window_state

        # Heater efficiency calibration:
        # 0 = Off
        # 1 = 10°C
        # 2 = 15°C
        # 3 = 20°C
        # 4 = 25°C
        # 5 = 30°C 

        # Calculation logic:
        # To maintain 22°C when outside is 0°C (Diff = 22):
        # Loss = 22 * 0.1 = 2.2 units.
        # We want this to happen at Valve Level ~3.5 (between 20°C and 25°C mark).
        # So: 3.5 * efficiency = 2.2  --> Efficiency approx 0.6
        self.heater_efficiency = 0.6

    def set_window(self, state):
        self.insulation_loss = 0.2 if state else 0.1
        self.window_open = state

    def update_temperature(self, T_outside, heater_power):
        """
        Simulates one minute of thermodynamics.
        power: 0-5 scale
        outside_temp: Degrees Celsius
        """
        # Heat loss to outside
        # if self.window_open:
        #     self.insulation_loss = 0.2
        # else:
        #     self.insulation_loss = 0.1
        heat_loss = self.insulation_loss * (self.T_inside - T_outside)
        # Heat gain from heater
        heat_gain = self.heater_efficiency * heater_power
        # Update inside temperature
        self.T_inside += heat_gain - heat_loss
        return self.T_inside
    

class ANFISThermostat:
    def __init__(self, n_rules=5, learning_rate=0.01, initial_heater_power=3.0):
        self.n_rules = n_rules
        self.learning_rate = learning_rate
        self.n_inputs = 2  # T_inside, T_outside

        # --- Initualize parameters ---
        # Membership function parameters: (n_inputs, n_rules, 2) for Gaussian MF (c, sigma)
        # Rules need to cover range of Errors (-5 to 5) and Outside Temps (-10 to 20)
        self.mu = np.random.uniform(-5, 10, (self.n_rules, self.n_inputs))
        self.sigma = np.random.uniform(2, 10, (self.n_rules, self.n_inputs))

        # Consequent parameters: (n_rules, n_inputs + 1) for linear function (p0, p1, bias)
        self.consequent = np.zeros((self.n_rules, self.n_inputs + 1))  # +1 for bias term
        
        # Initialize bias (c) to something non-zero so heater isn't off at start
        self.consequent[:, 2] = initial_heater_power    # Shape: (3, n_rules)

    def gaussian_mf(self, x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    

    def forward(self, error, T_outside):
        self.x_input = np.array([error, T_outside])  # Shape: (2,)

        # --- Fuzzification ---
        self.mu_values = self.gaussian_mf(self.n_inputs, self.mu, self.sigma)  # Shape: (n_inputs, n_rules)

        # --- Rule Evaluation ---
        self.w = np.prod(self.mu_values, axis=1)  # Shape: (n_rules,)

        # --- Normalization ---
        self.w_sum = np.sum(self.w) + 1e-6  # Avoid division by zero
        self.w_normalized = self.w / self.w_sum  # Shape: (n_rules,)

        # --- Linear Output ---
        x_bias = np.append(self.x_input, 1)  # Shape: (3,) [Error, T_outside, 1]
        self.linear_output = np.dot((self.consequent), x_bias)  # Shape: (n_rules,)

        # --- Aggregation ---
        raw_output = np.dot(self.w_normalized, self.linear_output)  # Scalar

        # --- Activation Function (Clamp to 0-5) ---
        heater_power = np.clip(raw_output, 0, 5)
        return heater_power
    

    def adapt(self, system_error):
        """
        system_error = Target - Current_Temp
        If error is positive (Too Cold), we raise weights to increase heating.
        """
         
        x_bias = np.append(self.x_input, 1)  
    
        for r in range(self.n_rules):
            grad = self.w_normalized[r] * x_bias   
            self.consequent[r] += self.learning_rate * system_error * grad  

            if self.w[r] > 0.05:
                diff = self.n_inputs - self.mu[r]   
                self.mu[r] += self.learning_rate * system_error * diff  


# --- SIMULATION ---

def main(mode="static"):

    if mode == "static":
        half_hourly_temps = [
            2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 3.0, 
            5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 14.5, 15.0, 15.0,
            15.0, 14.5, 14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 10.0, 9.0, 8.0, 6.5,
            6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 1.0
        ]

        window_events = [
            (420, 480, "Window opened (60m)"),   
            (780, 795, "Window opened (15m)"),    
            (1080, 1110, "Window opened (30m)") 
        ]

        T_target = 22.0
        T_inside = 20.0
        initial_heater_power = 3.0
        current_T_outside = 5.0  
        window_state = False

        room = Room(T_target, T_inside, window_state)
        thermostat = ANFISThermostat(n_rules=5, learning_rate=0.001, initial_heater_power=initial_heater_power)

        n_minutes = 1440
        history_time = []
        history_T_inside = []
        history_T_outside = []
        history_heater_power = []


        print("Starting simulation...")

        for t in range(n_minutes):

            # --- A.1 Update Outside Temperature ---
            current_hour = t // 30 
            current_T_outside = half_hourly_temps[current_hour]

            # --- A.2 Update window state ---
            for start, end, label in window_events:
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
                heater_power = 0.0
            else:
                heater_power = thermostat.forward(error, current_T_outside)
                # --- D. Learn ---
                thermostat.adapt(error)

            # --- E. Actuate ---
            new_T_inside = room.update_temperature(current_T_outside, heater_power)
            
            # --- F. Log Data ---
            history_time.append(t / 60.0)  # Convert to hours for plotting
            history_T_inside.append(new_T_inside)
            history_T_outside.append(current_T_outside)
            history_heater_power.append(heater_power)

        print("Simulation complete.")

        # --- PLOT RESULTS ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        hours_ticks = np.arange(0, 25, 2) 

        # Plot Temperatures
        ax1.plot(history_time, [T_target]*len(history_time), 'k--', label='Target (22°C)')
        ax1.plot(history_time, history_T_inside, 'r-', linewidth=2, label='Inside Temperature')
        ax1.plot(history_time, history_T_outside, 'b-', alpha=0.6, label='Outside Temperature')

        for start, end, label in window_events:
            ax1.axvspan(start/60, end/60, color='gray', alpha=0.3)
            ax1.text((start+end)/120, 10, label, ha='center' , fontsize=8, rotation=90)

        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title(f'ANFIS Adaptive Thermostat Simulation')
        ax1.set_xlim(0, 24)
        ax1.set_xticks(hours_ticks)
        ax1.legend()
        ax1.grid(True)

        # Plot Heater Output
        ax2.plot(history_time, history_heater_power, 'g-', label='Heater Power (0-5)', linewidth=2)
        ax2.set_ylabel('Heater Power (0-5)')
        ax2.set_xlabel('Time (Hours)')
        ax2.set_title('Control Action')
        ax2.fill_between(history_time, history_heater_power, color='green', alpha=0.1)
        ax2.grid(True)

        # Set Y-Axis to look like a Radiator Knob
        ax2.set_ylim(0, 6)
        ax2.set_yticks([0, 1, 2, 3, 4, 5])
        ax2.fill_between(history_time, history_heater_power, color='green', alpha=0.1)
        ax2.grid(True)
        ax2.set_xlim(0,24)
        ax2.set_xticks(hours_ticks)
        ax2.legend()

        for start, end, label in window_events:
            ax2.axvspan(start/60, end/60, color='gray', alpha=0.3)
            ax2.text((start+end)/120, 2, label, ha='center' , fontsize=8, rotation=90)

        plt.tight_layout()
        plt.show()


    elif mode == "dynamic":
        T_target = 22.0
        T_inside = 20.0
        initial_heater_power = 2.0 # Start closer to expected value
        
        state = {'window_open': False}

        room = Room(T_target, T_inside, state['window_open'])
        
        # FIX 1: DRASTICALLY LOWER LEARNING RATE
        # 0.1 is too high (causes oscillation). 0.01 is stable.
        thermostat = ANFISThermostat(n_rules=5, learning_rate=0.01, initial_heater_power=initial_heater_power)

        # Show last 300 minutes (5 hours) on screen
        max_len = 300 
        x_data = np.arange(max_len)
        
        y_temp = [T_target] * max_len
        y_level = [initial_heater_power] * max_len
        y_window = [0.0] * max_len
        y_outside = [5.0] * max_len

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)

        # Graph 1: Temps
        line_temp, = ax1.plot(x_data, y_temp, 'r-', lw=2, label='Inside Temp')
        line_outside, = ax1.plot(x_data, y_outside, 'c--', lw=1, label='Outside Temp')
        ax1.axhline(y=T_target, color='k', linestyle='--', label='Target (22°C)')
        
        ax1.set_ylim(0, 35)
        ax1.set_ylabel("Temp (°C)")
        ax1.set_title("ANFIS Control (1 Frame = 1 Minute)")
        ax1.legend(loc='upper left', ncol=3)

        # Graph 2: Controls
        line_level, = ax2.plot(x_data, y_level, 'g-', lw=2, label='Heater (0-5)')
        line_window, = ax2.plot(x_data, y_window, 'b-', lw=1, label='Window', alpha=0.5)

        ax2.set_ylim(0, 6)
        ax2.set_ylabel("Heater Level")
        ax2.legend(loc='upper left')

        status_text = ax1.text(0.02, 0.05, '', transform=ax1.transAxes)

        # Simulation Time Tracker
        global_sim_time_mins = 0

        def update(frame):
            nonlocal global_sim_time_mins
            global_sim_time_mins += 1

            # 1. Update Outside Temp (Sine wave for 24h cycle)
            # 1440 mins = 24 hours. Temp swings 0C to 10C.
            day_progress = (global_sim_time_mins % 1440) / 1440.0
            current_T_outside = 5.0 + 5.0 * np.sin(2 * np.pi * day_progress)

            # 2. Update Room Window
            room.set_window(state['window_open'])
            
            # 3. Calculate Error
            error = T_target - room.T_inside

            # 4. ANFIS DECISION
            if room.window_open:
                heater_power = 0.0
            else:
                heater_power = thermostat.forward(error, current_T_outside)
                # Force clip to 0-5 range just in case ANFIS goes crazy
                heater_power = max(0.0, min(5.0, heater_power))
                
                thermostat.adapt(error)

            # 5. PHYSICS UPDATE (1 Minute step)
            new_T_inside = room.update_temperature(current_T_outside, heater_power)

            # 6. LOGGING
            y_temp.pop(0)
            y_temp.append(new_T_inside)

            y_outside.pop(0)
            y_outside.append(current_T_outside)

            y_level.pop(0)
            y_level.append(heater_power)

            y_window.pop(0)
            y_window.append(0 if state['window_open'] else heater_power)

            line_temp.set_ydata(y_temp)
            line_outside.set_ydata(y_outside)
            line_level.set_ydata(y_level)
            line_window.set_ydata(y_window)

            status_text.set_text(
                f'Time: {global_sim_time_mins//60:02d}:{(global_sim_time_mins%60):02d} | '
                f'In: {new_T_inside:.2f}°C | Out: {current_T_outside:.2f}°C | '
                f'Heater: {heater_power:.2f}'
            )
            return line_temp, line_outside, line_level, line_window, status_text

        def toggle_window(event):
            state['window_open'] = not state['window_open']

        ax_btn = plt.axes([0.4, 0.05, 0.2, 0.075])
        btn = Button(ax_btn, 'Open/Close Window')
        btn.on_clicked(toggle_window)
        fig._btn_ref = btn 

        # FIX 2: Fast interval (20ms). 
        # This makes it look like "Fast Forward", updating 50 times per second.
        anim = FuncAnimation(fig, update, interval=20, blit=False)
        plt.show()




    else:
        print("Invalid mode selected. Choose 'static' or 'dynamic'.")


if __name__ == "__main__":
    mode = "static" 
    main(mode)