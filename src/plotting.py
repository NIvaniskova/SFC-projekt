import numpy as np
import matplotlib.pyplot as plt


def plot_one_run_static(T_target, history_time, history_T_inside, history_T_outside, history_heater_power, window_events):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        hours_ticks = np.arange(0, 25, 2) 

        # Plot temperatures
        ax1.plot(history_time, [T_target]*len(history_time), 'k--', label='Target (22°C)')
        ax1.plot(history_time, history_T_inside, 'r-', linewidth=2, label='Inside Temperature')
        ax1.plot(history_time, history_T_outside, 'b-', alpha=0.6, label='Outside Temperature')

        for start, end in window_events:
            ax1.axvspan(start/60, end/60, color='gray', alpha=0.3)
            ax1.text((start+end)/120, 10, s='window open', ha='center', fontsize=8, rotation=90)

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

        for start, end in window_events:
            ax2.axvspan(start/60, end/60, color='gray', alpha=0.3)
            ax2.text((start+end)/120, 2, ha='center', s='window open', fontsize=8, rotation=90)

        plt.tight_layout()
        plt.show()

