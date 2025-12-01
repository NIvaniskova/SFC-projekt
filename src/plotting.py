import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def plot_lr_comparison(learning_rates, df):
    plt.figure(figsize=(10, 6))

    for lr in learning_rates:
        subset = df[df["Learning_Rate"] == lr]
        plt.scatter(subset["RMSE"], subset["Chattering"], label=f"LR={lr}", alpha=0.6)

    means = df.groupby("Learning_Rate")[["RMSE", "Chattering"]].mean()
    for lr in learning_rates:
        plt.scatter(means.loc[lr, "RMSE"], means.loc[lr, "Chattering"], 
                    s=200, marker='X', edgecolors='black', linewidth=2, label=f"Mean {lr}")

    plt.xlabel("RMSE (Accuracy) -> Lower is better")
    plt.ylabel("Chattering (Instability) -> Lower is better")
    plt.title("Pareto Front: Accuracy vs. Stability")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_heatmap_gridsearch(rmse_pivot, chat_pivot):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot A: Accuracy (RMSE)
    sns.heatmap(rmse_pivot, annot=True, fmt=".3f", cmap="viridis_r", ax=ax1, 
                cbar_kws={'label': 'RMSE (Lower is Better)'})
    ax1.set_title("Accuracy Heatmap (RMSE)")
    ax1.set_ylabel("Number of Rules (Complexity)")
    ax1.set_xlabel("Learning Rate (Speed)")

    # Plot B: Stability (Chattering)
    sns.heatmap(chat_pivot, annot=True, fmt=".2f", cmap="magma_r", ax=ax2,
                cbar_kws={'label': 'Chattering (Lower is Better)'})
    ax2.set_title("Stability Heatmap (Valve Wear)")
    ax2.set_ylabel("Number of Rules")
    ax2.set_xlabel("Learning Rate")

    plt.suptitle("Hyperparameter Grid Search Results")
    plt.tight_layout()
    plt.show()

def print_results_gridsearch(df):

    pd.options.display.float_format = '{:.4f}'.format

    rmse_table = df.pivot(index='Rules', columns='Learning_Rate', values='RMSE')
    chat_table = df.pivot(index='Rules', columns='Learning_Rate', values='Chattering')

    print("\n" + "="*40)
    print(f"{'RMSE MATRIX (Accuracy)':^40}")
    print("="*40)
    print(rmse_table)

    print("\n" + "="*40)
    print(f"{'CHATTERING MATRIX (Stability)':^40}")
    print("="*40)
    print(chat_table)

    top_results = df.sort_values(by="RMSE").head(3)

    print("\n" + "="*40)
    print(f"{'TOP 3 CONFIGURATIONS':^40}")
    print("="*40)
    print(top_results[['Rules', 'Learning_Rate', 'RMSE', 'Chattering']].to_string(index=False))