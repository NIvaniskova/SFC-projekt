import random
import numpy as np

random.seed(42)

def generate_window_events():

    # window_events = [
    #     (305, 365, "Window opened (60m)"),
    #     (640, 655, "Window opened (15m)"),
    #     (910, 955, "Window opened (45m)"),
    #     (1200, 1230, "Window opened (30m)")
    # ]

    total_minutes = 1440
    earliest_start = 7 * 60      # 07:00 -> 420
    latest_end = 22 * 60         # 22:00 -> 1320

    durations = [15, 30, 45, 60]
    num_events = random.randint(3, 5)
    events = []
    used_ranges = []

    for _ in range(num_events):
        duration = random.choice(durations)

        while True:
            start = random.randint(earliest_start, latest_end - duration)
            end = start + duration

            # prevent overlaps
            if all(not (start < e and end > s) for s, e in used_ranges):
                break

        #label = f"Window opened ({duration}m)"
        event = (start, end)

        events.append(event)
        used_ranges.append(event)

    events.sort(key=lambda x: x[0])
    return events


def generate_outdoor_temps_30mins(min_temp=0, max_temp=15, peak_hour=15, noise=0.3):

    # half_hourly_temps = [
    #     2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 3.0, 
    #     5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 14.5, 15.0, 15.0,
    #     15.0, 14.5, 14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 10.0, 9.0, 8.0, 6.5,
    #     6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 1.0
    # ]

    points = 48
    times = np.arange(points)

    mean = (max_temp + min_temp) / 2
    amplitude = (max_temp - min_temp) / 2

    peak_index = peak_hour * 2  # 30-min intervals

    temps = mean + amplitude * np.sin(
        2 * np.pi * (times - peak_index) / points + np.pi/2
    )

    temps += np.random.normal(0, noise, size=points)

    return [round(float(t), 1) for t in temps]

if __name__ == "__main__":

    n_runs = 100

    output_path_outside_temperatures = ("../data/outside_temperatures.txt")
    output_path_window_events = ("../data/window_events.txt")

    with open(output_path_outside_temperatures, "w", encoding="utf-8") as file:
        for _ in range(50):
            data = generate_outdoor_temps_30mins()
            file.write(str(data) + "\n")

    with open(output_path_window_events, "w", encoding="utf-8") as file:
        for _ in range(50):
            data = generate_window_events()
            file.write(str(data) + "\n")

    