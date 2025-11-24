import numpy as np
import matplotlib.pyplot as plt


class Room:
    def __init__(self, T_target):
        self.T_inside = 20.0  # Initial inside temperature
        self.T_target = T_target
        self.insulation_loss = 0.1

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

    def update_temperature(self, T_outside, heater_power):
        """
        Simulates one minute of thermodynamics.
        power: 0-5 scale
        outside_temp: Degrees Celsius
        """
        # Heat loss to outside
        heat_loss = self.insulation_loss * (self.T_inside - T_outside)
        # Heat gain from heater
        heat_gain = self.heater_efficiency * heater_power
        # Update inside temperature
        self.T_inside += heat_gain - heat_loss
        return self.T_inside
    

class ANFISThermostat:
    def __init__(self, n_rules=5, learning_rate=0.01):
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
        self.consequent[:, 2] = 1.0    # Shape: (3, n_rules)

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
        

# --- Simulation ---
T_target = 22.0
room = Room(T_target)
thermostat = ANFISThermostat(n_rules=5, learning_rate=0.01)

n_minutes = 300
history_time = []
history_T_inside = []
history_T_outside = []
history_heater_power = []

current_T_outside = 5.0  # Initial outside temperature

print("Starting simulation...")

for t in range(n_minutes):

    # --- A. Scenario: Weather changes ---
    if t > 100:
        current_T_outside = 0.0  # Drop outside temp after 100 minutes
    if t > 200:
        current_T_outside = -5.0  # Further drop after 200 minutes
    current_T_outside = max(-10.0, current_T_outside)

    # --- B. Meassure ---
    error = T_target - room.T_inside

    # --- C. ANFIS Control ---
    # Inputs: Error, T_outside
    heater_power = thermostat.forward(error, current_T_outside)

    # --- D. Actuate ---
    new_T_inside = room.update_temperature(current_T_outside, heater_power)

    # --- E. Learn ---
    thermostat.adapt(error)

    # --- F. Log Data ---
    history_time.append(t)
    history_T_inside.append(new_T_inside)
    history_T_outside.append(current_T_outside)
    history_heater_power.append(heater_power)

# --- Plot Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Temperatures
ax1.plot(history_time, [T_target]*len(history_time), 'k--', label='Target (22°C)')
ax1.plot(history_time, history_T_inside, 'r-', linewidth=2, label='Inside Temp')
ax1.plot(history_time, history_T_outside, 'b-', alpha=0.6, label='Outside Temp')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title(f'ANFIS Adaptive Thermostat (Rules={thermostat.n_rules})')
ax1.legend()
ax1.grid(True)

# Plot Heater Output
ax2.plot(history_time, history_heater_power, 'g-', label='Heater Power (%)')
ax2.set_ylabel('Heater Power %')
ax2.set_xlabel('Time (minutes)')
ax2.set_title('Control Action')
ax2.fill_between(history_time, history_heater_power, color='green', alpha=0.1)
ax2.grid(True)

# Set Y-Axis to look like a Radiator Knob
ax2.set_ylim(0, 6)
ax2.set_yticks([0, 1, 2, 3, 4, 5])
ax2.fill_between(history_time, history_heater_power, color='green', alpha=0.1)
ax2.grid(True)

plt.tight_layout()
plt.show()
