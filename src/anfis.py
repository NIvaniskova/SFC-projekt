import numpy as np
import matplotlib.pyplot as plt


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
        heat_loss = self.insulation_loss * (self.T_inside - T_outside)
        heat_gain = self.heater_efficiency * heater_power
        self.T_inside += heat_gain - heat_loss
        return self.T_inside
    

class ANFISThermostat:
    def __init__(self, n_rules=5, learning_rate=0.01, initial_heater_power=3.0):
        self.n_rules = n_rules
        self.learning_rate = learning_rate
        self.n_inputs = 2  

        # --- Initualize parameters ---
        # Membership function parameters: (n_inputs, n_rules, 2) for Gaussian MF (c, sigma)
        self.mu = np.random.uniform(-5, 10, (self.n_rules, self.n_inputs))
        self.sigma = np.random.uniform(2, 10, (self.n_rules, self.n_inputs))

        # Consequent parameters: (n_rules, n_inputs + 1) for linear function (p0, p1, bias)
        self.consequent = np.zeros((self.n_rules, self.n_inputs + 1))  
        self.consequent[:, 2] = initial_heater_power    # Shape: (n_rules, 3)

    def gaussian_mf(self, x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    

    def forward(self, error, T_outside):
        self.x_input = np.array([error, T_outside])  # Shape: (2,)

        # --- Fuzzification ---
        self.mu_values = self.gaussian_mf(self.x_input, self.mu, self.sigma)  # Shape: (n_inputs, n_rules)

        # --- Rule Evaluation ---
        self.w = np.prod(self.mu_values, axis=1)  # Shape: (n_rules,)

        # --- Normalization ---
        self.w_sum = np.sum(self.w) + 1e-6  
        self.w_normalized = self.w / self.w_sum  # Shape: (n_rules,)

        # --- Linear Output ---
        x_bias = np.append(self.x_input, 1)  # Shape: (3,) [Error, T_outside, 1]
        self.linear_output = np.dot((self.consequent), x_bias)  # Shape: (n_rules,)

        # --- Aggregation ---
        raw_output = np.dot(self.w_normalized, self.linear_output)
        heater_power = np.clip(raw_output, 0, 5)    # Clip to (0-5) that corresponds to heater valve levels
        return heater_power
    

    def adapt(self, system_error):
        
        # Noise influence prevention - system reacts after some threshold value is crossed
        if abs(system_error) < 0.1:
            return 
        
        # Exploding gradients prevention - clip system error to prevent uncontrollable weights growing
        learning_error = np.clip(system_error, -2.0, 2.0)
 
        
        x_bias = np.append(self.x_input, 1) # p0*Error + p1*T_outside + 1*bias
    
        for r in range(self.n_rules):
            # Consequent update
            grad = self.w_normalized[r] * x_bias   
            self.consequent[r] += self.learning_rate * learning_error * grad  

            # Membership functions update
            if self.w[r] > 0.05:    # Rules with weak firing strengths are omitted
                diff = self.x_input - self.mu[r]   
                self.mu[r] += self.learning_rate * system_error * diff  
