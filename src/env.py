import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NomotoEnv(gym.Env):
    """
    Custom Gymnasium Environment utilizing the first-order Nomoto model for vessel kinematics:
    T * r_dot + r = K * delta
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Action space: continuous commanded rudder angle (rad), scaled between max and min rudder angles (rad)
        bound_rudder_angle = self.config.bound_rudder_angle_rad
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space: [heading_error (rad), yaw_rate (rad/s), rudder_angle (rad), integral_heading_error (rad*s)]
        # Heading error, yaw rate and the integral term are unbounded accumulators (the physics never
        # clips them), so they are declared as infinite to guarantee observations stay within the space.
        # Only the rudder angle is a genuinely bounded, hardware-limited quantity.
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -bound_rudder_angle, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, bound_rudder_angle, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # Vessel state variables (initialized dynamically during reset)
        self.psi: float          # yaw / heading error (radians)
        self.r: float            # yaw rate (radians / second)
        self.cmd_rudder: float   # physical rudder angle (radians)
        self.integral_psi: float # accumulated heading error (radians * seconds)
        self.K: float            # Ship's Turning Gain
        self.T: float            # Inertia of the Ship (Time Constant)
        self.steps: int          # Episode step counter
        self.stable_steps: int   # Counter for consecutive steps within success tolerance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 1. Randomize K and T using uniform distribution
        self.K = self.np_random.uniform(self.config.K_range[0], self.config.K_range[1])
        self.T = self.np_random.uniform(self.config.T_range[0], self.config.T_range[1])
        
        # 2. Reset ship state
        init_heading_rad = self.config.init_heading_error_rad
        self.psi = self.np_random.uniform(-init_heading_rad, init_heading_rad) # initial heading deviation
        self.integral_psi = 0.0
        self.r = 0.0        
        self.cmd_rudder = 0.0           
        self.steps = 0
        self.stable_steps = 0
        
        return self._get_obs(), {
            "K": self.K,             # Current randomized turning gain
            "T": self.T,             # Current randomized time constant
            "true_heading_deg": np.degrees(self.psi)
        }

    def step(self, action):
        # 1. Get action in radians (scaled from action space low/high of [-1, 1])
        cmd_rudder_rad = action[0] * self.config.bound_rudder_angle_rad
        
        # Track difference in rudder commands (delta) for wear and tear penalty
        prev_cmd = self.cmd_rudder
        self.cmd_rudder = cmd_rudder_rad
        delta = self.cmd_rudder - prev_cmd
        
        # 2. Physics update
        # Nomoto first-order model: T * r_dot + r = K * cmd_rudder
        r_dot = (self.K * self.cmd_rudder - self.r) / self.T   # Compute yaw acceleration (rad/s^2)
        self.r += r_dot * self.config.dt                       # Update yaw rate (rad/s)
        self.psi += self.r * self.config.dt                    # Update heading error (rad)
        
        # Cummulative heading error
        self.integral_psi += self.psi * self.config.dt

        # 3. Add Gaussian wave disturbance to yaw rate
        if self.config.yaw_rate_noise_flag:
            self.r += self.np_random.normal(0, self.config.yaw_rate_noise_std)
        
        # Check if heading error is within success tolerance
        if abs(self.psi) <= self.config.bound_trial_rad:
            self.stable_steps += 1
        else:
            self.stable_steps = 0  # Must be consecutive!
        
        # 4. Compute reward & done flags
        self.steps += 1
        is_success = self.stable_steps >= self.config.step_trial
        terminated = (self.steps >= self.config.max_steps) or is_success
        truncated = False
        
        # Quadratic reward formulation: R_t = -(w1*psi^2 + w2*cmd_rudder^2 + w3*delta^2)
        # Note: delta here is purely the absolute difference (delta = cmd_t - cmd_{t-1}), no dt division.
        reward = -(
            self.config.w1_heading_error * (self.psi ** 2) +
            self.config.w2_rudder_angle * (self.cmd_rudder ** 2) +
            self.config.w3_rudder_rate * (delta ** 2)
        )
        
        return self._get_obs(), float(reward), terminated, truncated, {
            "K": self.K,             # Current turning gain
            "T": self.T,             # Current time constant
            "true_heading_deg": np.degrees(self.psi),
            "is_success": is_success
        }

    def _get_obs(self):
        return np.array([self.psi, self.r, self.cmd_rudder, self.integral_psi], dtype=np.float32)
