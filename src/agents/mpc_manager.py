import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sparse
import osqp
from collections import deque
from src.agents.base_manager import BaseManager

class SysIDNet(nn.Module):
    """
    System Identification Neural Network (SysIDNet).
    
    A sequential deep learning model utilizing a Gated Recurrent Unit (GRU).
    Its objective is to perform real-time, online physical parameter estimation.
    By observing a rolling temporal history window of the vessel's movements
    (yaw rate 'r' and commanded rudder angle 'delta'), it learns to predict
    the underlying, randomized physical parameters of the ship:
    - K (Ship turning gain / steering responsiveness)
    - T (Time constant / rotational inertia)
    
    A Softplus activation is applied to the final regression head to mathematically
    guarantee that the predicted K and T are strictly positive values, preventing
    physics division-by-zero errors in downstream controllers.
    """
    def __init__(self, hidden_sizes):
        super().__init__()
        # Input features: [yaw_rate, rudder_angle]
        gru_hidden = hidden_sizes[0]
        linear_hidden = hidden_sizes[1]
        
        self.gru = nn.GRU(input_size=2, hidden_size=gru_hidden, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(gru_hidden, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, 2),
            nn.Softplus() # Guarantees positive output for K and T
        )

    def forward(self, history):
        # history shape: [batch_size, sequence_length, 2]
        gru_out, _ = self.gru(history)
        last_out = gru_out[:, -1, :] # Take the hidden state from the final sequence step
        predictions = self.regressor(last_out)
        return predictions # [K_hat, T_hat]

class MPC:
    """
    Model Predictive Control (MPC) Solver.
    
    A receding horizon optimal controller formulating the course-keeping problem as a
    Quadratic Program (QP) solved via the OSQP solver.
    
    At every timestep, it uses the discrete linear state-space representation of the
    1st-order Nomoto model in augmented, delta-input form:
    x[t+1] = A x[t] + B u[t]

    State  x = [psi (heading error), r (yaw rate), integral_psi (accumulated heading error), delta_{t-1} (previous rudder)]
    Control u = [v (change in rudder angle, i.e. Delta delta)]

    Using the predicted K_hat and T_hat from SysIDNet, it solves the QP over a prediction
    horizon 'N' to minimize the quadratic performance cost:
    J = sum(w1*psi^2 + w2*delta^2 + w3*v^2)
    (heading error via w1, physical rudder angle via w2 on the augmented state, rudder rate via w3 on the control;
    yaw rate r and integral_psi are not directly penalized in the objective)

    The solver strictly respects physical hardware constraints on the rudder angle boundaries
    [-bound_rudder_angle_rad, +bound_rudder_angle_rad].
    It executes the very first optimal action from the computed horizon sequence
    and recalculates the entire trajectory at the next step.
    """
    def __init__(self, cfg_env, cfg_mpc):
        self.cfg_env = cfg_env
        self.cfg_mpc = cfg_mpc
        self.N = self.cfg_mpc.prediction_horizon
        self.dt = self.cfg_env.dt
        
        # We will initialize the OSQP solver dynamically when K and T are predicted
        self.solver = None

    def setup_matrices(self, K_hat, T_hat):
        """
        Formulate discrete state-space matrices using State Augmentation (Delta-Input MPC).
        x[t+1] = A x[t] + B u[t]
        
        State x = [psi, r, integral_psi, delta_{t-1}]  (nx = 4)
        Control u = [v_t] (where v_t is the change in rudder angle, Delta delta)
        """
        dt2 = self.dt ** 2
        dt3 = self.dt ** 3

        # State transition matrix for augmented state vector
        A = np.array([
            [1.0, self.dt - (dt2 / T_hat), 0.0, (K_hat * dt2) / T_hat],      # Row 1: psi[t+1]
            [0.0, 1.0 - (self.dt / T_hat), 0.0, (K_hat * self.dt) / T_hat],  # Row 2: r[t+1]
            [self.dt, dt2 - (dt3 / T_hat), 1.0, (K_hat * dt3) / T_hat],      # Row 3: integral_psi[t+1]
            [0.0, 0.0, 0.0, 1.0]                                             # Row 4: delta[t] = delta[t-1] + v_t
        ])
        
        # Control input matrix (for v_t)
        B = np.array([
            [(K_hat * dt2) / T_hat],       # u_t directly affects psi
            [(K_hat * self.dt) / T_hat],   # u_t directly affects r
            [(K_hat * dt3) / T_hat],       # u_t directly affects integral_psi
            [1.0]                          # u_t (v_t) has a 1.0 immediate impact on delta[t]
        ])
        return A, B

    def solve(self, x0, u_prev, K_hat, T_hat):
        """
        Formulate and solve the QP problem for the receding horizon using the delta-input formulation.
        """
        A, B = self.setup_matrices(K_hat, T_hat)

        nx = A.shape[0]  # State dimension (now 4)
        nu = B.shape[1]  # Control dimension (now 1, v_t)
        
        # Objective matrices: J = x^T Q x + u^T R u
        # We penalize heading error (w1) and physical rudder angle (w2), which is now state 4!
        Q = sparse.diags([
            self.cfg_env.w1_heading_error, 
            0.0, 
            0.0,
            self.cfg_env.w2_rudder_angle
        ])
        QN = Q # Simple terminal heuristic

        # Control penalty matrix R now ONLY penalizes v_t (the rudder difference w3)
        R = sparse.diags([self.cfg_env.w3_rudder_rate])

        # Hessian Matrix
        P = sparse.block_diag([
            sparse.kron(sparse.eye(self.N), Q), 
            QN,
            sparse.kron(sparse.eye(self.N), R)
        ], format='csc')
        
        # Linear objective term is set to zero since we only care about quadratic penalties
        q = np.zeros((self.N + 1) * nx + self.N * nu)

        # Equality Constraints: x[t+1] = A x[t] + B u[t]
        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(self.N + 1, k=-1), A)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), B)
        A_eq = sparse.hstack([Ax, Bu])
        
        # Initial State Equality Constraint: x[0] = x0
        l_eq = np.zeros((self.N + 1) * nx)
        x0_augmented = np.array([x0[0], x0[1], x0[2], u_prev])
        l_eq[:nx] = -x0_augmented
        u_eq = l_eq # strict equality

        # Inequality Constraints: 
        # 1. Hardware bounds on physical Rudder Angle (which is now state #4)
        # 2. Hardware bounds on Rudder Rate (which is our control u_k)
        
        # Extract state #4 from all steps in the horizon.
        # Since we have N + 1 states in the trajectory, we multiply N+1 by the state vector selector
        state_4_extractor = sparse.kron(sparse.eye(self.N + 1), sparse.csc_matrix([0, 0, 0, 1]))
        
        # Extract u_k from all steps
        # This matches the shape of the decision vector variables (N+1)*nx + N*nu
        control_extractor = sparse.hstack([
            sparse.csc_matrix((self.N * nu, (self.N + 1) * nx)), 
            sparse.eye(self.N * nu)
        ])
        
        # Combine state constraint extractor and control constraint extractor.
        # They must have the same column width, which is the dimension of the decision vector: (N+1)*nx + N*nu
        state_extractor_augmented = sparse.hstack([
            state_4_extractor,
            sparse.csc_matrix(((self.N + 1), self.N * nu))
        ])
        
        A_ineq = sparse.vstack([state_extractor_augmented, control_extractor])
        
        # Bound vectors
        rudder_max = self.cfg_env.bound_rudder_angle_rad
        rate_max = rudder_max # Optional: if you want to bound how fast the rudder can move per step
        
        l_ineq = np.hstack([
            np.full((self.N + 1), -rudder_max),
            np.full(self.N * nu, -rate_max)
        ])
        u_ineq = np.hstack([
            np.full((self.N + 1), rudder_max),
            np.full(self.N * nu, rate_max)
        ])

        # Stack constraints
        A_osqp = sparse.vstack([A_eq, A_ineq], format='csc')
        l_osqp = np.hstack([l_eq, l_ineq])
        u_osqp = np.hstack([u_eq, u_ineq])

        # Initialize OSQP solver
        self.solver = osqp.OSQP()
        self.solver.setup(P=P, q=q, A=A_osqp, l=l_osqp, u=u_osqp, verbose=False)

        res = self.solver.solve()
        
        if res.info.status != 'solved':
            return None # Solver failed
            
        # Extract the sequence of optimal Delta delta (v_t)
        v_opt = res.x[-self.N * nu:]
        
        # The true optimal rudder angle to command is the current rudder + the optimal change
        optimal_rudder_angle = u_prev + v_opt[0]
        return optimal_rudder_angle

class MPCManager(BaseManager):
    """
    Manager for the Two-Stage SysID Neural Network + MPC agent.
    """
    def build_model(self, verbose=None):
        if self.env is None:
            self._create_env(n_envs=1)
            
        # Instantiate the SysID Net and Optimizer using the hidden_sizes tuple from config
        self.sysid_net = SysIDNet(hidden_sizes=self.cfg.rl.sysid.hidden_sizes)
        self.optimizer = optim.Adam(self.sysid_net.parameters(), lr=self.cfg.rl.sysid.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Instantiate the mathematical OSQP MPC Solver wrapper
        self.mpc_solver = MPC(self.cfg.env, self.cfg.rl.mpc)

    def load_model(self, path):
        if self.env is None:
            self._create_env(n_envs=1)
        self.build_model()
        print(f"Loading SysID network weights from {path}.pth...")
        self.sysid_net.load_state_dict(torch.load(f"{path}.pth", weights_only=True))

    def save_model(self):
        candidate_path = self.get_save_path(extension=".pth")
        torch.save(self.sysid_net.state_dict(), f"{candidate_path}.pth")
        print(f"SysID Neural Network saved successfully to: {candidate_path}.pth")

    def train(self, save=True):
        if not hasattr(self, 'sysid_net'):
            raise ValueError("Model not built! Call build_model() first.")
            
        print(f"Stage 1: Collecting {self.cfg.rl.sysid.excitation_steps} steps of random-excitation data...")
        dataset_x = []
        dataset_y = []
        
        # 1. Collect Random Excitation Data
        obs = self.env.reset()
        history_buffer = []
        history_len = self.cfg.rl.sysid.history_length
        
        for step in range(self.cfg.rl.sysid.excitation_steps):
            # Pure random noise controller to train for ship physics
            # sample() returns an array of shape (1,)
            action = np.array([self.env.action_space.sample()]) 
            next_obs, reward, done, info = self.env.step(action)
            
            # obs is [[psi, r, delta, int_psi]]
            # Append (yaw_rate, rudder) using the rudder stored in the observation so the pairing
            # matches evaluation: each yaw rate is paired with the rudder that actually produced it.
            history_buffer.append([obs[0][1], obs[0][2]])
            
            # Maintain rolling window
            if len(history_buffer) > history_len:
                history_buffer.pop(0)
                
            # If buffer is full, we can save a valid training sample
            if len(history_buffer) == history_len:
                dataset_x.append(list(history_buffer))
                dataset_y.append([info[0]["K"], info[0]["T"]]) # True parameters extracted from env info
                
            if done[0]:
                history_buffer.clear()
            
            obs = next_obs
            
        print("Data collection complete. Training SysID Neural Network...")
        
        # 2. Supervised Learning Loop
        X = torch.tensor(dataset_x, dtype=torch.float32)
        Y = torch.tensor(dataset_y, dtype=torch.float32)
        
        # Create minibatches
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(self.cfg.rl.sysid.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.sysid_net(batch_x)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % self.cfg.rl.sysid.log_interval == 0:
                print(f"Epoch {epoch}/{self.cfg.rl.sysid.epochs} - MSE Loss: {epoch_loss/len(dataloader):.6f}")

        print("SysID Training Complete!")
        if save:
            self.save_model()

    def evaluate(self, n_episodes=None, render=False):
        if not hasattr(self, 'sysid_net'):
            raise ValueError("SysID Model not loaded! Call load_model() first.")
            
        episodes_to_run = (
            n_episodes
            if n_episodes is not None
            else self.cfg.rl.eval.get("n_eval_episodes", 10)
        )
            
        print(f"Evaluating SysID+MPC over {episodes_to_run} episodes...")
        total_rewards = []
        
        for ep in range(episodes_to_run):
            obs = self.env.reset()
            ep_reward = 0.0
            history_buffer = []
            history_len = self.cfg.rl.sysid.history_length
            
            while True:
                # 1. Fill history buffer (Zero-padding if episode just started)
                current_r = obs[0][1]
                current_delta = obs[0][2]
                history_buffer.append([current_r, current_delta])
                if len(history_buffer) > history_len:
                    history_buffer.pop(0)
                    
                pad_len = history_len - len(history_buffer)
                padded_history = [[0.0, 0.0]] * pad_len + history_buffer
                
                # 2. Predict K and T
                with torch.no_grad():
                    hist_tensor = torch.tensor([padded_history], dtype=torch.float32)
                    predictions = self.sysid_net(hist_tensor)
                    K_hat, T_hat = predictions[0].numpy()
                
                # 3. Formulate State and Solve MPC
                # OSQP state: [psi, r, integral_psi]
                state_vec = np.array([obs[0][0], obs[0][1], obs[0][3]])
                
                # u_prev is the current physical rudder angle (obs[0][2])
                u_prev = obs[0][2]
                
                optimal_action_rad = self.mpc_solver.solve(state_vec, u_prev, K_hat, T_hat)
                
                # If solver failed to find solution, take 0 action to coast safely
                if optimal_action_rad is None:
                    optimal_action_rad = 0.0
                    
                # 4. Map physical radians back to [-1, 1] normalized action space for the Gym env
                normalized_action = optimal_action_rad / self.cfg.env.bound_rudder_angle_rad
                normalized_action = np.clip(normalized_action, -1.0, 1.0)
                
                # 5. Step Environment
                # SB3 vectorized env step expects action shape to be [batch, action_dim], i.e., [[action_val]]
                action_to_step = np.array([[normalized_action]])
                obs, reward, done, info = self.env.step(action_to_step)
                ep_reward += reward[0]
                
                if done[0]:
                    break
                    
            total_rewards.append(ep_reward)
            
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"Evaluation Results: Mean Reward = {mean_reward:.2f}, Standard Deviation = {std_reward:.2f}")
        return mean_reward

    def optimize_hyperparameters(self):
        print("Hyperparameter optimization for SysID is not strictly necessary (supervised learning task).")
        pass
