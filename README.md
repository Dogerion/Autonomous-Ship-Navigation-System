# MARIN: Hybrid Autonomous Surface Vessel (ASV) Navigation System

MARIN is a state-of-the-art hybrid autonomous surface vessel navigation platform. Designed for high-fidelity maritime research, this project focuses on robust **course-keeping under stochastic wave disturbances and heavy ship dynamics domain randomization**.

The platform is designed around the **Strategy Design Pattern**, allowing researchers to run fair head-to-head benchmarks between two advanced control architectures:
1.  **Implicit Adaptive Reinforcement Learning**: A Recurrent PPO (`RecurrentPPO`) agent with an LSTM neural network brain that implicitly learns to identify and adapt to ship dynamics over time.
2.  **Explicit Adaptive Model Predictive Control (MPC)**: A modular two-stage controller consisting of a Gated Recurrent Unit (GRU) System Identification network (`SysIDNet`) for online parameter estimation, coupled to an augmented-state, delta-input receding horizon Linear Quadratic Quadratic Programming (LQP) solver (`OSQP`).

---

## Key Features & Mathematical Innovations

### 1. Unified Physics Sandbox (Gymnasium)
The vessel kinematics strictly follow the discrete-time approximation of the **First-Order Nomoto Model**:
$$ T \dot{r} + r = K \delta $$
Where $r$ is the yaw rate, $\delta$ is the rudder angle, $K$ is the steering gain, and $T$ is the ship's turning inertia.
*   **Domain Randomization**: Every episode reset, $K$ and $T$ are sampled from uniform distributions, forcing controllers to adapt dynamically to completely different vessels (ranging from speedboats to heavy tankers).
*   **Integrated State ($\int \psi$)**: The environment observation space is augmented with the time-integral of the heading error ($\int \psi$) to natively eliminate steady-state offsets under constant side-currents or waves.
*   **Wave Disturbances**: Stochastic Gaussian noise is injected into the yaw rate calculation at every step to simulate dynamic ocean waves.

### 2. State-Augmented Delta-Input MPC
Unlike standard MPC implementations which suffer from complex tridiagonal Hessian matrices and non-zero linear objective vectors due to actuator rate penalties ($\Delta \delta^2$), our custom MPC solver uses **augmented state space representation**:
$$ x_k = \begin{bmatrix} \psi_k \\ r_k \\ I_{\psi, k} \\ \delta_{k-1} \end{bmatrix}, \quad u_k = \begin{bmatrix} v_k \end{bmatrix} $$
Where the control input is the *change* in rudder command $v_k = \Delta \delta_k$.
*   **Mathematical Elegance**: This formulation completely absorbs the rudder rate penalties into the state variables, reducing the Hessian $P$ to a **pure diagonal block matrix** and the linear cost vector $q$ to exactly **$\vec{0}$**.
*   **Causality Alignment**: The state transition matrix $A$ and control matrix $B$ are fully coupled to perfectly mirror the discrete integration steps of the Gymnasium environment, eliminating model-mismatch oscillations.

### 3. MLOps & Rigorous Scientific Benchmarking
*   **Strategy & Factory Patterns**: The local local module structure under `src/agents/` ensures both agents share a parent `BaseManager` which orchestrates TensorBoard logs, checkpoint directory setups, and deterministic seeding.
*   **Mathematical Reproducibility**: Using NumPy's `SeedSequence`, a single master seed generates mathematically non-overlapping child seeds for environments, model weights, and optimization studies, eliminating "lucky seed" bias.
*   **Optuna Hyperparameter Sweeping**: Ready-to-use hyperparameter optimization pipeline with integrated pruning and customizable search parameters.

---

## Directory Structure

```text
MARIN/
├── pyproject.toml          # Project dependencies (managed by uv)
├── main.py                 # Core entry point (routes commands)
├── README.md               # You are here!
├── conf/                   # Hydra hierarchical configs
│   ├── config.yaml         # Master config & routing parameters
│   └── rl/
│       ├── ppo.yaml        # SB3 RecurrentPPO parameters
│       └── sysid_mpc.yaml  # PyTorch GRU & OSQP MPC parameters
├── docs/                   # Mathematical documentation
│   ├── nomoto_model.md     # Nomoto physics primer
│   └── mpc_formulation.md  # Continuous-to-discrete QP proof
├── src/                    # Source application code
│   ├── __init__.py         
│   ├── env.py              # Custom Gymnasium vessel environment
│   ├── utils.py            # Dynamic factory router utilities
│   └── agents/             # Modular agent managers
│       ├── __init__.py
│       ├── base_manager.py # Shared MLOps parent manager
│       ├── ppo_manager.py  # SB3 RecurrentPPO execution manager
│       └── mpc_manager.py  # SysID NN + OSQP MPC execution manager
```

---

## Setup & Installation

This project uses the modern, lightning-fast Python package installer **`uv`**.

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Dogerion/MARIN.git
    cd MARIN
    ```
2.  **Synchronize Dependencies**:
    Create the virtual environment and install all packages in `pyproject.toml` with a single command:
    ```bash
    uv sync
    ```
3.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

---

## Usage Guide

You can run training, evaluation, and optimization pipelines for either agent directly from the command line with zero hardcoded arguments.

### 1. Implicit Recurrent PPO (RL)
*   **Train PPO**:
    ```bash
    python main.py rl=ppo agent_type=ppo mode=train
    ```
*   **Evaluate Trained PPO**:
    ```bash
    python main.py rl=ppo agent_type=ppo mode=eval load_path=models/autonomous-vessel-rl/ppo_lstm_vessel
    ```
*   **Tune PPO Hyperparameters (Optuna)**:
    ```bash
    python main.py rl=ppo agent_type=ppo mode=optimize
    ```

### 2. Explicit SysID + State-Augmented MPC
*   **Train SysID Neural Network**:
    Saves PyTorch weights after collecting PRBS random excitation ship data:
    ```bash
    python main.py rl=sysid_mpc agent_type=sysid_mpc mode=train
    ```
*   **Evaluate SysID + MPC Closed Loop**:
    Sparsely resolves optimal rudder commands recursively utilizing the GRU predictions:
    ```bash
    python main.py rl=sysid_mpc agent_type=sysid_mpc mode=eval load_path=models/autonomous-vessel-rl/ppo_lstm_vessel
    ```

---

## Monitoring & Visualization

All training metrics and evaluation trajectories are logged to **TensorBoard**.

1.  Start the TensorBoard server:
    ```bash
    tensorboard --logdir ./runs/
    ```
2.  Open your browser and navigate to **`http://localhost:6006`** to compare heading convergence, rudder wear, and success rates.

---

## Detailed Scientific Documentation

For a comprehensive mathematical breakdown of the system mechanics, see:
*   [docs/nomoto_model.md](./docs/nomoto_model.md) — Under the hood of the continuous-to-discrete physics.
*   [docs/mpc_formulation.md](./docs/mpc_formulation.md) — The complete QP proof and augmented-state matrix derivations.
