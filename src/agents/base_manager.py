import os
from datetime import datetime
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from src.env import NomotoEnv
from src.visualize import animate_trajectory

class BaseManager:
    """
    Base Manager class defining the template setup and interface 
    for all Reinforcement Learning and Control agents.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = None
        self.model = None
        
        # Setup paths
        self.run_dir = f"./tensorboard_runs/{self.cfg.project_name}"
        self.model_dir = f"./models/{self.cfg.project_name}"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_base_name = self.cfg.model_name

        # Generate independent child seeds
        seed_seq = np.random.SeedSequence(self.cfg.seed)
        env_seq, algo_seq, optuna_seq = seed_seq.spawn(3)
        self.env_seed = int(env_seq.generate_state(1)[0])
        self.algo_seed = int(algo_seq.generate_state(1)[0])
        self.optuna_seed = int(optuna_seq.generate_state(1)[0])

    def _create_env(self, n_envs=1):
        """Internal helper to instantiate vectorized Nomoto environments."""
        self.env = make_vec_env(
            lambda: NomotoEnv(self.cfg.env), 
            n_envs=n_envs,
            seed=self.env_seed
        )

    def get_model_path(self):
        """
        Canonical base path (without extension) for the model identified by `model_name`,
        i.e. models/{project_name}/{model_name}. Used for both saving and loading; saving
        with an existing name overwrites it.
        """
        return os.path.join(self.model_dir, self.model_base_name)

    # --- Abstract Methods (Children must implement these) ---
    def build_model(self, verbose=None):
        raise NotImplementedError

    def load_model(self, path=None):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def train(self, save=True):
        raise NotImplementedError

    def evaluate(self, n_episodes=None, render=False):
        raise NotImplementedError

    def optimize_hyperparameters(self):
        raise NotImplementedError

    def _rollout_episode(self):
        """
        Run a single episode with the (already loaded) controller and return its trajectory:
        a list of per-step dicts with keys {psi, r, delta, integral_psi, reward, K, T}.
        Children implement this using their own action-selection logic.
        """
        raise NotImplementedError

    def visualize(self, agent_name, sysid=False, save=True, show=True):
        """
        Animate one episode: a 2D top-down boat path plus synced psi/delta/r time-series.
        Requires a loaded model (call load_model first). When `save` is True, the animation is
        written as a GIF to visuals/{project_name}/{model_name}/ before the live window opens.

        Args:
            agent_name: label for the title/filename (e.g. "ppo", "sysid_mpc").
            sysid: if True, also plot the SysID network's predicted K/T vs the true K/T.
        """
        traj = self._rollout_episode()

        save_path = None
        if save:
            out_dir = os.path.join("visuals", self.cfg.project_name, self.cfg.model_name)
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(out_dir, f"{agent_name}_{timestamp}.gif")

        animate_trajectory(
            traj,
            dt=self.cfg.env.dt,
            rudder_bound=self.cfg.env.bound_rudder_angle_rad,
            agent_name=agent_name,
            save_path=save_path,
            show=show,
            sysid=sysid,
        )

        if save_path is not None:
            print(f"Saved visualization to: {save_path}")
