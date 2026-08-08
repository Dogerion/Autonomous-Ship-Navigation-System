import os
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from src.env import NomotoEnv

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
        self.run_dir = f"./runs/{self.cfg.project_name}"
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

    def get_save_path(self, extension=".zip"):
        """
        Helper method to generate a safe path with an incrementing suffix 
        to avoid overwriting existing models.
        Returns the base path (without extension) suitable for SB3 or PyTorch.
        """
        candidate_path = os.path.join(self.model_dir, self.model_base_name)
        if os.path.exists(f"{candidate_path}{extension}"):
            counter = 1
            while os.path.exists(f"{candidate_path}_{counter}{extension}"):
                counter += 1
            candidate_path = f"{candidate_path}_{counter}"
        return candidate_path

    # --- Abstract Methods (Children must implement these) ---
    def build_model(self, verbose=None):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def train(self, save=True):
        raise NotImplementedError

    def evaluate(self, n_episodes=None, render=False):
        raise NotImplementedError

    def optimize_hyperparameters(self):
        raise NotImplementedError
