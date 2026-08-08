import os
import optuna
import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from src.env import NomotoEnv

class RLManager:
    """
    RL manager class to handle the RecurrentPPO agent on the NomotoEnv.
    Handles environment creation, training, evaluation, saving, loading, and hyperparameter optimization.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = None
        self.model = None
        
        # Setup paths (TensorBoard runs, saved models) based on project config
        self.run_dir = f"./runs/{self.cfg.project_name}" # TensorBoard logs
        self.model_dir = f"./models/{self.cfg.project_name}" # Saved models
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_base_name = self.cfg.model_name

        # Generate independent child seeds for environment, algorithm, and hyperparameter optimization
        seed_seq = np.random.SeedSequence(self.cfg.seed)
        env_seq, ppo_seq, optuna_seq = seed_seq.spawn(3)
        self.env_seed = int(env_seq.generate_state(1)[0])
        self.ppo_seed = int(ppo_seq.generate_state(1)[0])
        self.optuna_seed = int(optuna_seq.generate_state(1)[0])

    def _create_env(self, n_envs=1):
        """Internal helper to instantiate vectorized Nomoto environments."""
        self.env = make_vec_env(
            lambda: NomotoEnv(self.cfg.env), 
            n_envs=n_envs,
            seed=self.env_seed
        )

    def build_model(self, verbose=None):
        """Instantiates a fresh RecurrentPPO model based on config."""
        if self.env is None:
            self._create_env()

        # Determine verbosity (allow override during hyperparameter optimization)
        model_verbose = verbose if verbose is not None else self.cfg.rl.get("verbose", 1)

        # Build the RecurrentPPO model with the specified hyperparameters
        self.model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=self.env,
            learning_rate=self.cfg.rl.learning_rate,
            n_steps=self.cfg.rl.n_steps,
            batch_size=self.cfg.rl.batch_size,
            n_epochs=self.cfg.rl.n_epochs,
            gamma=self.cfg.rl.gamma,
            gae_lambda=self.cfg.rl.gae_lambda,
            clip_range=self.cfg.rl.clip_range,
            ent_coef=self.cfg.rl.ent_coef,
            tensorboard_log=self.run_dir,
            verbose=model_verbose,
            seed=self.ppo_seed
        )

    def load_model(self, path):
        """Loads a pre-trained RecurrentPPO model from disk."""
        if self.env is None:
            self._create_env()
        print(f"Loading model from {path}...")
        self.model = RecurrentPPO.load(path, env=self.env)

    def save_model(self):
        """
        Saves the current model to disk with a unique filename to avoid overwriting.
        """
        if self.model is None:
            raise ValueError("No model is initialized to save!")
            
        # Standardize base path (SB3 saves as .zip automatically, so we don't append it here)
        candidate_path = os.path.join(self.model_dir, self.model_base_name)
        
        # Check if the file already exists and append a suffix if necessary
        if os.path.exists(f"{candidate_path}.zip"):
            counter = 1
            # Loop until we find a suffix index that does not exist
            while os.path.exists(f"{candidate_path}_{counter}.zip"):
                counter += 1
            candidate_path = f"{candidate_path}_{counter}"

        # Save model
        self.model.save(candidate_path)
        print(f"Model saved successfully to: {candidate_path}.zip")

    def train(self, save=True):
        """
        Executes the main training loop.
        """
        if self.model is None:
            raise ValueError("Model not built! Call build_model() first.")
            
        print(f"Starting training for {self.cfg.rl.total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=self.cfg.rl.total_timesteps,
            log_interval=self.cfg.rl.log_interval,
            tb_log_name=self.model_base_name
        )
        
        # Save the model after training completes
        if save:
            self.save_model()

    def evaluate(self, n_episodes=None, render=False):
        """
        Evaluates the loaded model on the environment without updating weights.
        """
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() must be called first.")
            
        # Determine the number of episodes (falling back to config)
        episodes_to_run = (
            n_episodes
            if n_episodes is not None
            else self.cfg.rl.eval.get("n_eval_episodes", 10)
        )
            
        print(f"Evaluating model over {episodes_to_run} episodes...")
        mean_reward, std_reward = evaluate_policy(
            self.model, self.env, n_eval_episodes=episodes_to_run, return_episode_rewards=False
        )
        print(f"Evaluation Results: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward

    def optimize_hyperparameters(self):
        """
        Hyperparameter optimization using Optuna.
        """
        n_trials = self.cfg.rl.hpo.n_trials
        print(f"Starting Optuna optimization for {n_trials} trials...")

        def objective(trial):
            # 1. Suggest hyperparameters from config bounds
            lr = trial.suggest_float("learning_rate", 
                                     self.cfg.rl.hpo.learning_rate_min, 
                                     self.cfg.rl.hpo.learning_rate_max, 
                                     log=True)
            gamma = trial.suggest_float("gamma", 
                                        self.cfg.rl.hpo.gamma_min, 
                                        self.cfg.rl.hpo.gamma_max)
            
            # 2. Copy the current config and apply the trial's suggested hyperparameters
            trial_cfg = self.cfg.copy()          # OmegaConf native deepcopy
            trial_cfg.rl.learning_rate = lr
            trial_cfg.rl.gamma = gamma
            
            # 3. Instantiate a clean, nested RLManager for this trial
            trial_manager = RLManager(trial_cfg)
            
            # 4. Run the pipeline completely from our built-in methods!
            trial_manager.build_model(verbose=0) # Silence PPO logging
            trial_manager.train(save=False)      # Do not write temporary trial models to disk
            
            # 5. Evaluate and return score
            mean_reward = trial_manager.evaluate(n_episodes=self.cfg.rl.hpo.n_eval_episodes)
            return mean_reward

        # Create and run the Optuna study with the seeded sampler
        sampler = optuna.samplers.TPESampler(seed=self.optuna_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        
        # Print results
        print("\n=========================================")
        print("Optimization Finished!")
        print(f"Best Trial Score: {study.best_value:.2f}")
        print("Best Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        print("=========================================")
