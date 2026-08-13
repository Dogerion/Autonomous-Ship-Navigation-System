import numpy as np
import optuna
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.agents.base_manager import BaseManager

class PPOManager(BaseManager):
    """
    Manager for the RecurrentPPO agent.
    """
    def build_model(self, verbose=None):
        if self.env is None:
            self._create_env()

        # Determine verbosity (allow override during hyperparameter optimization)
        model_verbose = verbose if verbose is not None else self.cfg.rl.get("verbose", 1)

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
            seed=self.algo_seed
        )

    def load_model(self, path=None):
        if path is None:
            path = self.get_model_path()
        if self.env is None:
            self._create_env()
        print(f"Loading PPO model from {path}...")
        self.model = RecurrentPPO.load(path, env=self.env)

    def save_model(self):
        if self.model is None:
            raise ValueError("No model is initialized to save!")
            
        save_path = self.get_model_path()
        self.model.save(save_path)
        print(f"PPO Model saved successfully to: {save_path}.zip")

    def train(self, save=True):
        if self.model is None:
            raise ValueError("Model not built! Call build_model() first.")
            
        print(f"Starting PPO training for {self.cfg.rl.total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=self.cfg.rl.total_timesteps,
            log_interval=self.cfg.rl.log_interval,
            tb_log_name=self.model_base_name
        )
        
        if save:
            self.save_model()

    def evaluate(self, n_episodes=None, render=False):
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() must be called first.")
            
        episodes_to_run = (
            n_episodes
            if n_episodes is not None
            else self.cfg.rl.eval.get("n_eval_episodes", 10)
        )
            
        print(f"Evaluating PPO model over {episodes_to_run} episodes...")
        mean_reward, std_reward = evaluate_policy(
            self.model, self.env, n_eval_episodes=episodes_to_run, return_episode_rewards=False
        )
        print(f"Evaluation Results: Mean Reward = {mean_reward:.2f}, Standard Deviation = {std_reward:.2f}")
        return mean_reward

    def _rollout_episode(self):
        """Run one deterministic episode, threading the LSTM state, returning its trajectory."""
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() first.")

        obs = self.env.reset()
        lstm_states = None
        episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        traj = []
        K_true = T_true = None

        while True:
            # Record the state the ship is currently in (before applying this step's action).
            step_rec = {
                "psi": float(obs[0][0]),
                "r": float(obs[0][1]),
                "delta": float(obs[0][2]),
                "integral_psi": float(obs[0][3]),
            }

            action, lstm_states = self.model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, done, info = self.env.step(action)
            episode_starts = done

            step_rec["reward"] = float(reward[0])
            if K_true is None:
                K_true, T_true = float(info[0]["K"]), float(info[0]["T"])
            traj.append(step_rec)

            if done[0]:
                break

        for rec in traj:
            rec["K"], rec["T"] = K_true, T_true
        return traj

    def optimize_hyperparameters(self):
        n_trials = self.cfg.rl.hpo.n_trials
        print(f"Starting PPO Optuna optimization for {n_trials} trials...")

        def objective(trial):
            lr = trial.suggest_float("learning_rate", 
                                     self.cfg.rl.hpo.learning_rate_min, 
                                     self.cfg.rl.hpo.learning_rate_max, 
                                     log=True)
            gamma = trial.suggest_float("gamma", 
                                        self.cfg.rl.hpo.gamma_min, 
                                        self.cfg.rl.hpo.gamma_max)
            
            trial_cfg = self.cfg.copy()
            trial_cfg.rl.learning_rate = lr
            trial_cfg.rl.gamma = gamma
            
            # Using the factory pattern implicitly here by re-instantiating itself
            trial_manager = PPOManager(trial_cfg)
            trial_manager.build_model(verbose=0)
            trial_manager.train(save=False)
            
            mean_reward = trial_manager.evaluate(n_episodes=self.cfg.rl.hpo.n_eval_episodes)
            return mean_reward

        sampler = optuna.samplers.TPESampler(seed=self.optuna_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        
        print("\n=========================================")
        print("Optimization Finished!")
        print(f"Best Trial Score: {study.best_value:.2f}")
        print("Best Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        print("=========================================")
