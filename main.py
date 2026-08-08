import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import agent_selecter

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=========================================")
    print("MARIN Hybrid Navigation System Loaded")
    print("=========================================")
    print(f"Project Name: {cfg.project_name}")
    print(f"Agent Type: {cfg.agent_type}")
    print(f"Mode: {cfg.mode}")
    print("=========================================")
    
    # Dynamically acquire the correct wrapper manager
    manager = agent_selecter(cfg)
    
    if cfg.mode == "train":
        manager.build_model()
        manager.train()
    elif cfg.mode == "eval":
        if not cfg.load_path:
            raise ValueError("Must provide 'load_path' in config to run evaluation mode.")
        manager.load_model(cfg.load_path)
        manager.evaluate()
    elif cfg.mode == "optimize":
        manager.optimize_hyperparameters()
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

if __name__ == "__main__":
    main()
