import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from src.utils import agent_selecter

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=========================================")
    print("MARIN Hybrid Navigation System Loaded")
    print("=========================================")
    agent_type = HydraConfig.get().runtime.choices.rl
    print(f"Project Name: {cfg.project_name}")
    print(f"Agent Type: {agent_type}")
    print(f"Mode: {cfg.mode}")
    print("=========================================")
    
    # Dynamically acquire the correct wrapper manager
    manager = agent_selecter(cfg)
    
    if cfg.mode == "train":
        manager.build_model()
        manager.train()
    elif cfg.mode == "eval":
        manager.load_model()
        manager.evaluate()
    elif cfg.mode == "optimize":
        manager.optimize_hyperparameters()
    elif cfg.mode == "visualize":
        manager.load_model()
        manager.visualize(agent_name=cfg.model_name, sysid=(agent_type == "sysid_mpc"))
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

if __name__ == "__main__":
    main()
