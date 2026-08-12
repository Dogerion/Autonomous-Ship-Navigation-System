from hydra.core.hydra_config import HydraConfig
from src.agents.ppo_manager import PPOManager
from src.agents.mpc_manager import MPCManager

def agent_selecter(cfg):
    """
    Factory function acting as a Router wrapper.
    Returns the initialized manager corresponding to the selected `rl` config group.
    """
    agent = HydraConfig.get().runtime.choices.rl

    if agent == "ppo":
        return PPOManager(cfg)
    elif agent == "sysid_mpc":
        return MPCManager(cfg)
    else:
        raise ValueError(f"Unknown agent (rl group): {agent}")
