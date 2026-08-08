from src.agents.ppo_manager import PPOManager
from src.agents.mpc_manager import MPCManager

def agent_selecter(cfg):
    """
    Factory function acting as a Router wrapper.
    Returns the initialized manager corresponding to the configured agent_type.
    """
    agent_type = cfg.agent_type
    
    if agent_type == "ppo":
        return PPOManager(cfg)
    elif agent_type == "sysid_mpc":
        return MPCManager(cfg)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
