import hydra
from omegaconf import OmegaConf, DictConfig

# %%
def print_cfg(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
