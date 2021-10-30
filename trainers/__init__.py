# -*- coding: utf-8 -*-
"""Executor

These functions are for execution.

"""

from .default_trainer import DefaultTrainer
from .expert_trainer import ExpertTrainer
from .stmoe_trainer import STMoETrainer
def get_trainer(cfg: object) -> object:
    """Get trainer

    Args:
        cfg: Config of the project.

    Returns:
        Trainer object.

    Raises:
        NotImplementedError: If the model you want to use is not suppoeted.

    """

    

    if cfg.model.name == "predrnn":
        return ExpertTrainer(cfg)
    elif cfg.model.name == "unet":
        return ExpertTrainer(cfg)
    elif cfg.model.name == "stmoe":
        return STMoETrainer(cfg)
    else:
        raise NotImplementedError(f"Not supported model: {cfg.model.name}")