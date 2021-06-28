# -*- coding: utf-8 -*-
"""Executor

These functions are for execution.

"""

from trainers.default_trainer import DefaultTrainer
from .unet_trainer import UnetTrainer

def get_trainer(cfg: object) -> object:
    """Get trainer

    Args:
        cfg: Config of the project.

    Returns:
        Trainer object.

    Raises:
        NotImplementedError: If the model you want to use is not suppoeted.

    """

    trainer_name = cfg.train.trainer.name

    if trainer_name == "default":
        return DefaultTrainer(cfg)