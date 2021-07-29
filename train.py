import logging
import os
import sys

import torch
import hydra
from omegaconf import DictConfig
from trainers import get_trainer



@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    trainer=get_trainer(cfg)
    try:
        trainer.train()
    except KeyboardInterrupt:
        torch.save(trainer.net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
if __name__ == '__main__':
    main()