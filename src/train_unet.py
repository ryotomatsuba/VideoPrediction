from models.unet import UNet
import logging
import os
import sys

import torch
import hydra
from omegaconf import DictConfig
from trainers import UnetTrainer



@hydra.main(config_path="../configs", config_name="unet")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=4, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if cfg.model.load:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(
            torch.load(cfg.model.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')

    # faster convolutions, but more memory
    # cudnn.benchmark = True
    trainer=UnetTrainer(cfg)
    try:
        trainer.train(net)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
if __name__ == '__main__':
    main()