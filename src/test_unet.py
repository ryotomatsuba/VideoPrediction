import logging
import os
import sys
import time
import torch
import hydra
from omegaconf import DictConfig
from models.unet import UNet
from torch.utils.data import DataLoader,random_split
from data.dataset import MnistDataset
from utils.draw import save_gif


def test(cfg):
    net = UNet(n_channels=4, n_classes=1, bilinear=True)
    dataset = MnistDataset(cfg)
    test_loader = DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=cfg.test.num_workers, pin_memory=True, drop_last=True)
    # define model
    device = torch.device(cfg.test.device)
    net = UNet(n_channels=4, n_classes=1, bilinear=True)
    net.to(device=device)
    net.load_state_dict(
        torch.load(cfg.model.load, map_location=device)
    )
    logging.info(f'Model loaded from {cfg.model.load}')
    net.eval()
    device = next(net.parameters()).device

    X = iter(test_loader).__next__()
    X = X.to(device=device, dtype=torch.float32)
    input_num = cfg.dataset.input_num
    batch_size, total_num, height, width = X.shape
    preds = torch.empty(batch_size, 0, height, width).to(device=device)
    input_X = X[:,0:input_num]
    for t in range(input_num,total_num):
        start = time.time()
        with torch.no_grad():
            pred = net(input_X)
        elapsed_time = time.time() - start
        logging.info(f"elapsed_time:{elapsed_time}[sec]")
        input_X = torch.cat((input_X[:,1:],pred),dim=1) # use output image to pred next frame
        preds = torch.cat((preds,pred),dim=1) 
    X, preds = X.to(device="cpu"), preds.to(device="cpu")
    for i in range(cfg.test.batch_size):
        save_gif(X[i], preds[i], save_path = f"pred_test_{i}.gif", suptitle=f"test")
        logging.info(f'gif saved to {hydra.utils.to_absolute_path(f"pred_test_{i}.gif")}')


@hydra.main(config_path="../configs", config_name="unet")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test(cfg)


if __name__ == '__main__':
    main()