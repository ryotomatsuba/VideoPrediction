from .moving_mnist import MnistDataset
from .typhoon import TyphoonDataset
from .human_action import ActionDataset


def get_dataset(cfg):
    name=cfg.dataset.name
    if name == 'moving_mnist':
        return MnistDataset(cfg)
    elif name == 'typhoon':
        return TyphoonDataset(cfg)
    elif name == 'human_action':
        return ActionDataset(cfg)
    else:
        raise ValueError(f'Unknown dataset: {name}')