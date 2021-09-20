from .moving_mnist import MnistDataset
from .typhoon import TyphoonDataset
from .human_action import ActionDataset
from .traffic import TrafficDataset
from .video_data import VideoDataset


def get_dataset(cfg):
    name=cfg.dataset.name
    if name == 'moving_mnist':
        return MnistDataset(cfg)
    elif name == 'typhoon':
        return TyphoonDataset(cfg)
    elif name == 'human_action':
        return ActionDataset(cfg)
    elif name == 'traffic':
        return TrafficDataset(cfg)
    elif name == 'video_data':
        return VideoDataset(cfg)
    else:
        raise ValueError(f'Unknown dataset: {name}')