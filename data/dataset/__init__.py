from .moving_image import MovingImageDataset, MixImageDataset
from .typhoon import TyphoonDataset
from .human_action import ActionDataset
from .traffic import TrafficDataset
from .video_data import VideoDataset



SUPPORTED_DICT = {
    "moving_image": MovingImageDataset,
    "typhoon": TyphoonDataset,
    "human_action": ActionDataset,
    "traffic": TrafficDataset,
    "video_data": VideoDataset,
    "mix_image": MixImageDataset,
}

def get_dataset(cfg):
    name=cfg.dataset.name
    if name not in SUPPORTED_DICT:
        raise ValueError(f"Dataset {name} not supported")
    else:
        return SUPPORTED_DICT[name](cfg)