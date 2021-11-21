import numpy as np

from models.unet import UNet
from predict import predict

net = UNet(n_channels=4, n_classes=1, bilinear=True) # define model

def test_pred():
    input_video=np.ones((1,4,128,128))
    pred=predict(net,input_video)
    assert pred.shape == (1,6,128,128)
    assert pred.dtype == np.float32
