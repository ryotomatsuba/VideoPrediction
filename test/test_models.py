import unittest
import torch
from torchsummary import summary
from models.unet import UNet
from models.predrnn import PredRNN

class Test(unittest.TestCase):
    def test_unet(self):
        batch, input_frame, h ,w = 1, 4, 128, 128
        net=UNet(n_channels=input_frame, n_classes=1)
        summary(net,(4,128,128),device="cpu")
        input = torch.rand(batch, input_frame, h ,w)
        output = net(input)
        self.assertEqual(list(output.shape),[batch, 1 ,h ,w])
    
    

if __name__=="__main__":
    unittest.main()