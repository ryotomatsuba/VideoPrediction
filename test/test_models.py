import unittest
import torch
from models.unet import UNet
from models.predrnn import PredRNN


batch, total_length, input_frame, h ,w, c= 1, 10, 4, 128, 128, 1
input = torch.rand(batch, input_frame, h ,w)

class Test(unittest.TestCase):
    def test_unet(self):
        net=UNet(n_channels=input_frame, n_classes=1)
        output = net(input)
        self.assertEqual(list(output.shape),[batch, 1 ,h ,w])
    
    def test_predrnn(self):
        input = torch.rand(batch, total_length, h ,w)
        net=PredRNN(input_num=input_frame,total_length=total_length)
        output = net(input)
        self.assertEqual(list(output.shape),[batch, total_length-input_frame, h ,w, c])

if __name__=="__main__":
    unittest.main()