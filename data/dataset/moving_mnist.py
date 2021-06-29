import torch
import numpy as np
from torch.utils.data import Dataset
class MnistDataset(Dataset):
    
    def __init__(self, data, input_num=4):
        self.data = data
        self.input_num=input_num

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X=self.data[index]
        X=torch.from_numpy(X).type(torch.FloatTensor)
        return X
        

if __name__=="__main__":
  data=np.load("/home/data/ryoto/Datasets/mnist/dev_data_10.npy")
  dataset=MnistDataset(data)
  print(dataset[0][0].shape)