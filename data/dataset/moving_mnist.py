from keras.backend import batch_dot
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
# Defining some data preparing and preprocessing functions

def Scaler(array):
    return np.log(array+0.01)


def invScaler(array):
    return np.exp(array) - 0.01


def pad_to_shape(array, from_shape=900, to_shape=928, how="mirror"):
    # calculate how much to pad in respect with native resolution
    padding = int( (to_shape - from_shape) / 2)
    # for input shape as (batch, W, H, channels)
    if how == "zero":
        array_padded = np.pad(array, ((0,0),(padding,padding),(padding,padding),(0,0)), mode="constant", constant_values=0)
    elif how == "mirror":
        array_padded = np.pad(array, ((0,0),(padding,padding),(padding,padding),(0,0)), mode="reflect")
    return array_padded


def pred_to_rad(pred, from_shape=928, to_shape=900):
    # pred shape 12,928,928
    padding = int( (from_shape - to_shape) / 2)
    return pred[::, padding:padding+to_shape, padding:padding+to_shape].copy()


def data_preprocessing(X):
    
    # 0. Right shape for batch
    X = np.moveaxis(X, 0, -1)
    X = X[np.newaxis, ::, ::, ::]
    # 1. To log scale
    X = Scaler(X)
    # 2. from 900x900 to 928x928
    X = pad_to_shape(X)
    
    return X


def data_postprocessing(nwcst):

    # 0. Squeeze empty dimensions
    nwcst = np.squeeze(np.array(nwcst))
    # 1. Convert back to rainfall depth
    nwcst = invScaler(nwcst)
    # 2. Convert from 928x928 back to 900x900
    nwcst = pred_to_rad(nwcst)
    # 3. Return only positive values
    nwcst = np.where(nwcst>0, nwcst, 0)
    return nwcst

class Dataset(tf.keras.utils.Sequence):
    
    def __init__(self, data, input_num=4,batch_size=1):
        self.data = data
        self.input_num=input_num
        self.batch_size=batch_size
    
    def get_index(self,i):
      x = self.data[i][0:self.input_num]
      x = data_preprocessing(np.stack(x,0))
      x = np.squeeze(x)
      y = self.data[i][self.input_num:]
      y = data_preprocessing(np.array(y))
      y = np.squeeze(y)
      return x.astype('float32'),y.astype('float32')
    
    def __getitem__(self, index):
      X = []
      Y = []

      for i in range(index*self.batch_size,(index+1)*self.batch_size):
        x,y = self.get_index(i)
        X.append(x[np.newaxis,:])
        Y.append(y[np.newaxis,:])

      return np.array(X),np.array(Y)
        
    def __len__(self):
      return len(self.data)//self.batch_size


if __name__=="__main__":
  data=np.load("/home/data/ryoto/Datasets/mnist_train.npy")
  data=Dataset(data)
  print(data)