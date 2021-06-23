# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/PradyumnaGupta/rainnet/blob/master/RainNet_Training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # RainNet Training Notebook
# 
# Author:[Pradyumna Gupta](https://github.com/PradyumnaGupta)
# 
# This notebook is created and executed on Google Colaboratory with Google Drive Mounted.
# %% [markdown]
# * Mounting Google Drive
# 
# 

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# * Importing libraries

# %%
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
import os

import matplotlib.pyplot as plt 
import imageio
import PIL
from PIL import ImageFile
import cv2
import numpy as np
from IPython.display import display
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.nn import functional as F
# import pretrainedmodels
import torch.optim as optim
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
import math
import time
import albumentations
import random
from tqdm import tqdm
import tensorflow as tf

ImageFile.LOAD_TRUNCATED_IMAGES = True
get_ipython().magic('matplotlib inline')

# %% [markdown]
# * Defining some data preparing and preprocessing functions

# %%
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

# %% [markdown]
# * Constructing Dataloaders

# %%
class Dataset(tf.keras.utils.Sequence):
    
    def __init__(
            self, 
            dataset_dict,
            image_names,
            batch_size
    ):
        self.keys = image_names
        self.dataset = dataset_dict
        self.bs = batch_size

    def get_index(self,i):
      x = []
      for j in range(4):
        try:
          arr = np.array(self.dataset.get(self.keys[i+j]))
        except:
          print(i,j)
        x.append(arr)
      
      x = data_preprocessing(np.stack(x,0))
      # x = np.transpose(np.squeeze(x),(2,0,1)) 
      x = np.squeeze(x)
      y = np.squeeze(data_preprocessing(np.array(self.dataset[self.keys[i+3]])[np.newaxis,:,:]))

      return x.astype('float32'),y.astype('float32')
    
    def __getitem__(self, index):
        
      X = []
      Y = []

      for i in range(index*self.bs,(index+1)*self.bs):
        x,y = self.get_index(i)
        X.append(x[np.newaxis,:])
        Y.append(y[np.newaxis,:])

      return X,Y
        
    def __len__(self):
      return (len(self.keys) - 4)//self.bs

# %% [markdown]
# * Loading the dataset dictionary.
# 
# The dataset consists of a large dictionary where the keys are the image names and the value is the actual image in the form of a numpy array.The dataset can be downloaded from [here](https://drive.google.com/file/d/1sZI4TbFkgJcpkZDBfbuWR_JBZSNhKow4/view?usp=sharing).

# %%
import h5py
dataset_dict = h5py.File('drive/MyDrive/RYDL.hdf5', 'r')

# %% [markdown]
# * Loading the image names.
# 
# The image names are present in the keys txt file which can be downloaded from [here](https://drive.google.com/file/d/1DvVUyrUvL4P8TRr_y_r5NrOxlRjwgWQr/view?usp=sharing). 
# 
# The data from the year 2012-2016 will be used for training purposes and the data from the year 2017 will be used for validation purposes.

# %%
import ast
with open('drive/MyDrive/RYDL_keys.txt','r') as f:
  image_names = ast.literal_eval(f.read())
image_names = [name for name in image_names if name[:4]>'2012']

train_images = [name for name in tqdm(image_names) if "2017" not in name]
val_images = [name for name in tqdm(image_names) if name[0:4]=="2017"]

# %% [markdown]
# * Instantiating the dataloaders.

# %%
train_dataset = Dataset(
    dataset_dict=dataset_dict,
    image_names=train_images,
    batch_size=1
)

valid_dataset = Dataset(
    dataset_dict=dataset_dict,
    image_names=val_images,
    batch_size=1
)

# %% [markdown]
# * Constructing the RainNet model.

# %%
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def rainnet(input_shape=(928, 928, 4), mode="regression"):

    """
    The function for building the RainNet (v1.0) model from scratch
    using Keras functional API.
    Parameters:
    input size: tuple(W x H x C), where W (width) and H (height)
    describe spatial dimensions of input data (e.g., 928x928 for RY data);
    and C (channels) describes temporal (depth) dimension of 
    input data (e.g., 4 means accounting four latest radar scans at time
    t-15, t-10, t-5 minutes, and t)
    
    mode: "regression" (default) or "segmentation". 
    For "regression" mode the last activation function is linear, 
    while for "segmentation" it is sigmoid.
    To train RainNet to predict continuous precipitation intensities use 
    "regression" mode. 
    RainNet could be trained to predict the exceedance of specific intensity 
    thresholds. For that purpose, use "segmentation" mode.
    """

    inputs = Input(input_shape)

    conv1f = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1f = Activation("relu")(conv1f)
    conv1s = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1f)
    conv1s = Activation("relu")(conv1s)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1s)

    conv2f = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2f = Activation("relu")(conv2f)
    conv2s = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2f)
    conv2s = Activation("relu")(conv2s)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2s)

    conv3f = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3f = Activation("relu")(conv3f)
    conv3s = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3f)
    conv3s = Activation("relu")(conv3s)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3s)

    conv4f = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4f = Activation("relu")(conv4f)
    conv4s = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4f)
    conv4s = Activation("relu")(conv4s)
    drop4 = Dropout(0.5)(conv4s)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5f = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5f = Activation("relu")(conv5f)
    conv5s = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5f)
    conv5s = Activation("relu")(conv5s)
    drop5 = Dropout(0.5)(conv5s)

    up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4s], axis=3)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3s], axis=3)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Activation("relu")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2s], axis=3)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Activation("relu")(conv8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Activation("relu")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1s], axis=3)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(up9)
    conv9 = Activation("relu")(conv9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Activation("relu")(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    if mode == "regression":
        outputs = Conv2D(1, 1, activation='linear')(conv9)
    elif mode == "segmentation":
        outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model

# %% [markdown]
# * Instantiating and compiling the model with Adam optimizer and Log_Cosh loss function.
# 

# %%
model = rainnet()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4),loss='log_cosh')

# %% [markdown]
# * Start Training

# %%
model.fit(x=train_dataset,validation_data=valid_dataset,epochs=10)


# %%



