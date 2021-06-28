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
from models.rainnet import rainnet
from data.dataset.moving_mnist import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True





# Instantiating the dataloaders.
train_data=np.load("/home/data/ryoto/Datasets/mnist_train.npy")
train_dataset = Dataset(train_data)
valid_data=np.load("/home/data/ryoto/Datasets/mnist_val.npy")
valid_dataset = Dataset(valid_data)



# Instantiating and compiling the model with Adam optimizer and Log_Cosh loss function.

model = rainnet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),loss='log_cosh')

# Start Training

model.fit(x=train_dataset,validation_data=valid_dataset,epochs=10)




