__author__ = 'yunbo'

import numpy as np
import torch
def reshape_patch(img_tensor, patch_size):
    if img_tensor.ndim==4:
        img_tensor=img_tensor[...,np.newaxis]
    assert 5 == img_tensor.ndim
    batch_size,seq_length,img_height,img_width,num_channels= img_tensor.shape
    a = torch.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    a.permute(0,1,2,4,3,5,6)  # transpose 
    patch_tensor = torch.reshape(a, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    if patch_tensor.ndim == 5:
        # if tensor is image sequence
        batch_size,seq_length,patch_height,patch_width, channels= patch_tensor.shape
        img_channels = channels // (patch_size*patch_size)
        a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                    patch_height, patch_width,
                                    patch_size, patch_size,
                                    img_channels])
        a.permute(0,1,2,4,3,5,6)  # transpose 
        img_tensor = torch.reshape(a, [batch_size, seq_length,
                                    patch_height * patch_size,
                                    patch_width * patch_size,
                                    img_channels])
        if img_channels==1:
            img_tensor=img_tensor[...,0]
        return img_tensor
    elif patch_tensor.ndim == 4:
        # if tensor is image
        batch_size,patch_height,patch_width, channels= patch_tensor.shape
        img_channels = channels // (patch_size*patch_size)
        a = torch.reshape(patch_tensor, [batch_size,
                                    patch_height, patch_width,
                                    patch_size, patch_size,
                                    img_channels])
        a.permute(0,1,3,2,4,5)  # transpose 
        img_tensor = torch.reshape(a, [batch_size,
                                    1,patch_height * patch_size,
                                    patch_width * patch_size,
                                    img_channels])
        if img_channels==1:
            img_tensor=img_tensor[...,0]
        return img_tensor

