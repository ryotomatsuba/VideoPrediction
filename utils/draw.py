import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

def save_gif(gt_images,pd_images,save_path="result.gif",suptitle="",interval = 500, greyscale=False):
    """
    params:
        gt_images,pd_images:(frame,width,hight)
        gt_imagesは入力+正解の出力
        pd_imagesは予測の出力のみ
    """
    if torch.is_tensor(gt_images):
        gt_images=gt_images.cpu().detach().numpy()
    if torch.is_tensor(pd_images):
        pd_images=pd_images.cpu().detach().numpy()
    vmin=0
    vmax=np.max(gt_images)
    input_length=len(gt_images)-len(pd_images)
    cmap = 'Greys_r' if greyscale else 'jet'
    def update(i, ax1, ax2):
        ax1.set_title(f'gt{i}') 
        ax1.imshow(gt_images[i], vmin = vmin, vmax = vmax, cmap = cmap) 
        if i>=input_length:
            ax2.set_title(f'est{i}')
            ax2.imshow(pd_images[i-input_length], vmin=vmin , vmax = vmax, cmap = cmap)
    fig, (ax1, ax2) = plt.subplots(1,2)  
    fig.suptitle(suptitle)
    im2=ax2.imshow(np.zeros_like(gt_images[0]), vmin=vmin , vmax = vmax, cmap = cmap)
    if not greyscale:
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2,cax=cax)
    ani = animation.FuncAnimation(fig, update, fargs = (ax1, ax2), interval = interval, frames = len(gt_images))
    ani.save(save_path, writer = 'imagemagick')
    plt.close()


def draw_image(gt_images,pd_images,save_path="result.png",suptitle=""):
    vmin=0
    vmax=np.max(gt_images)
    total_length=len(gt_images)
    input_length=len(gt_images)-len(pd_images)
    size=2.8
    fig, ax = plt.subplots(2,total_length,figsize=(total_length*size,2.2*size),)

    fig.suptitle(suptitle)
    for i in range(total_length):
        ax[0][i].set_title(f'gt{i}') 
        ax[0][i].imshow(gt_images[i], vmin = vmin, vmax = vmax, cmap = 'jet') 
        if i>=input_length:
            ax[1][i].set_title(f'est{i}')
            ax[1][i].imshow(pd_images[i-input_length], vmin=vmin , vmax = vmax, cmap = 'jet')
        else:
            ax[1][i].imshow(np.zeros_like(gt_images[0]), vmin = vmin, vmax = vmax, cmap = 'jet') 
    im=ax[1][0].imshow(np.zeros_like(gt_images[0]), vmin=vmin , vmax = vmax, cmap = 'jet')
    
    divider = make_axes_locatable(ax[0][-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
    plt.savefig(save_path)
    plt.close()

def save_weight_gif(pd_images,weights,save_path="weights.gif",suptitle="weights",interval = 500, greyscale=False):
    """
    params:
        pd_images:(frame,width,hight)
        weights: Gating Networkの出力 (frame,expert,height,width,)

    """
    if torch.is_tensor(pd_images):
        pd_images=pd_images.cpu().detach().numpy()
    if torch.is_tensor(weights):
        weights=weights.cpu().detach().numpy()
    num_frame, num_expert, height, width = weights.shape
    assert len(pd_images)==num_frame
    if num_expert<3:
        # add channel
        add_channel=np.zeros((num_frame,3-num_expert,height,width))
        w_images=np.concatenate((weights,add_channel),axis=1)
        w_images = w_images.transpose(0,2,3,1)

    vmin=0
    vmax=np.max(pd_images)
    cmap = 'Greys_r' if greyscale else 'jet'
   
    def update(i, ax1, ax2):
        ax1.set_title(f'pd{i}') 
        ax1.imshow(pd_images[i], vmin = vmin, vmax = vmax, cmap = cmap) 
        ax2.set_title(f'w{i}')
        ax2.imshow(w_images[i])

    fig, (ax1, ax2) = plt.subplots(1,2)  
    fig.suptitle(suptitle)
    ani = animation.FuncAnimation(fig, update, fargs = (ax1, ax2), interval = interval, frames = num_frame)
    ani.save(save_path, writer = 'imagemagick')
    plt.close()



import unittest
class Test(unittest.TestCase):
    def test_weight_gif(self):
        pd_images = np.random.rand(6,128,128)
        weights = np.random.rand(6,2,128,128)
        save_weight_gif(pd_images,weights)
    
    def test_save_gif(self):
        gt_images = np.random.rand(20,128,128)
        pd_images = np.random.rand(10,128,128)
        save_gif(gt_images,pd_images,save_path="test.gif",)

    def test_save_gif_grey(self):
        gt_images = np.random.rand(20,128,128)
        pd_images = np.random.rand(10,128,128)
        save_gif(gt_images,pd_images,save_path="test.gif", greyscale=True,suptitle="grey")
    
    def test_torch_tensor(self):
        gt_images = torch.rand(10,128,128)
        pd_images = torch.rand(6,128,128)
        weights = torch.rand(6,2,128,128)
        save_weight_gif(pd_images,weights)
        save_gif(gt_images,pd_images,save_path="test.gif",)

    

if __name__ == '__main__':
    unittest.main()