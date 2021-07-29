import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def save_gif(gt_images,pd_images,save_path="result.gif",suptitle="",interval = 500, greyscale=False):
    """
    params:
        gt_images,pd_images:(frame,width,hight)
        gt_imagesは入力+正解の出力
        pd_imagesは予測の出力のみ
    """
    gt_images=np.array(gt_images)
    pd_images=np.array(pd_images)
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
    if not greyscale:
        im2=ax2.imshow(np.zeros_like(gt_images[0]), vmin=vmin , vmax = vmax, cmap = cmap)
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

def save_weight_gif(weights,start,save_path="weights.gif",suptitle="weights",interval = 500):
    """
    params:
        weights: numpy.ndarray Gating Networkの出力
                (frame,height,width,expert)

    """
    experts = ['translation', 'rotation', 'growth/decay']
    vmin=0
    vmax=np.max(weights)
    def update(i, ax1, ax2, ax3):
        ax1.set_title(f'{experts[0]}_{start+i}') 
        ax1.imshow(weights[i, :, :, 0], vmin = vmin, vmax = vmax, cmap = 'jet') 
        ax2.set_title(f'{experts[1]}_{start+i}')
        ax2.imshow(weights[i, :, :, 1], vmin=vmin , vmax = vmax, cmap = 'jet')
        ax3.set_title(f'{experts[2]}_{start+i}')
        ax3.imshow(weights[i, :, :, 2], vmin=vmin , vmax = vmax, cmap = 'jet')

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)  
    fig.suptitle(suptitle)
    im3=ax3.imshow(np.zeros_like(weights[0, :, :, 2]), vmin=vmin , vmax = vmax, cmap = 'jet')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3,cax=cax)
    ani = animation.FuncAnimation(fig, update, fargs = (ax1, ax2, ax3), interval = interval, frames = len(weights))
    ani.save(save_path, writer = 'imagemagick')
    plt.close()


import unittest
class Test(unittest.TestCase):
    def test_weight_gif(self):
        W = np.load("/home/lab/ryoto/src/STMoE_experiments/test/W.npy")
        save_weight_gif(W[0],4,save_path="weights_3.gif",suptitle="weights:epoch=3")
    
    def test_save_gif_grey(self):
        gt_images = np.random.rand(10,128,128)
        pd_images = np.random.rand(10,128,128)
        save_gif(gt_images,pd_images,save_path="test.gif", greyscale=True,)
    
    def test_save_gif(self):
        gt_images = np.random.rand(10,128,128)
        pd_images = np.random.rand(10,128,128)
        save_gif(gt_images,pd_images,save_path="test.gif",)

if __name__ == '__main__':
    unittest.main()