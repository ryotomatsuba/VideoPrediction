
import urllib.request
from io import BytesIO

import cv2
import numpy as np
import torchvision

def get_mnist_images():
    """get mnist images
    Return: 
        mnist_images: shape(60000, 28, 28)
    """
    req = urllib.request.Request('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')
    with urllib.request.urlopen(req) as response:
        data = response.read()
        mnist_images = np.load(BytesIO(data),allow_pickle=True)['x_train'].reshape(60000, 28, 28)
    return mnist_images

def get_box_images(num_images=1000):
    """get box images
    Return:
        box_images: shape(num_images, 28, 28)
    """
    box_images=np.ones((num_images, 28, 28))
    for i in range(num_images):
        box_images[i]*=np.random.randint(0,255)
    return box_images

def get_cifar10_images(num_images=1000):
    """get cifar10 images
    Return: 
        cifar10_images: shape(num_images, 28, 28)
    """
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((28,28)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
    ])
    cifar10_dataset=torchvision.datasets.CIFAR10(root='/data/Datasets/', download=True, transform=transform)
    cifar10_images=np.zeros((num_images, 28, 28))
    for i in range(num_images):
        cifar10_images[i]=cifar10_dataset[i][0].numpy()
    cifar10_images*=255
    return cifar10_images

def get_wave_images(freq_type="low",num_images=1000):
    """get sine wave images
    Parameters:
        freq_type: "low", "middle" or "high"
        num_images: number of images
    Return:
        wave_images: shape(num_images, 28, 28)
    """
    width=28
    if freq_type=="low":
        freq_range=np.linspace(0,width/6,num_images)
    elif freq_type=="middle":
        freq_range=np.linspace(width/6,width/3,num_images)
    elif freq_type=="high":
        freq_range=np.linspace(width/3,width/2,num_images)
    else:
        raise ValueError(f'freqency {freq_type} is not supported')
    wave_images=np.zeros((num_images,width,width))
    h_line=np.arange(0,width).reshape(-1, 1)#horizonal
    v_line=np.arange(0,width)#vertical
    
    for i in range(num_images):
        n=np.random.choice(freq_range)
        omega=2*np.pi/width*n
        wave_images[i]+=np.sin(omega*h_line)
        wave_images[i]+=np.sin(omega*v_line)

    wave_images*=255
    return wave_images

def get_mix_images(images1,images2):
    """mix two types of images
    Params:
        images1: shape(num_images1, 28, 28)
        images2: shape(num_images2, 28, 28)
    Return:
        mix_images: shape(num_mix_images, 28, 28)
    """
    if len(images1)>len(images2):
        images1, images2=images2,images1
    assert len(images1)<=len(images2)
    mix_images=np.zeros(images1.shape)
    for i in range(images1.shape[0]):
        mix_images[i][:14]=images1[i][:14]
        mix_images[i][14:]=images2[i][14:]
    return mix_images


if __name__=="__main__":
    images=get_box_images(10)
    print(images.shape)
    print(images.min(),images.max())
    #save image
    cv2.imwrite('get_image.png',images[0])
