import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.fourier import fourier_transform, inverse_fourier_transform, divide_by_frequency



def test_fourier_transform():
    # read greyscale image
    cifar10_image=cv2.imread('/workspace/test/images/cifar10.png',cv2.IMREAD_GRAYSCALE)  
    assert cifar10_image.shape == (28,28)
    img=cifar10_image
    img_fourier = fourier_transform(img)
    assert img_fourier.shape == img.shape
    assert img_fourier.dtype == np.complex
    image_back = inverse_fourier_transform(img_fourier)
    assert image_back.shape == img.shape
    assert np.allclose(img, image_back)



def test_divide_by_frequency():
    # read greyscale image
    cifar10_image=cv2.imread('/workspace/test/images/cifar10.png',cv2.IMREAD_GRAYSCALE)  
    assert cifar10_image.shape == (28,28)
    img=cifar10_image
    low_im, high_im = divide_by_frequency(img,7)
    assert low_im.dtype == np.float
    assert high_im.dtype == np.float
    assert low_im.shape == img.shape
    assert high_im.shape == img.shape

if __name__ == "__main__":
    test_divide_by_frequency()