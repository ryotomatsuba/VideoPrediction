import numpy as np
from matplotlib import image, pyplot as plt
from data.dataset.get_images import get_cifar10_images

def fourier_transform(img):
    """Calculate the fourier transform of an image.
    Params:
        img: numpy array of shape (height, width)
    Returns:
        fourier: numpy array of shape (height, width)
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def inverse_fourier_transform(fourier):
    """Calculate the inverse fourier transform of an image.
    Params:
        fourier: numpy array of shape (height, width)
    Returns:
        img: numpy array of shape (height, width)
    """
    f_ishift = np.fft.ifftshift(fourier)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def low_pass_filter(img, cut_off):
    """Calculate the low pass filter of an image.
    Params:
        img: numpy array of shape (height, width)
        cut_off: frequency above this value is set to 0
    Returns:
        img_filtered: numpy array of shape (height, width)
    """
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    assert cut_off > 0 and cut_off < crow and cut_off < ccol, "cut_off must be greater than 0 and less than half of the image size"
    fourier = fourier_transform(img)
    fshift = np.zeros((rows, cols), np.complex)
    fshift[crow - cut_off:crow + cut_off, ccol - cut_off:ccol + cut_off] = fourier[crow - cut_off:crow + cut_off, ccol - cut_off:ccol + cut_off]
    img_filtered = inverse_fourier_transform(fshift)
    return img_filtered


def high_pass_filter(img, cut_off):
    """Calculate the high pass filter of an image.
    Params:
        img: numpy array of shape (height, width)
        cut_off: frequency below this value is set to 0
    Returns:
        img_filtered: numpy array of shape (height, width)
    """
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    assert cut_off > 0 and cut_off < crow and cut_off < ccol, "cut_off must be greater than 0 and less than half of the image size"
    fourier = fourier_transform(img)
    fourier[crow - cut_off:crow + cut_off, ccol - cut_off:ccol + cut_off] = 0
    img_filtered = inverse_fourier_transform(fourier)
    return img_filtered

def divide_by_frequency(image, cut_off):
    """Divide an image by its frequency.
    Params:
        image: numpy array of shape (height, width)
        cut_off: frequency below this value is set to 0
    Returns:
        [low_image, high_image]: numpy array of shape (height, width)
    """
    return  low_pass_filter(image, cut_off),high_pass_filter(image, cut_off)



if __name__=="__main__":
    #images=get_cifar10_images(num_images=10)
    img=images[1]
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    cut_off=5
    ax[1].imshow(high_pass_filter(img,cut_off),cmap='gray')
    ax[2].imshow(low_pass_filter(img,cut_off),cmap='gray')
    # save the image
    fig.savefig('high_pass_filter.png')