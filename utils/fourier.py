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


def low_pass_filter(img, cutoff):
    """Calculate the low pass filter of an image.
    Params:
        img: numpy array of shape (height, width)
        cutoff: frequency above this value is set to 0
    Returns:
        img_filtered: numpy array of shape (height, width)
    """
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    assert cutoff > 0 and cut_off < crow and cut_off < ccol, "cutoff must be greater than 0 and less than half of the image size"
    fourier = fourier_transform(img)
    fshift = np.zeros((rows, cols), np.complex)
    fshift[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = fourier[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff]
    img_filtered = inverse_fourier_transform(fshift)
    return img_filtered


def high_pass_filter(img, cutoff):
    """Calculate the high pass filter of an image.
    Params:
        img: numpy array of shape (height, width)
        cutoff: frequency below this value is set to 0
    Returns:
        img_filtered: numpy array of shape (height, width)
    """
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    assert cutoff > 0 and cut_off < crow and cut_off < ccol, "cutoff must be greater than 0 and less than half of the image size"
    fourier = fourier_transform(img)
    fourier[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0
    img_filtered = inverse_fourier_transform(fourier)
    return img_filtered

# pytest
def test_fourier_transform():
    images=get_cifar10_images(num_images=10)
    img=images[1]
    img_fourier = fourier_transform(img)
    assert img_fourier.shape == img.shape
    assert img_fourier.dtype == np.complex
    image_back = inverse_fourier_transform(img_fourier)
    assert image_back.shape == img.shape
    assert np.allclose(img, image_back)



if __name__=="__main__":
    images=get_cifar10_images(num_images=10)
    img=images[1]
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    cut_off=5
    ax[1].imshow(high_pass_filter(img,cut_off),cmap='gray')
    ax[2].imshow(low_pass_filter(img,cut_off),cmap='gray')
    # save the image
    fig.savefig('high_pass_filter.png')
    # pytest
    test_fourier_transform()