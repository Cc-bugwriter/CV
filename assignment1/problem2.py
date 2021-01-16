import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    return np.load(path)


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    H, W = bayerdata.shape

    red = np.zeros(bayerdata.shape)
    green = np.zeros(bayerdata.shape)
    blue = np.zeros(bayerdata.shape)
    for i in range(H):
        for j in range(W):
            if i % 2 == 0:
                if j % 2 == 0:
                    green[i, j] = bayerdata[i, j]
                else:
                    red[i, j] = bayerdata[i, j]
            else:
                if j % 2 == 0:
                    blue[i, j] = bayerdata[i, j]
                else:
                    green[i, j] = bayerdata[i, j]

    return red, green, blue


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    H, W = r.shape
    image = np.zeros((H, W, 3))
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b

    return image


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    H, W = r.shape
    image = np.zeros((H, W, 3))

    # bilinear interpolation kernel
    rb_k = 1/4*np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])

    g_k = 1/4*np.array([[0, 1, 0],
                    [1, 4, 1],
                    [0, 1, 0]])

    image[:, :, 0] = convolve(r, rb_k, mode='nearest')
    image[:, :, 1] = convolve(g, g_k, mode='nearest')
    image[:, :, 2] = convolve(b, rb_k, mode='nearest')

    return image
