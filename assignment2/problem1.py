from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve
import math

def loadimg(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """

    #
    # You code here
    im = Image.open(path)
    img = np.array(im.getdata()).reshape(im.size[1], im.size[0])
    return (img - np.min(img))/ (np.max(img) - np.min(img))
    #


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """

    #
    # You code here
    g = np.empty(fsize)

    for i in range(fsize[0]):
        for j in range(fsize[1]):
            g[i, j] = 1 / (2 * math.pi * sigma ** 2) * \
                      math.exp(-((i - np.median(range(fsize[0]))) ** 2 + (j - np.median(range(fsize[1]))) ** 2)
                               / (2 * (sigma ** 2)))

    return g / np.sum(abs(g))  # normalized
    #


def binomial2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """

    #
    # You code here
    def compute_binomial_coef(size):
        """
        Create a 1D binomial coefficient array
        Args:
            size: int, the size of local raw/ column

        Returns:
            coef: [size + 1, 1] array, 1D binomial coefficient np.array
        """
        N = size + 1
        coef = np.zeros((N, 1))
        for i in range(N):
            coef[i] = np.math.factorial(size) / (np.math.factorial(size - i) * np.math.factorial(i))
        return coef

    coef_v = compute_binomial_coef(fsize[0] - 1)  # raw coeff [n, 1]
    coef_h = compute_binomial_coef(fsize[1] - 1).reshape((1, -1))   # column coeff [1, m]

    bino_coef = coef_v @ coef_h  # without normalized
    return bino_coef / np.sum(abs(bino_coef))
    #


def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """

    #
    # You code here
    img_f = convolve(img, f, mode="wrap")

    img_down = np.zeros((math.ceil(img.shape[0]/2), math.ceil(img.shape[1]/2)))
    for i in range(img_down.shape[0]):
        for j in range(img_down.shape[1]):
            img_down[i, j] = deepcopy(img_f[2 * i, 2 * j])
    return img_down
    #


def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """

    #
    # You code here
    img_up = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    for i in range(img_up.shape[0]):
        for j in range(img_up.shape[1]):
            if (i % 2 == 0) and (j % 2 == 0):
                img_up[i, j] = deepcopy(img[math.ceil(i/2), math.ceil(j/2)])

    img_up = convolve(img_up, f, mode="wrap")

    return img_up * 4  # scale factor of 4
    #


def gaussianpyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    #
    # You code here
    pyramid_g_list = []
    for i in range(nlevel):
        if i != 0:
            img = downsample2(img, f)
        pyramid_g_list.append(img)

    return pyramid_g_list
    #


def laplacianpyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    #
    # You code here
    laplacian_list = []
    for i in range(len(gpyramid)):
        if i + 1 < len(gpyramid):
            G_i = deepcopy(gpyramid[i])
            G_exp = upsample2(gpyramid[i+1], f)
            L_i = G_i - G_exp
        else:
            L_i = deepcopy(gpyramid[i])
        laplacian_list.append(L_i)

    return laplacian_list
    #


def reconstructimage(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """

    #
    # You code here
    size = lpyramid[0].shape

    # Reconstructed Laplacian pyramid
    lpyramid_rec = []
    for i in reversed(range(size[0])):
        if i == len(lpyramid) - 1:
            img_rec = deepcopy(lpyramid[i])
        else:
            G_exp = upsample2(img_rec, f)
            img_rec = lpyramid[i] + G_exp
        lpyramid_rec.insert(0, img_rec)

    img = lpyramid_rec[0] / np.max(lpyramid_rec[0])
    return img
    #


def amplifyhighfreq(lpyramid, l0_factor=1.2, l1_factor=1.1):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """

    #
    # You code here
    lpyramid_amp = deepcopy(lpyramid)
    lpyramid_amp[0] = lpyramid_amp[0] * l0_factor
    lpyramid_amp[1] = lpyramid_amp[1] * l1_factor

    return lpyramid_amp
    #


def createcompositeimage(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """
    #
    # You code here
    size = pyramid[0].shape
    for i in range(len(pyramid)):
        if i == 0:
            img = deepcopy(pyramid[i])
            img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normaliz in [0, 1]
        else:
            img_temp = deepcopy(pyramid[i])
            img_temp = (img_temp - np.min(img_temp)) / (np.max(img_temp) - np.min(img_temp))   # Normalize in [0, 1]
            padding = np.zeros((size[0] - img_temp.shape[0], img_temp.shape[1]))
            img_temp = np.vstack((img_temp, padding))
            img = np.hstack((img, img_temp))

    return img
    #
