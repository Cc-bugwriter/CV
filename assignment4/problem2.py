from functools import partial
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.ndimage import convolve

####################
# Provided functions
####################


conv2d = partial(convolve, mode="mirror")


def gauss2d(fsize, sigma):
    """ Create a 2D Gaussian filter

    Args:
        fsize: (w, h) dimensions of the filter
        sigma: width of the Gaussian filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def downsample(img, fsize=(5, 5), sigma=1.4):
    """
    Downsampling an image by a factor of 2

    Args:
        img: image as (h, w) np.array
        fsize and sigma: parameters for Gaussian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as (h/2, w/2) np.array
    """
    g_k = gauss2d(fsize, sigma)
    img = conv2d(img, g_k)
    return img[::2, ::2]


def gaussian_pyramid(img, nlevels=3, fsize=(5, 5), sigma=1.4):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        fsize: gaussian kernel size
        sigma: sigma of gaussian kernel

    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    pyramid = [img]
    for i in range(0, nlevels - 1):
        pyramid.append(downsample(pyramid[i], fsize, sigma))

    return pyramid


def resize(arr, shape):
    """ Resize an image to target shape

    Args:
        arr: image as (h, w) np.array
        shape: target size (h', w') as tuple

    Returns:
        resized image as (h', w') np.array
    """
    return np.array(Image.fromarray(arr).resize(shape[::-1]))


######################
# Basic Lucas-Kanade #
######################


def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives

    Args:
        im1: first image as (h, w) np.array
        im2: second image as (h, w) np.array

    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
                    as (h, w) np.array
    """

    #
    # You code here
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    sobel_x = sobel_x / np.sum(abs(sobel_x))
    sobel_y = sobel_x.T
    Ix = conv2d(im1, sobel_x)
    Iy = conv2d(im1, sobel_y)
    It = im2 - im1

    return Ix, Iy, It
    #


def compute_motion(Ix, Iy, It, patch_size=15):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t each as (h, w) np.array
        patch_size: specifies the side of the square region R in Eq. (1)
    Returns:
        u: optical flow in x direction as (h, w) np.array
        v: optical flow in y direction as (h, w) np.array
    """

    #
    # You code here
    # sum in region
    kernel_patch = np.ones((patch_size, patch_size))

    I_xx = conv2d(Ix * Ix, kernel_patch)
    I_xy = conv2d(Ix * Iy, kernel_patch)
    I_yy = conv2d(Iy * Iy, kernel_patch)
    I_xt = conv2d(Ix * It, kernel_patch)
    I_yt = conv2d(Iy * It, kernel_patch)

    h, w = Ix.shape

    u = np.empty((h, w))
    v = np.empty((h, w))
    # LK in center of region
    for i in range(h):
        for j in range(w):
            str_tensor = np.array([[I_xx[i, j], I_xy[i, j]],
                                   [I_xy[i, j], I_yy[i, j]]])
            time_tensor = - np.vstack((I_xt[i, j], I_yt[i, j]))
            flow = np.linalg.inv(str_tensor) @ time_tensor
            u[i, j] = flow[0].reshape(-1)
            v[i, j] = flow[1].reshape(-1)

    return u, v
    #


def warp(im, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        im: input image as (h, w) np.array
        u, v: optical flow in x and y direction each as (h, w) np.array

    Returns:
        im_warp: warped image as (h, w) np.array
    """

    #
    # # You code here
    h, w = im.shape
    mesh_v, mesh_h = np.mgrid[0:h, 0:w].astype(np.float)

    # transformation
    mesh_x_warp = mesh_h + u    # -> horizontal direction
    mesh_y_warp = mesh_v + v    # -> vertical direction

    # preprocess for griddata
    values = im.flatten()

    mesh_x_warp = mesh_x_warp.flatten()
    mesh_y_warp = mesh_y_warp.flatten()
    points = np.stack([mesh_x_warp, mesh_y_warp], axis=-1)

    # interpolation to view img
    im_warp = interpolate.griddata(points, values, (mesh_h, mesh_v), method='linear', fill_value=0)

    return im_warp
    #


def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade.
    Args:
        im1, im2: Images as (h, w) np.array

    Returns:
        Cost as float scalar
    """

    #
    # You code here
    error = (im1 - im2) * (im1 - im2)
    return np.sum(error)
    #


###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(pyramid1, pyramid2, n_iter=10):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.

    Args:
        pyramid1, pyramid2: Gaussian pyramids corresponding to
                            im1 and im2, in fine to coarse order
        n_iter: number of refinement iterations

    Returns:
        u: OF in x direction as np.array
        v: OF in y direction as np.array
    """

    #
    # You code here
    # global
    u = np.zeros(pyramid1[-1].shape)
    v = np.zeros(pyramid1[-1].shape)

    for i in reversed(range(len(pyramid1))):
        if i == len(pyramid1) - 1:
            print("ok")
            # first iteration
        else:
            u = resize(u, pyramid1[i].shape) * 2.0
            v = resize(u, pyramid1[i].shape) * 2.0
        for j in range(n_iter):
            pyr_wrap = warp(pyramid1[i], u, v)
            Ix, Iy, It = compute_derivatives(pyr_wrap, pyramid2[i])
            u_delta, v_delta = compute_motion(Ix, Iy, It)

            u_delta = np.nan_to_num(u_delta)
            v_delta = np.nan_to_num(v_delta)

            u += u_delta
            v += v_delta

    return u, v

    #
