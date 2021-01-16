import math
import numpy as np
from scipy import ndimage


def gauss2d(sigma, fsize):
    """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """

    g = np.empty(fsize)

    for i in range(fsize[0]):
        for j in range(fsize[1]):
            g[i, j] = 1 / (2 * math.pi * sigma ** 2) * \
                      math.exp(-((i - np.median(range(fsize[0]))) ** 2 + (j - np.median(range(fsize[1]))) ** 2)
                               / (2 * (sigma ** 2)))

    return g / np.sum(abs(g))  # normalized


def createfilters():
    """
  Returns:
    fx, fy: filters as described in the problem assignment
  """
    # central differences
    f_d = np.array([1, 0, -1]).reshape((-1, 3))/2

    # smoothing with Gaussian filter
    sigma = 0.9
    f_avg = gauss2d(sigma, (3, 1))

    # combine filter
    fy = f_avg @ f_d
    fx = fy.T

    return fx, fy


def filterimage(I, fx, fy):
    """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

    Ix = ndimage.convolve(I, fx)
    Iy = ndimage.convolve(I, fy)
    return Ix, Iy


def detectedges(Ix, Iy, thr=0.029750135 + 0.050867412):
    """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """
    # threshold derivative
    # threshold = 0.029750135 + 0.050867412
    # 0.029750135 is the mean value of gradient magnitude in all of image
    # 0.050867412 ist the standard deviation of gradient magnitude in all of image
    # with empirical rule in statistics, using the addition of mean value and standard deviation
    # could guarantee this threshold more than 68.27% of all gradient magnitude (1 sigma rule in Normal distribution)

    # gradient magnitude
    I_magn = np.sqrt(Ix ** 2 + Iy ** 2)

    # thresholding
    edges = I_magn
    edges[edges <= thr] = 0

    return edges

def nonmaxsupp(edges, Ix, Iy):
    """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """

    # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]

    # You code here
    # gradient magnitude
    ref = np.sqrt(Ix ** 2 + Iy ** 2)
    edges2 = edges

    # theta in [-90, -67.5]
    for i in range(ref.shape[0]-1):  # avoid IndexError
        for j in range(ref.shape[1]):
            if edges2[i, j] != 0 and i != 0:  # avoid IndexError
                if edges2[i+1, j] > edges2[i, j] or edges2[i-1, j] > edges2[i, j]:
                    edges2[i, j] = 0

    # handle left-to-right edges: theta in (-22.5, 22.5]

    # You code here
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]-1):  # avoid IndexError
            if edges2[i, j] != 0 and j != 0:
                if edges2[i, j+1] > edges2[i, j] or edges2[i, j-1] > edges2[i, j]:
                    edges2[i, j] = 0

    # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

    # Your code here
    for i in range(ref.shape[0] - 1):  # avoid IndexError
        for j in range(ref.shape[1] - 1):  # avoid IndexError
            if edges2[i, j] != 0 and (i != 0 or j != 0):  # avoid IndexError
                if edges2[i - 1, j + 1] > edges2[i, j] or edges2[i + 1, j - 1] > edges2[i, j]:
                    edges2[i, j] = 0

    # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

    # Your code here
    for i in range(ref.shape[0] - 1):  # avoid IndexError
        for j in range(ref.shape[1] - 1):  # avoid IndexError
            if edges2[i, j] != 0 and (i != 0 or j != 0):  # avoid IndexError
                if edges2[i+1, j+1] > edges2[i, j] or edges2[i - 1, j - 1] > edges2[i, j]:
                    edges2[i, j] = 0

    return edges2