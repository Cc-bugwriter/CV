import numpy as np
from scipy.ndimage import convolve, maximum_filter


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """

    #
    # You code here
    # smoothing image by gaussian kernel
    img_smooth = convolve(img, gauss, mode='mirror')

    # compute 2nd derivative in x and y direction
    I_x = convolve(img_smooth, fx, mode='mirror')
    I_xx = convolve(I_x, fx, mode='mirror')
    I_y = convolve(img_smooth, fy, mode='mirror')
    I_yy = convolve(I_y, fy, mode='mirror')

    I_xy = convolve(I_x, fy, mode='mirror')

    return I_xx, I_yy, I_xy
    #


def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """

    #
    # You code here
    h, w = I_xx.shape
    H_scaled = np.empty((h, w))

    for i in range(h):
        for j in range(w):
            H_scaled[i, j] = sigma ** 4 * (I_xx[i, j] * I_yy[i, j] - I_xy[i, j] ** 2)

    return H_scaled
    #


def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """

    #
    # You code here
    criterion_max_filter = maximum_filter(criterion, size=5, mode='constant')

    criterion_non_max = np.zeros(criterion_max_filter.shape)
    criterion_non_max[np.where(criterion == criterion_max_filter)] = \
        criterion[np.where(criterion == criterion_max_filter)]

    rows, cols = np.nonzero(criterion_non_max > threshold)

    # throw away interest points near by image boundary
    for i in range(rows.size):
        if (rows[i] < 5 or rows[i] >= criterion_max_filter.shape[0] - 5) or \
                (cols[i] < 5 or cols[i] >= criterion_max_filter.shape[1] - 5):
            rows[i] = 0
            cols[i] = 0

    return rows[rows.nonzero()], cols[cols.nonzero()]
    #
