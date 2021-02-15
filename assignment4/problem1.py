import numpy as np
import matplotlib.pyplot as plt



def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2, 2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """

    #
    # You code here
    # second svd to F_
    U_F, D_F, V_F = np.linalg.svd(A)
    D_F[-1] = 0.0
    D_matrix = np.eye(D_F.size) * D_F
    A_hat = U_F @ D_matrix @ V_F

    return A_hat
    #



def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    #
    # You code here
    n = p1.shape[0]
    A = np.empty((n, 9))

    for i in range(n):
        x_, y_ = p1[i, :-1]
        x, y = p2[i, :-1]
        A[i] = np.array([x * x_, y * x_, x_, x * y_, y * y_, y_, x, y, 1.0])
    # first svd to A
    _, _, V = np.linalg.svd(A)
    F_ = V[-1, :].reshape(3, 3)

    F = enforce_rank2(F_)

    return F
    #



def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    #
    # You code here
    p1_, T1 = condition_points(p1)
    p2_, T2 = condition_points(p2)

    F_hat = compute_fundamental(p1_, p2_)

    # unconditioned coordinates
    F = T1.T @ F_hat @ T2

    return F
    #



def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """

    #
    # You code here
    # epipolar lines in other img
    # l:[a, b, c] -> ax+by+c=0
    p1_ = np.concatenate([p1, np.ones((p1.shape[0], 1))], axis=1)
    l1 = p1_ @ F.T  # -> (F @ p1_.T).T

    X1 = np.zeros(p1.shape[0])
    Y1 = (- l1[:, 0] * X1 - l1[:, 2]) / l1[:, 1]
    X2 = img.shape[1] * np.ones(p1.shape[0])
    Y2 = (- l1[:, 0] * X2 - l1[:, 2]) / l1[:, 1]

    return X1, X2, Y1, Y2
    #



def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """

    #
    # You code here
    residual = np.abs(np.diagonal(p1 @ F @ p2.T))
    max_residual = np.max(residual)
    avg_residual = np.mean(residual)

    return max_residual, avg_residual
    #


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    #
    # You code here
    # epipolar point in img
    # methode 1 : use eig
    eig_val, eig_vec = np.linalg.eig(F @ F.T)
    index = np.argmin(eig_val)
    e1 = eig_vec[:-1, index] / eig_vec[-1, index]

    eig_val, eig_vec = np.linalg.eig(F.T @ F)
    index = np.argmin(eig_val)
    e2 = eig_vec[:-1, index] / eig_vec[-1, index]

    # methode 2 : use svd
    u, s, vh = np.linalg.svd(F)
    e1 = u[:-1, -1] / u[-1, -1]
    e2 = vh[-1, :-1] / vh[-1, -1]

    return e1, e2
    #
