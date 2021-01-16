import numpy as np
import os
from PIL import Image


def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images.
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        imgs: (N, H, W) numpy array
    """
    
    #
    # You code here
    # for root, dirs, files in walk(path):

    # create a list to catch each single image data
    img_lst = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name[-4:] == ext:
                img = Image.open(os.path.join(root, name))
                img_lst.append(np.array(img.getdata()).reshape((img.size[1], img.size[0])))

    # merge all array in a 3D array
    img_arr = np.empty((len(img_lst), img_lst[0].shape[0], img_lst[0].shape[1]))
    for i in range(len(img_lst)):
        img_arr[i, :, :] = img_lst[i]

    return img_arr


def vectorize_images(imgs):
    """Turns an  array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images into
    an (N, M) array where M=H*W is the image dimension.
    
    Args:
        imgs: (N, H, W) numpy array
    
    Returns:
        x: (N, M) numpy array
    """
    
    #
    # You code here
    x = np.empty((imgs.shape[0], imgs.shape[1] * imgs.shape[2]))
    for i in range(x.shape[0]):
        x[i, :] = imgs[i, :, :].flatten()

    return x
    #


def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an numpy array with N M-dimensional features
    
    Returns:
        mean_face: (M,) numpy array representing the mean face
        u: (M, M) numpy array, bases with D principal components
        cumul_var: (N, ) numpy array, corresponding cumulative variance
    """

    #
    # You code here
    mean_face = np.mean(X, axis=0)

    # to turbo the computation process, here choose SVD of X features instead of eig of covariance_matrix
    _, S, u = np.linalg.svd(X)
    # corresponding cumulative variance is the project of covariance_matrix in principal components
    cumul_var = S * S / (len(X) - 1)  # unbiased estimation

    return mean_face, u.T, cumul_var
    #


def basis(u, cumul_var, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) numpy array containing principal components.
        For example, i'th vector is u[:, i]
        cumul_var: (N, ) numpy array, variance along the principal components.
    
    Returns:
        v: (M, D) numpy array, contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """
    
    #
    # You code here
    for i in range(len(cumul_var)):
        v = u[:, :i + 1]
        if (np.sum(cumul_var[:i + 1]) / np.sum(cumul_var)) >= p:
            break
    print(v.shape)
    return v
    #


def compute_coefficients(face_image, mean_face, u):
    """Computes the coefficients of the face image with respect to
    the principal components u after projection.
    
    Args:
        face_image: (M, ) numpy array (M=h*w) of the face image a vector
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        a: (D, ) numpy array, containing the coefficients
    """
    
    #
    # You code here
    return u.T @ (face_image - mean_face)
    #


def reconstruct_image(a, mean_face, u):
    """Reconstructs the face image with respect to
    the first D principal components u.
    
    Args:
        a: (D, ) numpy array containings the image coefficients w.r.t
        the principal components u
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (M, ) numpy array, projected vector of face_image on 
        principal components
    """
    
    #
    # You code here
    return mean_face + u @ a
    #


def compute_similarity(Y, x, u, mean_face):
    """Compute the similarity of an image x to the images in Y
    based on the cosine similarity.

    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) image we would like to retrieve
        u: (M, D) bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector

    Returns:
        sim: (N, ) numpy array containing the cosine similarity values
    """

    #
    # You code here
    x_pca = compute_coefficients(x, mean_face, u)

    sim = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        y_pca = compute_coefficients(Y[i, :].T, mean_face, u)
        # compute similarity
        sim[i] = (y_pca.T @ x_pca) / (np.linalg.norm(y_pca) * np.linalg.norm(x_pca))

    return sim
    #


def search(Y, x, u, mean_face, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) numpy array, image we would like to retrieve
        u: (M, D) numpy arrray, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y_sort: (top_n, M) numpy array containing the top_n most similar images
        sorted by similarity
    """

    #
    # You code here
    sim = compute_similarity(Y, x, u, mean_face)
    # sort the similarity reversely (Up: highest Down: lowest)
    sim_sort = np.sort(sim)
    sim_sort = sim_sort[::-1]

    # search top-n candidates
    Y_sort = np.empty((top_n, Y.shape[1]))
    for i in range(top_n):
        for k in range(len(sim)):
            if sim[k] == sim_sort[i]:
                Y_sort[i, :] = Y[k, :]

    return Y_sort
    #


def interpolate(x1, x2, u, mean_face, n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (M, ) numpy array, the first image
        x2: (M, ) numpy array, the second image
        u: (M, D) numpy array, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        n: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate n equally-spaced points on a line
    
    Returns:
        Y: (n, M) numpy arrray, interpolated results.
        The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """

    #
    # You code here
    # obtain projection coefficients
    a_1 = compute_coefficients(x1, mean_face, u)
    a_2 = compute_coefficients(x2, mean_face, u)

    a_steps = np.linspace(a_1, a_2, num=n)

    Y = np.empty((n, x1.shape[0]))
    for i in range(n):
        Y[i, :] = reconstruct_image(a_steps[i], mean_face, u)

    return Y
    #
