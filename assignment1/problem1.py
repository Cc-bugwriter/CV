import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """

    # Display data as an image, i.e., on a 2D regular raster.
    fig = plt.figure()
    plt.imshow(img)
    # Display all open figures.
    fig.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    # Save an array to a binary file in NumPy .npy format.
    np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    # Load arrays or pickled objects from .npy, .npz or pickled files.
    return np.load(path)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    return np.fliplr(img)


def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    fig = plt.figure()
    # display the normal image in subplot 1
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img1)
    plt.title("normal image")

    # display the mirrored image in subplot 2
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img2)
    plt.title("mirrored image")

    fig.show()
