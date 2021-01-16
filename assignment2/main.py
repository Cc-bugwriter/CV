import matplotlib.pyplot as plt
import numpy as np


#
# Problem 1
#
from problem1 import *

def problem1():
    """Example code implementing the steps in problem 1"""

    def showimage(img):
        plt.figure(dpi=150)
        plt.imshow(img, cmap="gray", interpolation="none")
        plt.axis("off")
        plt.show()

    # default values
    fsize = (5, 5)
    sigma = 1.4
    nlevel = 6

    # load image and build Gaussian pyramid
    img = loadimg("data/a2p1.png")
    gf = gauss2d(sigma, fsize)
    gpyramid = gaussianpyramid(img, nlevel, gf)
    showimage(createcompositeimage(gpyramid))

    # build Laplacian pyramid from Gaussian pyramid
    bf = binomial2d(fsize)
    lpyramid = laplacianpyramid(gpyramid, bf)

    # amplifiy high frequencies of Laplacian pyramid
    lpyramid_amp = amplifyhighfreq(lpyramid)
    showimage(createcompositeimage(lpyramid_amp))

    # reconstruct sharpened image from amplified Laplacian pyramid
    img_rec = reconstructimage(lpyramid_amp, bf)
    showimage(createcompositeimage((img, img_rec, img_rec - img)))


#
# Problem 2
#
import problem2 as p2

def problem2():
    """Example code implementing the steps in Problem 2"""

    def show_images(ims, hw, title='', size=(8, 2)):
        assert ims.shape[0] < 10, "Too many images to display"
        n = ims.shape[0]
        
        # visualising the result
        fig = plt.figure(figsize=size)
        for i, im in enumerate(ims):
            fig.add_subplot(1, n, i + 1)
            plt.imshow(im.reshape(*hw), "gray")
            plt.axis("off")
        fig.suptitle(title)


    # Load images
    imgs = p2.load_faces("./data/yale_faces")
    y = p2.vectorize_images(imgs)
    hw = imgs.shape[1:]
    print("Loaded array: ", y.shape)

    # Using 2 random images for testing
    test_face2 = y[0, :]
    test_face = y[-1, :]
    show_images(np.stack([test_face, test_face2], 0), hw,  title="Sample images")

    # Compute PCA
    mean_face, u, cumul_var = p2.compute_pca(y)

    # Compute PCA reconstruction
    # percentiles of total variance
    ps = [0.5, 0.75, 0.9, 0.95]
    ims = []
    for i, p in enumerate(ps):
        b = p2.basis(u, cumul_var, p)
        a = p2.compute_coefficients(test_face2, mean_face, b)
        ims.append(p2.reconstruct_image(a, mean_face, b))

    show_images(np.stack(ims, 0), hw, title="PCA reconstruction")

    # fix some basis
    b = p2.basis(u, cumul_var, 0.95)

    # Image search
    top5 = p2.search(y, test_face2, b, mean_face, 5)
    show_images(top5, hw, title="Image Search")

    # Interpolation
    ints = p2.interpolate(test_face2, test_face, b, mean_face, 5)
    show_images(ints, hw, title="Interpolation")

    plt.show()


if __name__ == "__main__":
    # problem1()
    # problem2()

    def show_images(ims, hw, title='', size=(8, 2)):
        assert ims.shape[0] < 10, "Too many images to display"
        n = ims.shape[0]

        # visualising the result
        fig = plt.figure(figsize=size)
        for i, im in enumerate(ims):
            fig.add_subplot(1, n, i + 1)
            # plt.imshow(im.reshape((96, 84)), "gray")
            plt.imshow(im.reshape(*hw), "gray")
            # print(*hw)
            plt.axis("off")
        fig.suptitle(title)


    # Load images
    imgs = p2.load_faces("./data/yale_faces")
    y = p2.vectorize_images(imgs)
    hw = imgs.shape[1:]

    # Using 2 random images for testing
    test_face2 = y[0, :]
    test_face = y[-1, :]
    # show_images(np.stack([test_face, test_face2], 0), hw, title="Sample images")

    # X = y
    # Y = y
    # Compute PCA
    mean_face, u, cumul_var = p2.compute_pca(y)

    # # Compute PCA reconstruction
    # # percentiles of total variance
    # ps = [0.5, 0.75, 0.9]
    # ims = []
    # for i, p in enumerate(ps):
    #     b = p2.basis(u, cumul_var, p)
    #     a = p2.compute_coefficients(test_face2, mean_face, b)
    #     ims.append(p2.reconstruct_image(a, mean_face, b))
    #
    # show_images(np.stack(ims, 0), hw, title="PCA reconstruction")

    # fix some basis
    b = p2.basis(u, cumul_var, 0.99)
    show_images(np.stack([test_face2], 0), hw, title="target images")

    # Image search
    x = test_face2
    top5 = p2.search(y, test_face2, b, mean_face, 5)
    show_images(top5, hw, title="Image Search")

    # # Interpolation
    # ints = p2.interpolate(test_face2, test_face, b, mean_face, 5)
    # show_images(ints, hw, title="Interpolation")

    plt.show()