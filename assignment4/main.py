import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import flow_to_color


#
# Problem 1
#

import problem1 as P1
def problem1():

    def cart2hom(p_c):
        p_h = np.concatenate([p_c, np.ones((p_c.shape[0], 1))], axis=1)
        return p_h

    def plot_epipolar(F, points1, points2, im1, im2):
        n = points1.shape[0]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(im1, interpolation=None)
        ax1.axis("off")
        ax1.scatter(points1[:,0], points1[:,1])
        ax1.set_title("Epipolar lines in the left image")
        # determine coordinates of the epipolar lines
        # at the left and right borders of image 1
        X1, X2, Y1, Y2 = P1.draw_epipolars(F, points2, im1)
        for i in range(n):
            ax1.plot([X1[i], X2[i]], [Y1[i], Y2[i]], "r")
        ax1.set_ylim(top=0)

        ax2.imshow(im2, interpolation=None)
        ax2.axis("off")
        ax2.scatter(points2[:,0], points2[:,1])
        ax2.set_title("Epipolar lines in the right image")
        # determine coordinates of the epipolar lines
        # at the left and right borders of image 2
        X1, X2, Y1, Y2 = P1.draw_epipolars(F.T, points1, im2)
        for i in range(n):
            ax2.plot([X1[i], X2[i]], [Y1[i], Y2[i]], "r")
        ax2.set_ylim(top=0)

    im1 = plt.imread("data/a4p1a.png")
    im2 = plt.imread("data/a4p1b.png")

    data = np.load("data/points.npz")
    points1 = data['points1']
    points2 = data['points2']

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im1, interpolation=None)
    ax1.axis("off")
    ax1.scatter(points1[:,0], points1[:,1])
    ax1.set_title("Keypoints in the left image")
    ax2.imshow(im2, interpolation=None)
    ax2.axis("off")
    ax2.scatter(points2[:,0], points2[:,1])
    ax2.set_title("Keypoints in the right image")

    # convert to homogeneous coordinates
    p1 = cart2hom(points1)
    p2 = cart2hom(points2)
    # compute fundamental matrix
    F = P1.eight_point(p1, p2)
    # plot the epipolar lines
    plot_epipolar(F, points1, points2, im1, im2)

    # compute the residuals
    max_residual, avg_residual = P1.compute_residuals(p1, p2, F)
    print('Max residual: ', max_residual)
    print('Average residual: ', avg_residual)

    # compute the epipoles
    e1, e2 = P1.compute_epipoles(F)
    print('Epipole in image 1: ', e1)
    print('Epipole in image 2: ', e2)

    plt.show()


#
# Problem 2
#

import problem2 as p2

def problem2():
    def show_two(gray_im, flow_im):
        fig = plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
        (ax1, ax2) = fig.subplots(1, 2)
        ax1.imshow(gray_im, "gray", interpolation=None)
        ax2.imshow(flow_im)
        plt.show()

    # Loading the image and scaling to [0, 1]
    im1 = np.array(Image.open("data/a4p2a.png")) / 255.0
    im2 = np.array(Image.open("data/a4p2b.png")) / 255.0

    #
    # Basic implementation
    #
    Ix, Iy, It = p2.compute_derivatives(im1, im2) # gradients
    u, v = p2.compute_motion(Ix, Iy, It) # flow

    # stacking for visualization
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_two(im1, rgb_image)

    # warping 1st image to the second
    im1_warped = p2.warp(im1, u, v)
    cost = p2.compute_cost(im1_warped, im2)
    print(f"Cost (basic): {cost:4.3e}")

    #
    # Iterative coarse-to-fine implementation
    #
    n_iter = 4 # number of iterations
    n_levels = 3 # levels in Gaussian pyramid

    pyr1 = p2.gaussian_pyramid(im1, nlevels=n_levels)
    pyr2 = p2.gaussian_pyramid(im2, nlevels=n_levels)

    u, v = p2.coarse_to_fine(pyr1, pyr2, n_iter)

    # warping 1st image to the second
    im1_warped = p2.warp(im1, u, v)
    cost = p2.compute_cost(im1_warped, im2)
    print(f"Cost (coarse-to-fine): {cost:4.3e}")

    # stacking for visualization
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_two(im1, rgb_image)


if __name__ == "__main__":
    # problem1()
    # problem2()

    def show_two(gray_im, flow_im):
        fig = plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
        (ax1, ax2) = fig.subplots(1, 2)
        ax1.imshow(gray_im, "gray", interpolation=None)
        ax2.imshow(flow_im)
        plt.show()


    # Loading the image and scaling to [0, 1]
    im1 = np.array(Image.open("data/a4p2a.png")) / 255.0
    im2 = np.array(Image.open("data/a4p2b.png")) / 255.0


    # Basic implementation

    Ix, Iy, It = p2.compute_derivatives(im1, im2)  # gradients
    u, v = p2.compute_motion(Ix, Iy, It)  # flow

    # stacking for visualization
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_two(im1, rgb_image)

    # warping 1st image to the second
    im1_warped = p2.warp(im1, u, v)
    cost = p2.compute_cost(im1_warped, im2)
    print(f"Cost (basic): {cost:4.3e}")


    # Iterative coarse-to-fine implementation

    n_iter = 4  # number of iterations
    n_levels = 3  # levels in Gaussian pyramid

    pyr1 = p2.gaussian_pyramid(im1, nlevels=n_levels)
    pyr2 = p2.gaussian_pyramid(im2, nlevels=n_levels)

    u, v = p2.coarse_to_fine(pyr1, pyr2, n_iter)

    # warping 1st image to the second
    im1_warped = p2.warp(im1, u, v)
    cost = p2.compute_cost(im1_warped, im2)
    print(f"Cost (coarse-to-fine): {cost:4.3e}")

    # stacking for visualization
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_two(im1, rgb_image)

