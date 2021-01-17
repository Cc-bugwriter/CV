import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from PIL import Image


from problem1 import *
def problem1():
    def load_img(path):
        color = Image.open(path)
        gray = color.convert("L")
        color = np.array(color) / 255
        gray = np.array(gray) / 255
        return color, gray

    def show_points(img, rows, cols):
        plt.figure()
        plt.imshow(img, interpolation="none")
        plt.plot(cols, rows ,"xr", linewidth=8)
        plt.axis("off")
        plt.show()

    def plot_heatmap(img, title=""):
        plt.imshow(img, "jet", interpolation="none")
        plt.axis("off")
        plt.title(title)

    # Set paramters and load the image
    sigma = 5
    threshold = 1.5e-3
    color, gray = load_img("data/a3p1.png")

    # Generate filters and compute Hessian
    fx, fy = derivative_filters()
    gauss = gauss2d(sigma, (25, 25))
    I_xx, I_yy, I_xy = compute_hessian(gray, gauss, fx, fy)

    # Show components of Hessian matrix
    plt.figure()
    plt.subplot(1,3,1)
    plot_heatmap(I_xx, "I_xx")
    plt.subplot(1,3,2)
    plot_heatmap(I_yy, "I_yy")
    plt.subplot(1,3,3)
    plot_heatmap(I_xy, "I_xy")
    plt.show()

    # Compute and show Hessian criterion
    criterion = compute_criterion(I_xx, I_yy, I_xy, sigma)
    plot_heatmap(criterion, "Determinant of Hessian")
    plt.show()

    # Show all interest points where criterion is greater than threshold
    rows, cols = np.nonzero(criterion > threshold)
    show_points(color, rows, cols)

    # Apply non-maximum suppression and show remaining interest points
    rows, cols = nonmaxsuppression(criterion, threshold)
    show_points(color, rows, cols)


from problem2 import Problem2
def problem2():

    def show_matches(im1, im2, pairs):
        plt.figure()
        plt.title("Keypoint matches")
        plt.imshow(np.append(im1, im2, axis=1), "gray", interpolation=None)
        plt.axis("off")
        shift = im1.shape[1]
        colors = pl.cm.viridis( np.linspace(0, 1 , pairs.shape[0]))
        for i in range(pairs.shape[0]):
            plt.scatter(x=pairs[i,0], y=pairs[i,1], color=colors[i])
            plt.scatter(x=pairs[i,2]+shift, y=pairs[i,3], color=colors[i])

    def show_image(im, title=""):
        plt.figure()
        plt.title(title)
        plt.imshow(im, "gray", interpolation=None)
        plt.axis("off")

    def stitch_images(im1, im2, H):
        h, w = im1.shape
        warped = np.zeros((h, 2*w))
        warped[:,:w] = im1
        im2 = Image.fromarray(im2)
        im3 = im2.transform(size=(2*w, h),
                                    method=Image.PERSPECTIVE,
                                    data=H.ravel(),
                                    resample=Image.BICUBIC)
        im3 = np.array(im3)
        warped[im3 > 0] = im3[im3 > 0]
        return warped

    # RANSAC Parameters
    ransac_threshold = 5.0  # inlier threshold
    p = 0.35                # probability that any given correspondence is valid
    k = 4                   # number of samples drawn per iteration
    z = 0.99                # total probability of success after all iterations

    P2 = Problem2()

    # load images
    im1 = plt.imread("data/a3p2a.png")
    im2 = plt.imread("data/a3p2b.png")

    # load keypoints
    data = np.load("data/keypoints.npz")
    keypoints1 = data['keypoints1']
    keypoints2 = data['keypoints2']

    # load SIFT features for the keypoints
    data = np.load("data/features.npz")
    features1 = data['features1']
    features2 = data['features2']

    # find matching keypoints
    distances = P2.euclidean_square_dist(features1,features2)
    pairs = P2.find_matches(keypoints1, keypoints2, distances)
    show_matches(im1, im2, pairs)

    # Compute homography matrix via ransac
    n_iters = P2.ransac_iters(p, k, z)
    H, num_inliers, inliers = P2.ransac(pairs, n_iters, k, ransac_threshold)
    print('Number of inliers:', num_inliers)
    warped = stitch_images(im1, im2, H)
    show_image(warped, "Ransac Homography")

    # recompute homography matrix based on inliers
    H = P2.recompute_homography(inliers)
    warped = stitch_images(im1, im2, H)
    show_image(warped, "Recomputed Homography")
    plt.show()


if __name__ == "__main__":
    problem1()
    # problem2()

