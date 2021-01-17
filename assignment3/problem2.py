import numpy as np


class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """
    #
    # You code here
    #

    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """
        
    #
    # You code here
    #


    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """
        
    #
    # You code here
    #


    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """

    #
    # You code here
    #


    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices shoul be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

    #
    # You code here
    #


    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """

    #
    # You code here
    #


    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
    #
    # You code here
    #


    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
    #
    # You code here
    #


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
    #
    # You code here
    #



    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """
    #
    # You code here
    #


    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
    #
    # You code here
    #