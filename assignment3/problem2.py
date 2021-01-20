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
        distances = np.empty((features2.shape[1], features1.shape[1]))
        for i in range(features2.shape[1]):
            for j in range(features1.shape[1]):
                distances[i, j] = np.linalg.norm(features2[:, i] - features1[:, j]) ** 2

        return distances
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
        n, m = distances.shape
        pairs = np.empty((min(n, m), 4))

        if n < m:
            for i in range(n):
                pairs[i, :2] = p1[np.where(distances[i, :] == min(distances[i, :]))]
                pairs[i, 2:] = p2[i, :]
        else:
            for i in range(m):
                pairs[i, :2] = p1[i, :]
                pairs[i, 2:] = p2[np.where(distances[:, i] == min(distances[:, i]))]

        return pairs
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
        # p1_ = np.copy(p1)
        # p2_ = np.copy(p2)
        # np.random.shuffle(p1_)
        # np.random.shuffle(p2_)
        #
        # index_1 = np.random.randint(p1.shape[0], size=k)
        # sample1 = p1_[index_1]
        # index_2 = np.random.randint(p2.shape[0], size=k)
        # sample2 = p2_[index_2]

        index_ = np.random.randint(p1.shape[0], size=k)
        sample1 = p1[index_]
        sample2 = p2[index_]

        return sample1, sample2
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

        # s = 1.0 / 2.0 * np.max(np.linalg.norm(points, axis=1))
        s = 1.0 / 2.0 * np.max(np.abs(points), axis=0)
        t = np.mean(points, axis=0)

        # T = np.array([[1.0 / s, 0., -t[0] / s],
        #               [0., 1.0 / s, -t[1] / s],
        #               [0., 0., 1.0]])
        T = np.array([[1.0 / s[0], 0., -t[0] / s[0]],
                      [0., 1.0 / s[1], -t[1] / s[1]],
                      [0., 0., 1.0]])

        points_ = np.hstack((points, np.ones(points.shape[0]).reshape(-1, 1)))
        ps = (T @ points_.T).T

        return ps, T
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
        A = np.empty((2 * p1.shape[0], 9))

        for i in range(p1.shape[0]):
            x_ = p2[i, 0]
            y_ = p2[i, 1]
            A[2 * i, :] = np.array(
                [0, 0, 0, p1[i, 0], p1[i, 1], p1[i, 2], -p1[i, 0] * y_, -p1[i, 1] * y_, -p1[i, 2] * y_])
            A[2 * i + 1, :] = np.array(
                [-p1[i, 0], -p1[i, 1], -p1[i, 2], 0, 0, 0, p1[i, 0] * x_, p1[i, 1] * x_, p1[i, 2] * x_])

        # SVD decomposition
        _, _, vh = np.linalg.svd(A)

        HC = vh[-1, :].reshape(3, -1)
        H = np.linalg.pinv(T2) @ HC @ T1
        # normalize
        HC = HC / HC[-1, -1]
        H = H / H[-1, -1]

        return H, HC
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
        p_ = np.copy(p)
        p_ = np.hstack((p_, np.ones(p_.shape[0]).reshape(-1, 1)))

        points = (H @ p_.T).T

        # normalization to reduce 3nd dimension in points (go back to non-homogenous coordinate)
        for i, point in enumerate(points):
            if point[-1] != 0:
                point = point / point[-1]
            else:
                point = np.zeros((1, 3))
            points[i, :] = point

        return points[:, :2]
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

        dist = np.linalg.norm(self.transform_pts(p1, H) - p2, axis=1) ** 2 + \
               np.linalg.norm(p1 - self.transform_pts(p2, np.linalg.pinv(H)), axis=1) ** 2

        return dist
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
        if (dist <= threshold).any():
            N = np.sum(dist <= threshold)
            inliers = pairs[np.where(dist <= threshold)]
        else:
            N = 0
            inliers = np.zeros((N, 4))

        return N, inliers
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
        return int(np.ceil(np.log(1 - z) / np.log(1 - p ** k)))
        #

    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            k: number of samples drawn per iteration
            threshold: inlier detection threshold

        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """

        #
        # You code here
        dist_ = 2e4
        H = np.empty((3, 3))
        max_inliers = 0
        best_inliers = 0
        inliers = []

        for i in range(n_iters):
            # estimate homography based on random sample paris
            p1, p2 = self.pick_samples(pairs[:, :2], pairs[:, 2:], k)
            p1_ps, T_p1 = self.condition_points(p1)
            p2_ps, T_p2 = self.condition_points(p2)
            H_i, _ = self.compute_homography(p1_ps, p2_ps, T_p1, T_p2)

            # transform all points and measure distance
            dist = self.compute_homography_distance(H_i, pairs[:, :2], pairs[:, 2:])
            N_i, inliers_i = self.find_inliers(pairs, dist, threshold)

            # means of the homography distance specified above
            dist_mean = np.mean(dist)

            # if N_i > best_inliers or \
            #         (N_i == best_inliers and dist_mean < dist_):
            if N_i > best_inliers:
            # if len(inliers_i) != 0 and dist_mean < dist_:
                H = H_i
                dist_ = dist_mean
                best_inliers = N_i
                # print(N_i)
                # print(len(inliers_i))
                # print(dist_mean)
                max_inliers += N_i
                for inlier in inliers_i:
                    inliers.append(inlier.tolist())

            # if len(inliers_i) != 0:
            #     max_inliers += N_i
            #     for inlier in inliers_i:
            #         inliers.append(inlier.tolist())

        inliers_array = np.asarray(inliers)

        return H, max_inliers, inliers_array
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
        if len(inliers) == 0:
            H = np.empty((3, 3))
            # possible H matrix
            # [[1.17392289e+00 -5.77082033e-02 -2.10266472e+02]
            #  [1.56935833e-01  1.22326315e+00 -5.52129965e+01]
            #  [4.72118770e-04 3.05875508e-04  1.00000000e+00]]
            # [[7.22089504e-01 - 5.36837222e-01 -7.43742921e+00]
            #  [2.53040613e-01  4.23592935e-01 -1.26596114e+00]
            #  [9.52979225e-04 - 2.22816468e-03 1.00000000e+00]]
            # [[2.55254493e+01  9.28108309e+00 -6.18116175e+03]
            #  [1.01401621e+01  1.93294021e+01 -3.12980935e+03]
            #  [4.62266803e-02  1.53791051e-02 1.00000000e+00]]
            # [[1.01437245e+00 -2.95655209e-01 -1.39701205e+02]
            #  [2.06548555e-01 1.59763011e+00 -1.18115538e+02]
            #  [-4.09437377e-04 1.99710535e-03 1.00000000e+00]]
            # [[2.12898423e+00 -3.50242016e-01 -3.59719197e+02]
            #  [5.40436917e-01 8.53914674e-01 -4.92351929e+01]
            #  [4.01816657e-03 -2.38609861e-03 1.00000000e+00]]
            # [[-2.17149512e+01  3.47410783e+00 3.85870287e+03]
            #  [-1.35288452e+01 -8.55272136e+00 2.65301309e+03]
            # [-5.55880467e-02 4.92369876e-03 1.00000000e+00]]
            # [[-2.52809922e+01 -4.49018097e+00  5.87029674e+03]
            #  [-1.39336260e+01 -2.64284143e+01  4.64840959e+03]
            #  [-5.23800378e-02 -3.91470768e-02 1.00000000e+00]]
        else:
            p1_ps, T_p1 = self.condition_points(inliers[:, :2])
            p2_ps, T_p2 = self.condition_points(inliers[:, 2:])
            H, _ = self.compute_homography(p1_ps, p2_ps, T_p1, T_p2)

        return H
        #
