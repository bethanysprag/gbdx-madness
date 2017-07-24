"""
This ir-mad function is copied from the work of Morton J. Canty
with some minor changes from the source code he provided here:
https://github.com/mortcanty/CRCPython
"""
import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy import linalg, stats


lib = ctypes.cdll.LoadLibrary('./task/madness/prov_means.so')

# At one point if the path wasn't as below it would error in docker.
# keeping this around incase the error returns.
# lib = ctypes.cdll.LoadLibrary('/task/madness/prov_means.so')

provmeans = lib.provmeans
provmeans.restype = None
c_double_p = ctypes.POINTER(ctypes.c_double)
provmeans.argtypes = [ndpointer(np.float64),
                      ndpointer(np.float64),
                      ctypes.c_int,
                      ctypes.c_int,
                      c_double_p,
                      ndpointer(np.float64),
                      ndpointer(np.float64)]


def geneiv(A, B):
    """Solves the generalized eigenvalue problem,
       From:
       https://github.com/mortcanty/CRCPython
    """
    Li = np.linalg.inv(linalg.cholesky(B).T)
    C = Li*A*(Li.T)
    C = np.asmatrix((C + C.T)*0.5, np.float32)

    eivs, V = np.linalg.eig(C)
    return eivs, Li.T*V


class Cpm(object):
    '''Provisional means algorithm'''
    def __init__(self, N):
        self.mn = np.zeros(N)
        self.cov = np.zeros((N, N))
        self.sw = 0.0000001

    def update(self, Xs, Ws=None):
        n, N = np.shape(Xs)

        if Ws is None:
            Ws = np.ones(n)

        sw = ctypes.c_double(self.sw)

        mn = self.mn
        cov = self.cov

        provmeans(Xs, Ws, N, n, ctypes.byref(sw), mn, cov)

        self.sw = sw.value
        self.mn = mn
        self.cov = cov

    def covariance(self):
        c = np.mat(self.cov / (self.sw-1.0))
        d = np.diag(np.diag(c))
        return c + c.T - d

    def means(self):
        return self.mn


def mad(t1, t2, penalty=0.9, stop_criteria=0.001,  num_iters=100):
    """Perform ir-mad.

    Performs the iteratively reweighted multivariate
    alteration detection algorithm as described here:

    :param t1: numpy array for t1
    :param t2: numpy array for t2
    :param penalty: the penalty, added to matrices to ensure they are
                    invertable.
    :param num_iters: number of iterations for iterative reweighting
    :returns: mad transformed im1, mad transformed im2_new,
              mad transformed difference image, chisq
    """
    rows, cols, bands = t1.shape

    lam = penalty
    # this is added below to a couple matrices
    # to guarentee that they are invertable.

    cpm = Cpm(2*bands)

    oldrho = np.zeros(bands)

    tile = np.zeros((cols, 2*bands))

    delta = 1.0
    itr = 0

    # these get assigned later in the second loop,
    # this just keeps my linter happy.
    sigMADs = means1 = means2 = A = B = 0

    while True:
        # exit criteria.
        if (delta < stop_criteria) or (itr > num_iters):
            break

        for row in range(rows):
            # a tile is the rows of both the
            # input images concatenated together.
            # We iteratively calculate the weights
            # for the images row wise.

            tile[:, :bands] = t1[row, :, :]
            tile[:, bands:] = t2[row, :, :]

            if itr > 0:
                # If not the first mad iteration
                # Create canonical variates.
                # Subtract canonical variates of img1 from img2
                # creating the mad variates.
                # means, sigMADs, A, B only exists on second
                # iteration.
                mads = np.asarray((tile[:, :bands]-means1)*A -
                                  (tile[:, bands:]-means2)*B)

                # In first mad iteration (sigMADs = 0)
                # so this would error.
                chisqr = np.sum((mads/sigMADs)**2, axis=1)

                # Weights are the probabilities from the previous iteration
                wts = 1-stats.chi2.cdf(chisqr, [bands])

                # we update the wts for each row
                cpm.update(tile, wts)

            else:
                # if the first mad iteration.
                # update the weighted cov matrix
                # and the means with the tiles.
                cpm.update(tile)

        # get weighted covariance matrices and means
        S = cpm.covariance()
        means = cpm.means()

        # reset prov means object
        # for next iteration.
        cpm.__init__(2*bands)

        # s11 = covariance matrix of the first image, N_bands X N_bands
        # s22 = covariance matrix of second image
        s11 = S[:bands, :bands]
        s22 = S[bands:, bands:]

        # lam is just a small value to ensure that
        # s22 and s11 are not degenerate.
        s11 = (1-lam)*s11 + lam*np.eye(bands)
        s22 = (1-lam)*s22 + lam*np.eye(bands)

        s12 = S[:bands, bands:]
        s21 = S[bands:, :bands]

        # multiply each covariance matrix it's inverse
        c1 = s12*linalg.inv(s22)*s21
        b1 = s11
        c2 = s21*linalg.inv(s11)*s12
        b2 = s22

        # solution of generalized eigenproblems
        if bands > 1:
            # We are getting the eigenvalues of
            # A*A^-1, A
            mu2a, A = geneiv(c1, b1)
            mu2b, B = geneiv(c2, b2)

            # sort eigenvectors
            idx = np.argsort(mu2a)
            A = A[:, idx]

            idx = np.argsort(mu2b)
            B = B[:, idx]
            mu2 = mu2b[idx]

        else:
            # if single band image
            mu2 = c1/b1
            A = 1/np.sqrt(b1)
            B = 1/np.sqrt(b2)

        # canonical correlations
        # why do we throw awaya mu2a?
        # i am guessing that the eigenvectors
        # are the same.
        mu = np.sqrt(mu2)

        a2 = np.diag(A.T*A)
        b2 = np.diag(B.T*B)

        try:
            sigma = np.sqrt((2-lam*(a2+b2))/(1-lam)-2*mu)
            rho = mu*(1-lam) / np.sqrt((1-lam*a2)*(1-lam*b2))
        except RuntimeWarning:
            # we break out and just use the data from the previous
            # iteration
            break

        # stopping criterion
        delta = max(abs(rho-oldrho))
        print os.getpid(), delta, rho
        oldrho = rho

        # tile the sigmas and means
        # numpy tile is the same as repmat in matlab
        sigMADs = np.tile(sigma, (cols, 1))
        means1 = np.tile(means[:bands], (cols, 1))
        means2 = np.tile(means[bands:], (cols, 1))

        # ensure sum of positive correlations between X and U is positive
        D = np.diag(1/np.sqrt(np.diag(s11)))
        s = np.ravel(np.sum(D*s11*A, axis=0))
        A = A*np.diag(s/np.abs(s))

        # ensure positive correlation between each pair of canonical variates
        cov = np.diag(A.T*s12*B)
        B = B*np.diag(cov/np.abs(cov))
        itr += 1

    # pre-allocate output arrays
    im1_new = np.zeros((rows, cols, bands))
    im2_new = np.zeros((rows, cols, bands))
    out_mads = np.zeros((rows, cols, bands)).astype(np.float32)
    chisq = np.zeros((rows, cols)).astype(np.float32)

    # apply the final A and B to the original input images to minimize
    # their variance with respect to one another and then find the
    # difference image. (find the difference between the canonical
    # variates to get the mad variates)

    for row in range(rows):
        im1_new[row, :, :] = (t1[row, :, :bands]-means1)*A
        im2_new[row, :, :] = (t2[row, :, :bands]-means2)*B

        # make difference image
        out_mads[row, :, :] = ((t1[row, :, :]-means1)*A -
                               (t2[row, :, :]-means2)*B)

        chisq[row, :] = np.sum((out_mads[row, :, :]/sigMADs)**2, axis=1)

    # calculate prob from chisq with n_bands degrees of freedom
    # prob = 1-gammainc(chisq, bands)
    # prob = 1-stats.chi2.cdf(chisq, [bands])

    return im1_new, im2_new, out_mads, chisq
