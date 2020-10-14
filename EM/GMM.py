import numpy as np
from scipy.stats import multivariate_normal


def e_step(X, pi, mu, sigma):
    """

    :param X: training example of shape Nxd
    :param pi: prior cluster probability (dimension cx1)
    :param mu: mean of each cluster cxd
    :param sigma: covariance matrix (cxdxd)
    :return: gamma, distribution of latent variables
    """
    N = X.shape[0]
    c = mu.shape[0]

    gamma = np.zeros((N, c))

    for i in range(c):
        gamma[:, i] = pi[i] * multivariate_normal.pdf(X, mean=mu[i, :], cov=sigma[i, ...])

    gamma = gamma / np.sum(gamma, axis=1)
    return gamma


def m_step(X, gamma):
    """
    :type gamma: posterior distribution

    """
    N = X.shape[0]
    C = gamma.shape[1]
    d = X.shape[1]
    mu = np.zeros((C, d))
    sigma = np.zeros((C, d, d))
    pi = np.sum(gamma, axis=0) / (1.0 * N)

    for i in range(C):
        mu[i, :] = np.sum(X * gamma[:, i][:, np.newaxis], axis=0) / (pi[i] * N)
        X_i = X - mu[i, :][np.newaxis, :]
        sigma[i, :, :] = np.dot((X_i * gamma[:, i][:, np.newaxis]).T, X_i) / (pi[i] * N)

    return pi, mu, sigma



