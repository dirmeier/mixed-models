import scipy as sp


def block_diag(m, times):
    return sp.linalg.block_diag(*[m for _ in range(times)])


def as_ranef_cov(nu):
    Q = sp.zeros(shape=(2, 2))
    Q[sp.tril_indices(2)] = nu
    return Q


def diag(n, nu):
    return sp.diag(nu * sp.ones(n))


def marginal_variance(U, G, R):
    return U.dot(G).dot(U.T) + R
