import scipy as sp


def block_diag(m, times):
    return sp.linalg.block_diag(*[m for _ in range(times)])


def cholesky_factor(nu):
    assert len(nu) == 3
    Q = sp.zeros(shape=(2, 2))
    Q[sp.tril_indices(2)] = nu
    Q = Q.dot(Q.T)
    return Q


def diag(n, nu):
    return sp.diag(nu * sp.ones(n))


def marginal_variance(U, G, R):
    return U.dot(G).dot(U.T) + R


def ranef_variance(nu, q):
    return block_diag(cholesky_factor(nu), q)


def v(sigma, nu, n, q, U):
    G = ranef_variance(nu, q)
    R = diag(n, sigma)
    V = marginal_variance(U, G, R)
    return V, G, R
