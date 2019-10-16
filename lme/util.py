import scipy as sp
from scipy.linalg import inv


def block_diag(m, times):
    return sp.linalg.block_diag(*[m for _ in range(times)])


def as_ranef_cov(nu):
    assert len(nu) == 3
    Q = sp.zeros(shape=(2, 2))
    Q[sp.tril_indices(2)] = nu
    Q = Q.dot(Q.T)
    return Q


def diag(n, nu):
    return sp.diag(nu * sp.ones(n))


def marginal_variance(U, G, R):
    return U.dot(G).dot(U.T) + R


def v(sigma, nu,  n, q, U):
    G = block_diag(as_ranef_cov(nu), q)
    R = diag(n, sigma)
    V = marginal_variance(U, G, R)
    return V, G, R


def wls(y, X, W):
    XT_Winv = X.T.dot(inv(W))
    return inv(XT_Winv.dot(X)).dot(XT_Winv).dot(y)


def solve_gamma(y, X, G, U, V, bhat):
    return G.dot(U.T).dot(inv(V)).dot(y - X.dot(bhat))
