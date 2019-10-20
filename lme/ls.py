import jax.scipy as sp
from jax.scipy.linalg import inv

def wls(y, X, W):
    XT_Winv = X.T.dot(inv(W))
    return inv(XT_Winv.dot(X)).dot(XT_Winv).dot(y)


def solve_gamma(y, X, G, U, V, bhat):
    return G.dot(U.T).dot(inv(V)).dot(y - X.dot(bhat))


def working_response(y, X, U, beta, gamma, invlink, grad):
    eta = X.dot(beta) + U.dot(gamma)
    mean = invlink(eta)
    deriv = grad(eta)
    Dinv = sp.diag(1 / deriv)
    working = eta + Dinv.dot(y - mean)
    return working