import numpy as np
from numpy.linalg import inv


def wls(y, X, W):
    XT_Winv = X.T.dot(inv(W))
    return inv(XT_Winv.dot(X)).dot(XT_Winv).dot(y)


def solve_gamma(y, X, G, U, V, bhat):
    return G.dot(U.T).dot(inv(V)).dot(y - X.dot(bhat))


def working_response(y, X, U, beta, gamma, invlink, grad):
    eta = X.dot(beta) + U.dot(gamma)
    mean = invlink(eta)
    deriv = grad(eta)
    Dinv = np.diag(1 / deriv)
    working = eta + Dinv.dot(y - mean)
    return working


def working_weight(y, X, U, beta, gamma, invlink, grad):
    eta = X.dot(beta) + U.dot(gamma)
    return grad(eta)**2 / invlink(eta)


def irls(X, U, G, w, y):
    
    C = np.hstack([X, U])
    W = np.diag(w)
    CTW = C.T.dot(W)
    CTWC = CTW.dot(C)
    B = np.zeros_like(CTWC)
    B[(B.shape[0] - G.shape[0]):, (B.shape[1] - G.shape[1]):] = np.linalg.inv(G)
    est = np.linalg.inv(CTWC + B).dot(C.T).dot(W).dot(y)
    return est[:X.shape[1]] , est[X.shape[1]:]
