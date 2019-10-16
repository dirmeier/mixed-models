import scipy as sp

from lme.util import block_diag


def _profile_loglik(family):
    def _gaussian(nu, y, X, U):
        n = len(y)
        q = int(U.shape[1] / 2)

        Q = sp.zeros(shape=(2, 2))
        Q[sp.tril_indices(2)] = nu[1:]
        G = block_diag(Q, q)
        R = sp.diag(nu[0] * sp.ones(n))

        v_nu = U.dot(G).dot(U.T) + R
        v_nu_inv = sp.linalg.inv(v_nu)
        x_vinv_x = X.T.dot(v_nu_inv).dot(X)
        b_hat = \
            sp.linalg.inv(x_vinv_x) \
            .dot(X.T)\
            .dot(v_nu_inv)\
            .dot(y)

        x_bhat = sp.dot(X, b_hat)
        wls = (y - x_bhat).T.dot(v_nu_inv).dot(y - x_bhat)
        res = (sp.log(sp.linalg.det(v_nu)) + wls).flatten()
        return res, x_vinv_x

    return {
        'gaussian': _gaussian
    }.get(family, "gaussian")
