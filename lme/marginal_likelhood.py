import scipy as sp

from lme.util import v


def profile_mll(family):
    def _gaussian(par, y, X, U):
        sigma, nu = par[0], par[1:]
        n = len(y)
        q = int(U.shape[1] / 2)

        V, _, _ = v(sigma, nu, n, q, U)
        V_inv = sp.linalg.inv(V)

        X_Vinv_X = X.T.dot(V_inv).dot(X)
        b_hat = \
            sp.linalg.inv(X_Vinv_X) \
            .dot(X.T) \
            .dot(V_inv) \
            .dot(y)

        x_bhat = sp.dot(X, b_hat)
        wls = (y - x_bhat).T.dot(V_inv).dot(y - x_bhat)
        res = (sp.log(sp.linalg.det(V)) + wls).flatten()
        return res, X_Vinv_X

    return {
        'gaussian': _gaussian
    }.get(family, "gaussian")


def restricted_mll(family):
    pll = profile_mll(family)

    def _gaussian(nu, y, X, Z):
        pll_res, V_inv = pll(nu, y, X, Z)
        res = pll_res + sp.log(sp.linalg.det(V_inv)).flatten()
        return res, V_inv

    return {
        'gaussian': _gaussian
    }.get(family, "gaussian")
