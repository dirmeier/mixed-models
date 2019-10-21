import scipy as sp
from lme.util import v, ranef_variance, marginal_variance


def profile_mll(family):
    def _gaussian(par, y, X, U):
        sigma, nu = par[0], par[1:]
        n = len(y)
        q = int(U.shape[1] / 2)

        V, _, _ = v(sigma, nu, n, q, U)
        V_inv = sp.linalg.inv(V)

        X_Vinv_X = X.T.dot(V_inv).dot(X)
        b_hat = sp.linalg.inv(X_Vinv_X).dot(X.T).dot(V_inv).dot(y)

        x_bhat = sp.dot(X, b_hat)
        wls = (y - x_bhat).T.dot(V_inv).dot(y - x_bhat)
        res = (sp.log(sp.linalg.det(V)) + wls).flatten()
        return res, X_Vinv_X

    def _poisson(par, y, X, U, W, b_hat):
        nu = par
        q = int(U.shape[1] / 2)
    
        V = marginal_variance(U, sp.linalg.inv(ranef_variance(nu, q)), W)
        V_inv = sp.linalg.inv(V)
        X_Vinv_X = X.T.dot(V_inv).dot(X)
    
        x_bhat = sp.dot(X, b_hat)
        wls = (y - x_bhat).T.dot(V_inv).dot(y - x_bhat)
        res = (sp.log(sp.linalg.det(V)) + wls).flatten()
        return res, X_Vinv_X

    return {"gaussian": _gaussian, "poisson": _poisson}.get(family, _gaussian)


def restricted_mll(family):
    pll = profile_mll(family)

    def _gaussian(nu, y, X, Z):
        pll_res, X_Vinv_X = pll(nu, y, X, Z)
        res = pll_res + sp.log(sp.linalg.det(X_Vinv_X)).flatten()
        return res, X_Vinv_X

    def _poisson(par, y, X, U, W, b_hat):
        pll_res, X_Vinv_X = pll(par, y, X, U, W, b_hat)
        res = pll_res + sp.log(sp.linalg.det(X_Vinv_X)).flatten()
        return res, X_Vinv_X

    return {"gaussian": _gaussian, "poisson": _poisson}.get(family, _gaussian)
