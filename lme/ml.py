import scipy as sp

from lme.util import block_diag, as_ranef_cov, diag, marginal_variance


def profile_loglik(family):
    def _gaussian(nu, y, X, U):
        n = len(y)
        q = int(U.shape[1] / 2)

        G = block_diag(as_ranef_cov(nu[1:]), q)
        R = diag(n, nu[0])
        V = marginal_variance(U, G, R)
        V_inv = sp.linalg.inv(V)

        X_Vinv_X = X.T.dot(V_inv).dot(X)
        b_hat = \
            sp.linalg.inv(X_Vinv_X) \
            .dot(X.T)\
            .dot(V_inv)\
            .dot(y)

        x_bhat = sp.dot(X, b_hat)
        wls = (y - x_bhat).T.dot(V_inv).dot(y - x_bhat)
        res = (sp.log(sp.linalg.det(V)) + wls).flatten()
        return res, X_Vinv_X

    return {
        'gaussian': _gaussian
    }.get(family, "gaussian")


def ml(y, X, U, family="gaussian"):
    nu0 = sp.array([1, 1, 0.5, 1])
    pll = profile_loglik(family)
    fn = lambda nu, y, X, U: sp.asscalar(pll(nu, y, X, U)[0])

    optim = sp.optimize.minimize(
      fn, nu0, args=(y, X, U),
      method='Nelder-Mead')
    return optim
