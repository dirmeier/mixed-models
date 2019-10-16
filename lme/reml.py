import scipy as sp

from lme.ml import profile_loglik


def restricted_loglik(family):
    pll = profile_loglik(family)

    def _gaussian(nu, y, X, Z):
        pll_res, V_inv = pll(nu, y, X, Z)
        res = pll_res + sp.log(sp.linalg.det(V_inv)).flatten()
        return res, V_inv

    return {
        'gaussian': _gaussian
    }.get(family, "gaussian")


def reml(y, X, U, family="gaussian"):
    nu0 = sp.array([1, 1, 0.5, 1])
    pll = restricted_loglik(family)
    fn = lambda nu, y, X, U: sp.asscalar(pll(nu, y, X, U)[0])

    optim = sp.optimize.minimize(
      fn, nu0, args=(y, X, U),
      method='L-BFGS-B',
      bounds=((0.1, None), (None, 5), (None, 5), (None, 5)))
    return optim
