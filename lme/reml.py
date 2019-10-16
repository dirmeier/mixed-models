import scipy as sp

from lme.ml import _profile_loglik


def _restricted_loglik(family):
    pll = _profile_loglik(family)
    def _gaussian(nu, y, X, Z):
        pll_res, v_nu_inv = pll(nu, y, X, Z)
        res = pll_res + sp.log(sp.linalg.det(v_nu_inv)).flatten()
        return res, v_nu_inv

    return {
        'gaussian': _gaussian
    }.get(family, "gaussian")


def predict_ranef_variance(y, X, U, family="gaussian"):
    nu0 = sp.array([1, 1, 0.5, 1])
    pll = _restricted_loglik(family)
    fn = lambda nu, y, X, U: sp.asscalar(pll(nu, y, X, U)[0])

    print("start")
    optim = sp.optimize.minimize(
      fn, nu0, args=(y, X, U),
      method='L-BFGS-B',
      bounds=((0.1, None), (None, 5), (None, 5), (None, 5)))
    print(optim)
    Q = sp.zeros(shape=(2, 2))
    Q[sp.tril_indices(2)] = optim.x[1:]
    print(Q.dot(Q.T))
    return optim
