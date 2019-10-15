import scipy


def _profile_loglik(family):
    def _gaussian(nu, Y, X, Z):

        G = scipy.zeros(shape=(Z.shape[1], Z.shape[1]))
        R = scipy.zeros(shape=(Y.shape[1], Y.shape[1]))

        v_nu = Z.matmul(G).matmul(Z.T) + R
        v_nu_inv = scipy.linalg.inv(v_nu)
        b_hat = \
            scipy.linalg.inv(X.T.matmul(v_nu_inv).matmul(X)) \
            .matmul(X.T)\
            .matmul(v_nu_inv)\
            .dot(y)
        x_bhat = scipy.dot(X, b_hat)
        wls = (y - x_bhat).T.matmul(v_nu_inv).matmul(y - x_bhat)
        res = .5 * (scipy.log(scipy.linalg.det(v_nu)) + wls)
        return res

    return {
        'gaussian': _gaussian
    }.get(family, "gaussian")


def _restricted_loglik(family):
    def _gaussian(nu, Y, X):
        return 1

    return {
        'gaussian': _gaussian
    }.get(family, "gaussian")


def predict_ranef_variance(Y, X, Z, family="gaussian"):
    logf = _restricted_loglik(family)
    nu0 = scipy.ones(shape=(10,))
    optim = scipy.optimize.minimize(logf, nu0)
    return optim
