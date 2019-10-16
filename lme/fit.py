import scipy as sp


def optim(fn, y, X, U):
    optim = sp.optimize.minimize(
      fn, sp.array([1, 1, 0.5, 1]), args=(y, X, U),
      method='L-BFGS-B',
      bounds=((0.01, None), (None, None), (None, None), (None, None)))
    sigma, nu = optim.x[0], optim.x[1:]

    return {
        'sigma': sigma,
        'nu': nu
    }