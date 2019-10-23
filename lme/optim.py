import scipy as sp


def optim(fn, init, bounds, args, iter=1000):
    optim = sp.optimize.minimize(
        fn,
        init,
        args=args,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": iter},
    )
    return optim
