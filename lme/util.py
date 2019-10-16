import scipy as sp


def block_diag(m, times):
    return sp.linalg.block_diag(*[m for _ in range(times)])
