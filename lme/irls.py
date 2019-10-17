
def working_observations(y, X, U, beta, gamma, invlink):
    eta = X.dot(beta) + U.dot(gamma)
    mean = invlink(eta)
    working = eta + (y - mean) / mean
    return working
