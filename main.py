import scipy as sp
import pandas
import pandas as pd
from patsy import dmatrices
import scipy.stats as st
from sklearn.preprocessing import LabelEncoder

from lme.optim import optim
from lme.ls import wls, solve_gamma, working_response
from lme.marginal_likelhood import restricted_mll, profile_mll
from lme.util import block_diag, as_ranef_cov, v

rmvnorm = st.multivariate_normal.rvs
rpois = st.poisson.rvs


def _build_ranef_model_matrix(tab, factor, ranef):
    inter_tab = tab[[factor, ranef]].copy()
    inter_tab['grp'] = LabelEncoder().fit_transform(tab.Subject)
    inter_tab = inter_tab[[factor, "grp"]].pivot(columns=factor).reindex()
    inter_tab.values[sp.isfinite(inter_tab.values)] = 1
    inter_tab.values[sp.isnan(inter_tab.values)] = 0

    slope_tab = tab[[factor, ranef]].copy().pivot(columns=factor).reindex()
    slope_tab.values[sp.isnan(slope_tab.values)] = 0

    Z = pd.concat([inter_tab, slope_tab], axis=1, sort=True)
    Z = Z.reindex(sorted(Z.columns, key=lambda x: x[1]), axis=1)

    return Z.values


def lme():
    n, p = X.shape
    q = int(U.shape[1] / 2)

    sd = 0.1
    beta = sp.array([2, 1])
    Q = sd * sp.array([[1, 0.25], [0.25, 1]])
    gamma = rmvnorm(mean=sp.zeros(q * 2), cov=block_diag(Q, q))

    y = rmvnorm(mean=X.dot(beta) + U.dot(gamma),
                cov=sp.diag(sd * sp.ones(n)))

    pll = restricted_mll("gaussian")
    fn = lambda nu, y, X, U: sp.asscalar(pll(nu, y, X, U)[0])
    optimz = optim(fn, y, X, U)
    sd_hat, nu_hat = optimz['sigma'], optimz['nu']

    print("sigma/sigma_hat:\n{}/{}\n".format(sd, sd_hat))
    print("Q/Q_hat:\n{}/\n{}\n".format(Q, as_ranef_cov(nu_hat)))

    V_hat, G_hat, R_hat = v(sd_hat, nu_hat, n, q, U)
    b_hat = wls(y, X, V_hat)
    gamma_hat = solve_gamma(y, X, G_hat, U, V_hat, b_hat)

    print("beta/beta_hat:\n{}/{}\n".format(beta, b_hat))
    print("gamma/gammahat:\n{}/\n{}\n".format(gamma, gamma_hat))


def glme():
    n, p = X.shape
    q = int(U.shape[1] / 2)

    sd = 0.1
    beta = sp.array([2, 1])
    Q = sd * sp.array([[1, 0.25], [0.25, 1]])
    gamma = rmvnorm(mean=sp.zeros(q * 2), cov=block_diag(Q, q))

    y = rpois(mu=sp.exp(X.dot(beta) + U.dot(gamma)))

    pll = restricted_mll("poisson")
    fn = lambda nu, y, X, U: sp.asscalar(pll(nu, y, X, U)[0])

    b_tilde = bold = sp.ones(shape=p)
    g_tilde = gold = sp.ones(shape=q * 2)
    while True:
        y_tilde = working_response(y, X, U, b_tilde, g_tilde, sp.absolute)
        optimz = optim(fn, y_tilde, X, U, iter=1)
        sd_hat, nu_hat = optimz['sigma'], optimz['nu']
        V_hat, G_hat, R_hat = v(sd_hat, nu_hat, n, q, U)
        b_tilde = wls(y, X, V_hat)
        g_tilde = solve_gamma(y, X, G_hat, U, V_hat, b_tilde)

        if sp.sum((b_tilde - bold) ** 2) < 0.1 and \
           sp.sum((g_tilde - gold) ** 2) < 0.1:
           break
        bold, gold = b_tilde, g_tilde


    print("sigma/sigma_hat:\n{}/{}\n".format(sd, sd_hat))
    print("Q/Q_hat:\n{}/\n{}\n".format(Q, as_ranef_cov(nu_hat)))

    print("beta/beta_hat:\n{}/{}\n".format(beta, b_tilde))
    print("gamma/gammahat:\n{}/\n{}\n".format(gamma, g_tilde))


if __name__ == "__main__":
    sp.random.seed(42)
    tab = pandas.read_csv("./data/sleepstudy.csv")
    _, X = dmatrices("Reaction~ Days", tab)
    U = _build_ranef_model_matrix(tab, "Subject", "Days")

    glme()
