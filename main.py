import pandas as pd
import scipy as sp
import scipy.stats as st
from patsy import dmatrices
from sklearn.preprocessing import LabelEncoder

from lme.ls import working_response, working_weight, irls
from lme.marginal_likelhood import restricted_mll
from lme.optim import optim
from lme.util import block_diag, cholesky_factor, ranef_variance, diag

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

    return sp.asarray(Z.values)


def lme():
    y = rmvnorm(mean=X.dot(beta) + U.dot(gamma),
                cov=sp.diag(sd * sp.ones(n)))

    pll = restricted_mll("gaussian")
    fn = lambda nu, y, X, U: sp.asscalar(pll(nu, y, X, U)[0])
    opt = optim(fn,
                sp.array([2, 5, 0.5, 1]),
                ((0.01, None), (None, None), (None, None), (None, None)),
                args=(y, X, U))
    
    sd_hat, nu_hat = opt.x[0], opt.x[1:]
    G_hat = ranef_variance(nu_hat, q)
    R_hat = diag(n, sd_hat)
    b_hat, gamma_hat = irls(X, U, G_hat, 1/sp.diag(R_hat), y)

    print("sigma/sigma_hat:\n{}/{}\n".format(sd, sd_hat))
    print("Q/Q_hat:\n{}/\n{}\n".format(Q, cholesky_factor(nu_hat)))
    print("beta/beta_hat:\n{}/{}\n".format(beta, b_hat))
    print("gamma/gammahat:\n{}/\n{}\n".format(gamma, gamma_hat))


def glme():
    y = rpois(mu=sp.exp(X.dot(beta) + U.dot(gamma)))

    pll = restricted_mll("poisson")
    fn = lambda nu, y, X, U, W, b: sp.asscalar(pll(nu, y, X, U, W, b)[0])

    b_tilde = bold = sp.ones(shape=p)
    g_tilde = gold = sp.ones(shape=q * 2)
    
    while True:
        y_tilde = working_response(y, X, U, b_tilde, g_tilde, sp.exp, sp.exp)
        w_tilde = working_weight(y, X, U, b_tilde, g_tilde, sp.exp, sp.exp)
        nu_hat = optim(fn,
                       sp.array([1, 0.5, 0.1]),
                       bounds=((None, None), (None, None), (None, None)),
                       args=(y_tilde, X, U, sp.diag(w_tilde), b_tilde),
                       iter=1).x
        
        G_hat = ranef_variance(nu_hat, q)
        b_tilde, g_tilde = irls(X, U, G_hat, w_tilde, y_tilde)

        if sp.sum((b_tilde - bold) ** 2) < 0.00001 and \
           sp.sum((g_tilde - gold) ** 2) < 0.00001:
           break
        bold, gold = b_tilde, g_tilde
    
    print("beta/beta_hat:\n{}/{}\n".format(beta, b_tilde))
    print("gamma/gammahat:\n{}/\n{}\n".format(gamma, g_tilde))


if __name__ == "__main__":
    import scipy
    scipy.random.seed(2)
    tab = pd.read_csv("./data/sleepstudy.csv")
    _, X = dmatrices("Reaction~ Days", tab)
    X = sp.asarray(X) / sp.asarray(X).mean()
    U = _build_ranef_model_matrix(tab, "Subject", "Days")

    n, p = X.shape
    q = int(U.shape[1] / 2)

    sd = 0.1
    beta = sp.array([1, 2])
    Q = sd * sp.array([[1, 0.5], [0.5, 1]])
    gamma = rmvnorm(mean=sp.zeros(q * 2), cov=block_diag(Q, q))
    
    glme()
