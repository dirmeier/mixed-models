import numpy as np
import pandas
import pandas as pd
from patsy import dmatrices
import scipy.stats as st
from sklearn.preprocessing import LabelEncoder

from lme.reml import predict_ranef_variance
from lme.util import block_diag

rmvnorm = st.multivariate_normal.rvs


def _build_ranef_model_matrix(tab, factor, ranef):
    inter_tab = tab[[factor, ranef]].copy()
    inter_tab['grp'] = LabelEncoder().fit_transform(tab.Subject)
    inter_tab = inter_tab[[factor, "grp"]].pivot(columns=factor).reindex()
    inter_tab.values[np.isfinite(inter_tab.values)] = 1
    inter_tab.values[np.isnan(inter_tab.values)] = 0

    slope_tab = tab[[factor, ranef]].copy().pivot(columns=factor).reindex()
    slope_tab.values[np.isnan(slope_tab.values)] = 0

    Z = pd.concat([inter_tab, slope_tab], axis=1, sort=True)
    Z = Z.reindex(sorted(Z.columns, key=lambda x: x[1]), axis=1)

    return Z.values


if __name__ == "__main__":
    tab = pandas.read_csv("./data/sleepstudy.csv")
    _, X = dmatrices("Reaction~ Days", tab)
    X = np.asarray(X)
    U = _build_ranef_model_matrix(tab, "Subject", "Days")

    n, p = X.shape
    q = int(U.shape[1] / 2)

    sd = 0.1
    beta = np.array([2, 1])
    Q = sd * np.array([[1, 0.25], [0.25, 1]])
    V = U.dot(block_diag(Q, q)).dot(U.T) + np.diag(sd * np.ones(n))
    print(Q)
    y = rmvnorm(size=1, mean=X.dot(beta), cov=V)

    predict_ranef_variance(y, X, U)
