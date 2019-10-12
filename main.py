import numpy as np
import pandas
import pandas as pd
from patsy import dmatrices
from sklearn.preprocessing import LabelEncoder

tab = pandas.read_csv("./data/sleepstudy.csv")

Y, X = dmatrices("Reaction ~ Days", tab)


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

    return Z


Z = _build_ranef_model_matrix(tab, "Subject", "Days")

print(Z)