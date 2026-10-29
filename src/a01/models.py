"""Models M0-M4 and nested tuning (protocol v1.0, section 5)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261029


class MedianByGroup(BaseEstimator, RegressorMixin):
    """M0: median of the training target within a categorical column (global median for unseen levels)."""

    def __init__(self, group_col="property_type"):
        self.group_col = group_col

    def fit(self, X, y):
        s = pd.Series(np.asarray(y), index=X.index)
        self.medians_ = s.groupby(X[self.group_col]).median().to_dict()
        self.global_ = float(s.median())
        return self

    def predict(self, X):
        return X[self.group_col].map(self.medians_).fillna(self.global_).to_numpy(dtype=float)


def _pre(numeric, categorical, scale):
    # Median imputation is fitted on the training fold only (the original 2016 release has a few NaNs).
    steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(steps), numeric),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])


GRIDS = {
    "M2_ridge": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
    "M3_random_forest": {"model__max_features": [0.33, 0.66], "model__min_samples_leaf": [1, 5]},
    "M4_hist_gradient_boosting": {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]},
}


def make_pipeline(name, numeric, categorical):
    if name == "M2_ridge":
        est, scale = Ridge(), True
    elif name == "M3_random_forest":
        est, scale = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED), False
    elif name == "M4_hist_gradient_boosting":
        est, scale = HistGradientBoostingRegressor(max_iter=500, early_stopping=True, random_state=SEED), False
    else:
        raise ValueError(name)
    return Pipeline([("pre", _pre(numeric, categorical, scale)), ("model", est)])


def fit_predict(name, train, test, numeric, categorical, y_col, inner="grouped", group_col="OSEBuildingID",
                type_col="property_type"):
    """Fit one model with nested tuning on `train`; return (test predictions, train predictions, best params)."""
    Xtr, Xte, ytr = train[numeric + categorical], test[numeric + categorical], train[y_col].to_numpy()
    if name == "M0_median_by_type":
        m = MedianByGroup(type_col).fit(train[[type_col]], ytr)
        return m.predict(test[[type_col]]), m.predict(train[[type_col]]), {}
    if name == "M1_persistence":
        return test["persist_log"].to_numpy(), train["persist_log"].to_numpy(), {}
    pipe = make_pipeline(name, numeric, categorical)
    if inner == "grouped":
        cv, groups = GroupKFold(n_splits=3, shuffle=True, random_state=SEED), train[group_col].to_numpy()
    else:
        cv, groups = KFold(n_splits=3, shuffle=True, random_state=SEED), None
    gs = GridSearchCV(pipe, GRIDS[name], cv=cv, scoring="neg_mean_absolute_error", n_jobs=4 if
                      name != "M3_random_forest" else 1, refit=True)
    gs.fit(Xtr, ytr, groups=groups)
    return gs.predict(Xte), gs.predict(Xtr), {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
