import numpy as np
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 108
TEMPORAL_CUTOFF = 2010


def models():
    return {
        "ols": make_pipeline(StandardScaler(), LinearRegression()),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED),
    }


def in_sample_adj_r2(X, y):
    return sm.OLS(y, sm.add_constant(X)).fit().rsquared_adj


def cv_r2(model, X, y, folds=5):
    scores = cross_val_score(clone(model), X, y, cv=KFold(folds, shuffle=True, random_state=SEED), scoring="r2")
    return scores.mean(), scores.std()


def cv_predictions(model, X, y, folds=5):
    return cross_val_predict(clone(model), X, y, cv=KFold(folds, shuffle=True, random_state=SEED))


def temporal_r2(model, X, y, years, cutoff=TEMPORAL_CUTOFF):
    train, test = np.asarray(years) < cutoff, np.asarray(years) >= cutoff
    if train.sum() == 0 or test.sum() == 0:
        return np.nan
    fitted = clone(model).fit(X[train], y[train])
    return r2_score(y[test], fitted.predict(X[test]))
