import joblib
import numpy as np
import pandas as pd
from typing import Union, Optional

from msitrees.tree import MSIDecisionTreeClassifier


class MSIRandomForestClassifier:

    def __init__(self, n_estimators: int = 100,
                 bagging_fraction: float = 1.0,
                 feature_fraction: float = 1.0,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 **kwargs):
        self._estimators = []
        self._fitted = False
        self._shape = None
        self._ncls = None
        self._ndim = None
        self._importances = None
        self._n_estimators = n_estimators
        self._frac_bagging = bagging_fraction
        self._frac_feats = feature_fraction
        self._n_jobs = n_jobs
        self._random_state = random_state

    def __repr__(self):
        pass

    @property
    def feature_importances_(self):
        pass

    def _add_estimator(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def predict_log_proba(self):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass
