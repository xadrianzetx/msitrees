import joblib
import numpy as np
import pandas as pd
from typing import Union, Optional

from msitrees._base import MSIBaseClassifier
from msitrees.tree import MSIDecisionTreeClassifier


class MSIRandomForestClassifier(MSIBaseClassifier):

    def __init__(self, n_estimators: int = 100,
                 bootstrap: bool = True,
                 feature_sampling: bool = True,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self._estimators = []
        self._importances = None
        self._n_estimators = n_estimators
        self._bootstrap = bootstrap
        self._sample_feats = feature_sampling
        self._n_jobs = n_jobs
        self._random_state = random_state

    def __repr__(self):
        return 'MSIRandomForestClassifier()'

    @property
    def feature_importances_(self):
        pass

    def _add_estimator(self, x: np.ndarray,
                       y: np.ndarray,
                       seed: int) -> MSIDecisionTreeClassifier:
        """Builds new estimator for ensemble"""
        if self._bootstrap:
            # sample dataset with replacement
            # before fitting new estimator
            indices = np.arange(start=0, stop=self._shape[0])

            if self._random_state:
                np.random.seed(seed)

            bstrap_idx = np.random.choice(indices, size=self._shape[0])
            x = x[bstrap_idx, :]
            y = y[bstrap_idx]

        # fit decision tree classifier
        clf = MSIDecisionTreeClassifier()
        clf._internal_fit(
            x, y,
            n_class=self._ncls,
            sample_features=self._sample_feats
        )

        return clf

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def predict_log_proba(self):
        pass

    def score(self):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass
