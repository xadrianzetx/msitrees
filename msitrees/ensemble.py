import joblib
import numpy as np
import pandas as pd
from typing import Union, Optional

from msitrees._base import MSIBaseClassifier
from msitrees.tree import MSIDecisionTreeClassifier


class MSIRandomForestClassifier(MSIBaseClassifier):

    def __init__(self, n_estimators: int = 100,
                 bootstrap: bool = True,
                 max_features: Union[int, float, str] = 'auto',
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self._estimators = []
        self._importances = None
        self._n_estimators = n_estimators
        self._bootstrap = bootstrap
        self._max_features = max_features
        self._n_jobs = n_jobs
        self._random_state = random_state

    def __repr__(self):
        return 'MSIRandomForestClassifier()'

    @property
    def feature_importances_(self):
        pass

    def _num_features_per_tree(self) -> Optional[int]:
        """Resolves number of features to use
        when fitting estimator.
        """
        if self._max_features is None:
            # no feature subsampling
            return self._shape[1]

        if isinstance(self._max_features, int):
            # user specified number of features
            # to use directly
            if self._max_features > self._shape[1]:
                raise ValueError('max_features is greater than '
                                 'overall number of features in dataset')
            return self._max_features

        if isinstance(self._max_features, float):
            # number of features to use specified
            # as fraction of overall shape
            if self._max_features > 1.0:
                raise ValueError('Feature fraction used to build '
                                 'a tree cannot be greater than 1')
            return round(self._max_features * self._shape[1])

        modes = {
            **dict.fromkeys(['auto', 'sqrt'], int(np.sqrt(self._shape[1]))),
            'log2': np.log2(self._shape[1])
        }

        try:
            num_features = modes[self._max_features]
            return num_features

        except KeyError:
            msg = 'Allowed feature fraction modes are {} got {}.'
            allowed = list(modes.keys())
            raise ValueError(msg.format(allowed, self._max_features))

    def _add_estimator(self, x: np.ndarray,
                       y: np.ndarray) -> MSIDecisionTreeClassifier:
        """Builds new estimator for ensemble

        TODO feature fraction should + should track which
        features went to which estimator if feature fraction
        is used
        """
        if self._bootstrap:
            # sample dataset with replacement
            # before fitting new estimator
            indices = np.arange(start=0, stop=self._shape[0])
            bstrap_idx = np.random.choice(indices, size=self._shape[0])
            x = x[bstrap_idx, :]
            y = y[bstrap_idx]

        # fit decision tree classifier
        clf = MSIDecisionTreeClassifier()
        clf._internal_fit(x, y, n_class=self._ncls)

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
