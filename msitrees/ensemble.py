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
        """Builds new estimator to ensemble"""
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

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series],
            y: Union[np.ndarray, pd.Series]) -> 'MSIRandomForestClassifier()':
        """
        TODO docstrings
        """

        x = self._validate_input(x, expected_dim=2)
        y = self._validate_input(y, expected_dim=1)

        if x.shape[0] != y.shape[0]:
            raise ValueError('Cannot match arrays with shapes '
                             '{} and {}'.format(x.shape, y.shape))

        # check if class labels are
        # label encoded from 0 to N
        if y.min() > 0:
            raise ValueError('Class labels should start from 0')

        classes = np.unique(y)
        n_class = len(classes)

        # check if classes go in sequence by one
        if n_class != max(classes) + 1:
            raise ValueError('Y is mislabeled')

        self._ncls = len(np.unique(y))
        self._shape = x.shape
        self._ndim = x.ndim

        # select seeding for each estimator.
        # ommitted by _add_estimator() if global
        # random state is not set
        maxint = np.iinfo(np.int32).max
        np.random.seed(self._random_state)
        seeds = np.random.randint(low=0, high=maxint, size=self._n_estimators)

        # build ensemble
        self._estimators = joblib.Parallel(n_jobs=self._n_jobs)(
            joblib.delayed(self._add_estimator)(x, y, seed) for seed in seeds
        )
        self._fitted = True

        return self

    def predict(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        TODO docstrings
        """

        x = self._validate_input(x, expected_dim=2, inference=True)
        stack = joblib.Parallel(n_jobs=self._n_jobs)(
            joblib.delayed(e._internal_predict)(x) for e in self._estimators
        )

        # transpose prediction stack so that prediction
        # classes for one obs are in one dim array
        stack = np.array(stack).T
        pred = [np.argmax(np.unique(x, return_counts=True)[1]) for x in stack]

        return np.array(pred)

    def predict_proba(self, x: Union[np.array, pd.DataFrame]) -> np.ndarray:
        """
        TODO docstrings
        """

        x = self._validate_input(x, expected_dim=2, inference=True)
        stack = joblib.Parallel(n_jobs=self._n_jobs)(
            joblib.delayed(e._internal_predict_proba)(x) for e in self._estimators
        )

        return np.array(stack).mean(axis=0)

    def predict_log_proba(self, x: Union[np.array, pd.DataFrame]) -> np.ndarray:
        """
        TODO docstrings
        """

        probas = self.predict_proba(x)
        logprob = [np.log(p) for p in probas]
        return np.array(logprob)

    def score(self):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass
