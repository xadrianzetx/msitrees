import joblib
import inspect
import numpy as np
import pandas as pd
from collections import Counter
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
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.feature_sampling = feature_sampling
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __repr__(self):
        return 'MSIRandomForestClassifier()'

    @property
    def feature_importances_(self):
        """Returns feature importances

        Each feature importance is calculated as normalized
        sum of gini based information gain at nodes where
        split was made on that particular feature. For random forest
        classifier, final importance is a mean over all estimators.

        Returns
        -------
        importances : np.ndarray
            Normalized array of feature importances.
        """

        if not self._fitted:
            return np.array([])

        allimp = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(getattr)(e, 'feature_importances_') for e in self._estimators
        )

        # calculate overall feature importance
        # as the mean over all estimators
        importances = np.mean(allimp, axis=0)

        return importances / sum(importances)

    def _add_estimator(self, x: np.ndarray,
                       y: np.ndarray,
                       seed: int) -> MSIDecisionTreeClassifier:
        """Adds new estimator to ensemble"""

        if self.bootstrap:
            # sample dataset with replacement
            # before fitting new estimator
            indices = np.arange(start=0, stop=self._shape[0])

            if self.random_state:
                np.random.seed(seed)

            bstrap_idx = np.random.choice(indices, size=self._shape[0])
            x = x[bstrap_idx, :]
            y = y[bstrap_idx]

        # fit decision tree classifier
        clf = MSIDecisionTreeClassifier()
        clf._internal_fit(
            x, y,
            n_class=self._ncls,
            sample_features=self.feature_sampling
        )

        return clf

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series],
            y: Union[np.ndarray, pd.Series]) -> 'MSIRandomForestClassifier':
        """Fits random forest classifier to training dataset.

        Parameters
        ----------
        x : np.ndarray
            Training data of shape (n_samples, n_features)
            or (n_samples, ). All values have to be numerical,
            so perform any required encoding before calling this
            method.

        y: np.ndarray
            Ground truth data of shape (n_samples, ). All values
            have to be numerical, so perform any required encoding
            before calling this method.

        Returns
        -------
        self : MSIRandomForestClassifier
            Fitted estimator.
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
        np.random.seed(self.random_state)
        seeds = np.random.randint(low=0, high=maxint, size=self.n_estimators)

        # build ensemble
        self._estimators = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._add_estimator)(x, y, seed) for seed in seeds
        )
        self._fitted = True

        return self

    def predict(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predicts class labels for each sample in input data X

        Parameters
        ----------
        x : np.ndarray
            Array of samples with shape (n_samples, n_features).
            Class label is predicted for each sample.

        Returns
        -------
        pred : np.ndarray
            Array with shape (n_samples, )
            Class label prediction for each sample.
        """

        x = self._validate_input(x, expected_dim=2, inference=True)
        stack = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(e._internal_predict)(x) for e in self._estimators
        )

        # transpose prediction stack so that prediction
        # classes for one obs are in one dim array
        stack = np.array(stack).T
        pred = [Counter(s).most_common(1)[0][0] for s in stack]

        return np.array(pred)

    def predict_proba(self, x: Union[np.array, pd.DataFrame]) -> np.ndarray:
        """Predicts class probability for each sample in input data X.

        Probability is defined as fraction of class
        label in a leaf.

        Parameters
        ----------
        x : np.ndarray
            Array of samples with shape (n_samples, n_features).
            Class probabilities are predicted for each sample.

        Returns
        -------
        probas : np.ndarray
            Array with shape (n_samples, n_targets)
            Array of probabilities. Each index corresponds to
            class label and holds predicted porbability of this class.
        """

        x = self._validate_input(x, expected_dim=2, inference=True)
        stack = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(e._internal_predict_proba)(x) for e in self._estimators
        )

        return np.array(stack).mean(axis=0)

    def predict_log_proba(self, x: Union[np.array, pd.DataFrame]) -> np.ndarray:
        """Predicts class log probability for each sample in input data X.

        Probability is defined as fraction of class
        label in a leaf.

        Parameters
        ----------
        x : np.ndarray
            Array of samples with shape (n_samples, n_features).
            Class log probabilities are predicted for each sample.

        Returns
        -------
        logprobas : np.ndarray
            Array with shape (n_samples, n_targets)
            Array of log probabilities. Each index corresponds to
            class label and holds predicted log porbability of this class.
        """

        probas = self.predict_proba(x)
        logprob = [np.log(p) for p in probas]
        return np.array(logprob)

    def score(self, x: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.DataFrame]) -> float:
        """Predicts class label for each sample in X
        and computes accuracy score wrt. ground truth.

        Parameters
        ----------
        x : np.ndarray
            Array of samples with shape (n_samples, n_features).
            Class label is predicted for each sample.

        y : np.ndarray
            Array of ground truth labels.

        Returns
        -------
        accuracy : float
            Accuracy score for predicted class labels.
        """

        pred = self.predict(x)
        accuracy = sum(pred == y) / len(y)
        return accuracy

    def get_params(self, **kwargs) -> dict:
        """Get parameters for this estimator

        Notes
        -----
        scikit-learn API compatibility.
        """

        spec = inspect.getfullargspec(self.__init__)
        arguments = filter(lambda arg: arg != 'self', spec.args)
        params = {arg: getattr(self, arg) for arg in arguments}

        return params

    def set_params(self, **params) -> 'MSIRandomForestClassifier':
        """Set the parameters of this estimator

        Notes
        -----
        scikit-learn API compatibility.
        """

        allowed_params = inspect.getfullargspec(self.__init__)
        filtered_params = params.keys() & allowed_params.args

        for param in filtered_params:
            setattr(self, param, params[param])

        return self
