# MIT License

# Copyright (c) 2020 xadrianzetx

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import joblib
import inspect
import numpy as np
import pandas as pd
from collections import Counter
from typing import Union, Optional

from msitrees._base import MSIBaseClassifier
from msitrees.tree import MSIDecisionTreeClassifier


class MSIRandomForestClassifier(MSIBaseClassifier):
    """MSI Random Forest Classifier

    A collection of MSI based decision tree classifiers
    fitted on bootstrapped sub-samples of dataset. Final
    class label is decided by majority voting between all
    estimators.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of tree estimators to fit

    bootstrap : bool, default=True
        When true, each estimator in will
        be fitted with bootstrap sub-sample of
        original dataset.

    feature_sampling : bool, default=True
        When true, number of features considered
        at each split will be equal to sqrt(n_features).
        This is equivalent of sklearn max_features param
        set to 'auto'.

    n_jobs : int, default=-1
        Number of parallel jobs to run. When set to -1
        all CPUs are used. 1 means no parallel processing.

    random_state : int, default=None
        Sets seed for class instance. Used to control
        bootstrap. Note that this seeds only generator
        that sets random state for each estimator, so
        trees generally should have their own unique seeds.
        When random_state is set to a number, those seeds will
        be reproduced at each run. Parameter set to None should
        result in different estimator seeding each time.

    Attributes
    ----------
    estimators : list
        list with all fited MSIDecisionTreeClassifier instances.

    fitted : bool
        Boolean variable indicating if tree was previously
        fitted.

    shape : tuple
        Shape of dataset X with (n_samples, n_features)
        or None if tree was not yet fitted.

    ncls : int
        Number of classification categories or None
        if tree was not yet fitted.

    ndim : int
        Number of dataset X dimensions. 1 if n_features eq 1,
        2 if n_features > 1 or None if tree was not yet fitted.

    importances : np.ndarray
        Array with feature importances or None if tree was not fitted.

    References
    ----------
    - [1] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8767915
    - [2] https://www.cs.bu.edu/teaching/c/tree/breadth-first/
    - [3] https://en.wikipedia.org/wiki/Kolmogorov_complexity
    - [4] https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from msitrees.ensemble import MSIRandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> data = load_iris()
    >>> clf = MSIRandomForestClassifier()
    >>> cross_val_score(clf, data['data'], data['target'], cv=10)
    ...
    array([1.        , 1.        , 1.        , 1.        , 0.93333333,
       0.86666667, 0.93333333, 0.86666667, 0.8       , 1.        ])
    """

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
