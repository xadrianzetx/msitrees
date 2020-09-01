import bz2
import numpy as np
import pandas as pd
from typing import Union

import msitrees._core as core
from msitrees._node import MSINode


class MSIDecisionTreeClassifier:
    """MSI Decision Tree Classifier

    Based on breadth-first tree traversal, this no-hyperparameter
    tree building algorithm tries to create new decision nodes by
    performing temporary split for each candidate node
    (at any point of time all current leaves are considered candidate)
    one by one, and keeping one which decreases overall cost
    function the most. New branches created with this operation are
    added to candidate pool. Best split points are estimated with
    gini based information gain. Training ends when any new split would
    only add needless complexity to the tree.

    Cost function follows paper implementation [1] based on harmonic
    mean of model inaccuracy and surfeit, but with modifications to
    approximation of I(X, M) for performance and reusability reasons.

    Attributes
    ----------
    root : MSINode
        Root of a decision tree. All decision and leaf
        nodes are children of this node.

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
        Array with feature importances or None if tree was not
        yet.

    References
    ----------
    - [1] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8767915
    - [2] https://www.cs.bu.edu/teaching/c/tree/breadth-first/
    - [3] https://en.wikipedia.org/wiki/Kolmogorov_complexity

    Examples
    --------
    >>> from msitrees.tree import MSIDecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> data = load_iris()
    >>> clf = MSIDecisionTreeClassifier()
    >>> cross_val_score(clf, data['data'], data['target'], cv=10)
    ...
    array([1.        , 1.        , 1.        , 0.93333333, 0.93333333,
        0.8       , 0.93333333, 0.86666667, 0.8       , 1.        ])
    """

    def __init__(self):
        self._root = MSINode()
        self._fitted = False
        self._shape = None
        self._ncls = None
        self._ndim = None
        self._importances = None

    @property
    def feature_importances_(self):
        """Returns feature importances

        Feature importance at each node is specified
        as weighted gini based information gain. Feature
        importance for each feature is normalized sum of
        importances at nodes where split was made
        on particular feature.

        Returns
        -------
        importances : np.ndarray
            Normalized array of feature importances.
        """

        if self._fitted:
            return self._importances / sum(self._importances)
        return np.array([])

    def _get_class_and_proba(self, y: np.ndarray) -> dict:
        """Wraps get_class_and_proba call"""
        label, proba = core.get_class_and_proba(y, self._ncls)
        return {'y': label, 'proba': proba}

    def _get_best_split(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Wraps classif_best_split call"""
        nfeats = self._shape[1] if self._ndim == 2 else 1
        *criteria, importance, valid = core.classif_best_split(
            x, y, nfeats, self._shape[0])
        named_criteria = {'feature': criteria[0], 'split': criteria[1]}

        return named_criteria, valid, importance

    def _calculate_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculates cost of growing new branch in a decision tree"""
        # approximate surfeit 1 - K(X)/M of a model
        # by  calculating 1 - Comp(M)/M where M
        # is a dict representation of decision tree
        stroot = str(self._root).encode()
        byteroot = bz2.compress(stroot, compresslevel=9)
        redu = 1 - len(byteroot) / len(stroot)

        # approximation of Kolmogorov complexity
        # I(X, M) = Comp(E)/Comp(X)  from paper
        # was simplified to 1 - model accuracy
        # It still has the same unit as surfeit, but it's
        # faster to calculate and should work with eg. MAPE
        # for regression tasks (or just about anything that maps
        # model error to [0, 1])
        y_pred = self._predict_in_training(x)
        hits = sum(y == y_pred)
        iacc = 1 - (hits / self._shape[0])

        if redu < iacc:
            # discard initial underfitted
            # tree candidates
            redu = 1

        cost = 2 / ((1 / redu) + (1 / (iacc + 1e-100)))

        return cost

    def _get_indices(self, x: np.ndarray, feature: int, split: float) -> tuple:
        """Returns new dataset indices wrt. best split"""
        if self._ndim == 2:
            idx_left = np.where(x[:, feature] < split)[0]
            idx_right = np.where(x[:, feature] >= split)[0]

        else:
            idx_left = np.where(x < split)[0]
            idx_right = np.where(x >= split)[0]

        return idx_left, idx_right

    def _build_tree(self, x: np.ndarray, y: np.ndarray):
        """Builds MSI classification tree"""
        min_cost = np.inf
        self._root.indices = np.arange(x.shape[0])
        candidates = [self._root.id]

        while True:
            # tree builder concludes at the point
            # where creating new node from current
            # candidates does not bring decrease in
            # cost function
            best_cand = None

            for cand in candidates:
                node = self._root.get_node_by_id(cand)
                sub_x, sub_y = x[node.indices], y[node.indices]
                criteria, valid, importance = self._get_best_split(sub_x, sub_y)

                if not valid:
                    # best split would only
                    # populate one branch, so its
                    # pointless
                    continue

                idx_left, idx_right = self._get_indices(sub_x, **criteria)
                cp_left = self._get_class_and_proba(sub_y[idx_left])
                cp_right = self._get_class_and_proba(sub_y[idx_right])
                bkp = MSINode(y=node.y, proba=node.proba, indices=node.indices)

                # grow candidate branch
                # and calculate its cost
                node.reset()
                node.left = MSINode(**cp_left)
                node.right = MSINode(**cp_right)
                node.set_split_criteria(**criteria)
                cost = self._calculate_cost(x, y)

                if cost < min_cost:
                    min_cost = cost
                    best_cand = node.id
                    best_criteria = criteria
                    best_importance = importance
                    data_left = {'indices': bkp.indices[idx_left], **cp_left}
                    data_right = {'indices': bkp.indices[idx_right], **cp_right}

                # revert node to original state
                node.reset()
                node.indices = bkp.indices
                node.y = bkp.y
                node.proba = bkp.proba

            if best_cand:
                # grow permanent branches on best split criteria
                node = self._root.get_node_by_id(best_cand)
                node.reset()
                node.set_split_criteria(**best_criteria)
                node.left = MSINode(**data_left)
                node.right = MSINode(**data_right)
                self._importances[best_criteria['feature']] += best_importance
                candidates.remove(best_cand)
                candidates.extend([node.left.id, node.right.id])

            else:
                # no more candidates to check
                break

    def _validate_input(self, data: Union[np.ndarray, pd.DataFrame, pd.Series],
                        expected_dim: int, inference: bool = False) -> np.ndarray:
        """Honeypot for incorrect input spec"""
        allowed_types = (
            np.ndarray,
            pd.core.frame.DataFrame,
            pd.core.frame.Series
        )

        if type(data) not in allowed_types:
            raise TypeError('Supported input types: np.ndarray, '
                            'pd.core.frame.DataFrame, pd.core.frame.Series got'
                            ' {}'.format(type(data)))

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values

        if data.size == 0:
            raise ValueError('Empty array passed to fit() or predict()')

        if data.ndim > expected_dim:
            raise ValueError('Data with incorrect number of dimensions '
                             'passed to fit() or predict(). Max dim is '
                             '{}, got {}'.format(expected_dim, data.ndim))

        if not np.issubdtype(data.dtype, np.number):
            raise ValueError('Non numeric value found in data')

        if not np.isfinite(data).all():
            raise ValueError('Data contains nan or inf')

        if inference:
            # additional checks on prediction time
            if not self._fitted:
                raise ValueError('Fit the model first.')

            if self._ndim == 2 and data.shape[-1] != self._shape[-1]:
                raise ValueError('Number of features does not match'
                                 ' data model was trained on. Expected'
                                 ' {}, got {}'
                                 .format(self._shape[-1], data.shape[-1]))

        return data

    def get_depth(self):
        """Returns decision tree depth

        Returns
        -------
        depth : int
            Maximum depth of fitted decision tree.
        """

        if self._fitted:
            return self._root.count_nodes_to_bottom() - 1
        return 0

    def get_n_leaves(self) -> int:
        '''Returns number of tree leaves

        Returns
        -------
        num_leaves : int
            Number of leaf nodes in fitted tree.
        '''

        if self._fitted:
            return self._root.count_tree_nodes(leaf_only=True)
        return 0

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series],
            y: Union[np.ndarray, pd.Series]) -> 'MSIDecisionTreeClassifier':
        """Fits decision tree from training dataset.

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
        self : MSIDecisionTreeClassifier
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
        self._ncls = len(classes)

        # check if classes go in sequence by one
        if self._ncls != max(classes) + 1:
            raise ValueError('Y is mislabeled')

        x = x.astype(np.float)
        y = y.astype(np.int)
        self._shape = x.shape
        self._ndim = x.ndim

        if x.ndim == 2:
            self._importances = np.zeros(shape=(x.shape[1], ))

        else:
            self._importances = np.zeros(shape=(1, ))

        self._build_tree(x, y)
        self._fitted = True

        return self

    def _predict_in_training(self, x: np.ndarray) -> np.ndarray:
        """Predicts class labels for input data X

        Notes
        -----
        Overrides input validation and should be used
        only inside cost function.
        """

        pred = [self._root.predict(obs)[0] for obs in x]
        return np.array(pred)

    def score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predicts from X and computes accuracy score wrt. y

        Parameters
        ----------
        x : np.ndarray
            Array of samples with shape (n_samples, n_features).
            Class label is predicted for each sample.

        y : np.ndarray
            Array of ground truth labels.

        Returns
        -------
        mean_acc : float
            Accuracy score for predicted class labels.
        """

        self._validate_input(x, expected_dim=2, inference=True)
        pred = self._predict_in_training(x)
        mean_acc = sum(pred == y) / len(y)

        return mean_acc

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts class labels for input data X

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

        self._validate_input(x, expected_dim=2, inference=True)
        pred = self._predict_in_training(x)
        return pred

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probability for input data X.

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

        self._validate_input(x, expected_dim=2, inference=True)
        pred = [self._root.predict(obs)[1] for obs in x]
        return np.array(pred)

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class log probability for input data X.

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

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn API compatibility"""
        return {}

    def set_params(self, **params: dict) -> 'MSIDecisionTreeClassifier':
        """scikit-learn API compatibility"""
        return self
