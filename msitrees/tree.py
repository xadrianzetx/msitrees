import bz2
import numpy as np
import msitrees._core as core
from msitrees._node import MSINode


class MSIDecisionTreeClassifier:

    def __init__(self):
        self._root = MSINode()
        self._fitted = False
        self._shape = None
        self._ncls = None
        self._ndim = None

    @property
    def feature_importances(self):
        pass

    def _get_class_and_proba(self, y: np.ndarray) -> dict:
        """Wraps get_class_and_proba call"""
        label, proba = core.get_class_and_proba(y, self._ncls)
        return {'y': label, 'proba': proba}

    def _get_best_split(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Wraps classif_best_split call"""
        nfeats = self._shape[1] if self._ndim == 2 else 1
        *criteria, valid = core.classif_best_split(x, y, nfeats)
        named_criteria = {'feature': criteria[0], 'split': criteria[1]}
        return named_criteria, valid

    def _calculate_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates cost of growing new branch in a decision tree

        Follows paper implementation based on harmonic mean of
        model inaccuracy and surfeit, but with modifications to
        approximation of I(X, M) for performance and reusability
        reasons.

        References
        ----------
        [1] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8767915
        [2] https://en.wikipedia.org/wiki/Kolmogorov_complexity
        """
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

        cost = 2 / ((1 / redu) + (1 / iacc))

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
        """
        tree builder - initial
        """
        min_cost = np.inf
        self._root.indices = np.arange(x.shape[0])
        candidates = [self._root.id]

        while True:
            best_cand = None

            for cand in candidates:
                node = self._root.get_node_by_id(cand)
                sub_x, sub_y = x[node.indices], y[node.indices]
                criteria, valid = self._get_best_split(sub_x, sub_y)

                if not valid:
                    continue

                idx_left, idx_right = self._get_indices(sub_x, **criteria)
                cp_left = self._get_class_and_proba(sub_y[idx_left])
                cp_right = self._get_class_and_proba(sub_y[idx_right])
                bkp = MSINode(y=node.y, proba=node.proba, indices=node.indices)

                node.reset()
                node.left = MSINode(**cp_left)
                node.right = MSINode(**cp_right)
                node.set_split_criteria(**criteria)
                cost = self._calculate_cost(x, y)

                if cost < min_cost:
                    min_cost = cost
                    best_cand = node.id
                    data_left = {'indices': bkp.indices[idx_left], **cp_left}
                    data_right = {'indices': bkp.indices[idx_right], **cp_right}

                node.reset()
                node.indices = bkp.indices
                node.y = bkp.y
                node.proba = bkp.proba

            if best_cand:
                node = self._root.get_node_by_id(best_cand)
                node.reset()
                node.set_split_criteria(**criteria)
                node.left = MSINode(**data_left)
                node.right = MSINode(**data_right)
                candidates.remove(best_cand)
                candidates.extend([node.left.id, node.right.id])

            else:
                # no more candidates to check
                break

    def _validate_before_predict(self, x):
        # TODO check fitted flags and data shapes
        pass

    def get_depth(self):
        pass

    def get_n_leaves(self) -> int:
        '''Returns number of tree leaves'''
        if self._fitted:
            return self._root.count_tree_nodes(leaf_only=True)
        return 0

    def fit(self):
        pass

    def _predict_in_training(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data X

        Overrides input validation and should be used
        only inside cost function.
        """
        pred = [self._root.predict(obs)[0] for obs in x]
        return np.array(pred)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data X

        Params
        ----------
            x: np.array
            Array of samples with shape (n_samples, n_features).
            Class label is predicted for each sample.

        Returns
        ----------
            np.array
            Array with shape (n_samples, )
            Class label prediction for each sample.
        """
        self._validate_before_predict(x)
        pred = self._predict_in_training(x)
        return pred

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probability for input data X.
        Probability is defined as fraction of class
        label in a leaf.

        Params
        ----------
            x: np.array
            Array of samples with shape (n_samples, n_features).
            Class probabilities are predicted for each sample.

        Returns
        ----------
            np.array
            Array with shape (n_samples, n_targets)
            Array of probabilities. Each index corresponds to
            class label and holds predicted porbability of this class.
        """
        self._validate_before_predict(x)
        pred = [self._root.predict(obs)[1] for obs in x]
        return np.array(pred)

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class log probability for input data X.
        Probability is defined as fraction of class
        label in a leaf.

        Params
        ----------
            x: np.array
            Array of samples with shape (n_samples, n_features).
            Class log probabilities are predicted for each sample.

        Returns
        ----------
            np.array
            Array with shape (n_samples, n_targets)
            Array of log probabilities. Each index corresponds to
            class label and holds predicted log porbability of this class.
        """
        probas = self.predict_proba(x)
        logprob = [np.log(p) for p in probas]
        return np.array(logprob)
