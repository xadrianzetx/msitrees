import bz2
import numpy as np
import msitrees._core as core
from msitrees._node import MSINode


class MSIDecisionTreeClassifier:

    def __init__(self):
        self._root = MSINode()
        self._n_classes = None
        self._x_shape = None

    @property
    def feature_importances(self):
        pass

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
        y_pred = self.predict(x)
        hits = sum(y == y_pred)
        iacc = 1 - (hits / self._x_shape[0])

        if redu < iacc:
            # discard initial underfitted
            # tree candidates
            redu = 1

        cost = 2 / ((1 / redu) + (1 / iacc))

        return cost

    def _build_tree(self):
        pass

    def get_depth(self):
        pass

    def get_n_leaves(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass
