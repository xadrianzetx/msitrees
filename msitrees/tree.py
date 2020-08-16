import bz2
import numpy as np
import msitrees._core as core
from msitrees._node import MSINode


class MSIDecisionTreeClassifier:

    def __init__(self):
        self._root = MSINode()
        self._n_classes = None

    @property
    def feature_importances(self):
        pass

    def _calculate_cost(self):
        pass

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
