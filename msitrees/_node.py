import uuid
import json
import numpy as np
from typing import Optional


class MSINode:

    def __init__(self, left: Optional['MSINode'] = None,
                 right: Optional['MSINode'] = None,
                 indices: Optional[list] = None,
                 feature: Optional[int] = None,
                 split: Optional[float] = None,
                 proba: Optional[np.ndarray] = None,
                 y: Optional[int] = None):
        """
        MSINode
        """
        self.id = uuid.uuid4().hex
        self.left = left
        self.right = right
        self.indices = indices
        self.feature = feature
        self.split = split
        self.proba = proba
        self.y = y

    def __repr__(self):
        num_nodes = self.count_tree_nodes(leaf_only=False)
        return 'Tree root/node with {} children'.format(num_nodes)

    def __str__(self):
        r = self._get_tree_structure()
        return json.dumps(r)

    def _get_tree_structure(self) -> dict:
        if self.y is not None:
            return {'leaf': self.y}

        else:
            node_left = self.left._get_tree_structure()
            node_right = self.right._get_tree_structure()
            return {
                'feature': self.feature,
                'split': self.split,
                'left': node_left,
                'right': node_right
            }

    def count_tree_nodes(self, leaf_only: bool) -> int:
        """
        Counts number of leaf nodes or total number
        of nodes for current node.
        """
        if self.y is not None:
            return 1

        lcount = self.left.count_tree_nodes(leaf_only) if self.left else 0
        rcount = self.right.count_tree_nodes(leaf_only) if self.right else 0
        total = lcount + rcount

        return total if leaf_only else total + 1

    def get_node_by_id(self, id: str) -> 'MSINode':
        if self.id == id:
            return self

        ncl = self.left.get_node_by_id(id) if self.left else None
        ncr = self.right.get_node_by_id(id) if self.right else None

        return ncl or ncr

    def predict(self, x: np.ndarray) -> tuple:
        if self.y is not None:
            return (self.y, self.proba)

        if x[self.feature] < self.split:
            pred = self.left.predict(x)

        else:
            pred = self.right.predict(x)

        return pred
