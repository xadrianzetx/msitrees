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
        pass

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

    def get_node_by_id(self, id: str) -> 'MSINode':
        if self.id == id:
            return self

        ncl = self.left.get_node_by_id(id) if self.left else None
        ncr = self.right.get_node_by_id(id) if self.right else None

        return ncl or ncr

    def predict(self, x: np.ndarray) -> tuple:
        if self.y:
            return (self.y, self.proba)

        if x[self.feature] < self.split:
            pred = self.left.predict(x)

        else:
            pred = self.right.predict(x)

        return pred
