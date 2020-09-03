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

import uuid
import json
import numpy as np
from typing import Optional


class MSINode:
    """A fundamental building block of MSI tree.

    Parameters
    ----------
    left : MSINode, default=None
        Left branch of the node. Can be another
        decision node or leaf node. If either
        left or right branch is None, then node
        is treated as leaf node.

    right : MSINode, default=None
        Right branch of the node. Can be another
        decision node or leaf node. If either
        left or right branch is None, then node
        is treated as leaf node.

    indices : list, default=None
        List of indices indicating subset
        of data used to perform split whithin
        a node.

    feature : int, default=None
        Index of a feature on which optimal
        split was performed.

    split : float, default=None
        Feature value on which optimal
        split was performed.

    proba : float, default=None
        Probability of predicted class label.
        Calculated as ratio of majority class
        label to all labels present in a node.

    y : int, default=None
        Predicted class label. Calculated as
        majority label within a node.

    Attributes
    ----------
    id : str
        Unique node id.

    Notes
    -----
    For internal use only.
    """

    def __init__(self, left: Optional['MSINode'] = None,
                 right: Optional['MSINode'] = None,
                 indices: Optional[list] = None,
                 feature: Optional[int] = None,
                 split: Optional[float] = None,
                 proba: Optional[np.ndarray] = None,
                 y: Optional[int] = None):
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
        """Recursively builds dict tree representation"""
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

    def reset(self) -> None:
        """Resets all node attributes to None, except id"""
        attrs = [k for k in self.__dict__.keys() if k != 'id']
        for attr in attrs:
            setattr(self, attr, None)

    def set_split_criteria(self, feature: int, split: float) -> None:
        """Sets feature index and split point.

        Parameters
        ----------
        feature : int
            Index of a feature on which optimal
            split was performed.

        split : float
            Feature value on which optimal
            split was performed
        """

        self.feature = feature
        self.split = split

    def count_tree_nodes(self, leaf_only: bool) -> int:
        """Counts number of leaf nodes or total number
        of child nodes for current node.

        Parameters
        ----------
        leaf_only : bool
            When true only leaf nodes are counted,
            otherwise both decision nodes and leaf
            nodes are.

        Returns
        -------
        total : int
            Number of nodes
        """

        if self.y is not None:
            return 1

        lcount = self.left.count_tree_nodes(leaf_only) if self.left else 0
        rcount = self.right.count_tree_nodes(leaf_only) if self.right else 0
        total = lcount + rcount

        return total if leaf_only else total + 1

    def count_nodes_to_bottom(self) -> int:
        """Return total depth of a tree including
        current node.

        Returns
        -------
        count : int
            Maximum depth of tree
        """

        if self.y is not None:
            return 1

        lcount = self.left.count_nodes_to_bottom() if self.left else 0
        rcount = self.right.count_nodes_to_bottom() if self.right else 0

        return max([lcount, rcount]) + 1

    def get_node_by_id(self, id: str) -> 'MSINode':
        """Retrieves node with specified id.

        Parameters
        ----------
        id : str
            Node id.

        Returns
        -------
        node : MSINode
            Node with specified id. If such node
            does not exist, then returns None
        """

        if self.id == id:
            return self

        ncl = self.left.get_node_by_id(id) if self.left else None
        ncr = self.right.get_node_by_id(id) if self.right else None

        return ncl or ncr

    def predict(self, x: np.ndarray) -> tuple:
        """Predicts class label and probability
        for input x.

        Returns
        -------
        pred : tuple
            Tuple with predicted values for input.
            Position 0 is class label, position 1
            is class probability.
        """

        if self.y is not None:
            return (self.y, self.proba)

        testpt = x if np.isscalar(x) else x[self.feature]

        if testpt < self.split:
            pred = self.left.predict(x)

        else:
            pred = self.right.predict(x)

        return pred
