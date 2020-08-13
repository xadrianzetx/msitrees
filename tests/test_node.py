import unittest
import numpy as np
from msitrees._node import MSINode


class MockupTreeFactory:

    def __init__(self, depth, prune=False):
        self.ids = []
        self.depth = depth
        self.prune = prune

    def _add_node(self, d):
        if d == self.depth:
            node = MSINode(y=0)
            self.ids.append(node.id)
            return node

        prune = [False, False]
        depth = d + 1
        lchild, rchild = None, None

        if self.prune:
            prune = np.random.binomial(1, 0.7, 2)
            prune = prune > 0

        if not prune[0]:
            lchild = self._add_node(depth)

        if not prune[1]:
            rchild = self._add_node(depth)

        node = MSINode(left=lchild, right=rchild)
        self.ids.append(node.id)

        return node

    def build(self):
        depth = 1
        left = self._add_node(depth)
        right = self._add_node(depth)
        root = MSINode(left=left, right=right)
        self.ids.append(root.id)

        return root
