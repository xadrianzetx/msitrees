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


class TestMSINode(unittest.TestCase):

    def test_get_by_id(self):
        factory = MockupTreeFactory(3)
        root = factory.build()
        id = np.random.choice(factory.ids)
        node = root.get_node_by_id(id)
        self.assertEqual(node.id, id)

    def test_get_by_id_pruned(self):
        factory = MockupTreeFactory(3, prune=True)
        root = factory.build()
        id = np.random.choice(factory.ids)
        node = root.get_node_by_id(id)
        self.assertEqual(node.id, id)

    def test_node_value_assigned(self):
        factory = MockupTreeFactory(1)
        root = factory.build()
        id = np.random.choice(factory.ids)
        node = root.get_node_by_id(id)
        node.proba = 1.0

        # one of those should now
        # have value assigned
        lproba = root.left.proba
        rproba = root.right.proba
        proba = lproba or rproba
        self.assertIsNotNone(proba)

    def test_node_delete(self):
        # TODO
        pass

    def test_node_count(self):
        depth = 4
        factory = MockupTreeFactory(depth)
        root = factory.build()
        expected = 2 ** (depth + 1) - 1
        count = root._count_child_nodes()
        self.assertEqual(count, expected)

    def test_tree_representation(self):
        # TODO
        pass

    # TODO Test root.predict


if __name__ == "__main__":
    unittest.main()
