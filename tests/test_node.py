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
        factory.ids.remove(root.id)
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
        depth = 4
        factory = MockupTreeFactory(depth)
        root = factory.build()
        # cut left branch of the tree
        root.left = None
        expected = 2 ** (depth + 1) / 2
        count = root.count_tree_nodes(leaf_only=False)
        self.assertEqual(count, expected)

    def test_node_count(self):
        depth = 4
        factory = MockupTreeFactory(depth)
        root = factory.build()
        expected = 2 ** (depth + 1) - 1
        count = root.count_tree_nodes(leaf_only=False)
        self.assertEqual(count, expected)

    def test_tree_representation(self):
        expected = {
            'feature': None,
            'split': None,
            'left': {'leaf': 0},
            'right': {'leaf': 0}
        }
        factory = MockupTreeFactory(1)
        root = factory.build()
        r = root._get_tree_structure()
        self.assertDictEqual(r, expected)

    def test_tree_traversal(self):
        x = np.array(
            [[0., 0., 0.],
            [0., 1., 0.]]
        )
        y = np.array([0, 1])
        factory = MockupTreeFactory(1)
        root = factory.build()

        # set up fake split points
        root.split = 1.
        root.feature = 1
        root.left.y = 0
        root.right.y = 1

        self.assertEqual(root.predict(x[0])[0], y[0])
        self.assertEqual(root.predict(x[1])[0], y[1])

    def test_node_reset_id_preserved(self):
        node = MSINode()
        id = node.id
        node.reset()
        self.assertEqual(node.id, id)

    def test_node_reset(self):
        node = MSINode(y=1, feature=1, proba=0.5)
        node.reset()
        self.assertIsNone(node.y)
        self.assertIsNone(node.feature)
        self.assertIsNone(node.proba)


if __name__ == "__main__":
    unittest.main()
