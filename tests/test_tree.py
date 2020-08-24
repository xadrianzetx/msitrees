import unittest
import numpy as np
from msitrees._node import MSINode
from msitrees.tree import MSIDecisionTreeClassifier


class TestMSIDecisionTreeClassifier(unittest.TestCase):

    def test_cost_binary(self):
        """Test cost calculation on xor problem"""
        # Define mock decision tree with one decision
        # node missing, giving out 0.75 acc
        left_branch = MSINode(
            feature=1,
            split=1.0,
            left=MSINode(y=0),
            right=MSINode(y=1)
        )
        right_branch = MSINode(y=1)
        mocktree = MSIDecisionTreeClassifier()
        mocktree._root.feature = 0
        mocktree._root.split = 1.0
        mocktree._root.left = left_branch
        mocktree._root.right = right_branch

        x = np.array(
            [[1., 0.],
             [1., 1.],
             [0., 1.],
             [0., 0.]]
        )
        y = np.array([1, 0, 1, 0])
        mocktree._shape = x.shape
        cost = mocktree._calculate_cost(x, y)
        self.assertAlmostEqual(cost, 0.2618, places=4)

    def test_cost_multiclass(self):
        """Test cost calculation on one-hot reverse problem"""
        # Define mock decision tree with one decision
        # node missing, giving out 0.6 acc
        mocktree = MSIDecisionTreeClassifier()
        mocktree._root.feature = 0
        mocktree._root.split = 1.0
        mocktree._root.left = MSINode(y=2)
        mocktree._root.right = MSINode(y=1)

        x = np.array(
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]
        )
        y = np.array([1, 2, 3])
        mocktree._shape = x.shape
        cost = mocktree._calculate_cost(x, y)
        self.assertAlmostEqual(cost, 0.5000, places=4)

    def test_input_x_empty(self):
        x = np.array([])
        y = np.array([1, 0])
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_y_empty(self):
        x = np.array([1., 0.])
        y = np.array([])
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_x_type_not_supported(self):
        x = [1, 2]
        y = np.array([1, 2])
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(TypeError):
            tree.fit(x, y)

    def test_input_y_type_not_supported(self):
        x = np.array([1, 2])
        y = [1, 2]
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(TypeError):
            tree.fit(x, y)

    def test_input_x_wrong_dim(self):
        x = np.zeros(shape=(1, 10, 10))
        y = np.zeros(shape=(10, ))
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_y_onehot(self):
        x = np.zeros(shape=(10, 10))
        y = np.zeros(shape=(10, 10))
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_x_not_numeric(self):
        # one of x features is not numeric
        pass

    def test_y_not_numeric(self):
        pass

    def test_x_has_nan(self):
        # data has nans
        pass

    def test_y_has_nan(self):
        pass

    def test_input_y_starts_1(self):
        # classes are label encoded but go
        # from 1 to N instead from 0
        pass

    def test_input_y_classes_not_in_series(self):
        # classes go from 0 to N but with gaps inbetween
        pass

    def test_fit_xor(self):
        # test if this can fit something at all
        pass

    def test_fit_onedim(self):
        # test fit to one dimensional x
        pass

    def test_fit_bc(self):
        # test fit on binary
        pass

    def test_fit_iris(self):
        # test fit on multiclass problem
        pass

    def test_fit_bc_pandas(self):
        pass

    def test_fit_iris_pandas(self):
        pass

    # TODO test predict methods


if __name__ == "__main__":
    unittest.main()
