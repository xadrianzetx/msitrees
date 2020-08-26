import unittest
import numpy as np
import pandas as pd
from msitrees._node import MSINode
from msitrees.tree import MSIDecisionTreeClassifier

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
        x = np.array(['1', '2', '3'])
        y = np.zeros(10)
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_y_not_numeric(self):
        x = np.zeros((10, 10))
        y = np.array(['1', '2', '3'])
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_x_has_nan(self):
        x = np.zeros((10, 10))
        y = np.zeros(10)
        x[0, 0] = np.nan
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_y_has_nan(self):
        x = np.zeros((10, 10))
        y = np.zeros(10)
        y[0] = np.nan
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_x_has_infinite(self):
        x = np.zeros((10, 10))
        y = np.zeros(10)
        x[0, 0] = np.inf
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

        x[0, 0] = -np.inf

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_y_has_infinite(self):
        x = np.zeros((10, 10))
        y = np.zeros(10)
        y[0] = np.inf
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

        y[0] = -np.inf

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_y_starts_1(self):
        x = np.zeros((3, 3))
        y = np.array([1, 2, 3])
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_y_classes_not_in_series(self):
        x = np.zeros((3, 3))
        y = np.array([0, 1, 3])
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_data_equal_length(self):
        x = np.zeros((10, 10))
        y = np.zeros((9, ))
        tree = MSIDecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_fit_xor(self):
        x = np.array(
            [[1, 0],
             [0, 1],
             [0, 0],
             [1, 1]]
        )
        y = np.array([1, 1, 0, 0])
        tree = MSIDecisionTreeClassifier()
        tree.fit(x, y)
        pred = tree.predict(x)
        acc = sum(pred == y) / len(y)
        nl = tree.get_n_leaves()
        self.assertEqual(acc, 1.0)
        self.assertEqual(nl, 4)

    def test_fit_onedim(self):
        data = load_iris()
        x_train, x_valid, y_train, y_valid = train_test_split(
            data['data'], data['target'], random_state=42)
        x_train = x_train[:, :2]
        x_valid = x_valid[:, :2]
        tree = MSIDecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_valid)
        acc = accuracy_score(y_valid, pred)
        self.assertGreater(acc, 0.5)

    def test_fit_bc(self):
        '''Test fit on binary dataset'''
        data = load_breast_cancer()
        x_train, x_val, y_train, y_val = train_test_split(
            data['data'], data['target'],
            random_state=42
        )
        tree = MSIDecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_val)
        acc = accuracy_score(y_val, pred)
        nl = tree.get_n_leaves()
        self.assertAlmostEqual(acc, 0.95104, places=4)
        self.assertEqual(nl, 12)

    def test_fit_iris(self):
        """Test fit on multiclass dataset"""
        data = load_iris()
        x_train, x_val, y_train, y_val = train_test_split(
            data['data'], data['target'],
            random_state=42
        )
        tree = MSIDecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_val)
        acc = accuracy_score(y_val, pred)
        nl = tree.get_n_leaves()
        self.assertAlmostEqual(acc, 0.97368, places=4)
        self.assertEqual(nl, 4)

    def test_fit_bc_pandas(self):
        data = load_breast_cancer()
        x_train, x_val, y_train, y_val = train_test_split(
            data['data'], data['target'],
            random_state=42
        )
        x_train = pd.DataFrame(x_train)
        y_train = pd.Series(y_train)
        tree = MSIDecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_val)
        acc = accuracy_score(y_val, pred)
        nl = tree.get_n_leaves()
        self.assertAlmostEqual(acc, 0.95104, places=4)
        self.assertEqual(nl, 12)

    def test_fit_iris_pandas(self):
        data = load_iris()
        x_train, x_val, y_train, y_val = train_test_split(
            data['data'], data['target'],
            random_state=42
        )
        x_train = pd.DataFrame(x_train)
        y_train = pd.Series(y_train)
        tree = MSIDecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_val)
        acc = accuracy_score(y_val, pred)
        nl = tree.get_n_leaves()
        self.assertAlmostEqual(acc, 0.97368, places=4)
        self.assertEqual(nl, 4)

    # TODO test predict methods


if __name__ == "__main__":
    unittest.main()
