import unittest
import numpy as np
import pandas as pd
from msitrees.ensemble import MSIRandomForestClassifier

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


RANDOM_STATE = 42


class TestMSIRandomForestClassifier(unittest.TestCase):

    def test_input_x_empty(self):
        x = np.array([])
        y = np.array([1, 0])
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_y_empty(self):
        x = np.array([1., 0.])
        y = np.array([])
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_x_type_not_supported(self):
        x = [1, 2]
        y = np.array([1, 2])
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(TypeError):
            tree.fit(x, y)

    def test_input_y_type_not_supported(self):
        x = np.array([1, 2])
        y = [1, 2]
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(TypeError):
            tree.fit(x, y)

    def test_input_x_wrong_dim(self):
        x = np.zeros(shape=(1, 10, 10))
        y = np.zeros(shape=(10, ))
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_y_onehot(self):
        x = np.zeros(shape=(10, 10))
        y = np.zeros(shape=(10, 10))
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_x_not_numeric(self):
        x = np.array(['1', '2', '3'])
        y = np.zeros(10)
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_y_not_numeric(self):
        x = np.zeros((10, 10))
        y = np.array(['1', '2', '3'])
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_x_has_nan(self):
        x = np.zeros((10, 10))
        y = np.zeros(10)
        x[0, 0] = np.nan
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_y_has_nan(self):
        x = np.zeros((10, 10))
        y = np.zeros(10)
        y[0] = np.nan
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_x_has_infinite(self):
        x = np.zeros((10, 10))
        y = np.zeros(10)
        x[0, 0] = np.inf
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

        x[0, 0] = -np.inf

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_y_has_infinite(self):
        x = np.zeros((10, 10))
        y = np.zeros(10)
        y[0] = np.inf
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

        y[0] = -np.inf

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_y_starts_1(self):
        x = np.zeros((3, 3))
        y = np.array([1, 2, 3])
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_input_y_classes_not_in_series(self):
        x = np.zeros((3, 3))
        y = np.array([0, 1, 3])
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

        with self.assertRaises(ValueError):
            tree.fit(x, y)

    def test_data_equal_length(self):
        x = np.zeros((10, 10))
        y = np.zeros((9, ))
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)

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
        tree = MSIRandomForestClassifier(
            bootstrap=False,
            feature_sampling=False,
            random_state=RANDOM_STATE
        )
        tree.fit(x, y)
        pred = tree.predict(x)
        acc = sum(pred == y) / len(y)
        # importances = tree.feature_importances_
        self.assertEqual(acc, 1.0)
        # TODO test feature importances
        # self.assertEqual(sum(importances), 1.0)
        # np.testing.assert_allclose(importances, np.array([0., 1.]))

    def test_fit_onedim(self):
        data = load_iris()
        x_train, x_valid, y_train, y_valid = train_test_split(
            data['data'], data['target'], random_state=42)
        x_train = x_train[:, :2]
        x_valid = x_valid[:, :2]
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)
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
        tree = MSIRandomForestClassifier(random_state=RANDOM_STATE)
        tree.fit(x_train, y_train)
        pred = tree.predict(x_val)
        acc = accuracy_score(y_val, pred)
        # importances = tree.feature_importances_
        self.assertGreater(acc, 0.95)
        # TODO importances
        # self.assertAlmostEqual(sum(importances), 1.0)

    def test_fit_iris(self):
        """Test fit on multiclass dataset"""
        data = load_iris()
        x_train, x_val, y_train, y_val = train_test_split(
            data['data'], data['target'],
            random_state=42
        )
        tree = MSIRandomForestClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_val)
        acc = accuracy_score(y_val, pred)
        # importances = tree.feature_importances_
        self.assertGreater(acc, 0.95)
        # self.assertEqual(sum(importances), 1.0)


if __name__ == "__main__":
    unittest.main()
