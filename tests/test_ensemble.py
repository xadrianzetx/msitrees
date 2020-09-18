import unittest
import numpy as np
import pandas as pd
from msitrees.ensemble import MSIRandomForestClassifier


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


if __name__ == "__main__":
    unittest.main()
