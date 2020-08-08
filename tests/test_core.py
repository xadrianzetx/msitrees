import unittest
import numpy as np
from msitrees._core import gini_impurity, entropy


class TestGiniImpurity(unittest.TestCase):

    def test_input_type_list(self):
        try:
            gini_impurity([0, 0])

        except TypeError:
            self.fail('Exception on allowed input type - list')

    def test_input_type_tuple(self):
        try:
            gini_impurity((0, 0))

        except TypeError:
            self.fail('Exception on allowed input type - tuple')

    def test_input_type_numpy(self):
        try:
            gini_impurity(np.array([0, 0]))

        except TypeError:
            self.fail('Exception on allowed input type - np.ndarray')

    def test_input_int(self):
        with self.assertRaises(ValueError):
            gini_impurity(0)

    def test_input_other(self):
        with self.assertRaises(TypeError):
            gini_impurity('foo')

        with self.assertRaises(TypeError):
            gini_impurity({'foo': 1})

    def test_input_wrong_shape(self):
        with self.assertRaises(ValueError):
            gini_impurity(np.array([[1, 0], [1, 0]]))

    def test_input_empty_list(self):
        with self.assertRaises(ValueError):
            gini_impurity([])

    def test_input_empty_array(self):
        with self.assertRaises(ValueError):
            gini_impurity(np.array([]))

    def test_binary_max_impurity(self):
        arr = np.array([1, 0, 1, 0])
        gini = gini_impurity(arr)
        self.assertAlmostEqual(gini, 0.5)

    def test_binary_min_impurity(self):
        arr = np.array([0, 0, 0, 0])
        gini = gini_impurity(arr)
        self.assertAlmostEqual(gini, 0.0)

    def test_multiclass_max_impurity(self):
        arr = np.array(list(range(5)))
        max_impurity = 1 - (1 / arr.shape[0])
        gini = gini_impurity(arr)
        self.assertAlmostEqual(gini, max_impurity)


class TestEntropy(unittest.TestCase):

    def test_input_type_list(self):
        try:
            entropy([0, 0])

        except TypeError:
            self.fail('Exception on allowed input type - list')

    def test_input_type_tuple(self):
        try:
            entropy((0, 0))

        except TypeError:
            self.fail('Exception on allowed input type - tuple')

    def test_input_type_numpy(self):
        try:
            entropy(np.array([0, 0]))

        except TypeError:
            self.fail('Exception on allowed input type - np.ndarray')

    def test_input_int(self):
        with self.assertRaises(ValueError):
            entropy(0)

    def test_input_other(self):
        with self.assertRaises(TypeError):
            entropy('foo')

        with self.assertRaises(TypeError):
            entropy({'foo': 1})

    def test_input_wrong_shape(self):
        with self.assertRaises(ValueError):
            entropy(np.array([[1, 0], [1, 0]]))

    def test_input_empty_list(self):
        with self.assertRaises(ValueError):
            entropy([])

    def test_input_empty_array(self):
        with self.assertRaises(ValueError):
            entropy(np.array([]))

    def test_binary_max_impurity(self):
        arr = np.array([1, 0, 1, 0])
        hs = entropy(arr)
        self.assertAlmostEqual(hs, 1.)

    def test_binary_min_impurity(self):
        arr = np.array([0, 0, 0, 0])
        hs = entropy(arr)
        self.assertAlmostEqual(hs, 0.)

    def test_multiclass_max_impurity(self):
        arr = np.array([1, 2, 3, 4])
        hs = entropy(arr)
        self.assertAlmostEqual(hs, 2.)


if __name__ == "__main__":
    unittest.main()
