import unittest
import numpy as np
from msitrees._core import (
    gini_impurity,
    gini_information_gain,
    entropy,
    get_class_and_proba
)


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


class TestGiniInformationGain(unittest.TestCase):

    def test_input_type_list(self):
        yl = [0, 0, 0]
        yr = [1, 1, 1]
        yall = [0, 0, 0, 1, 1, 1]

        try:
            gini_information_gain(yl, yr, yall)

        except TypeError:
            self.fail('Exception on allowed input type - list')

    def test_input_type_tuple(self):
        yl = (0, 0, 0)
        yr = (1, 1, 1)
        yall = (0, 0, 0, 1, 1, 1)

        try:
            gini_information_gain(yl, yr, yall)

        except TypeError:
            self.fail('Exception on allowed input type - tuple')

    def test_input_type_numpy(self):
        yl = np.array([0, 0, 0])
        yr = np.array([1, 1, 1])
        yall = np.array([0, 0, 0, 1, 1, 1])

        try:
            gini_information_gain(yl, yr, yall)

        except TypeError:
            self.fail('Exception on allowed input type - np.ndarray')

    def test_input_int(self):
        yl = np.array([0, 0, 0])
        yr = np.array([1, 1, 1])
        yall = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaises(ValueError):
            gini_information_gain(0, yr, yall)

        with self.assertRaises(ValueError):
            gini_information_gain(yl, 0, yall)

        with self.assertRaises(ValueError):
            gini_information_gain(yl, yr, 0)

    def test_input_other(self):
        yl = np.array([0, 0, 0])
        yr = np.array([1, 1, 1])
        yall = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaises(TypeError):
            gini_information_gain('foo', yr, yall)

        with self.assertRaises(TypeError):
            gini_information_gain(yl, 'foo', yr)

        with self.assertRaises(TypeError):
            gini_information_gain(yl, yr, 'foo11')

    def test_input_wrong_shape(self):
        badshape = np.array([[1], [1], [1]])
        yl = np.array([0, 0, 0])
        yr = np.array([1, 1, 1])
        yall = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaises(ValueError):
            gini_information_gain(badshape, yr, yall)

        with self.assertRaises(ValueError):
            gini_information_gain(yl, badshape, yall)

        with self.assertRaises(ValueError):
            gini_information_gain(yl, yr, badshape)

    def test_input_empty_array(self):
        yl = np.array([0, 0, 0])
        yr = np.array([1, 1, 1])
        yall = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaises(ValueError):
            gini_information_gain([], yr, yall)

        with self.assertRaises(ValueError):
            gini_information_gain(yl, [], yall)

        with self.assertRaises(ValueError):
            gini_information_gain(yl, yr, [])

    def test_binary_perfect_split(self):
        yl = np.array([0, 0])
        yr = np.array([1, 1])
        yall = np.array([0, 0, 1, 1])

        gain = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.5)

    def test_binary_noisy_split(self):
        yl = np.array([0, 1])
        yr = np.array([1, 0])
        yall = np.array([0, 0, 1, 1])

        gain = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.0)

    def test_binary_uneven_split(self):
        yl = np.array([0, 0])
        yr = np.array([1, 1, 1])
        yall = np.array([0, 0, 1, 1, 1])

        gain = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.48)

    def test_multiclass_perfect_split(self):
        yl = np.array([1, 1])
        yr = np.array([2, 2])
        yall = np.array([2, 2, 1, 1])

        gain = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.5)

    def test_multiclass_noisy_split(self):
        yl = np.array([2, 1])
        yr = np.array([1, 2])
        yall = np.array([2, 2, 1, 1])

        gain = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.0)

    def test_multiclass_uneven_split(self):
        yl = np.array([1, 1])
        yr = np.array([2, 2, 3])
        yall = np.array([2, 2, 1, 1, 3])

        gain = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.3733, places=4)


class TestGetClassProba(unittest.TestCase):

    def test_input_type_list(self):
        y = [1, 1, 0, 0]

        try:
            get_class_and_proba(y)

        except TypeError:
            self.fail('Exception on allowed input type - list')

    def test_input_type_tuple(self):
        y = (1, 1, 0, 0)

        try:
            get_class_and_proba(y)

        except TypeError:
            self.fail('Exception on allowed input type - tuple')

    def test_input_type_numpy(self):
        y = np.array([1, 1, 0, 0])

        try:
            get_class_and_proba(y)

        except TypeError:
            self.fail('Exception on allowed input type - np.ndarray')

    def test_input_int(self):
        with self.assertRaises(ValueError):
            get_class_and_proba(0)

    def test_input_other(self):
        with self.assertRaises(TypeError):
            get_class_and_proba('foo')

    def test_input_wrong_shape(self):
        badshape = np.array([[1], [1], [1]])

        with self.assertRaises(ValueError):
            get_class_and_proba(badshape)

    def test_input_empty_array(self):
        with self.assertRaises(ValueError):
            get_class_and_proba([])

    def test_binary_class_major(self):
        y = np.array([0, 0, 1, 1, 1])
        label, _ = get_class_and_proba(y)
        self.assertEqual(int(label), 1)

    def test_binary_class_draw(self):
        y = np.array([0, 0, 1, 1])
        label, _ = get_class_and_proba(y)
        self.assertEqual(int(label), 0)

    def test_multiclass_class_major(self):
        y = np.array([1, 2, 3, 3])
        label, _ = get_class_and_proba(y)
        self.assertEqual(int(label), 3)

    def test_multiclass_class_draw(self):
        y = np.array([1, 1, 2, 2, 3, 3])
        label, _ = get_class_and_proba(y)
        self.assertEqual(int(label), 1)

    def test_binary_proba_major(self):
        y = np.array([0, 0, 1, 1, 1])
        _, proba = get_class_and_proba(y)
        self.assertAlmostEqual(proba, 0.6)

    def test_binary_proba_draw(self):
        y = np.array([0, 0, 1, 1])
        _, proba = get_class_and_proba(y)
        self.assertAlmostEqual(proba, 0.5)

    def test_multiclass_proba_major(self):
        y = np.array([1, 2, 3, 3])
        _, proba = get_class_and_proba(y)
        self.assertAlmostEqual(proba, 0.5)

    def test_multiclass_proba_draw(self):
        y = np.array([1, 1, 2, 2, 3, 3])
        _, proba = get_class_and_proba(y)
        self.assertAlmostEqual(proba, 0.33333, places=5)


if __name__ == "__main__":
    unittest.main()
