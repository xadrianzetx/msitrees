import unittest
import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_iris
)
from msitrees._core import (
    gini_impurity,
    gini_information_gain,
    entropy,
    get_class_and_proba,
    classif_best_split
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

        gain, imp = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.5)
        self.assertAlmostEqual(imp, 0.5)

    def test_binary_noisy_split(self):
        yl = np.array([0, 1])
        yr = np.array([1, 0])
        yall = np.array([0, 0, 1, 1])

        gain, imp = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.0)
        self.assertAlmostEqual(imp, 0.5)

    def test_binary_uneven_split(self):
        yl = np.array([0, 0])
        yr = np.array([1, 1, 1])
        yall = np.array([0, 0, 1, 1, 1])

        gain, imp = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.48)
        self.assertAlmostEqual(imp, 0.48)

    def test_multiclass_perfect_split(self):
        yl = np.array([1, 1])
        yr = np.array([2, 2])
        yall = np.array([2, 2, 1, 1])

        gain, imp = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.5)
        self.assertAlmostEqual(imp, 0.5)

    def test_multiclass_noisy_split(self):
        yl = np.array([2, 1])
        yr = np.array([1, 2])
        yall = np.array([2, 2, 1, 1])

        gain, imp = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.0)
        self.assertAlmostEqual(imp, 0.5)

    def test_multiclass_uneven_split(self):
        yl = np.array([1, 1])
        yr = np.array([2, 2, 3])
        yall = np.array([2, 2, 1, 1, 3])

        gain, _ = gini_information_gain(yl, yr, yall)
        self.assertAlmostEqual(gain, 0.3733, places=4)


class TestGetClassProba(unittest.TestCase):

    def test_input_type_list(self):
        y = [1, 1, 0, 0]

        try:
            get_class_and_proba(y, 2)

        except TypeError:
            self.fail('Exception on allowed input type - list')

    def test_input_type_tuple(self):
        y = (1, 1, 0, 0)

        try:
            get_class_and_proba(y, 2)

        except TypeError:
            self.fail('Exception on allowed input type - tuple')

    def test_input_type_numpy(self):
        y = np.array([1, 1, 0, 0])

        try:
            get_class_and_proba(y, 2)

        except TypeError:
            self.fail('Exception on allowed input type - np.ndarray')

    def test_input_int(self):
        with self.assertRaises(ValueError):
            get_class_and_proba(0, 0)

    def test_input_other(self):
        with self.assertRaises(TypeError):
            get_class_and_proba('foo', 0)

    def test_input_wrong_shape(self):
        badshape = np.array([[1], [1], [1]])

        with self.assertRaises(ValueError):
            get_class_and_proba(badshape, 2)

    def test_input_empty_array(self):
        with self.assertRaises(ValueError):
            get_class_and_proba([], 0)

    def test_binary_class_major(self):
        y = np.array([0, 0, 1, 1, 1])
        label, _ = get_class_and_proba(y, 2)
        self.assertEqual(label, 1)

    def test_binary_class_draw(self):
        y = np.array([0, 0, 1, 1])
        label, _ = get_class_and_proba(y, 2)
        self.assertEqual(label, 0)

    def test_multiclass_class_major(self):
        y = np.array([0, 1, 2, 2])
        label, _ = get_class_and_proba(y, 3)
        self.assertEqual(label, 2)

    def test_multiclass_class_draw(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        label, _ = get_class_and_proba(y, 3)
        self.assertEqual(label, 0)

    def test_binary_proba_major(self):
        y = np.array([0, 0, 1, 1, 1])
        label, proba = get_class_and_proba(y, 2)
        self.assertAlmostEqual(proba[label], 0.6)
        self.assertAlmostEqual(proba[0], 1 - 0.6)

    def test_binary_proba_draw(self):
        y = np.array([0, 0, 1, 1])
        label, proba = get_class_and_proba(y, 2)
        self.assertAlmostEqual(proba[label], 0.5)
        self.assertAlmostEqual(proba[1], 1 - 0.5)

    def test_multiclass_proba_major(self):
        y = np.array([0, 1, 2, 2])
        label, proba = get_class_and_proba(y, 3)
        self.assertAlmostEqual(proba[label], 0.5)
        self.assertAlmostEqual(proba[0], 0.25)
        self.assertAlmostEqual(proba[1], 0.25)

    def test_multiclass_proba_draw(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        label, proba = get_class_and_proba(y, 3)
        self.assertAlmostEqual(proba[label], 0.33333, places=5)
        self.assertAlmostEqual(proba[1], 0.33333, places=5)
        self.assertAlmostEqual(proba[2], 0.33333, places=5)

    def test_padding_left(self):
        # binary classification, but
        # leaf only has class 1 - test
        # if 0 is represented with proba 0.
        y = np.array([1, 1])
        _, proba = get_class_and_proba(y, 2)
        self.assertEqual(len(proba), 2)
        self.assertEqual(proba[0], 0.0)
        self.assertEqual(proba[1], 1.0)

    def test_padding_inner(self):
        # multiclass classification
        # with classes 0, 1, 2 but leaf
        # only has class 1 and 2. check
        # if class 1 is represented with proba 0.
        y = np.array([0, 0, 2, 2])
        _, proba = get_class_and_proba(y, 3)
        self.assertEqual(len(proba), 3)
        self.assertEqual(proba[1], 0.0)
        self.assertAlmostEqual(proba[0], 0.5)
        self.assertAlmostEqual(proba[2], 0.5)

    def test_padding_right(self):
        # binary classification, but
        # leaf only has class 0 - test
        # if 1 is represented with proba 0.
        y = np.array([0, 0])
        _, proba = get_class_and_proba(y, 2)
        self.assertEqual(len(proba), 2)
        self.assertEqual(proba[1], 0.0)
        self.assertEqual(proba[0], 1.0)


class TestClassifBestSplit(unittest.TestCase):

    def test_input_x_list(self):
        x = [[1., 0.], [1., 1.]]
        y = np.array([1, 0])

        try:
            classif_best_split(x, y, 2, 2)

        except TypeError:
            self.fail('Exception on allowed data type')

    def test_input_x_numpy(self):
        x = np.array([[1., 0.], [1., 1.]])
        y = np.array([1, 0])

        try:
            classif_best_split(x, y, 2, 2)

        except TypeError:
            self.fail('Exception on allowed data type')

    def test_input_y_list(self):
        x = [[1., 0.], [1., 1.]]
        y = [1, 0]

        try:
            classif_best_split(x, y, 2, 2)

        except TypeError:
            self.fail('Exception on allowed data type')

    def test_input_y_numpy(self):
        x = np.array([[1., 0.], [1., 1.]])
        y = [1, 0]

        try:
            classif_best_split(x, y, 2, 2)

        except TypeError:
            self.fail('Exception on allowed data type')

    def test_x_one_dim_binary(self):
        x = np.array([1., 0.])
        y = np.array([1, 0])
        feature, value, _, valid = classif_best_split(x, y, 1, 2)
        self.assertEqual(int(feature), 0)
        self.assertEqual(value, 1.)
        self.assertTrue(valid)

    def test_leftmost_split_binary(self):
        """Best split is on feature 0"""
        x = np.array([[1., 0., 0.], [0., 0., 0.]])
        y = np.array([1, 0])
        feature, value, _, _ = classif_best_split(x, y, 3, 2)
        self.assertEqual(int(feature), 0)
        self.assertEqual(value, 1.)

    def test_rightmost_split_binary(self):
        """Best split is on feature N"""
        x = np.array([[0., 0., 1], [0., 0., 0]])
        y = np.array([1, 0])
        feature, value, _, _ = classif_best_split(x, y, 3, 2)
        self.assertEqual(int(feature), 2)
        self.assertEqual(value, 1.)

    def test_node_empty_binary(self):
        """
        Test if useless split is bypassed
        eg. split on min or max value
        """
        x = np.array([1., 1.])
        y = np.array([0, 0])
        feature, value, _, valid = classif_best_split(x, y, 1, 2)
        self.assertEqual(int(feature), 0)
        self.assertEqual(value, 0.)
        self.assertFalse(valid)

    def test_one_dim_multicls(self):
        x = np.array([1., 1., 0., 0.])
        y = np.array([1, 1, 2, 2])
        feature, value, _, valid = classif_best_split(x, y, 1, 4)
        self.assertEqual(int(feature), 0)
        self.assertEqual(value, 1.)
        self.assertTrue(valid)

    def test_leftmost_split_multicls(self):
        """Best split is on feature 0"""
        x = np.array([[1., 0., 0.], [0., 0., 0.]])
        y = np.array([1, 2])
        feature, value, _, _ = classif_best_split(x, y, 3, 2)
        self.assertEqual(int(feature), 0)
        self.assertEqual(value, 1.)

    def test_rightmost_split_multicls(self):
        """Best split is on feature N"""
        x = np.array([[0., 0., 1], [0., 0., 0]])
        y = np.array([1, 2])
        feature, value, _, _ = classif_best_split(x, y, 3, 2)
        self.assertEqual(int(feature), 2)
        self.assertEqual(value, 1.)

    def test_node_empty_multicls(self):
        x = np.array([1., 1.])
        y = np.array([2, 2])
        feature, value, _, valid = classif_best_split(x, y, 1, 2)
        self.assertEqual(int(feature), 0)
        self.assertEqual(value, 0.)
        self.assertFalse(valid)

    def test_firstsplit_bc(self):
        """
        Test if first split on breast cancer
        dataset (binary classification)
        is performed correctly
        """
        data = load_breast_cancer()
        nfeats = data['data'].shape[1]
        nobs = data['data'].shape[0]
        feature, value, importance, _ = classif_best_split(
            data['data'],
            data['target'],
            nfeats,
            nobs
        )
        self.assertEqual(int(feature), 20)
        self.assertAlmostEqual(value, 16.82)
        self.assertAlmostEqual(importance, 0.398062, places=5)

    def test_firstsplit_iris(self):
        """
        Test if first split on iris
        dataset (multiclass classification)
        is performed correctly
        """
        data = load_iris()
        nfeats = data['data'].shape[1]
        nobs = data['data'].shape[0]
        feature, value, importance, _ = classif_best_split(
            data['data'],
            data['target'],
            nfeats,
            nobs
        )
        self.assertEqual(int(feature), 2)
        self.assertAlmostEqual(value, 3.0)
        self.assertAlmostEqual(importance, 1.0)


if __name__ == "__main__":
    unittest.main()
