import unittest
import numpy as np
from msitrees._core import gini_impurity


class TestGiniImpurity(unittest.TestCase):

    def test_input_type_list(self):
        try:
            gini_impurity([0, 0])

        except RuntimeError:
            # wrong exception to test against, FIXME
            self.fail('Exception on allowed input type')

    def test_input_type_numpy(self):
        try:
            gini_impurity(np.array([[0, 0]]))

        except RuntimeError:
            self.fail('Exception on allowed input type')

    def test_input_other(self):
        pass

    def test_input_empty_list(self):
        pass

    def test_input_empty_array(self):
        pass

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


if __name__ == "__main__":
    unittest.main()
