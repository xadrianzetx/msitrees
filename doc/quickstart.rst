Example
=======

.. code-block:: python

   from msitrees.tree import MSIDecisionTreeClassifier
   from sklearn.datasets import load_iris
   from sklearn.model_selection import cross_val_score

   data = load_iris()
   clf = MSIDecisionTreeClassifier()
   cross_val_score(clf, data['data'], data['target'], cv=10)

   # array([1.        , 1.        , 1.        , 0.93333333, 0.93333333,
      #    0.8       , 0.93333333, 0.86666667, 0.8       , 1.        ])