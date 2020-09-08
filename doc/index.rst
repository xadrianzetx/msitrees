Getting started
===============

msitrees is a set of machine learning models based on `minimum surfeit and inaccuracy <https://ieeexplore.ieee.org/document/8767915>`_ decision tree algorithm. So whats cool about them? No hyperparameters to optimize for base learner. Tree is regularized internally to avoid overfitting by design. Quoting authors of the paper:

 To achieve this, the algorithm must automatically understand when growing the decision tree adds needless complexity, and must
 measure such complexity in a way that is commensurate to some prediction quality aspect, e.g., inaccuracy. We argue that a
 natural way to achieve the above objectives is to define both the inaccuracy and the complexity using the concept of Kolmogorov
 complexity.

For convenience, msitrees comes with scikit-learn style API and can be used with sklearn functions accepting ``estimator`` object as parameter.

Contents
========

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api_guide/index
