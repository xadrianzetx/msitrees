Getting started
===============

:code:`msitrees` is a set of machine learning models based on `minimum surfeit and inaccuracy <https://ieeexplore.ieee.org/document/8767915>`_ decision tree algorithm. 
The main difference to other `CART <https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29>`_ methods is, 
that there is no hyperparameters to optimize for base learner. Tree is regularized internally to avoid overfitting by design. Quoting authors of the paper:

 To achieve this, the algorithm must automatically understand when growing the decision tree adds needless complexity, and must
 measure such complexity in a way that is commensurate to some prediction quality aspect, e.g., inaccuracy. We argue that a
 natural way to achieve the above objectives is to define both the inaccuracy and the complexity using the concept of Kolmogorov
 complexity.

Contents
========

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api_guide/index
