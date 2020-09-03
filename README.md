# msitrees

msitrees is a set of machine learning models based on [minimum surfeit and inaccuracy](https://ieeexplore.ieee.org/document/8767915) decision tree algorithm. So whats cool about them? No hyperparameters to optimize for base learner. Tree is regularized internally to avoid overfitting by design. Quoting authors of the paper:

> To achieve this, the algorithm must automatically understand when growing the decision tree adds needless complexity, and must
> measure such complexity in a way that is commensurate to some prediction quality aspect, e.g., inaccuracy. We argue that a
> natural way to achieve the above objectives is to define both the inaccuracy and the complexity using the concept of Kolmogorov
> complexity.

For convenience, msitrees comes with scikit-learn style API and can be used with sklearn functions accepting ```estimator``` object as parameter.

## Instalation

### With pip

```bash
soon
```

### From source

```bash
git clone https://github.com/xadrianzetx/msitrees.git
cd msitrees
python setup.py install
```

Windows builds require at least [MSVC2015](https://www.microsoft.com/en-gb/download/details.aspx?id=48145)

## Quick start

```python
from msitrees.tree import MSIDecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

data = load_iris()
clf = MSIDecisionTreeClassifier()
cross_val_score(clf, data['data'], data['target'], cv=10)

# array([1.        , 1.        , 1.        , 0.93333333, 0.93333333,
    #    0.8       , 0.93333333, 0.86666667, 0.8       , 1.        ])
```

## Reference documentation

Soon.
