<p align="center">
<a href='https://msitrees.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/msitrees/badge/?version=latest' alt='Documentation Status' />
</a>

<a href='https://github.com/xadrianzetx/msitrees/actions'>
    <img src='https://github.com/xadrianzetx/msitrees/workflows/Linux%20build/badge.svg' alt='Build linux' />
</a>

<a href='https://github.com/xadrianzetx/msitrees/actions'>
    <img src='https://github.com/xadrianzetx/msitrees/workflows/Windows%20build/badge.svg' alt='Build windows' />
</a>

<a href="https://badge.fury.io/py/msitrees">
    <img src="https://badge.fury.io/py/msitrees.svg" alt="PyPI version">
</a>

</p>

# msitrees

```msitrees``` is a set of machine learning models based on [minimum surfeit and inaccuracy](https://ieeexplore.ieee.org/document/8767915) decision tree algorithm. The main difference to other [CART](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29) methods is, that there is no hyperparameters to optimize for base learner. Tree is regularized internally to avoid overfitting by design. Quoting authors of the paper:

> To achieve this, the algorithm must automatically understand when growing the decision tree adds needless complexity, and must
> measure such complexity in a way that is commensurate to some prediction quality aspect, e.g., inaccuracy. We argue that a
> natural way to achieve the above objectives is to define both the inaccuracy and the complexity using the concept of Kolmogorov
> complexity.

## Installation

### With pip

```bash
pip install msitrees
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

API documentation is available [here](https://msitrees.readthedocs.io/en/latest/index.html).

## Zero hyperparameter based approach

MSI based algorithm should have performance comparable to CART decision tree where best hyperparameters were established with
some sort of search. We are going to compare ```MSIRandomForestClassifier``` with ```scikit-learn``` implementation of random forest algorithm with hyperparameters grid searched using ```optuna```. Both algorithms will be limited to 100 estimators, and measured by comparing accuracy on validation set of MNIST dataset.

```python
   import optuna
   from sklearn.ensemble import RandomForestClassifier

   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   data = load_digits()
   x_train, x_valid, y_train, y_valid = train_test_split(data['data'], data['target'], random_state=42)

   def objective(trial):
      params = {
          'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
          'max_depth': trial.suggest_int('max_depth', 8, 20),
          'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
          'random_state': 42,
          'n_estimators': 100
      }

      clf = RandomForestClassifier(**params)
      clf.fit(x_train, y_train)
      pred = clf.predict(x_valid)
      score = accuracy_score(y_valid, pred)

      return score
   
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_jobs=-1, show_progress_bar=True, n_trials=500)
   
   # fit benchmark model on best params
   benchmark = RandomForestClassifier(**study.best_params)
   benchmark = benchmark.fit(x_train, y_train)

   pred = benchmark.predict(x_valid)
   accuracy_score(y_valid, pred)
   # 0.9711111111111111
```

Since MSI based algorithm has no additional hyperparameters, code is sparse.

```python
   from msitrees.ensemble import MSIRandomForestClassifier

   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   data = load_digits()
   x_train, x_valid, y_train, y_valid = train_test_split(data['data'], data['target'], random_state=42)

   clf = MSIRandomForestClassifier(n_estimators=100)
   clf.fit(x_train, y_train)
   pred = msiclf.predict(x_valid)
   accuracy_score(y_valid, pred)
   # 0.9733333333333334
```

Results for both random forest algorithms are comparable. Furthermore, median depth of a tree estimator is equal for both methods,
even though MSI has no explicit parameter controlling tree depth.

```python
   np.median([e.get_depth() for e in benchmark.estimators_])
   # 12.0
   np.median([e.get_depth() for e in clf._estimators])
   # 12.0
```