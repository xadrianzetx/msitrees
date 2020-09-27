Examples
========

Minimal example
---------------

:code:`msitrees` follows :code:`scikit-learn` API style, which allows for fast model iteration. Below is an example, where decision
tree classifier is fitted and scored over 10 fold cross validation using :code:`cross_val_score()`.

.. code-block:: python

   from msitrees.tree import MSIDecisionTreeClassifier
   from sklearn.datasets import load_iris
   from sklearn.model_selection import cross_val_score

   data = load_iris()
   clf = MSIDecisionTreeClassifier()
   cross_val_score(clf, data['data'], data['target'], cv=10)

   # array([1.        , 1.        , 1.        , 0.93333333, 0.93333333,
      #    0.8       , 0.93333333, 0.86666667, 0.8       , 1.        ])

Model preservation
------------------

Model preservation is possible with :code:`pickle` module.

.. code-block:: python

   import pickle

   with open('model.pkl', 'wb') as file:
      pickle.dump(clf, file)

The same can be used to load the model back.

.. code-block:: python

   with open('model.pkl', 'rb') as file:
      loaded_model = pickle.load(file)

Zero hyperparameter based approach
----------------------------------

MSI based algorithm should have performance comparable to CART decision tree where best hyperparameters were established with
some sort of search. We are going to compare :code:`MSIRandomForestClassifier` with :code:`scikit-learn` implementation of random forest 
algorithm with hyperparameters grid searched using :code:`optuna`. Both algorithms will be limited to 100 estimators, and measured
by comparing accuracy on validation set of MNIST dataset.

.. code-block:: python

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

Since MSI based algorithm has no additional hyperparameters, code is sparse.

.. code-block:: python

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

Results for both random forest algorithms are comparable. Furthermore, median depth of a tree estimator is equal for both methods,
even though MSI has no explicit parameter controlling tree depth.

.. code-block:: python

   np.median([e.get_depth() for e in benchmark.estimators_])
   # 12.0
   np.median([e.get_depth() for e in clf._estimators])
   # 12.0