import numpy as np
import pandas as pd
from typing import Union


class MSIBaseClassifier:

    def __init__(self):
        self._fitted = False
        self._shape = None
        self._ncls = None
        self._ndim = None

    def _validate_input(self, data: Union[np.ndarray, pd.DataFrame, pd.Series],
                        expected_dim: int, inference: bool = False) -> np.ndarray:
        """Honeypot for incorrect input spec"""
        allowed_types = (
            np.ndarray,
            pd.core.frame.DataFrame,
            pd.core.frame.Series
        )

        if type(data) not in allowed_types:
            raise TypeError('Supported input types: np.ndarray, '
                            'pd.core.frame.DataFrame, pd.core.frame.Series got'
                            ' {}'.format(type(data)))

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values

        if data.size == 0:
            raise ValueError('Empty array passed to fit() or predict()')

        if data.ndim > expected_dim:
            raise ValueError('Data with incorrect number of dimensions '
                             'passed to fit() or predict(). Max dim is '
                             '{}, got {}'.format(expected_dim, data.ndim))

        if not np.issubdtype(data.dtype, np.number):
            raise ValueError('Non numeric value found in data')

        if not np.isfinite(data).all():
            raise ValueError('Data contains nan or inf')

        if inference:
            # additional checks on prediction time
            if not self._fitted:
                raise ValueError('Fit the model first.')

            if self._ndim == 2 and data.shape[-1] != self._shape[-1]:
                raise ValueError('Number of features does not match'
                                 ' data model was trained on. Expected'
                                 ' {}, got {}'
                                 .format(self._shape[-1], data.shape[-1]))

        return data
