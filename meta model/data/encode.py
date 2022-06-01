import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def binary(df: pd.DataFrame) -> pd.DataFrame:

    df_ = df.applymap(lambda v: {
        'y': 1.0, '1.0': 1.0,
        'n': 0.0, '0.0': 0.0,
        'nan': np.nan
    }[str(v).lower()])

    return df_


def binary_one_hot(df: pd.DataFrame) -> pd.DataFrame:

    df_ = df.applymap(lambda v: {
        'y': 'Y', '1.0': 'Y', '1': 'Y', 1: 'Y',
        'n': 'N', '0.0': 'N', '0': 'N', 0: 'N',
        'nan': 'U'
    }[str(v).lower()])

    df_ = pd.concat([df_, pd.get_dummies(df_)], axis=1)
    df_.drop(df.columns.to_list(), axis=1, inplace=True)

    return df_


class BinaryNominalEncoder(BaseEstimator, TransformerMixin):

    mapping = {
        'y': 'Y', '1.0': 'Y', '1': 'Y',
        'n': 'N', '0.0': 'N', '0': 'N',
        'nan': np.nan
    }

    def fit(self, X, y=None):

        return self

    def transform(self, X: pd.DataFrame, y=None):

        assert type(X) == pd.DataFrame, 'BinaryNominalEncoder takes DataFrame only'
        return X.applymap(self._transform_value)

    def _transform_value(self, value):
        str_value = str(value).lower()

        if str_value in self.mapping:
            return self.mapping[str_value]
        else:
            return value
