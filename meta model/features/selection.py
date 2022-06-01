import os
from typing import List, Tuple, Union, Optional

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, ARDRegression
from sklearn.preprocessing import MinMaxScaler, minmax_scale

from sal.data import Attrs
from sal.data.explore import pairwise_correlation


def drop_irrelevant_attributes(df: pd.DataFrame):

    df.drop(inplace=True, columns=Attrs.irrelevant)


def drop_constant_attributes(df: pd.DataFrame):

    df.drop(inplace=True, columns=df.columns[df.nunique(dropna=False) <= 1])


def drop_date_attributes(df: pd.DataFrame):

    df.drop(inplace=True, columns=Attrs.dates)


def drop_features_with_low_information_gain(df: pd.DataFrame):

    for col in ['ECOGCAT', 'AGEGR']:
        if col in df.columns:
            df.drop(inplace=True, columns=[col])


def drop_labels(data: Union[pd.DataFrame, List[pd.DataFrame]], more: Optional[List[str]] = None) -> None:

    label_names = [
        'OSTM', 'OSSTAT', 'RFSTM', 'RFSSTAT', 'EFSTM', 'EFSSTAT', 'CR1', 'ED60', 'ED30',
        'CISTAT', 'CR1DTC', 'RFSDTC', 'OSDTC', 'DTHDTC', 'EFSDTC'
    ]

    if more is not None:
        label_names.extend(more)

    def _drop_labels(df: pd.DataFrame):
        labels = []

        for feature in df.columns:
            for label_name in label_names:
                if str(feature).startswith(label_name):
                    labels.append(str(feature))

        df.drop(columns=labels, inplace=True)

    if isinstance(data, pd.DataFrame):
        _drop_labels(data)
    else:
        for df in data:
            _drop_labels(df)


def feature_support_by_correlation(X: pd.DataFrame, y: pd.Series) -> List[float]:

    cor_list = []
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1] if np.std(X[i]) > 0 else 0
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    return minmax_scale(np.abs(cor_list))


def feature_support_by_chi2(X: pd.DataFrame, y: pd.Series, num_feats: int, classification: bool) -> List[float]:

    if classification:
        X_norm = MinMaxScaler().fit_transform(X)

        chi_selector = SelectKBest(chi2, k=num_feats)
        chi_selector.fit(X_norm, y)
        scores = chi_selector.scores_
        scores = [0 if np.isnan(i) else i for i in scores]
        return minmax_scale(scores)
    else:
        return len(X.columns) * [0]


def feature_support_by_rfe(X: pd.DataFrame, y: pd.Series, classification: bool) -> List[float]:

    X_norm = MinMaxScaler().fit_transform(X)

    if classification:
        estimator = LogisticRegression(max_iter=1000)
    else:
        estimator = ARDRegression()

    rfe_selector = RFE(estimator=estimator, step=10)
    rfe_selector.fit(X_norm, y)

    if classification:
        _coef = rfe_selector.estimator_.coef_[0]
    else:
        _coef = rfe_selector.estimator_.coef_

    _support = rfe_selector.get_support()
    indices = np.where(_support == True)[0]
    support = len(X.columns) * [0]

    for i in range(len(indices)):
        support[indices[i]] = _coef[i]

    return minmax_scale(np.abs(support))


def feature_support_by_lasso(X: pd.DataFrame, y: pd.Series, num_feats: int, classification: bool) -> List[float]:

    X_norm = MinMaxScaler().fit_transform(X)

    if classification:
        estimator = LogisticRegression(penalty="l2", max_iter=1000)
    else:
        estimator = ARDRegression()

    embeded_lr_selector = SelectFromModel(estimator, max_features=num_feats)
    embeded_lr_selector.fit(X_norm, y)

    if classification:
        _coef = embeded_lr_selector.estimator_.coef_[0]
    else:
        _coef = embeded_lr_selector.estimator_.coef_

    return minmax_scale(np.abs(_coef))


def feature_support_by_random_forest(X: pd.DataFrame, y: pd.Series, num_feats: int, classification: bool)\
        -> List[float]:

    if classification:
        estimator = RandomForestClassifier(n_estimators=100)
    else:
        estimator = RandomForestRegressor(n_estimators=100)

    embeded_rf_selector = SelectFromModel(estimator, max_features=num_feats)
    embeded_rf_selector.fit(X, y)

    return minmax_scale(embeded_rf_selector.estimator_.feature_importances_)


def feature_support_combined(X: pd.DataFrame, y: pd.Series, num_feats: int,
                             classification: bool = True) -> pd.DataFrame:

    cor_support = feature_support_by_correlation(X, y)
    chi2_support = feature_support_by_chi2(X, y, num_feats, classification)
    rfe_support = feature_support_by_rfe(X, y, classification)
    lasso_support = feature_support_by_lasso(X, y, num_feats, classification)
    rf_support = feature_support_by_random_forest(X, y, num_feats, classification)

    features = pd.DataFrame({
        'name':  X.columns.tolist(),
        'correlation': cor_support,
        'chi2': chi2_support,
        'rfe': rfe_support,
        'lasso': lasso_support,
        'random_forest': rf_support
    })

    features['support'] = np.sum(features, axis=1)
    features = features.sort_values(['support', 'name'], ascending=False)
    features.index = range(1, len(features) + 1)

    return features


def drop_features_by_support(
        X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, classification: bool,
        top_feature_count: int, top_feature_support_min: int, experiment_report_path: str)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:

    feature_support = feature_support_combined(X_train, y_train, top_feature_count, classification=classification)
    feature_support.to_csv(os.path.join(experiment_report_path, 'feature_support.csv'))

    scatter_data = feature_support.sort_values(by='support', ascending=False).head(30)
    scatter = sns.scatterplot(data=scatter_data, x="support", y="name")
    scatter.set(xlabel="Feature Support (Importance)", ylabel="Feature")
    scatter.get_figure().savefig(os.path.join(experiment_report_path, 'feature_support.jpg'),
                                 pad_inches=0.0, bbox_inches='tight')
    plt.clf()

    features_correlated = pairwise_correlation(X_train).query('corr >= 1.0').reset_index().A.unique().tolist()
    features = feature_support.query('support >= {}'.format(top_feature_support_min)).name
    features = features[~features.isin(features_correlated)]
    return X_train[features], X_test[features]


class DefaultAttributeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, drop_text_attributes=True, drop_constant_attributes=True, drop_low_information_gain=True):
        self.drop_text_attributes = drop_text_attributes
        self.drop_constant_attributes = drop_constant_attributes
        self.drop_low_information_gain = drop_low_information_gain

    def fit(self, X, y=None):

        return self

    def transform(self, X: pd.DataFrame, y=None):

        X_ = X.copy()

        drop_irrelevant_attributes(X_)
        
        if self.drop_constant_attributes:
            drop_constant_attributes(X_)

        if self.drop_low_information_gain:
            drop_features_with_low_information_gain(X_)

        if self.drop_text_attributes:
            for col in Attrs.text:
                if col in X_.columns:
                    X_.drop(inplace=True, columns=[col])

        return X_
