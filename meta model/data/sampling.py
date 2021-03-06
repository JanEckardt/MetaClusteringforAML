import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def stratified_train_validation_test_split(df: pd.DataFrame, label_name: str, split_size=0.1, random_state=42)\
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series):

    df_ = df.copy()

    split = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=random_state)

    X_train = pd.DataFrame()
    X_val = pd.DataFrame()
    X_test = pd.DataFrame()

    for train, test in split.split(df_, df_[label_name]):
        X_train = df_.iloc[train].copy()
        X_test = df_.iloc[test].copy()

    df_.drop(index=X_test.index, inplace=True)

    for train, val in split.split(df_, df_[label_name]):
        X_train = df_.iloc[train].copy()
        X_val = df_.iloc[val].copy()

    y_train: pd.Series = X_train[label_name]
    y_val: pd.Series = X_val[label_name]
    y_test: pd.Series = X_test[label_name]

    return X_train, X_val, X_test, y_train, y_val, y_test


def stratified_train_test_split(df: pd.DataFrame, label_name: str, split_size=0.1, random_state=42)\
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):

    df_ = df.copy()

    split = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=random_state)

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()

    for train, test in split.split(df_, df_[label_name]):
        X_train = df_.iloc[train].copy()
        X_test = df_.iloc[test].copy()

    y_train: pd.Series = X_train[label_name]
    y_test: pd.Series = X_test[label_name]

    return X_train, X_test, y_train, y_test


def random_train_test_split(df: pd.DataFrame, label_name: str, test_size=0.1, random_state=42) \
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):

    return train_test_split(df.copy(), df[label_name].copy(), test_size=test_size, random_state=random_state)


def random_train_validation_test_split(df: pd.DataFrame, label_name: str, split_size=0.1, random_state=42) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series):

    X_train, X_test, y_train, y_test = train_test_split(
        df.copy(), df[label_name].copy(), test_size=split_size, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train.copy(), X_train[label_name].copy(), test_size=split_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test
