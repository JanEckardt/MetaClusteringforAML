import pandas as pd


def typed_view(path: str = 'data/interim/typed_view.csv'):

    df = pd.read_csv(path, na_values='NULL')
    df.set_index('SUBJID', inplace=True)
    return df.reindex(sorted(df.columns), axis=1)
