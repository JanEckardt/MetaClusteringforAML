import pandas as pd
from sklearn.preprocessing import StandardScaler


def z_score(X: pd.DataFrame) -> pd.DataFrame:

    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
