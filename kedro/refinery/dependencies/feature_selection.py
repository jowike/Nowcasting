import mifs
import pandas as pd


def mtsfs(ds: pd.DataFrame, series_name: str, method: str):

    df = ds.sort_index()
    df.index = pd.to_datetime(df.index)

    X = df.drop(columns=[series_name])
    y = df[[series_name]]

    feat_selector = mifs.MutualInformationFeatureSelector(
        method=method,
        n_features="auto",
        categorical=False
    )

    # find all relevant features
    try:
        feat_selector.fit(X.values, y.values.ravel())
    except ValueError as e:
        print(f"ERROR: Exception raised for {series_name}: {e}")  # https://github.com/danielhomola/mifs/issues/15

    # call transform() on X to filter it down to selected features
    X_support = pd.DataFrame(
        feat_selector.transform(X.values),
        columns=X.columns[feat_selector._support_mask],
        index=X.index,
    )
    to_write = pd.merge(X_support, y, left_index=True, right_index=True, how="right")

    return to_write