import numpy as np
import warnings
import pandas as pd
import warnings
from datetime import datetime


def retransform_(
    X: np.ndarray,
    Z: np.ndarray,
    Time: np.ndarray,
    Spec: dict,
    header: list,
    cutoff_date: datetime,
) -> np.ndarray:
    """
    Retransforms predictions back to their original form based on specified transformations.

    This function takes stationary (transformed) data and reverts it to its original scale using the
    transformation specifications provided. It handles various types of transformations such as linear
    levels, changes, percent changes, and logarithmic transformations.

    Parameters
    ----------
    X : np.ndarray
        Transformed data (stationary), with shape (H, N), where H is the number of periods and N is the
        number of series.
    Z : np.ndarray
        Nominal (raw) data in base units, with shape (H, N).
    Time : np.ndarray
        Observation periods for the time series data, typically as datetime objects.
    Spec : dict
        Model specification containing transformation details. It should include the following keys:
            - "transformation": List of transformation types for each series.
            - "frequency": List of frequencies (e.g., 'm' for monthly) for each series.
            - "seriesid": List of series identifiers.
            - "seriesname": List of series names.
    header : list
        Original data headers corresponding to each series.
    cutoff_date : datetime
        The date marking the beginning of the transformed records. All data from this date onward
        are treated as forecasts.

    Returns
    -------
    np.ndarray
        Retransformed data in original form, with shape (H, N). This array includes both historical
        data from `Z` and retransformed forecasts from `X`.

    Raises
    ------
    AssertionError
        If any header in `header` does not match the corresponding `Spec["seriesid"]`.
    Warning
        If an unknown transformation type is encountered for any series, the function will issue a
        warning and use the untransformed data for that series.

    Examples
    --------
    >>> import numpy as np
    >>> from datetime import datetime
    >>> X = np.random.randn(100, 3)
    >>> Z = np.random.randn(120, 3)
    >>> Time = np.array([datetime(2020, 1, 1) + np.timedelta64(i, 'M') for i in range(120)])
    >>> Spec = {
    ...     "transformation": ["lin", "chg", "log"],
    ...     "frequency": ["m", "m", "m"],
    ...     "seriesid": ["series1", "series2", "series3"],
    ...     "seriesname": ["Series 1", "Series 2", "Series 3"]
    ... }
    >>> header = ["series1", "series2", "series3"]
    >>> cutoff_date = datetime(2023, 1, 1)
    >>> V_final = retransform_data(X, Z, Time, Spec, header, cutoff_date)
    """

    # Filter indexes that denote predictions
    c_idx = np.where(Time >= cutoff_date)[0]

    # Get the number of periods of predictions
    T = c_idx.shape[0]
    # Get number of periods, series
    H = X.shape[0]
    N = X.shape[1]
    # Initialize T x N matrix filled with NaN's
    V = np.full((T, N), np.nan)

    for i in range(N):
        formula = Spec["transformation"][i]
        freq = Spec["frequency"][i]
        step = 1 if freq == "m" else 3
        t1 = step
        n = step / 12
        assert header[i] == Spec["seriesid"][i]
        series = Spec["seriesname"][i]

        # Apply inverse transformations based on formula
        if formula == "lin":  # Levels (No Transformation)
            V[:, i] = X[:, i][c_idx]
        elif formula == "chg":  # Change (Difference)
            # V[0:T:step, i] = np.cumsum(X[c_idx[0]:H:step, i]) + Z[c_idx[0] - step, i]  # Assuming X[t1-step] is a last historical value
            V[0:T:step, i] = np.add(
                X[c_idx[0] : H : step, i], Z[c_idx[0] - step : H - step : step, i]
            )
        elif formula == "ch1":
            V[0:T:step, i] = np.add(
                Z[c_idx[0] - 12 : H - 12 : step, i], X[c_idx[0] : H : step, i]
            )
        elif formula == "pch":  # Percent Change
            # V[0:T:step, i] = np.cumprod(1 + (X[c_idx[0]:H:step, i] / 100)) * Z[c_idx[0] - step, i]  # Assuming X[t1-step] is a last historical value
            V[0:T:step, i] = np.multiply(
                1 + (X[c_idx[0] : H : step, i] / 100),
                Z[c_idx[0] - step : H - step : step, i],
            )
        elif formula == "pc1":  # Year over Year Percent Change
            V[0:T:step, i] = np.multiply(
                1 + (X[c_idx[0] : H : step, i] / 100),
                Z[c_idx[0] - 12 : H - 12 : step, i],
            )
        elif formula == "pca":  # Percent Change (Annual Rate)
            # V[0:T:step, i] = np.cumprod((1 + X[c_idx[0]:H:step, i] / 100) ** n) * Z[c_idx[0] - step, i]
            V[0:T:step, i] = ((1 + X[c_idx[0] : H : step, i] / 100) ** n) * Z[
                c_idx[0] - step : H - step : step, i
            ]
        elif formula == "log":  # Natural Log
            V[:, i] = np.exp(X[:, i][c_idx])
        else:
            warnings.warn(
                f"Transformation '{formula}' not found for {series}. Using untransformed data."
            )
            V[:, i] = X[:, i][c_idx]

        V_final = np.full((H, N), np.nan)
        for _ in range(N):
            V_final[:, _] = np.concatenate((Z[: c_idx[0], _], V[:, _]))
    return V_final


# Example usage retransform_data(X, Z, Time, Spec, header, datetime(2023, 1, 1))

# def retransform_prediction(transf_series, base_series, Spec, series_name):

#     return R, Time
