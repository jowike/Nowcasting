import re
import numpy as np
import pandas as pd
from typing import List

from statsmodels.tsa.stattools import adfuller
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler

import warnings

warnings.filterwarnings("ignore")


def _convert_to_datetime(df: pd.DataFrame, colnames: List[str]) -> pd.DataFrame:
    for col in colnames:
        df[col] = pd.to_datetime(df[col])
    return df


def _align_dates(dataframe: pd.DataFrame, date_colname: str) -> pd.DataFrame:
    """Align reference dates to the 1st. days of months."""
    df = dataframe.reset_index(drop=True).reset_index()
    df[date_colname] = df[date_colname].apply(lambda x: x.replace(day=1))
    return df.drop(columns=["index"])


def lag_to_fill_ragged_edges(df):
    # Iterate over each column (representing different series)
    for col in df.columns:
        series = df[col]

        # Check if the series has missing values at the end
        if series.iloc[-1:].isna().all():
            # Find the position of the last non-NaN value
            last_non_nan = series.last_valid_index()

            # If valid non-NaN is found, create a lagged version of the column
            if last_non_nan is not None:
                shift_amount = len(series) - series.index.get_loc(last_non_nan) - 1
                df[col] = series.shift(shift_amount)

    return df


def cast_spec_to_dict(df):
    """
    Parse variable specifications from a DataFrame and return them as a dictionary.

    This function processes a DataFrame containing model specifications, ensuring
    column headers are lowercase and extracting required fields (`seriesid`, 
    `frequency`, and `transformation`). It raises an error if any of the required 
    columns are missing.

    Args:
        df (pd.DataFrame): The input DataFrame containing the variable specification data.

    Returns:
    - spec: dict, containing the variables specification.
    """
    raw_data = df.copy()
    # Convert all headers to lowercase for consistency
    raw_data.columns = raw_data.columns.str.lower()

    # Initialize spec dictionary
    spec = {}

    # Fields to extract from the Excel file
    field_names = ["seriesid", "seriesname", "frequency", "transformation", "units", "category", "model"]
    # Extract required fields and ensure they exist
    for field in field_names:
        if field not in raw_data.columns:
            raise ValueError(f"{field} column missing from model specification.")
        spec[field] = raw_data[field].tolist()    

    return spec


def suggest_transformation(unit):
    """
    Suggests a transformation based on the economic meaning of the unit using regex.
    """
    if re.search(r"(Billions|Millions|Thousands)\s+of\s+Chained\s+\d{4}\s+Dollars", unit):
        return "pc1"

    if re.search(r"Index", unit):
        return "ch1"  # Wskaźniki, np. CPI, używamy zmiany

    if re.search(r"Percent", unit):
        return "lin" # Jeśli jednostka to Percent, nie stosujemy transformacji

    if re.search(r"Percent\s+Change\s+at\s+Annual\s+Rate", unit):
        return "lin"  # Jeśli to już procentowa zmiana roczna, nie wymagamy dalszej transformacji

    if re.search(r"Level", unit):
        return "pc1"  # Poziomy (np. liczba osób, PKB) — stosujemy procentową zmianę roczną (pca)


    if re.search(r"Ratio", unit):
        return "lin"  # Wskaźniki proporcji — nie wymagają transformacji


    if re.search(r"Rate", unit):
        return "lin"  # Stopy procentowe (np. procentowa stopa bezrobocia) — pozostawiamy bez zmian


    if re.search(r"(Dollars|Euro|Yen|Pounds|Rupees|Franc|Pesos)\s+per\s+[A-Za-z]+", unit):
        return "pc1"  # Ceny jednostkowe (np. "Dollars per Gallon") — zmiana procentowa


    if re.search(r"Thousands\s+of\s+[A-Za-z]+", unit):
        return "pc1"  # Jednostki liczbowe w tysiącach — logarytmizacja

    return "ch1"

def test_variance(
    data: pd.DataFrame,
):
    # discard low-variance features
    transformer = MaxAbsScaler().fit(data)
    df_scaled = pd.DataFrame(transformer.transform(data), columns=data.columns)
    tsh = df_scaled.var().quantile(0.05)
    selector = VarianceThreshold()

    selector = VarianceThreshold(threshold=tsh)
    selector.fit(df_scaled)

    features = df_scaled.columns[selector.get_support(indices=True)]
    
    return list(set(data.columns)-set(features))

# stage 4.
def test_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """Detect series with non-stationarity effects"""

    def __adfuller_test(
        series: pd.Series, signif: float = 0.05, name: str = "", verbose: bool = False
    ):
        """Perform ADFuller to test for Stationarity of given series and print report"""
        r = adfuller(series, autolag="AIC")
        output = {
            "test_statistic": round(r[0], 4),
            "pvalue": round(r[1], 4),
            "n_lags": round(r[2], 4),
            "n_obs": r[3],
        }
        p_value = output["pvalue"]

        def adjust(val, length=6):
            return str(val).ljust(length)

        # Print Summary
        if verbose:
            print(f' Augmented Dickey-Fuller Test on "{name}"', "\n   ", "-" * 47)
            print(" Null Hypothesis: Data has unit root. Non-Stationary.")
            print(f" Significance Level = {signif}")
            print(f' Test Statistic = {output["test_statistic"]}')
            print(' No. Lags Chosen = {output["n_lags"]}')

            for key, val in r[4].items():
                print(f" Critical value {adjust(key)} = {round(val, 3)}")

            if p_value <= signif:
                print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
                print(" => Series is Stationary.")
            else:
                print(
                    f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis."
                )
                print(" => Series is Non-Stationary.")
        return p_value

    stat = []

    for _id in df.columns:
        series_df = df[[_id]]
        series = series_df.dropna().sort_index()
        
        test_pval = __adfuller_test(series=series, name=_id)
        if test_pval < 0.05:
            stat.append(_id)

    return list(set(df.columns) - set(stat))


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    EPSILON = 1e-10
    return _error(actual, predicted) / (actual + EPSILON)

def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

