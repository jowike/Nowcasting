import pandas as pd
from typing import List

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
    field_names = ['seriesid', 'seriesname', 'frequency', 'transformation', 'units', 'category']
    for field in field_names:
        if field in raw_data.columns:
            spec[field] = raw_data[field].tolist()
        else:
            raise ValueError(f"{field} column missing from model specification.")

    return spec

