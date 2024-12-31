import os
import pandas as pd
import numpy as np
from utils_ import _align_dates, _convert_to_datetime
import warnings

warnings.filterwarnings("ignore")


def prepare_real_time_vintage_data(
    ds,
    y_code,
    ref_date,
    series_code_col,
    ref_date_col,
    pub_date_col,
    series_val_col,
):
    """
    Prepares an actual real-time dataset for a given reference date, adjusting for data publication schedules.

    Parameters:
    - ds (pd.DataFrame): The dataset containing time series with reference and publication dates.
    - y_code (str): The code of the target variable (e.g., GDP).
    - ref_date (str): The reference date for which to prepare data.
    - series_code_col (str): Column name for the series code.
    - ref_date_col (str): Column name for the reference date.
    - pub_date_col (str): Column name for the publication date.
    - series_val_col (str): Column name for the series values.
    - out_path (str, optional): Directory path for saving the output file.

    Returns:
    - pd.DataFrame: A long-format DataFrame containing real-time dataset prepared for the target reference date.
    """

    def _format_individual_series(df, ref_date, pub_date_limit):
        """
        This function is structured to process each non-target variable code to create a long format series
        based on available data up to a specified publication date (it extracts the last release dates
        up to a specified publication limit).

        Parameters:
        - df (pd.DataFrame): Subset of the dataset containing only non-target series.
        - ref_date (pd.Timestamp): Date for which the data should be evaluated.
        - pub_date_limit (pd.Timestamp): Maximum publication date to consider.

        Returns:
        - pd.DataFrame: Long-format DataFrame for each non-target series up to the specified publication date.
        - list: List of series codes with missing data (null columns).
        """
        df_long = pd.DataFrame(
            columns=[series_code_col, ref_date_col, pub_date_col, series_val_col]
        )
        null_cols = []

        for variable_code in df[series_code_col].unique():
            series_long = df[df[series_code_col] == variable_code]

            series_pivot = series_long.pivot(
                index=pub_date_col, columns=ref_date_col, values=series_val_col
            )
            series_pivot = (
                series_pivot.reindex(
                    sorted(
                        pd.date_range(
                            min(series_pivot.columns),
                            max(series_pivot.columns),
                            freq="MS",
                        )
                    ),
                    axis=1,
                )
                .sort_index()
                .ffill()
            )
            if series_pivot.index.min().strftime("%Y-%m-%d") >= pub_date_limit:
                print(
                    f"The observation for {variable_code} was unavailable before {pub_date_limit}."
                )
                series = pd.DataFrame(
                    {series_val_col: np.nan, pub_date_col: np.datetime64("NaT")},
                    index=series_pivot.columns,
                )
            else:
                pivot_limit = series_pivot[series_pivot.index < pub_date_limit]

                last_release_dt = (
                    pivot_limit[
                        min(
                            max(pivot_limit.dropna(how="all", axis=1).columns), ref_date
                        )
                    ]
                    .dropna(how="all")
                    .last_valid_index()
                    .strftime("%Y-%m-%d")
                )

                series = pivot_limit.loc[last_release_dt].to_frame()
                series = series.loc[
                    series.first_valid_index() : min(
                        series.last_valid_index(), ref_date
                    )
                ]
                series = series.reindex(
                    sorted(
                        pd.date_range(min(series.index), max(series.index), freq="MS")
                    )
                )
                series.columns = [series_val_col]
                series[pub_date_col] = last_release_dt

            series[series_code_col] = variable_code

            if series[series_val_col].isnull().any():
                null_cols.append(variable_code)

            df_long = pd.concat([df_long, series.reset_index()])

        return df_long, null_cols

    # Start of the main function
    ref_date = pd.to_datetime(ref_date)
    ds = _align_dates(dataframe=ds, date_colname=ref_date_col)

    y_long = ds[ds[series_code_col] == y_code]
    y_pivot = y_long.pivot(
        index=pub_date_col, columns=ref_date_col, values=series_val_col
    )
    y_pivot = (
        y_pivot.reindex(
            sorted(
                pd.date_range(min(y_pivot.columns), max(y_pivot.columns), freq="MS")
            ),
            axis=1,
        )
        .sort_index()
        .ffill()
    )

    y_first_est_release_dt = y_pivot[ref_date].first_valid_index().strftime("%Y-%m-%d")
    X_df = ds[ds[series_code_col] != y_code]

    # Call the nested helper function for non-target series
    X_df_long, _ = _format_individual_series(
        df=X_df, ref_date=ref_date, pub_date_limit=y_first_est_release_dt
    )

    rt_vintage_dt = y_pivot[y_pivot.index < y_first_est_release_dt].index.max()
    y_series = y_pivot[y_pivot.columns[y_pivot.columns <= ref_date]].loc[rt_vintage_dt]
    y_series.loc[ref_date] = y_pivot.loc[y_first_est_release_dt, ref_date]

    y_long = (
        y_series.to_frame()
        .reset_index()
        .rename(columns={y_series.name: series_val_col})
    )
    y_long[series_code_col] = y_code
    y_long[pub_date_col] = np.where(
        y_long[ref_date_col] < ref_date, rt_vintage_dt, y_first_est_release_dt
    )

    df_long = pd.concat(
        [
            _convert_to_datetime(y_long, [ref_date_col, pub_date_col]),
            _convert_to_datetime(X_df_long, [ref_date_col, pub_date_col]),
        ]
    )

    return df_long
