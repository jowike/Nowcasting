import os
import pandas as pd
import numpy as np
from utils import _align_dates, _convert_to_datetime
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
    freq_desc_col,
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

    def _format_individual_series(df, ref_date, max_pub_date, freq_desc):
        """
        This function is structured to process each non-target variable code to create a long format series
        based on available data up to a specified publication date (it extracts the last release dates
        up to a specified publication limit).

        Parameters:
        - df (pd.DataFrame): Subset of the dataset containing only non-target series.
        - ref_date (pd.Timestamp): Date for which the data should be evaluated.
        - max_pub_date (pd.Timestamp): Maximum publication date to consider.

        Returns:
        - pd.DataFrame: Long-format DataFrame for each non-target series up to the specified publication date.
        - list: List of series codes with missing data (null columns).
        """
        df_long = pd.DataFrame(
            columns=[series_code_col, ref_date_col, pub_date_col, series_val_col, freq_desc_col]
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
                            freq=freq_desc,
                        )
                    ),
                    axis=1,
                )
                .sort_index()
                .ffill()
            )
            pivot_limit = series_pivot[series_pivot.index < max_pub_date]

            try:
                last_release_dt = (
                    pivot_limit[ref_date]
                    .dropna(how="all")
                    .last_valid_index()
                    .strftime("%Y-%m-%d")
                )
            except (KeyError, AttributeError):
                last_release_dt = None
                null_cols.append(variable_code)
                continue

            series = pivot_limit.loc[last_release_dt].to_frame()
            series = series.loc[
                series.first_valid_index() : min(series.last_valid_index(), ref_date)
            ]
            series = series.reindex(
                sorted(pd.date_range(min(series.index), max(series.index), freq="MS"))
            )

            series.columns = [series_val_col]
            series[series_code_col] = variable_code
            series[pub_date_col] = last_release_dt
            series[freq_desc_col] = freq_desc

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

    X_df_m = X_df.loc[X_df['FrequencyDescription'].str.contains(r'(Monthly)')]
    X_df_q = X_df.loc[X_df['FrequencyDescription'].str.contains(r'(Quarterly)')]

    # Call the nested helper function for non-target series
    cols = [series_code_col, ref_date_col, pub_date_col, series_val_col]
    X_df_long_m, _ = _format_individual_series(
        df=X_df_m[cols], ref_date=ref_date, max_pub_date=y_first_est_release_dt, freq_desc="MS"
    )
    X_df_long_q, _ = _format_individual_series(
        df=X_df_q[cols], ref_date=ref_date, max_pub_date=y_first_est_release_dt, freq_desc="QS"
    )
    X_df_long = pd.concat([X_df_long_m, X_df_long_q])

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
