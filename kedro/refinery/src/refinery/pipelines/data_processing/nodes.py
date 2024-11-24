import sys

sys.path.append("/Users/ejowik001/Desktop/Github/Nowcasting/kedro/refinery/src/scripts")

import pandas as pd
import numpy as np
from typing import List, Literal, Tuple
from datetime import datetime
from itertools import compress

from scripts.utils import _convert_to_datetime, cast_spec_to_dict
from scripts.data_revisions import prepare_real_time_vintage_data
from scripts.ragged_edges import shift_to_fill_trailing_nans
from scripts.load_spec import load_spec
from scripts.remNaNs_spline import remNaNs_spline
from scripts.load_data import load_data
from scripts.summarize import summarize


def prepare_vintage_data(
    ds: pd.DataFrame,
    parameters: dict,
    spec_options: dict = None,
) -> pd.DataFrame:
    """
    This function prepares retrospective dataset (vintage data) based on revision history
    """
    ds = _convert_to_datetime(
        ds, [parameters["ref_date_col"], parameters["pub_date_col"]]
    )

    if spec_options:
        Spec = load_spec(spec_options["filepath"])
        SeriesID, SeriesName, Units, UnitsTransformed, Frequency = (
            Spec["seriesid"],
            Spec["seriesname"],
            Spec["units"],
            Spec["unitstransformed"],
            Spec["frequency"],
        )
        df = ds.loc[ds[parameters["series_code_col"]].isin(SeriesID)]
    else:
        df = ds.loc[
            ds[parameters["freq_desc_col"]].isin(parameters["scope_freq_desc"])
        ].copy()

    cols = [
        parameters["series_code_col"],
        parameters["ref_date_col"],
        parameters["pub_date_col"],
        parameters["series_val_col"],
    ]

    if parameters["vintage_name"] == "real-time":
        df_long = prepare_real_time_vintage_data(
            ds=df[cols].drop_duplicates(),
            y_code=parameters["y_code"],
            ref_date=parameters["ref_date"],
            series_code_col=parameters["series_code_col"],
            ref_date_col=parameters["ref_date_col"],
            pub_date_col=parameters["pub_date_col"],
            series_val_col=parameters["series_val_col"],
        )
    # TODO: preliminary, current-vintage (pseudo-real-time)
    return df_long


def build_spec_from_source(ds: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Build a standardized specification of variables from the source data.

    This function filters and processes a DataFrame to create a standardized 
    specification of variables. It maps columns to consistent names, extracts 
    required information, and applies transformations to generate the final output.

    Args:
        df (pd.DataFrame): The source DataFrame containing the raw variable data.
        parameters (dict): A dictionary containing the following keys:
            - "freq_desc_col": Column name for frequency descriptions.
            - "scope_freq_desc": List of frequency descriptions to include (e.g., ["Monthly", "Quarterly"]).
            - "series_code_col": Column name for series codes.
            - "series_name_col": Column name for series names.
            - "unit_col": Column name for units.
            - "series_categ_col": Column name for series categories.
            - "default_transf_code": Default transformation code to apply.

    Returns:
        pd.DataFrame: A DataFrame with the following standardized columns:
            - "SeriesID"
            - "SeriesName"
            - "Frequency"
            - "Transformation"
            - "Units"
            - "Category"
    """
    # Define the output columns
    output_columns = [
        "SeriesID",
        "SeriesName",
        "Frequency",
        "Transformation",
        "Units",
        "Category",
    ]

    # Filter the source DataFrame based on the specified frequency descriptions
    df = ds.loc[
        ds[parameters["freq_desc_col"]].isin(parameters["scope_freq_desc"])
    ].copy()

    # Select and rename columns
    renamed_df = (
        df[
            [
                parameters["series_code_col"],
                parameters["freq_desc_col"],
                parameters["series_name_col"],
                parameters["unit_col"],
                parameters["series_categ_col"],
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                parameters["series_code_col"]: "SeriesID",
                parameters["series_name_col"]: "SeriesName",
                parameters["unit_col"]: "Units",
                parameters["series_categ_col"]: "Category",
            }
        )
    )

    # Map frequency descriptions to standardized frequency codes
    renamed_df["Frequency"] = renamed_df[parameters["freq_desc_col"]].apply(
        lambda x: "m" if "Monthly" in x else "q" if "Quarterly" in x else None
    )

    # Add a default transformation column
    renamed_df["Transformation"] = parameters["default_transf_code"]

    # Return the final DataFrame with standardized columns
    return renamed_df[output_columns]



def harmonize_ragged_edges(
    ds,
    ds_spec,
    parameters,
):
    to_write = pd.DataFrame()
    for freq_desc in ds_spec["Frequency"].unique():
        series_codes = ds_spec.loc[ds_spec["Frequency"] == freq_desc]["SeriesID"]
        subset = ds.loc[ds[parameters["series_code_col"]].isin(series_codes)]
        if subset.shape[0]:
            df_f_pivot = subset.pivot(
                index=parameters["ref_date_col"],
                columns=parameters["series_code_col"],
                values=parameters["series_val_col"],
            )
            res_i = shift_to_fill_trailing_nans(df_f_pivot)

            to_write = pd.concat(
                [to_write, pd.melt(res_i, value_vars=res_i.columns, ignore_index=False)]
            )

    to_write = to_write.reset_index().pivot(
        index=parameters["ref_date_col"],
        columns=parameters["series_code_col"],
        values="value",
    )
    return to_write


def transform_time_series(
    ds: pd.DataFrame,
    ds_spec: pd.DataFrame,
    parameters: dict,
    spec_options: dict = None,
):
    if parameters["sample_start"]:
        sample_start = pd.to_datetime(parameters["sample_start"], format="%Y-%m-%d")

    if spec_options:
        Spec = load_spec(spec_options["filepath"])

        X, Time, Z, header = load_data(ds, Spec, sample_start)

        # summarize data
        summarize(X.astype(float), Time, Spec)

        # Prepare data -----------------------------------------------------------
        Mx = np.nanmean(X, axis=0)
        Wx = np.nanstd(X, axis=0)
        xNaN = (X - Mx) / Wx  # Standardize series

        optNaN = {"method": 2, "k": 3}
        x_est, _, nanLE = remNaNs_spline(xNaN, optNaN)  # Impute series

        summarize(x_est, Time[~nanLE], Spec)

        X_df = pd.DataFrame(
            x_est, columns=header, index=Time[~nanLE]
        ).reset_index().rename(columns={"index": parameters["ref_date_col"]})  # Transformed, standarized, imputed data
        Z_df = pd.DataFrame(
            data=Z, columns=header, index=Time
        ).reset_index().rename(columns={"index": parameters["ref_date_col"]})  # Source data (just in cases)
    else:
        Spec = cast_spec_to_dict(ds_spec)

        X, Time, Z, header = load_data(ds, Spec, sample_start)

        # summarize data
        summarize(X.astype(float), Time, Spec)

        # Prepare data -----------------------------------------------------------
        Mx = np.nanmean(X, axis=0)
        Wx = np.nanstd(X, axis=0)
        xNaN = (X - Mx) / Wx  # Standardize series

        optNaN = {"method": 2, "k": 3}
        x_est, indNaN, nanLE = remNaNs_spline(xNaN, optNaN)  # Impute series

        x_header = list(compress(header, ~indNaN.all(axis=0)))
        X_est = x_est[:, ~indNaN.all(axis=0)]  # Drop all-NaN columns
        Spec = cast_spec_to_dict(ds_spec.loc[ds_spec["SeriesID"].isin(x_header)])

        summarize(X_est, Time[~nanLE], Spec)

        X_df = pd.DataFrame(
            X_est, columns=x_header, index=Time[~nanLE]
        ).reset_index().rename(columns={"index": parameters["ref_date_col"]})  # Transformed, standarized, imputed data

        Z_df = pd.DataFrame(
            data=Z, columns=header, index=Time
        ).reset_index().rename(columns={"index": parameters["ref_date_col"]})  # Source data (just in cases)

    return X_df, Z_df


# TODO: feature selection, stationarity-based filtering, vif fot the case when spec_options are undefined
