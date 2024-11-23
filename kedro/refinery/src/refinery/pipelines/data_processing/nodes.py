import sys

sys.path.append("/Users/ejowik001/Desktop/Github/Nowcasting/kedro/refinery/src/scripts")

import pandas as pd
import numpy as np
from typing import List, Literal, Tuple
from datetime import datetime

from scripts.utils import _convert_to_datetime, prepare_auto_spec
from scripts.data_revisions import prepare_real_time_vintage_data
from scripts.ragged_edges import shift_to_fill_trailing_nans
from scripts.load_spec import load_spec
from scripts.remNaNs_spline import remNaNs_spline
from scripts.load_data import load_data
from scripts.summarize import summarize


def prepare_vintage_data(
    ds: pd.DataFrame,
    parameters: dict,
    dataprep_options: dict,
    spec_options: dict = None,
) -> pd.DataFrame:
    """
    This function prepares retrospective dataset (vintage data) based on revision history
    """
    ds = _convert_to_datetime(
        ds, [dataprep_options["ref_date_col"], dataprep_options["pub_date_col"]]
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
        df = ds.loc[ds[dataprep_options["series_code_col"]].isin(SeriesID)]
    else:
        df = ds.copy()

    cols = [
        dataprep_options["series_code_col"],
        dataprep_options["ref_date_col"],
        dataprep_options["pub_date_col"],
        dataprep_options["series_val_col"],
    ]
    df = df[cols].drop_duplicates()

    if parameters["vintage_name"] == "real-time":
        df_long = prepare_real_time_vintage_data(
            ds=df,
            y_code=parameters["y_code"],
            ref_date=parameters["ref_date"],
            series_code_col=dataprep_options["series_code_col"],
            ref_date_col=dataprep_options["ref_date_col"],
            pub_date_col=dataprep_options["pub_date_col"],
            series_val_col=dataprep_options["series_val_col"],
        )
    # TODO: preliminary, current-vintage (pseudo-real-time)
    return df_long


def prepare_freq_details(ds: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    This function prepares retrospective dataset (vintage data) based on revision history
    """
    df = ds.copy()
    df = (
        df[[parameters["series_code_col"], parameters["freq_desc_col"]]]
        .drop_duplicates()
        .rename(columns={parameters["series_code_col"]: "SeriesID"})
    )
    df["Frequency"] = df[parameters["freq_desc_col"]].apply(
        lambda x: "m" if "Monthly" in x else "q" if "Quarterly" in x else None
    )

    return df[["SeriesID", "Frequency"]]


def harmonize_ragged_edges(
    ds,
    freq_details,
    parameters,
):
    to_write = pd.DataFrame()
    for freq_desc in freq_details["Frequency"].unique():
        series_codes = freq_details.loc[freq_details["Frequency"] == freq_desc][
            "SeriesID"
        ]
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
    freq_details: pd.DataFrame,
    parameters: dict,
    dataprep_options: dict,
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
        )  # Transformed, standarized, imputed data
        Z_df = pd.DataFrame(
            data=Z, columns=header, index=Time
        )  # Source data (just in cases)
    else:
        spec = freq_details.copy()
        spec["Transformation"] = dataprep_options["default_transf_code"]
        Spec = prepare_auto_spec(spec)
        X, Time, Z, header = load_data(ds, Spec, sample_start)

        print(Spec)
        # TODO

    return X_df.reset_index(), Z_df.reset_index()


# TODO: feature selection, stationarity-based filtering, vif fot the case when spec_options are undefined
