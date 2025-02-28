import sys

sys.path.append(
    "/Users/ejowik001/Desktop/Github/Nowcasting/kedro/refinery/dependencies/"
)

import pandas as pd
import numpy as np
from itertools import compress
from typing import List

from sklearn.metrics import r2_score
import os
from dateutil.relativedelta import relativedelta
from datetime import datetime

from utils_ import (
    _convert_to_datetime,
    cast_spec_to_dict,
    suggest_transformation,
)
from utils_ import test_variance as tvar
from utils_ import test_stationarity as tstat
from data_revisions import prepare_real_time_vintage_data
from ragged_edges import shift_to_fill_trailing_nans
from load_spec import load_spec
from remNaNs_spline import remNaNs_spline
from load_data import load_data
from summarize import summarize
from feature_selection import mtsfs
from estimation import (
    estimate_arima,
    estimate_automl,
    estimate_var,
    cast_to_base_unit,
    calculate_contributions,
    calculate_conf_bounds,
)
from retransform_prediction import retransform_
from plots import plot_prediction


def prepare_vintage_data(
    ds: pd.DataFrame,
    parameters: dict,
    spec_options: dict = None,
) -> List[pd.DataFrame]:
    """
    This function prepares retrospective dataset (vintage data) based on revision history
    """
    ds = _convert_to_datetime(
        ds, [parameters["ref_date_col"], parameters["pub_date_col"]]
    )

    if spec_options:
        Spec = load_spec(spec_options["filepath"])
        seriesid, SeriesName, Units, UnitsTransformed, frequency = (
            Spec["seriesid"],
            Spec["seriesname"],
            Spec["units"],
            Spec["unitstransformed"],
            Spec["frequency"],
        )
        df = ds.loc[ds[parameters["series_code_col"]].isin(seriesid)]
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
    spec = suggest_spec(ds, parameters, spec_options)
    return df_long, spec


def suggest_spec(
    ds: pd.DataFrame, parameters: dict, spec_options: dict = None
) -> pd.DataFrame:
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
            - "seriesid"
            - "SeriesName"
            - "frequency"
            - "Transformation"
            - "Units"
            - "Category"
    """
    # Define the output columns
    output_columns = [
        "seriesid",
        "seriesname",
        "frequency",
        "transformation",
        "units",
        "category",
        "region",
        "model"
    ]
    if spec_options:
        Spec = load_spec(spec_options["filepath"])
        Spec.pop("blocknames")
        df = pd.DataFrame(Spec)
        df = df.rename(columns={parameters["region_col"]: "region"})
        return df[output_columns]
    else:
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
                    parameters["region_col"],
                ]
            ]
            .drop_duplicates()
            .rename(
                columns={
                    parameters["series_code_col"]: "seriesid",
                    parameters["series_name_col"]: "seriesname",
                    parameters["unit_col"]: "units",
                    parameters["series_categ_col"]: "category",
                    parameters["region_col"]: "region",
                }
            )
        )

        # Map frequency descriptions to standardized frequency codes
        renamed_df["frequency"] = renamed_df[parameters["freq_desc_col"]].apply(
            lambda x: "m" if "Monthly" in x else "q" if "Quarterly" in x else None
        )

        # Add a default transformation column
        renamed_df["transformation"] = [
            suggest_transformation(unit) for unit in renamed_df["units"]
        ]
        renamed_df["model"] = np.nan

        # Return the final DataFrame with standardized columns
        return renamed_df[output_columns]


def harmonize_ragged_edges(
    ds,
    spec,
    parameters,
):
    to_write = pd.DataFrame()
    for freq_desc in spec["frequency"].unique():
        series_codes = spec.loc[spec["frequency"] == freq_desc]["seriesid"]
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
    spec: pd.DataFrame,
    parameters: dict,
    # spec_options: dict = None,
):
    if parameters["sample_start"]:
        sample_start = pd.to_datetime(parameters["sample_start"], format="%Y-%m-%d")
    Spec = cast_spec_to_dict(spec)

    if not np.isnan(Spec["model"]).all():
        # if spec_options:
        # Spec = load_spec(spec_options["filepath"])

        X, Time, Z, header = load_data(ds, Spec, sample_start)

        # summarize data
        summarize(X.astype(float), Time, Spec)

        # Prepare data -----------------------------------------------------------
        # Mx = np.nanmean(X, axis=0)
        # Wx = np.nanstd(X, axis=0)
        # xNaN = (X - Mx) / Wx  # Standardize series

        optNaN = {"method": 2, "k": 3}
        # x_est, _, nanLE = remNaNs_spline(xNaN, optNaN)  # Impute series
        x_est, _, nanLE = remNaNs_spline(X, optNaN)  # Impute series
        X[np.isnan(X)] = x_est[np.isnan(X)]

        summarize(X, Time[~nanLE], Spec)

        X_df = (
            pd.DataFrame(x_est, columns=header, index=Time[~nanLE])
            .reset_index()
            .rename(columns={"index": parameters["ref_date_col"]})
        )  # Transformed, standarized, imputed data
        Z_df = (
            pd.DataFrame(data=Z, columns=header, index=Time)
            .reset_index()
            .rename(columns={"index": parameters["ref_date_col"]})
        )  # Source data (just in cases)
    else:
        X, Time, Z, header = load_data(ds, Spec, sample_start)

        # summarize data
        summarize(X.astype(float), Time, Spec)

        # Prepare data -----------------------------------------------------------
        T, N = X.shape  # Gives dimensions for data input
        indNaN = np.isnan(X)  # Returns location of NaNs
        rem = (
            np.sum(indNaN, axis=0) > T * 0.8
        )  # Returns columns sum for NaN values. Marks true for rows with more than 80% NaN
        X = X[:, ~rem]
        x_header = list(compress(header, ~rem))

        # Mx = np.nanmean(X, axis=0)
        # Wx = np.nanstd(X, axis=0)
        # xNaN = (X - Mx) / Wx  # Standardize series

        optNaN = {"method": 2, "k": 3}
        # x_est, _, nanLE = remNaNs_spline(xNaN, optNaN)  # Impute series
        x_est, indNaN, nanLE = remNaNs_spline(X, optNaN)  # Impute series

        X[np.isnan(X)] = x_est[np.isnan(X)]

        indFin = np.isfinite(x_est)
        X = X[:, indFin.all(axis=0)]  # Drop all-NaN columns
        x_header = list(compress(x_header, indFin.all(axis=0)))

        Spec = cast_spec_to_dict(spec.loc[spec["seriesid"].isin(x_header)])

        summarize(X, Time[~nanLE], Spec)

        X_df = (
            pd.DataFrame(X, columns=x_header, index=Time[~nanLE])
            .reset_index()
            .rename(columns={"index": parameters["ref_date_col"]})
        )  # Transformed, standarized, imputed data

        Z_df = (
            pd.DataFrame(data=Z, columns=header, index=Time)
            .reset_index()
            .rename(columns={"index": parameters["ref_date_col"]})
        )  # Source data (just in cases)

    return X_df, Z_df


def test_variance(
    ds: pd.DataFrame,
    spec: pd.DataFrame,
    parameters: dict,
    # spec_options: dict = None,
):
    Spec = cast_spec_to_dict(spec)

    if not np.isnan(Spec["model"]).all():
        # if spec_options:
        to_write = ds.copy()
    else:
        ds = _convert_to_datetime(ds, [parameters["ref_date_col"]])

        ds = ds.set_index(parameters["ref_date_col"]).sort_index()
        X, y = ds.drop(columns=[parameters["y_code"]]), ds[[parameters["y_code"]]]

        x_est = X.drop(columns=tvar(data=X))

        to_write = pd.merge(x_est, y, left_index=True, right_index=True, how="right")
        to_write = to_write.reset_index()

    return to_write


# TODO: feature selection, stationarity-based filtering, vif fot the case when spec_options are undefined
def test_stationarity(
    ds: pd.DataFrame,
    spec: pd.DataFrame,
    parameters: dict,
    # spec_options: dict = None,
):
    Spec = cast_spec_to_dict(spec)

    if not np.isnan(Spec["model"]).all():
        # if spec_options:
        to_write = ds.copy()
    else:
        ds = _convert_to_datetime(ds, [parameters["ref_date_col"]])

        ds = ds.set_index(parameters["ref_date_col"]).sort_index()
        X, y = ds.drop(columns=[parameters["y_code"]]), ds[[parameters["y_code"]]]

        x_stat = X.drop(columns=tstat(X))

        to_write = pd.merge(x_stat, y, left_index=True, right_index=True, how="right")
        to_write = to_write.reset_index()

    return to_write


def apply_series_selection(
    ds: pd.DataFrame,
    spec: pd.DataFrame,
    parameters: dict,
    # spec_options: dict = None,
):
    ds = _convert_to_datetime(ds, [parameters["ref_date_col"]])
    ds = ds.set_index(parameters["ref_date_col"]).sort_index()

    Spec = cast_spec_to_dict(spec)

    if not np.isnan(Spec["model"]).all():
        # if spec_options:
        to_write = ds.copy()
    else:
        to_write = mtsfs(
            ds=ds, series_name=parameters["y_code"], method=parameters["mifs_method"]
        )
    return to_write.reset_index()


def estimate_ml_node(ds: pd.DataFrame, ds_base, spec, parameters: dict):
    # Example usage
    model_result = estimate_automl(
        ds=ds,
        ds_base=ds_base,
        spec=spec,
        ref_date_col=parameters["ref_date_col"],
        series_name=parameters["y_code"],
        reference_date=parameters["ref_date"],
        n_periods=parameters["backcasting_period"],
    )

    reference_date = pd.to_datetime(parameters["ref_date"]).date()
    lag_date = reference_date - relativedelta(months=1)
    # lag = model_result["pred_"]["backcast"].loc[(reference_date-relativedelta(months=1)).strftime("%Y-%m-%d")]
    pred = model_result["pred_"]["forecast"]
    coef_ = model_result["coef_"]
    values = model_result["values"]

    # Print the best model's details
    # print("============ Model Details ============")
    # print(f"Model                     : {model_result['best_model']}")
    # print(f"Reference Date            : {reference_date}")
    # print(f"Forecast                  : {pred:.4f}")
    # print(f"R-Squared (R²)            : {model_result['r_squared']:.4f}")
    # print(f"Mean Absolute Percentage Error (MAPE): {model_result['mape']:.2f}%")
    # print(f"Root Mean Square Error (RMSE) : {model_result['rmse']:.4f}")

    # print(calculate_contributions(coef_, pred, lag, values))

    formula = spec.loc[spec["seriesid"] == parameters["y_code"]][
        "transformation"
    ].item()
    unit = spec.loc[spec["seriesid"] == parameters["y_code"]]["units"].item()
    dt = model_result["pred_"]["backcast"].index

    pred = model_result["pred_"]["backcast"]
    actual = model_result["actual"].loc[dt]
    bounds = calculate_conf_bounds(pred, actual)

    # plot_prediction(
    #     dt=dt,
    #     y_pred=pred,
    #     y_actual=actual,
    #     mode="lines+markers",
    #     lower1=bounds["L1"],
    #     upper1=bounds["U1"],
    #     lower2=bounds["L2"],
    #     upper2=bounds["U2"],
    #     title=f'Series: {parameters["y_code"]}, Reference Date: {reference_date}, Unit: {unit} {formula}',
    #     plt_out_path=os.path.join(parameters["fig_out_dir"], f'{model_result["best_model"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}_pva.png')
    # )

    transf_pred = pd.concat(
        [
            model_result["pred_"]["backcast"],
            pd.Series(
                model_result["pred_"]["forecast"],
                index=[pd.to_datetime(model_result["pred_"]["reference_date"])],
            ),
        ]
    )
    transf_actual = model_result["actual"]

    # Retransform forecast
    Rhat, Time, cutoff_date = cast_to_base_unit(
        ds_base, spec, parameters["y_code"], transf_pred, dtype="pred"
    )
    R, _, _ = cast_to_base_unit(
        ds_base, spec, parameters["y_code"], transf_actual, dtype="actual"
    )

    # TBC
    header = [parameters["y_code"]]
    Rhat_df = pd.DataFrame(Rhat, columns=header, index=Time)
    R_df = pd.DataFrame(R, columns=header, index=Time)

    retr_forecast = Rhat_df.loc[reference_date].item()
    retr_actual = R_df.loc[reference_date].item()
    retr_lag = R_df.loc[lag_date].item()

    contributions = calculate_contributions(coef_, retr_forecast, retr_lag, values)

    # print("\n============ Forecast vs Actual ============")
    # print(f"Reference Date            : {reference_date}")
    # print(f"Retransformed Forecast    : {retr_forecast:,.2f}")
    # print(f"Actual Release            : {retr_actual:,.2f}")
    # print(
    #     f"Percentage Error (Level)  : {(retr_forecast - retr_actual) / retr_actual:.2%}"
    # )

    bounds_level = {}
    for key, value in bounds.items():
        data, dt_, _ = cast_to_base_unit(
            ds_base, spec, parameters["y_code"], value, dtype="pred"
        )
        tmp = pd.Series(data.reshape(1, -1)[0], index=dt_)
        bounds_level[key] = tmp.loc[dt]
    bounds_level["Predicted"] = Rhat_df.loc[dt][parameters["y_code"]]  # retransformed backcast
    retransformed_actual = R_df.loc[dt][parameters["y_code"]]  # retransformed actual

    # plot_prediction(
    #     dt=dt,
    #     y_pred=Rhat_df.loc[dt][parameters["y_code"]],
    #     y_actual=R_df.loc[dt][parameters["y_code"]],
    #     lower1=bounds_level["L1"],
    #     upper1=bounds_level["U1"],
    #     lower2=bounds_level["L2"],
    #     upper2=bounds_level["U2"],
    #     mode="lines+markers",
    #     title=f'Series: {parameters["y_code"]}, Reference Date: {reference_date}, Unit: {unit}',
    #     plt_out_path=os.path.join(parameters["fig_out_dir"], f'{model_result["best_model"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}_bpva.png')
    # )

    # Save everything to an Excel file with multiple sheets
    excel_file = os.path.join(parameters["model_output_directory"], parameters["ml_report_filename"])

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        # Save model details as a dataframe
        sheet1 = (
            pd.DataFrame.from_dict(
                {
                    "Model Name": model_result["best_model"],
                    "Series Code": parameters["y_code"],
                    "Reference Date": reference_date,
                    "R-Squared": model_result['r_squared'],
                    "MAPE": (retr_forecast - retr_actual) / retr_actual,
                    "RMSE": f"{model_result['rmse']:.4f}",
                },
                orient="index",
                columns=["Value"],
            )
            .reset_index()
            .rename(columns={"index": "Banner"})
        )
        sheet1.to_excel(writer, sheet_name="Model Details", index=False)
        # Save contributions
        contributions.to_excel(writer, sheet_name="Contributions")

        # Save confidence bounds
        pd.DataFrame(bounds_level).to_excel(writer, sheet_name="Confidence Bounds")

        # Save forecast and actual values
        R_df = R_df.rename(columns={parameters["y_code"]: "Actual"})
        Rhat_df = Rhat_df.rename(columns={parameters["y_code"]: "Predicted"})
        sheet3 = (
            pd.merge(Rhat_df, R_df, how="outer", left_index=True, right_index=True)
            .reset_index()
            .rename(columns={"index": "Reference Date"})
        )
        sheet3.to_excel(writer, sheet_name="Forecast vs Actual", index=False)

    return f"Results saved to {excel_file}"


def estimate_arima_node(ds: pd.DataFrame, ds_base, spec, parameters: dict):
    model_result = estimate_arima(
        ds=ds,
        ds_base=ds_base,
        spec=spec,
        ref_date_col=parameters["ref_date_col"],
        series_name=parameters["y_code"],
        reference_date=parameters["ref_date"],
        n_periods=parameters["backcasting_period"],
    )

    reference_date = pd.to_datetime(parameters["ref_date"]).date()

    # Print the best model's details
    # print("============ Model Details ============")
    # print(f"Model                     : {model_result['model']}")
    # print(f"Reference Date            : {reference_date}")
    # print(f"Forecast                  : {model_result['pred_']['forecast']:.4f}")
    # print(f"R-Squared (R²)            : {model_result['r_squared']:.4f}")
    # print(f"Mean Absolute Percentage Error (MAPE): {model_result['mape']:.2f}%")
    # print(f"Root Mean Square Error (RMSE) : {model_result['rmse']:.4f}")

    formula = spec.loc[spec["seriesid"] == parameters["y_code"]][
        "transformation"
    ].item()
    unit = spec.loc[spec["seriesid"] == parameters["y_code"]]["units"].item()

    dt = model_result["pred_"]["backcast"].index
    pred = model_result["pred_"]["backcast"]
    actual = model_result["actual"].loc[dt]

    bounds = calculate_conf_bounds(pred, actual)

    # plot_prediction(
    #     dt=dt,
    #     y_pred=pred,
    #     y_actual=actual,
    #     mode="lines+markers",
    #     title=f'Series: {parameters["y_code"]}, Reference Date: {reference_date}, Unit: {unit} {formula}',
    #     plt_out_path=os.path.join(parameters["fig_out_dir"], f"{model_result['model']}_Predicted_vs_Actual.png")
    # )

    transf_pred = pd.concat(
        [
            model_result["pred_"]["backcast"],
            pd.Series(
                model_result["pred_"]["forecast"],
                index=[pd.to_datetime(model_result["pred_"]["reference_date"])],
            ),
        ]
    )
    transf_actual = model_result["actual"]

    # Retransform forecast
    Rhat, Time, cutoff_date = cast_to_base_unit(
        ds_base, spec, parameters["y_code"], transf_pred, dtype="pred"
    )
    R, _, _ = cast_to_base_unit(
        ds_base, spec, parameters["y_code"], transf_actual, dtype="actual"
    )

    # TBC
    header = [parameters["y_code"]]
    Rhat_df = pd.DataFrame(Rhat, columns=header, index=Time)
    R_df = pd.DataFrame(R, columns=header, index=Time)

    retr_forecast = Rhat_df.loc[reference_date].item()
    retr_actual = R_df.loc[reference_date].item()

    reference_date = pd.to_datetime(parameters["ref_date"]).date()

    # print("\n============ Forecast vs Actual ============")
    # print(f"Reference Date            : {reference_date}")
    # print(f"Retransformed Forecast    : {retr_forecast:,.2f}")
    # print(f"Actual Release            : {retr_actual:,.2f}")
    # print(
    #     f"Percentage Error (Level)  : {(retr_forecast - retr_actual) / retr_actual:.2%}"
    # )

    # calculate confidence bounds
    bounds_level = {}
    for key, value in bounds.items():
        data, dt_, _ = cast_to_base_unit(
            ds_base, spec, parameters["y_code"], value, dtype="pred"
        )
        tmp = pd.Series(data.reshape(1, -1)[0], index=dt_)
        bounds_level[key] = tmp.loc[dt]
    bounds_level["Predicted"] = Rhat_df.loc[dt][parameters["y_code"]]  # retransformed backcast
    retransformed_actual = R_df.loc[dt][parameters["y_code"]]  # retransformed actual

    # plot_prediction(
    #     dt=dt,
    #     y_pred=Rhat_df.loc[dt][parameters["y_code"]],
    #     y_actual=R_df.loc[dt][parameters["y_code"]],
    #     mode="lines+markers",
    #     title=f'Series: {parameters["y_code"]}, Reference Date: {reference_date}, Unit: {unit}',
    #     plt_out_path=os.path.join(parameters["fig_out_dir"], f'{model_result["model"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}_bpva.png')
    # )

    # Save everything to an Excel file with multiple sheets
    excel_file = os.path.join(parameters["model_output_directory"], parameters["ar_report_filename"])

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        # Save model details as a dataframe
        sheet1 = (
            pd.DataFrame.from_dict(
                {
                    "Model Name": model_result["model"],
                    "Series Code": parameters["y_code"],
                    "Reference Date": reference_date,
                    "R-Squared": model_result['r_squared'],
                    "MAPE": ((bounds_level["Predicted"] - retransformed_actual) / retransformed_actual).mean(),
                    "RMSE": f"{model_result['rmse']:.4f}",
                },
                orient="index",
                columns=["Value"],
            )
            .reset_index()
            .rename(columns={"index": "Banner"})
        )
        sheet1.to_excel(writer, sheet_name="Model Details", index=False)

        # Save forecast and actual values
        R_df = R_df.rename(columns={parameters["y_code"]: "Actual"})
        Rhat_df = Rhat_df.rename(columns={parameters["y_code"]: "Predicted"})
        sheet2 = (
            pd.merge(Rhat_df, R_df, how="outer", left_index=True, right_index=True)
            .reset_index()
            .rename(columns={"index": "Reference Date"})
        )
        sheet2.to_excel(writer, sheet_name="Forecast vs Actual", index=False)

        # Save confidence bounds
        pd.DataFrame(bounds_level).to_excel(writer, sheet_name="Confidence Bounds")


    return f"Results saved to {excel_file}"


def estimate_var_node(ds: pd.DataFrame, ds_base, spec, parameters: dict):
    model_result = estimate_var(
        ds=ds,
        ds_base=ds_base,
        spec=spec,
        ref_date_col=parameters["ref_date_col"],
        series_name=parameters["y_code"],
        reference_date=parameters["ref_date"],
        n_periods=parameters["backcasting_period"],
    )

    reference_date = pd.to_datetime(parameters["ref_date"]).date()

    # print("============ Model Details ============")
    # print(f"Model                     : {model_result['model']}")
    # print(f"Reference Date            : {reference_date}")
    # print(f"Forecast                  : {model_result['pred_']['forecast']:.4f}")
    # print(f"R-Squared (R²)            : {model_result['r_squared']:.4f}")
    # print(f"Mean Absolute Percentage Error (MAPE): {model_result['mape']:.2f}%")
    # print(f"Root Mean Square Error (RMSE) : {model_result['rmse']:.4f}")

    formula = spec.loc[spec["seriesid"] == parameters["y_code"]][
        "transformation"
    ].item()
    unit = spec.loc[spec["seriesid"] == parameters["y_code"]]["units"].item()

    dt = model_result["pred_"]["backcast"].index
    pred = model_result["pred_"]["backcast"]
    actual = model_result["actual"].loc[dt]

    bounds = calculate_conf_bounds(pred, actual)

    # plot_prediction(
    #     dt=dt,
    #     y_pred=pred,
    #     y_actual=actual,
    #     mode="lines+markers",
    #     title=f'Series: {parameters["y_code"]}, Reference Date: {reference_date}, Unit: {unit} {formula}',
    #     plt_out_path=os.path.join(parameters["fig_out_dir"], f'{model_result["model"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}_pva.png')
    # )

    transf_pred = pd.concat(
        [
            model_result["pred_"]["backcast"],
            pd.Series(
                model_result["pred_"]["forecast"],
                index=[pd.to_datetime(model_result["pred_"]["reference_date"])],
            ),
        ]
    )
    transf_actual = model_result["actual"]

    # Retransform forecast
    Rhat, Time, cutoff_date = cast_to_base_unit(
        ds_base, spec, parameters["y_code"], transf_pred, dtype="pred"
    )
    R, _, _ = cast_to_base_unit(
        ds_base, spec, parameters["y_code"], transf_actual, dtype="actual"
    )

    # TBC
    header = [parameters["y_code"]]
    Rhat_df = pd.DataFrame(Rhat, columns=header, index=Time)
    R_df = pd.DataFrame(R, columns=header, index=Time)

    retr_forecast = Rhat_df.loc[reference_date].item()
    retr_actual = R_df.loc[reference_date].item()

    # print("\n============ Forecast vs Actual ============")
    # print(f"Reference Date            : {reference_date}")
    # print(f"Retransformed Forecast    : {retr_forecast:,.2f}")
    # print(f"Actual Release            : {retr_actual:,.2f}")
    # print(
    #     f"Percentage Error          : {(retr_forecast - retr_actual) / retr_actual:.2%}"
    # )

    # calculate confidence bounds
    bounds_level = {}
    for key, value in bounds.items():
        data, dt_, _ = cast_to_base_unit(
            ds_base, spec, parameters["y_code"], value, dtype="pred"
        )
        tmp = pd.Series(data.reshape(1, -1)[0], index=dt_)
        bounds_level[key] = tmp.loc[dt]
    bounds_level["Predicted"] = Rhat_df.loc[dt][parameters["y_code"]]  # retransformed backcast
    retransformed_actual = R_df.loc[dt][parameters["y_code"]]  # retransformed actual

    # plot_prediction(
    #     dt=dt,
    #     y_pred=Rhat_df.loc[dt][parameters["y_code"]],
    #     y_actual=R_df.loc[dt][parameters["y_code"]],
    #     mode="lines+markers",
    #     title=f'Series: {parameters["y_code"]}, Reference Date: {reference_date}, Unit: {unit}',
    #     plt_out_path=os.path.join(parameters["fig_out_dir"], f'{model_result["model"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}_bpva.png')
    # )

    # Save everything to an Excel file with multiple sheets
    excel_file = os.path.join(parameters["model_output_directory"], parameters["var_report_filename"])

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        # Save model details as a dataframe
        sheet1 = (
            pd.DataFrame.from_dict(
                {
                    "Model Name": model_result["model"],
                    "Series Code": parameters["y_code"],
                    "Reference Date": reference_date,
                    "R-Squared": model_result['r_squared'],
                    "MAPE": ((bounds_level["Predicted"] - retransformed_actual) / retransformed_actual).mean(),
                    "RMSE": f"{model_result['rmse']:.4f}",
                },
                orient="index",
                columns=["Value"],
            )
            .reset_index()
            .rename(columns={"index": "Banner"})
        )
        sheet1.to_excel(writer, sheet_name="Model Details", index=False)

        # Save forecast and actual values
        R_df = R_df.rename(columns={parameters["y_code"]: "Actual"})
        Rhat_df = Rhat_df.rename(columns={parameters["y_code"]: "Predicted"})
        sheet2 = (
            pd.merge(Rhat_df, R_df, how="outer", left_index=True, right_index=True)
            .reset_index()
            .rename(columns={"index": "Reference Date"})
        )
        sheet2.to_excel(writer, sheet_name="Forecast vs Actual", index=False)

        # Save confidence bounds
        pd.DataFrame(bounds_level).to_excel(writer, sheet_name="Confidence Bounds")

    return f"Results saved to {excel_file}"

def collect_results(variable, vintagedata, ts, parameters, s1, s2, s3):

    vintagedata[parameters["ref_date_col"]] = pd.to_datetime(vintagedata[parameters["ref_date_col"]])

    ar_xl = pd.ExcelFile(os.path.join(parameters["model_output_directory"], parameters["ar_report_filename"]))
    ar_sheets = {}
    for sheet_name in ar_xl.sheet_names:
        ar_sheets[sheet_name] = ar_xl.parse(sheet_name) 

    var_xl = pd.ExcelFile(os.path.join(parameters["model_output_directory"], parameters["var_report_filename"]))
    var_sheets = {}
    for sheet_name in var_xl.sheet_names:
        var_sheets[sheet_name] = var_xl.parse(sheet_name)

    ml_xl = pd.ExcelFile(os.path.join(parameters["model_output_directory"], parameters["ml_report_filename"]))
    ml_sheets = {}
    for sheet_name in ml_xl.sheet_names:
        ml_sheets[sheet_name] = ml_xl.parse(sheet_name) 

    to_write = {}

    # Cards
    contents = []

    # ARIMA
    reference_date = ar_sheets["Model Details"].loc[ar_sheets["Model Details"]["Banner"] == "Reference Date"]["Value"].item()
    forecast = ar_sheets["Forecast vs Actual"].loc[ar_sheets["Forecast vs Actual"]["Reference Date"] == reference_date]["Predicted"].item()
    lag = ar_sheets["Forecast vs Actual"].loc[ar_sheets["Forecast vs Actual"]["Reference Date"] == reference_date - relativedelta(months=1)]["Actual"].item()

    width = (ar_sheets['Confidence Bounds']["Predicted"] - ar_sheets['Confidence Bounds']["L1"]).tail(12).mean()

    contents.append({
        "Card": ar_sheets["Model Details"].loc[ar_sheets["Model Details"]["Banner"] == "Model Name"]["Value"].item(),
        "Value": forecast,
        "Since Last Month": (forecast - lag) / lag,
        "Prediction Range": width,
    })

    # VAR
    reference_date = var_sheets["Model Details"].loc[var_sheets["Model Details"]["Banner"] == "Reference Date"]["Value"].item()
    forecast = var_sheets["Forecast vs Actual"].loc[var_sheets["Forecast vs Actual"]["Reference Date"] == reference_date]["Predicted"].item()
    lag = var_sheets["Forecast vs Actual"].loc[var_sheets["Forecast vs Actual"]["Reference Date"] == reference_date - relativedelta(months=1)]["Actual"].item()

    width = (var_sheets['Confidence Bounds']["Predicted"] - var_sheets['Confidence Bounds']["L1"]).tail(12).mean()

    contents.append({
        "Card": var_sheets["Model Details"].loc[var_sheets["Model Details"]["Banner"] == "Model Name"]["Value"].item(),
        "Value": forecast,
        "Since Last Month": (forecast - lag) / lag,
        "Prediction Range": width,
    })

    # ML
    reference_date = ml_sheets["Model Details"].loc[ml_sheets["Model Details"]["Banner"] == "Reference Date"]["Value"].item()
    forecast = ml_sheets["Forecast vs Actual"].loc[ml_sheets["Forecast vs Actual"]["Reference Date"] == reference_date]["Predicted"].item()
    lag = ml_sheets["Forecast vs Actual"].loc[ml_sheets["Forecast vs Actual"]["Reference Date"] == reference_date - relativedelta(months=1)]["Actual"].item()

    width = (ml_sheets['Confidence Bounds']["Predicted"] - ml_sheets['Confidence Bounds']["L1"]).tail(12).mean()

    contents.append({
        "Card": "Confidence Interval",
        "Value": width,
        "Since Last Month": (forecast - lag) / lag,
        "Prediction Range": ""
    })

    to_write["Cards"] = pd.DataFrame(contents)


    # Nowcast Browser – Header
    reference_date = ml_sheets["Model Details"].loc[ml_sheets["Model Details"]["Banner"] == "Reference Date"]["Value"].item()
    forecast = ml_sheets["Forecast vs Actual"].loc[ml_sheets["Forecast vs Actual"]["Reference Date"] == reference_date]["Predicted"].item()
    lag = ml_sheets["Forecast vs Actual"].loc[ml_sheets["Forecast vs Actual"]["Reference Date"] == reference_date - relativedelta(months=1)]["Actual"].item()
    series_code = ml_sheets["Model Details"].loc[ml_sheets["Model Details"]["Banner"] == "Series Code"]["Value"].item()
    LastUpdatedOnSource = pd.to_datetime(vintagedata["LastUpdatedOnSource"]).max()

    contents = {
        "Value": forecast,
        "Since Last Month": (forecast - lag) / lag,
        "Series Name": variable.loc[variable["seriesid"] == series_code]["seriesname"].item(),
        "Series Code": series_code,
        "Reference Period": f"{reference_date.strftime('%b')} 1 - {reference_date.strftime('%b')} {pd.Period(reference_date.strftime('%Y-%m')).days_in_month}",
        "Region": variable.loc[variable["seriesid"] == series_code]["region"].item(),
        "Unit": variable.loc[variable["seriesid"] == series_code]["units"].item(),
        "Last Run Watermark": datetime.now().strftime('%d/%m/%Y %H:%M'),
        "Data as of": LastUpdatedOnSource.strftime('%d/%m/%Y %H:%M'),
        }

    to_write["Nowcast Browser – Header"] = pd.DataFrame.from_dict(contents, orient="index").reset_index().rename(columns={"index": "Banner", 0: "Value"})


    # Nowcast Browser – Base

    reference_date = ml_sheets["Model Details"].loc[ml_sheets["Model Details"]["Banner"] == "Reference Date"]["Value"].item()

    traces = ml_sheets["Forecast vs Actual"].loc[
        (ml_sheets["Forecast vs Actual"]["Reference Date"] >= (reference_date - relativedelta(months=18))) &\
        (ml_sheets["Forecast vs Actual"]["Reference Date"] <= reference_date)
            ]
    traces.loc[(traces["Reference Date"] == reference_date), "Actual"] = None
    to_write["Nowcast Browser – Base"] = traces


    # Local Explanation
    reference_date = ml_sheets["Model Details"].loc[ml_sheets["Model Details"]["Banner"] == "Reference Date"]["Value"].item()
    lag_date = reference_date - relativedelta(months=1)
    forecast = ml_sheets["Forecast vs Actual"].loc[ml_sheets["Forecast vs Actual"]["Reference Date"] == reference_date]["Predicted"].item()
    lag = ml_sheets["Forecast vs Actual"].loc[ml_sheets["Forecast vs Actual"]["Reference Date"] == reference_date - relativedelta(months=1)]["Actual"].item()

    impact_assessment = ml_sheets["Contributions"][["Unnamed: 0", "impact"]].rename(columns={"Unnamed: 0": "Series ID", "impact": "Impact"})

    # values = ts.loc[ts[parameters["ref_date_col"]] == lag_date][list(impact_assessment["Series ID"])]
    # values.index = ["Actual"]

    # impact_assessment = impact_assessment.merge(values.T.reset_index().rename(columns={"index": "Series ID"}), how="left", on="Series ID")
    # impact_assessment["Impact"] = impact_assessment["Impact"]
    # impact_assessment["Actual"] = impact_assessment["Actual"].map(lambda x: "{:,.2f}".format(x))
    # actuals = vintagedata.loc[
    #     (vintagedata[parameters["series_code_col"]].isin(list(impact_assessment["Series ID"]))) &\
    #     (vintagedata[parameters["ref_date_col"]] == lag_date)
    #     ][[parameters["series_code_col"], "Description", parameters["ref_date_col"], "LastUpdatedOnSource"]].groupby(
    #         [parameters["series_code_col"], "Description", parameters["ref_date_col"]]
    #         ).min().reset_index().rename(
    #         columns={parameters["series_code_col"]: "Series ID", "Description": "Data Series", "LastUpdatedOnSource": "Release Date"}
    #         )

    tmp = vintagedata.loc[
        (vintagedata[parameters["series_code_col"]].isin(list(impact_assessment["Series ID"]))) &\
        (vintagedata[parameters["ref_date_col"]] <= reference_date)
        ]

    actuals = tmp.merge(
        tmp.groupby([parameters["series_code_col"]]).agg({parameters["ref_date_col"]: "max"}).reset_index(), on=[parameters["series_code_col"], parameters["ref_date_col"]]
        )[[parameters["series_code_col"], "Description", parameters["ref_date_col"], "LastUpdatedOnSource"]].groupby(
            [parameters["series_code_col"], "Description", parameters["ref_date_col"]]
            ).min().reset_index().rename(
            columns={parameters["series_code_col"]: "Series ID", "Description": "Data Series", "LastUpdatedOnSource": "Release Date"}
            )


    impact_assessment = impact_assessment.merge(
        actuals,
        on="Series ID",
        how="left"
    )
    # impact_assessment["Release Date"] = pd.to_datetime(impact_assessment["Release Date"]).dt.strftime('%b-%d')

    to_write["Local Explanation"] = impact_assessment[["Release Date", "Series ID", "Data Series", "Impact"]]


    #  Global Explanation
    series_code = ml_sheets["Model Details"].loc[ml_sheets["Model Details"]["Banner"] == "Series Code"]["Value"].item()
    reference_date = ml_sheets["Model Details"].loc[ml_sheets["Model Details"]["Banner"] == "Reference Date"]["Value"].item()
    df = ts.set_index(parameters["ref_date_col"])[list(impact_assessment["Series ID"])+[series_code]]
    df.loc[reference_date, series_code] = None

    # Melt wide DataFrame to long format
    df_long = df.sort_index().loc[
        reference_date-relativedelta(months=12):reference_date
        ].reset_index().melt(
            id_vars=[parameters["ref_date_col"]], var_name='Variable Code', value_name='Variable Value'
            )

    to_write["Global Explanation"] = df_long


    # Model Assessment
    df = ml_sheets["Model Details"].loc[ml_sheets["Model Details"]["Banner"].isin(["R-Squared", "MAPE"])].rename(columns={"Banner": "Measure"})
    df["Measure"] = df["Measure"].replace({
        "R-Squared": "Adjusted R-Squared",
        "MAPE": "Average Error Rate"
    })


    to_write["Model Assessment"] = df

    excel_file = os.path.join(parameters["reporting_directory"], parameters["out_report_filename"])

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        for sheet_name, contents in to_write.items():
            contents.to_excel(writer, sheet_name=sheet_name, index=False)
    return f"Results saved to {excel_file}"