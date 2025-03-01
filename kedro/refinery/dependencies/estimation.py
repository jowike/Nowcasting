import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from utils_ import _convert_to_datetime, rmse, mape, cast_spec_to_dict
from retransform_prediction import retransform_
from retransform_data import retransform_data

from sklearn.metrics import r2_score
import pmdarima as pm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from lineartree import LinearForestRegressor, LinearBoostRegressor
from statsmodels.tsa.api import VAR


def ml_fit_predict(ds, ref_date_col, model, series_name, reference_date, n_periods):
    reference_date = pd.to_datetime(reference_date, format="%Y-%m-%d")

    df = _convert_to_datetime(df=ds, colnames=[ref_date_col])
    df = df.set_index(ref_date_col)

    T = pd.to_datetime(
        [
            (reference_date - relativedelta(months=i))
            for i in range(n_periods - 1, -1, -1)
        ],
        format="%Y-%m-%d",
    )

    # cascading model training
    yhat = pd.DataFrame()
    for test_date in T:
        X, y = df.drop(columns=[series_name]), df[series_name]

        train_index, test_index = y.loc[y.index < test_date].index, test_date

        X_train, y_train = X.loc[train_index], y.loc[train_index]
        X_test, y_test = X.loc[[test_index]], y.loc[[test_index]]

        assert X_test.shape[0] == y_test.shape[0] == 1

        model.fit(X_train, y_train)
        pred_ = model.predict(X_test)

        # Coefficients values
        try:
            coef_ = pd.Series(model.coef_, index=X_train.columns)
        except AttributeError:
            coef_ = pd.Series(model.feature_importances_, index=X_train.columns)
        except ValueError:
            coef_ = pd.Series(
                model.base_estimator_.fit(X_train, y_train).coef_[0],
                index=X_train.columns,
            )

        # TODO: calculate contributions for each forecast

        yhat = pd.concat(
            [
                yhat,
                pd.DataFrame(
                    {"y_pred": pred_, "y_actual": y_test},
                    index=[test_date],
                ),
            ]
        )

    return coef_, yhat, T, X_test.squeeze(axis=0), len(T)


def arima_fit_predict(ds, ref_date_col, series_name, reference_date, n_periods):
    def __arima_feed(series, h=6):
        series = series.dropna()
        arima_model = pm.auto_arima(series, stepwise=True)
        forecast = arima_model.predict(n_periods=h)
        forecast_index = pd.date_range(
            series.index[-1] + relativedelta(months=1), periods=h, freq="MS"
        )
        forecast_series = pd.Series(forecast, index=forecast_index)
        return forecast_series

    reference_date = pd.to_datetime(reference_date, format="%Y-%m-%d")

    df = _convert_to_datetime(df=ds, colnames=[ref_date_col])
    df = df.set_index(ref_date_col)

    test_dates = pd.to_datetime(
        [
            (reference_date - relativedelta(months=i))
            for i in range(n_periods - 1, -1, -1)
        ],
        format="%Y-%m-%d",
    )

    # cascading model training
    to_write = pd.DataFrame()
    for test_date in test_dates:
        # split between train and test subsets
        y_train, y_test = (
            df[[series_name]].loc[df.index < test_date],
            df[[series_name]].loc[df.index == test_date],
        )
        # AR forecast inference
        y_pred = y_train.apply(__arima_feed, h=y_test.shape[0], axis=0)

        to_write = pd.concat(
            [
                to_write,
                pd.merge(
                    y_pred.rename(columns={series_name: "y_pred"}),
                    y_test.rename(columns={series_name: "actual"}),
                    left_index=True,
                    right_index=True,
                ),
            ]
        )
    return to_write


def var_fit_predict(ds, ref_date_col, series_name, reference_date, n_periods):
    reference_date = pd.to_datetime(reference_date, format="%Y-%m-%d")

    df = _convert_to_datetime(df=ds, colnames=[ref_date_col])
    df = df.set_index(ref_date_col)

    test_dates = pd.to_datetime(
        [
            (reference_date - relativedelta(months=i))
            for i in range(n_periods - 1, -1, -1)
        ],
        format="%Y-%m-%d",
    )

    # cascading model training
    to_write = pd.DataFrame()
    for test_date in test_dates:
        X, y = df.drop(columns=[series_name]), df[series_name]

        train_index, test_index = y.loc[y.index < test_date].index, test_date

        X_train, y_train = X.loc[train_index], y.loc[train_index]
        X_test, y_test = X.loc[[test_index]], y.loc[[test_index]]

        assert X_test.shape[0] == y_test.shape[0] == 1
        train_data = pd.merge(X_train, y_train, left_index=True, right_index=True)
        y_test.index = pd.to_datetime(y_test.index)
        test_data = pd.merge(X_test, y_test, left_index=True, right_index=True)

        var_model = VAR(train_data)
        var_fit = var_model.fit(
            maxlags=1
        )  # You can adjust the maxlags based on the model's AIC/BIC criteria

        # train_preds = var_fit.fittedvalues

        lag_order = var_fit.k_ar
        forecast_input = train_data.values[-lag_order:]
        forecast_output = pd.DataFrame(
            var_fit.forecast(y=forecast_input, steps=len(test_data)),
            columns=test_data.columns,
        )

        to_write = pd.concat(
            [
                to_write,
                pd.DataFrame(
                    {
                        "y_pred": forecast_output[series_name].item(),
                        "actual": y_test.item(),
                    },
                    index=[test_date],
                ),
            ]
        )

    return to_write


def select_model_by_r2(models_results, y_actual):
    """
    Selects the best model based on R-squared score.

    Parameters:
    - models_results (dict): A dictionary where keys are model names and values are dicts with "backcast" and "forecast".
    - y_actual (pd.Series): Actual values for comparison.
    - reference_date (datetime or str): Date to exclude from evaluation (forecast point).

    Returns:
    - dict: A dictionary with the best model name, its R-squared score, and its predictions.
    """
    r2_scores = {
        model_name: r2_score(y_true=y_actual, y_pred=results["backcast"])
        for model_name, results in models_results.items()
    }

    # Find the best model
    best_model = max(r2_scores, key=r2_scores.get)

    return {
        "best_model": best_model,
        "r_squared": r2_scores[best_model],
        "coef_": models_results[best_model].pop("coef_"),
        "values": models_results[best_model].pop("values"),
        "pred_": models_results[best_model],
    }


# def cast_to_base_unit(ds, model_result, spec, series_name):
#     Spec = cast_spec_to_dict(spec.loc[spec["seriesid"] == series_name])

#     ## Retransform
#     ds = _convert_to_datetime(ds, ["ReferenceDate"])

#     dsrc = ds.set_index("ReferenceDate")

#     # def retransform_prediction(transf_series, base_series, Spec, series_name):
#     base_series = dsrc[series_name]
#     header = [series_name]

#     backcast = model_result["pred_"]["backcast"]
#     forecast = pd.Series(
#         model_result["pred_"]["forecast"],
#         index=[model_result["pred_"]["reference_date"]],
#     )

#     transf_pred = pd.concat([backcast, forecast])
#     transf_pred.index = pd.to_datetime(transf_pred.index)

#     transf_series = model_result["actual"]

#     Time = np.sort(
#         np.unique(np.concatenate((base_series.index.date, transf_pred.index.date)))
#     )
#     cutoff_date = transf_pred.index.min().date()

#     Z = base_series.reindex(Time).to_numpy().reshape(-1, 1)

#     Yhat = transf_pred.reindex(Time).to_numpy().reshape(-1, 1)
#     Y = transf_series.reindex(Time).to_numpy().reshape(-1, 1)

#     Rhat = retransform_(
#         X=Yhat, Z=Z, Time=Time, Spec=Spec, header=header, cutoff_date=cutoff_date
#     )
#     R = retransform_data(
#         X=Y, Z=Z, Time=Time, Spec=Spec, header=header, cutoff_date=cutoff_date
#     )

#     return Rhat, R, Time, cutoff_date


def cast_to_base_unit(
        ds, 
        # model_result, 
        spec, 
        series_name,
        series_values,
        dtype
        ):
    Spec = cast_spec_to_dict(spec.loc[spec["seriesid"] == series_name])
    ds = _convert_to_datetime(ds, ["ReferenceDate"])
    dsrc = ds.set_index("ReferenceDate")

    base_series = dsrc[series_name]
    header = [series_name]
    Time = np.sort(
        np.unique(np.concatenate((base_series.index.date, series_values.index.date)))
    )
    cutoff_date = series_values.index.min().date()

    Z = base_series.reindex(Time).to_numpy().reshape(-1, 1)
    Y = series_values.reindex(Time).to_numpy().reshape(-1, 1)

    ## Retransform
    if dtype == "actual":
        R = retransform_data(
            X=Y, Z=Z, Time=Time, Spec=Spec, header=header, cutoff_date=cutoff_date
        )
    elif dtype == "pred":
        R = retransform_(
            X=Y, Z=Z, Time=Time, Spec=Spec, header=header, cutoff_date=cutoff_date
        )
    else:
        raise Exception(f"ValueError: {dtype} not supported")

    return R, Time, cutoff_date



def estimate_automl(
    ds, ds_base, spec, ref_date_col, series_name, reference_date, n_periods
):
    """
    Automatically trains models, evaluates them, and selects the best one based on R-squared.

    Parameters:
    - ds (pd.DataFrame): Dataset containing features and target.
    - ref_date_col (str): Column name for reference dates.
    - series_name (str): Column name for the series to forecast.
    - reference_date (datetime or str): Date for forecasting and backcasting split.
    - n_periods (int): Number of periods to forecast.

    Returns:
    - dict: A dictionary containing the best model, its R-squared score, and predictions.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "LinearForest": LinearForestRegressor(
            base_estimator=Ridge(), random_state=42, max_features="log2"
        ),
        "LinearBoost": LinearBoostRegressor(
            base_estimator=Ridge(), random_state=42, max_features="log2"
        ),
        "RandomForestRegressor": RandomForestRegressor(),
    }

    models_results = {}
    n_est = 0
    for model_name, model in models.items():
        coef_, pred, T, values, n_iter= ml_fit_predict(
            ds=ds,
            ref_date_col=ref_date_col,
            model=model,
            series_name=series_name,
            reference_date=reference_date,
            n_periods=n_periods,
        )

        models_results[model_name] = {
            "backcast": pred["y_pred"].drop(reference_date),
            "forecast": pred["y_pred"].loc[reference_date],
            "reference_date": reference_date,
            "coef_": coef_,
            "values": values,
        }
        n_est = n_est + n_iter
    # Ensure all predictions align with the actuals index
    y_actual = ds.set_index(ref_date_col).loc[T].sort_index()[series_name]

    # Select the best model based on R-squared
    best_model_res = select_model_by_r2(models_results, y_actual.drop(reference_date))
    best_model_res["actual"] = y_actual
    best_model_res["rmse"] = rmse(
        actual=y_actual.drop(reference_date),
        predicted=models_results[model_name]["backcast"],
    )
    best_model_res["mape"] = mape(
        actual=y_actual.drop(reference_date),
        predicted=models_results[model_name]["backcast"],
    )
    best_model_res["n_est"] = n_est

    return best_model_res


def estimate_var(
    ds, ds_base, spec, ref_date_col, series_name, reference_date, n_periods
):
    var_pred = var_fit_predict(
        ds=ds,
        ref_date_col="ReferenceDate",
        series_name=series_name,
        reference_date=reference_date,
        n_periods=n_periods,
    )
    y_actual = var_pred["actual"]
    backcast = var_pred["y_pred"].drop(reference_date)

    model_info = {
        "model": "VAR",
        "r_squared": r2_score(y_true=y_actual.drop(reference_date), y_pred=backcast),
        "pred_": {
            "backcast": backcast,
            "forecast": var_pred["y_pred"].loc[reference_date],
            "reference_date": reference_date,
        },
        "actual": y_actual,
        "rmse": rmse(actual=y_actual.drop(reference_date), predicted=backcast),
        "mape": mape(actual=y_actual.drop(reference_date), predicted=backcast),
    }

    return model_info


def estimate_arima(
    ds, ds_base, spec, ref_date_col, series_name, reference_date, n_periods
):
    ar_pred = arima_fit_predict(
        ds=ds,
        ref_date_col="ReferenceDate",
        series_name=series_name,
        reference_date=reference_date,
        n_periods=n_periods,
    )
    y_actual = ar_pred["actual"]
    backcast = ar_pred["y_pred"].drop(reference_date)

    model_info = {
        "model": "ARIMA",
        "r_squared": r2_score(y_true=y_actual.drop(reference_date), y_pred=backcast),
        "pred_": {
            "backcast": backcast,
            "forecast": ar_pred["y_pred"].loc[reference_date],
            "reference_date": reference_date,
        },
        "actual": y_actual,
        "rmse": rmse(actual=y_actual.drop(reference_date), predicted=backcast),
        "mape": mape(actual=y_actual.drop(reference_date), predicted=backcast),
    }

    return model_info


def calculate_contributions(coef_, forecast, lag, values):
    var_imp = pd.merge(
        pd.DataFrame([values], index=["var"]).T,
        pd.DataFrame(coef_).rename(columns={0: "coef_"}),
        left_index=True,
        right_index=True,
    )
    var_imp["model_imp_"] = var_imp["var"] * var_imp["coef_"]

    # https://math.stackexchange.com/questions/452566/how-to-calculate-weight-of-positive-and-negative-values
    s = var_imp["model_imp_"]
    t = s - s.min() + 1
    weights = t / t.sum()

    s_weighted = weights * s
    s_weighted = s_weighted / np.abs(s_weighted.sum())

    assert np.abs(np.round(s_weighted.sum())) == 1

    var_imp["weight"] = s_weighted
    assert (np.sign(var_imp["weight"]) == np.sign(var_imp["model_imp_"])).all()
    assert np.isclose(var_imp["weight"].sum(), 1)

    var_imp["contrib"] = var_imp["weight"] * np.abs(forecast)
    assert (np.sign(var_imp["contrib"]) == np.sign(var_imp["model_imp_"])).all()
    assert np.isclose(np.abs(var_imp["contrib"].sum()), forecast)

    # (Forecast	− Lag) × Weight = Impact
    var_imp["impact"] = (forecast - lag) * var_imp["weight"]
    assert np.isclose(var_imp["impact"].sum(), (forecast - lag))

    return var_imp


def calculate_conf_bounds(pred, actual):
    # mape_series = np.abs((actual - pred) / actual).expanding(1).mean()
    # upper = (1+mape_series)*pred
    # lower = (1-mape_series)*pred
    std = (actual - pred).shift(1).expanding(2).std().fillna(0)
    low1, upp1 = pred - std, pred + std
    low2, upp2 = pred - 3 * std, pred + 3 * std
    return {"L1": low1, "U1": upp1, "L2": low2, "U2": upp2}
