import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import pmdarima as pm
from utils import _convert_to_datetime, rmse, mape

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from lineartree import LinearForestRegressor, LinearBoostRegressor
from statsmodels.tsa.api import VAR

def ml_predict(ds, ref_date_col, model, series_name, reference_date, n_periods):
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

        model.fit(X_train, y_train)

        to_write = pd.concat(
            [
                to_write,
                pd.DataFrame(
                    {"y_pred": model.predict(X_test), "y_actual": y_test},
                    index=[test_date],
                ),
            ]
        )

    return to_write, test_dates


def arima_predict(ds, ref_date_col, series_name, reference_date, n_periods):
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
                    y_test.rename(columns={series_name: "y_actual"}),
                    left_index=True,
                    right_index=True,
                ),
            ]
        )
    return to_write


def select_best_model_by_r2(models_results, y_actual):
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
        "predictions": models_results[best_model],
    }


def auto_train_evaluate(ds, ref_date_col, series_name, reference_date, n_periods):
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
    }

    models_results = {}

    for model_name, model in models.items():
        pred, T = ml_predict(
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
        }
    # Ensure all predictions align with the actuals index
    y_actual = ds.set_index(ref_date_col).loc[T].sort_index()[series_name]

    # Select the best model based on R-squared
    best_model_info = select_best_model_by_r2(
        models_results, y_actual.drop(reference_date)
    )
    best_model_info["actual"] = y_actual
    best_model_info["rmse"] = rmse(
        actual=y_actual.drop(reference_date),
        predicted=models_results[model_name]["backcast"],
    )
    best_model_info["mape"] = mape(
        actual=y_actual.drop(reference_date),
        predicted=models_results[model_name]["backcast"],
    )

    return best_model_info


def var_predict(ds, ref_date_col, series_name, reference_date, n_periods):
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
        y_test.index=pd.to_datetime(y_test.index)
        test_data = pd.merge(X_test, y_test, left_index=True, right_index=True)

        var_model = VAR(train_data)
        var_fit = var_model.fit(maxlags=1)  # You can adjust the maxlags based on the model's AIC/BIC criteria

        # train_preds = var_fit.fittedvalues

        lag_order = var_fit.k_ar
        forecast_input = train_data.values[-lag_order:]
        forecast_output = pd.DataFrame(var_fit.forecast(y=forecast_input, steps=len(test_data)), columns=test_data.columns)

        to_write = pd.concat(
            [
                to_write,
                pd.DataFrame(
                    {"y_pred": forecast_output[series_name].item(), "y_actual": y_test.item()},
                    index=[test_date],
                ),
            ]
        )

    return to_write