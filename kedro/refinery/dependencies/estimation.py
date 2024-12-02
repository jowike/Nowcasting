import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import pmdarima as pm
from utils import _convert_to_datetime

def fit_predict(ds, ref_date_col, model, series_name, reference_date, n_periods):
    reference_date = pd.to_datetime(reference_date, format="%Y-%m-%d")

    df = _convert_to_datetime(df=ds, colnames=[ref_date_col])
    df = df.set_index(ref_date_col)

    test_dates = pd.to_datetime([
        (
            reference_date
            - relativedelta(months=i)
        )
        for i in range(n_periods - 1, -1, -1)
    ], format="%Y-%m-%d")

    # cascading model training
    to_write = pd.DataFrame()
    for test_date in test_dates:
        X, y = df.drop(columns=[series_name]), df[series_name]

        train_index, test_index = y.loc[y.index < test_date].index, test_date

        X_train, y_train = X.loc[train_index],  y.loc[train_index]
        X_test, y_test = X.loc[[test_index]],  y.loc[[test_index]]

        assert X_test.shape[0] == y_test.shape[0] == 1

        model.fit(X_train, y_train)

        to_write = pd.concat([
            to_write, 
            pd.DataFrame({
                "y_pred": model.predict(X_test),
                "y_actual": y_test
                }, index=[test_date])
                ])
        
    return to_write

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

    test_dates = pd.to_datetime([
        (
            reference_date
            - relativedelta(months=i)
        )
        for i in range(n_periods - 1, -1, -1)
    ], format="%Y-%m-%d")
    
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
        
        to_write = pd.concat([
            to_write, 
            pd.merge(
                y_pred.rename(columns={series_name: "y_pred"}), 
                y_test.rename(columns={series_name: "y_actual"}),
                left_index=True, right_index=True
                )
                ])
    return to_write
