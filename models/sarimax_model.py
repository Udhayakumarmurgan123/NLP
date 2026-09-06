import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def train_sarima(train_data, test_data, order, seasonal_order):
    """
    Train SARIMA model

    Parameters
    ----------
    train_data : pandas.Series
        Training data
    test_data : pandas.Series
        Test data
    order : tuple
        (p, d, q) order for ARIMA
    seasonal_order : tuple
        (P, D, Q, s) seasonal order

    Returns
    -------
    model_fit : SARIMAXResults
        Fitted model
    predictions : pandas.Series
        Predictions on test set
    conf_int : pandas.DataFrame
        Confidence intervals for predictions
    """
    model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False)

    predictions = model_fit.forecast(steps=len(test_data))
    forecast_result = model_fit.get_forecast(steps=len(test_data))
    conf_int = forecast_result.conf_int()

    predictions.index = test_data.index
    conf_int.index = test_data.index

    return model_fit, predictions, conf_int


def train_sarimax(train_data, test_data, exog_train, exog_test, order, seasonal_order):
    """
    Train SARIMAX model with exogenous variables

    Parameters
    ----------
    train_data : pandas.Series
        Training data
    test_data : pandas.Series
        Test data
    exog_train : pandas.Series or DataFrame
        Exogenous variables for training
    exog_test : pandas.Series or DataFrame
        Exogenous variables for testing
    order : tuple
        (p, d, q) order for ARIMA
    seasonal_order : tuple
        (P, D, Q, s) seasonal order

    Returns
    -------
    model_fit : SARIMAXResults
        Fitted model
    predictions : pandas.Series
        Predictions on test set
    conf_int : pandas.DataFrame
        Confidence intervals for predictions
    """
    model = SARIMAX(
        train_data,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False)

    predictions = model_fit.forecast(steps=len(test_data), exog=exog_test)
    forecast_result = model_fit.get_forecast(steps=len(test_data), exog=exog_test)
    conf_int = forecast_result.conf_int()

    predictions.index = test_data.index