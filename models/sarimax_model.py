import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def train_sarima(train_data, test_data, order, seasonal_order):
    """
    Train SARIMA model
    
    Parameters:
    -----------
    train_data : pandas Series
        Training data
    test_data : pandas Series
        Test data
    order : tuple
        (p, d, q) order for ARIMA
    seasonal_order : tuple
        (P, D, Q, s) seasonal order
    
    Returns:
    --------
    model_fit : SARIMAX Results
        Fitted model
    predictions : pandas Series
        Predictions on test set
    conf_int : pandas DataFrame
        Confidence intervals for predictions
    """
    # Fit SARIMA model
    model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    model_fit = model.fit(disp=False)
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data))
    
    # Get confidence intervals
    forecast_result = model_fit.get_forecast(steps=len(test_data))
    conf_int = forecast_result.conf_int()
    
    # Set proper index
    predictions.index = test_data.index
    conf_int.index = test_data.index
    
    return model_fit, predictions, conf_int

def train_sarimax(train_data, test_data, exog_train, exog_test, order, seasonal_order):
    """
    Train SARIMAX model with exogenous variables
    
    Parameters:
    -----------
    train_data : pandas Series
        Training data
    test_data : pandas Series
        Test data
    exog_train : pandas Series or DataFrame
        Exogenous variables for training
    exog_test : pandas Series or DataFrame
        Exogenous variables for testing
    order : tuple
        (p, d, q) order for ARIMA
    seasonal_order : tuple
        (P, D, Q, s) seasonal order
    
    Returns:
    --------
    model_fit : SARIMAX Results
        Fitted model
    predictions : pandas Series
        Predictions on test set
    conf_int : pandas DataFrame
        Confidence intervals for predictions
    """
    # Fit SARIMAX model
    model = SARIMAX(
        train_data,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    model_fit = model.fit(disp=False)
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data), exog=exog_test)
    
    # Get confidence intervals
    forecast_result = model_fit.get_forecast(steps=len(test_data), exog=exog_test)
    conf_int = forecast_result.conf_int()
    
    # Set proper index
    predictions.index = test_data.index
    conf_int.index = test_data.index
    
    return model_fit, predictions, conf_int

def forecast_future(model_fit, steps, exog=None):
    """
    Generate future forecasts
    
    Parameters:
    -----------
    model_fit : SARIMAX Results
        Fitted model
    steps : int
        Number of steps to forecast
    exog : pandas Series or DataFrame or None
        Exogenous variables for forecasting
    
    Returns:
    --------
    forecast : pandas Series
        Future forecasts
    conf_int : pandas DataFrame
        Confidence intervals
    """
    # Get forecast
    if exog is not None:
        forecast_result = model_fit.get_forecast(steps=steps, exog=exog)
    else:
        forecast_result = model_fit.get_forecast(steps=steps)
    
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    return forecast, conf_int

def evaluate_model(actual, predicted):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    actual : pandas Series
        Actual values
    predicted : pandas Series
        Predicted values
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics

def grid_search_sarima(train_data, test_data, p_range, d_range, q_range, 
                        P_range, D_range, Q_range, s):
    """
    Perform grid search to find best SARIMA parameters
    
    Parameters:
    -----------
    train_data : pandas Series
        Training data
    test_data : pandas Series
        Test data
    p_range : list
        Range of p values to try
    d_range : list
        Range of d values to try
    q_range : list
        Range of q values to try
    P_range : list
        Range of P values to try
    D_range : list
        Range of D values to try
    Q_range : list
        Range of Q values to try
    s : int
        Seasonal period
    
    Returns:
    --------
    best_params : dict
        Best parameters found
    results : list
        List of all results
    """
    results = []
    best_aic = np.inf
    best_params = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            try:
                                order = (p, d, q)
                                seasonal_order = (P, D, Q, s)
                                
                                model = SARIMAX(
                                    train_data,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                
                                model_fit = model.fit(disp=False)
                                aic = model_fit.aic
                                
                                results.append({
                                    'order': order,
                                    'seasonal_order': seasonal_order,
                                    'AIC': aic
                                })
                                
                                if aic < best_aic:
                                    best_aic = aic
                                    best_params = {
                                        'order': order,
                                        'seasonal_order': seasonal_order,
                                        'AIC': aic
                                    }
                            except:
                                continue
    
    return best_params, results

def auto_arima_wrapper(train_data, seasonal=True, m=12, max_p=5, max_q=5, 
                        max_P=2, max_Q=2, max_d=2, max_D=1):
    """
    Wrapper for auto ARIMA model selection
    
    Parameters:
    -----------
    train_data : pandas Series
        Training data
    seasonal : bool
        Whether to use seasonal model
    m : int
        Seasonal period
    max_p, max_q, max_P, max_Q, max_d, max_D : int
        Maximum values for parameters
    
    Returns:
    --------
    best_model : dict
        Best model parameters
    """
    try:
        from pmdarima import auto_arima
        
        model = auto_arima(
            train_data,
            seasonal=seasonal,
            m=m,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            max_d=max_d,
            max_D=max_D,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        best_model = {
            'order': model.order,
            'seasonal_order': model.seasonal_order,
            'AIC': model.aic()
        }
        
        return best_model
    except ImportError:
        print("pmdarima not installed. Install with: pip install pmdarima")
        return None

def cross_validate_time_series(data, model_class, order, seasonal_order, 
                                n_splits=5, exog=None):
    """
    Perform time series cross-validation
    
    Parameters:
    -----------
    data : pandas Series
        Time series data
    model_class : class
        Model class (SARIMAX)
    order : tuple
        ARIMA order
    seasonal_order : tuple
        Seasonal order
    n_splits : int
        Number of splits for cross-validation
    exog : pandas Series or None
        Exogenous variables
    
    Returns:
    --------
    scores : list
        List of RMSE scores for each fold
    """
    scores = []
    split_size = len(data) // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = split_size * (i + 2)
        test_end = min(train_end + split_size, len(data))
        
        train = data[:train_end]
        test = data[train_end:test_end]
        
        if len(test) == 0:
            break
        
        if exog is not None:
            exog_train = exog[:train_end]
            exog_test = exog[train_end:test_end]
            
            model = model_class(train, exog=exog_train, order=order, 
                              seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            predictions = model_fit.forecast(steps=len(test), exog=exog_test)
        else:
            model = model_class(train, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            predictions = model_fit.forecast(steps=len(test))
        
        rmse = np.sqrt(mean_squared_error(test, predictions))
        scores.append(rmse)
    
    return scores
