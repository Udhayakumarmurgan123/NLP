import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')


def _load_uploaded_file(uploaded_file):
    """
    Load data from an uploaded CSV file.

    Parameters
    ----------
    uploaded_file : file-like object
        The CSV file to read.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a datetime index and the original columns.
    """
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def _generate_sample_data():
    """
    Generate synthetic energy consumption and temperature data
    (5 years of monthly observations).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by month start dates containing
        'consumption' and 'temperature' columns.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='MS')

    # Base consumption and trend
    base = 5000
    trend = np.linspace(0, 500, len(dates))

    # Seasonal pattern (higher in summer and winter)
    seasonal = (
        1500 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        + 800 * np.cos(2 * np.pi * np.arange(len(dates)) / 12)
    )

    # Random noise
    noise = np.random.normal(0, 200, len(dates))

    consumption = base + trend + seasonal + noise

    # Temperature (correlated with consumption)
    temp_base = 20
    temp_seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12 - np.pi / 2)
    temp_noise = np.random.normal(0, 3, len(dates))
    temperature = temp_base + temp_seasonal + temp_noise

    return pd.DataFrame(
        {
            'consumption': consumption,
            'temperature': temperature,
        },
        index=dates,
    )


def load_data(uploaded_file=None):
    """
    Load energy consumption data from a file or generate sample data.

    Parameters
    ----------
    uploaded_file : file-like object or None
        Uploaded CSV file; if None, synthetic data is generated.

    Returns
    -------
    pandas.DataFrame
        DataFrame with datetime index and 'consumption', 'temperature' columns.
    """
    if uploaded_file is not None:
        return _load_uploaded_file(uploaded_file)
    else:
        return _generate_sample_data()


def prepare_data(df, test_size=0.2):
    """
    Prepare data for modeling by splitting into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with consumption data.
    test_size : float
        Proportion of data to use for testing.

    Returns
    -------
    df : pandas.DataFrame
        Processed dataframe (sorted, with datetime index, no missing values).
    train_data : pandas.Series
        Training data for consumption.
    test_data : pandas.Series
        Test data for consumption.
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by date and drop missing values
    df = df.sort_index().dropna()

    # Split data
    train_size = int(len(df) * (1 - test_size))
    train_data = df['consumption'].iloc[:train_size]
    test_data = df['consumption'].iloc[train_size:]

    return df, train_data, test_data


def check_stationarity(timeseries):
    """
    Perform Augmented Dickey-Fuller test to check stationarity.

    Parameters
    ----------
    timeseries : pandas.Series
        Time series data.

    Returns
    -------
    dict
        Dictionary containing ADF test results and a boolean flag.
    """
    adf_test = adfuller(timeseries, autolag='AIC')
    return {
        'ADF Statistic': adf_test[0],
        'P-Value': adf_test[1],
        'Lags Used': adf_test[2],
        'Observations': adf_test[3],
        'Critical Values': adf_test[4],
        'Is Stationary': adf_test[1] < 0.05,
    }


def make_stationary(timeseries, method='difference'):
    """
    Make time series stationary using differencing or log transformation.

    Parameters
    ----------
    timeseries : pandas.Series
        Time series data.
    method : str
        Method to use ('difference', 'log', 'both').

    Returns
    -------
    pandas.Series
        Transformed stationary series.

    Raises
    ------
    ValueError
        If an unsupported method is provided.
    """
    if method == 'difference':
        return timeseries.diff().dropna()
    if method == 'log':
        return np.log(timeseries)
    if method == 'both':
        return np.log(timeseries).diff().dropna()
    raise ValueError("Method must be 'difference', 'log', or 'both'")


def detect_outliers(timeseries, threshold=3):
    """
    Detect outliers using the z-score method.

    Parameters
    ----------
    timeseries : pandas.Series
        Time series data.
    threshold : float
        Z-score threshold for outlier detection.

    Returns
    -------
    pandas.Series
        Boolean series indicating outliers.
    """
    mean = timeseries.mean()
    std = timeseries.std()
    z_scores = np.abs((timeseries - mean) / std)
    return z_scores > threshold


def create_lag_features(df, column, lags):
    """
    Create lagged features for a time series column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    column : str
        Column name to create lags for.
    lags : list[int]
        List of lag offsets.

    Returns
    -------
    pandas.DataFrame
        Dataframe with lag features added (rows with NaNs are dropped).
    """
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{column}_lag_{lag}'] = df_copy[column].shift(lag)
    return df_copy.dropna()


def calculate_rolling_stats(timeseries, window=12):
    """
    Calculate rolling mean and standard deviation.

    Parameters
    ----------
    timeseries : pandas.Series
        Time series data.
    window : int
        Rolling window size.

    Returns
    -------
    dict
        Dictionary containing rolling mean and standard deviation series.
    """
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    return {
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
    }