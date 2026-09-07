import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')


def _generate_sample_data():
    """
    Generate synthetic energy consumption and temperature data.
    This helper isolates the data creation logic from `load_data`,
    making the latter shorter and easier to test.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='MS')

    # Base consumption and trend
    base = 5000
    trend = np.linspace(0, 500, len(dates))

    # Seasonal pattern (higher in summer and winter due to AC/heating)
    seasonal = (
        1500 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        + 800 * np.cos(2 * np.pi * np.arange(len(dates)) / 12)
    )

    # Random noise
    noise = np.random.normal(0, 200, len(dates))

    # Combine components for consumption
    consumption = base + trend + seasonal + noise

    # Temperature data (correlated with consumption)
    temp_base = 20
    temp_seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12 - np.pi / 2)
    temp_noise = np.random.normal(0, 3, len(dates))
    temperature = temp_base + temp_seasonal + temp_noise

    return pd.DataFrame(
        {
            "consumption": consumption,
            "temperature": temperature,
        },
        index=dates,
    )


def load_data(uploaded_file=None):
    """
    Load energy consumption data from a CSV file or generate sample data.

    Parameters
    ----------
    uploaded_file : file-like or None
        CSV file containing a 'date' column. If None, synthetic data is generated.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with a datetime index and `consumption`, `temperature` columns.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        df = _generate_sample_data()

    return df


def prepare_data(df, test_size=0.2):
    """
    Prepare data for modeling by splitting into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with consumption data.
    test_size : float, optional
        Proportion of data to use for testing (default is 0.2).

    Returns
    -------
    df : pandas.DataFrame
        Processed dataframe (sorted, with datetime index, no missing values).
    train_data : pandas.Series
        Training subset of `consumption`.
    test_data : pandas.Series
        Testing subset of `consumption`.
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by date and drop missing values
    df = df.sort_index().dropna()

    # Split data
    train_size = int(len(df) * (1 - test_size))
    train_data = df["consumption"].iloc[:train_size]
    test_data = df["consumption"].iloc[train_size:]

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
    result : dict
        Dictionary containing ADF test results and a boolean flag `Is Stationary`.
    """
    adf_test = adfuller(timeseries, autolag="AIC")
    return {
        "ADF Statistic": adf_test[0],
        "P-Value": adf_test[1],
        "Lags Used": adf_test[2],
        "Observations": adf_test[3],
        "Critical Values": adf_test[4],
        "Is Stationary": adf_test[1] < 0.05,
    }


def make_stationary(timeseries, method="difference"):
    """
    Transform a time series to become stationary.

    Parameters
    ----------
    timeseries : pandas.Series
        Original time series.
    method : {'difference', 'log', 'both'}, optional
        Transformation method (default is 'difference').

    Returns
    -------
    pandas.Series
        Transformed stationary series.
    """
    if method == "difference":
        return timeseries.diff().dropna()
    if method == "log":
        return np.log(timeseries)
    if method == "both":
        return np.log(timeseries).diff().dropna()
    raise ValueError("Method must be 'difference', 'log', or 'both'")


def detect_outliers(timeseries, threshold=3):
    """
    Detect outliers using the z-score method.

    Parameters
    ----------
    timeseries : pandas.Series
        Time series data.
    threshold : float, optional
        Z-score threshold for outlier detection (default is 3).

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
    Create lagged features for a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    column : str
        Column name to create lags for.
    lags : list of int
        List of lag periods.

    Returns
    -------
    pandas.DataFrame
        Dataframe with lag features added (rows with NaNs are dropped).
    """
    df_copy = df.copy()
    for lag in lags:
        df_copy[f"{column}_lag_{lag}"] = df_copy[column].shift(lag)
    return df_copy.dropna()


def calculate_rolling_stats(timeseries, window=12):
    """
    Calculate rolling mean and standard deviation.

    Parameters
    ----------
    timeseries : pandas.Series
        Time series data.
    window : int, optional
        Rolling window size (default is 12).

    Returns
    -------
    dict
        Dictionary containing `rolling_mean` and `rolling_std` series.
    """
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    return {"rolling_mean": rolling_mean, "rolling_std": rolling_std}