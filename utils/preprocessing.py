import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

def load_data(uploaded_file=None):
    """
    Load energy consumption data from file or generate sample data
    
    Parameters:
    -----------
    uploaded_file : file or None
        Uploaded CSV file from Streamlit
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with datetime index and consumption, temperature columns
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        # Generate sample data (5 years of monthly data)
        np.random.seed(42)
        dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='MS')
        
        # Create realistic energy consumption pattern
        # Base consumption
        base = 5000
        
        # Trend (slight increase over time)
        trend = np.linspace(0, 500, len(dates))
        
        # Seasonal pattern (higher in summer and winter due to AC/heating)
        seasonal = 1500 * np.sin(2 * np.pi * np.arange(len(dates)) / 12) + \
                   800 * np.cos(2 * np.pi * np.arange(len(dates)) / 12)
        
        # Random noise
        noise = np.random.normal(0, 200, len(dates))
        
        # Combine components
        consumption = base + trend + seasonal + noise
        
        # Temperature data (correlated with consumption)
        temp_base = 20
        temp_seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12 - np.pi/2)
        temp_noise = np.random.normal(0, 3, len(dates))
        temperature = temp_base + temp_seasonal + temp_noise
        
        df = pd.DataFrame({
            'consumption': consumption,
            'temperature': temperature
        }, index=dates)
    
    return df

def prepare_data(df, test_size=0.2):
    """
    Prepare data for modeling by splitting into train and test sets
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with consumption data
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    df : pandas DataFrame
        Processed dataframe
    train_data : pandas Series
        Training data for consumption
    test_data : pandas Series
        Test data for consumption
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    # Remove any missing values
    df = df.dropna()
    
    # Split data
    train_size = int(len(df) * (1 - test_size))
    train_data = df['consumption'].iloc[:train_size]
    test_data = df['consumption'].iloc[train_size:]