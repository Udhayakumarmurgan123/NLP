import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------
# Parameter objects to replace long argument lists
# ----------------------------------------------------------------------
@dataclass
class TimeSeriesData:
    """
    Container for all time‑series inputs required by a SARIMA/SARIMAX model.

    Attributes
    ----------
    train : pd.Series
        The series used for fitting the model.
    test : pd.Series
        The series used for evaluating / forecasting.
    exog_train : pd.Series or pd.DataFrame, optional
        Exogenous regressors aligned with ``train``.  Used only for SARIMAX.
    exog_test : pd.Series or pd.DataFrame, optional
        Exogenous regressors aligned with ``test``.  Used only for SARIMAX.
    """
    train: pd.Series
    test: pd.Series
    exog_train: Optional[Union[pd.Series, pd.DataFrame]] = None
    exog_test: Optional[Union[pd.Series, pd.DataFrame]] = None


@dataclass
class SarimaxConfig:
    """
    Hyper‑parameter bundle for SARIMA / SARIMAX models.

    Parameters
    ----------
    order : tuple of int (p, d, q)
        Non‑seasonal ARIMA order.
    seasonal_order : tuple of int (P, D, Q, s)
        Seasonal ARIMA order.
    """
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]


# ----------------------------------------------------------------------
# Model training utilities
# ----------------------------------------------------------------------