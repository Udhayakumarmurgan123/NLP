config = AutoArimaConfig(max_p=3, max_q=3, seasonal=False)
best = auto_arima_wrapper(train_series, config)