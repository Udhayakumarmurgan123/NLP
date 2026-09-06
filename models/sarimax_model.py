# Make predictions
predictions = model_fit.forecast(steps=len(test_data))          # or with exog
# Get confidence intervals
forecast_result = model_fit.get_forecast(steps=len(test_data)) # or with exog
conf_int = forecast_result.conf_int()

# Set proper index
predictions.index = test_data.index
conf_int.index = test_data.index