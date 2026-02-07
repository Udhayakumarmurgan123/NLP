# ⚡ Seasonal Energy Consumption Forecasting Using SARIMAX

A comprehensive web application for forecasting energy consumption using SARIMA and SARIMAX time series models with an interactive Streamlit interface.

## 📋 Problem Statement

A power distribution company aims to predict monthly electricity consumption to ensure adequate supply, prevent outages, and optimize energy distribution. Energy usage data exhibits strong seasonal patterns and is significantly influenced by external factors such as temperature variations and public holidays.

## 🎯 Project Objectives

- Build SARIMA and SARIMAX models to capture seasonal trends
- Incorporate temperature as an exogenous variable
- Tune seasonal and non-seasonal parameters
- Perform residual diagnostics
- Validate forecast accuracy using performance metrics

## 📁 Project Structure

```
energy_forecasting_project/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   └── energy_consumption.csv      # Sample dataset
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py            # Data preprocessing utilities
│
└── models/
    ├── __init__.py
    └── sarimax_model.py            # SARIMAX model implementation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
# Navigate to your project directory
cd energy_forecasting_project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎮 Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📊 Features

### 1. **Data Overview**
- Dataset preview and statistical summary
- Time series visualization
- Data quality metrics

### 2. **Exploratory Analysis**
- Seasonal decomposition (Trend, Seasonal, Residual)
- Monthly consumption patterns
- Temperature vs Consumption correlation
- Stationarity testing (ADF test)

### 3. **Model Configuration**
- **Model Selection**: Choose between SARIMA or SARIMAX
- **Parameter Tuning**:
  - Non-seasonal parameters (p, d, q)
  - Seasonal parameters (P, D, Q, s)
- **Customizable Settings**: Adjust forecast periods

### 4. **Model Results**
- Comprehensive model summary
- Performance metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - R² Score
- Test set predictions with confidence intervals

### 5. **Diagnostics**
- Residual analysis plots
- Distribution check (histogram)
- Q-Q plot for normality
- ACF of residuals
- Ljung-Box test for white noise

### 6. **Future Forecasting**
- Generate forecasts for custom time periods
- Confidence interval visualization
- Downloadable forecast results (CSV)

## 🔧 Model Parameters Guide

### Non-Seasonal Parameters (p, d, q)
- **p**: Auto-Regressive order (number of lag observations)
- **d**: Degree of differencing (to make series stationary)
- **q**: Moving Average order (size of moving average window)

### Seasonal Parameters (P, D, Q, s)
- **P**: Seasonal Auto-Regressive order
- **D**: Seasonal differencing
- **Q**: Seasonal Moving Average order
- **s**: Seasonal period (12 for monthly data, 4 for quarterly)

### Recommended Starting Values
- **SARIMA**: (1,1,1)(1,1,1,12)
- **SARIMAX**: Same as SARIMA with temperature as exogenous variable

## 📈 Dataset Information

### Sample Dataset Includes:
- **Date Range**: January 2019 - December 2023 (60 months)
- **Features**:
  - `date`: Monthly timestamps
  - `consumption`: Energy consumption in kWh
  - `temperature`: Average monthly temperature in °C

### Data Characteristics:
- Strong seasonal pattern (summer/winter peaks)
- Slight upward trend
- Temperature correlation with consumption

## 📝 Using Your Own Data

To use your own dataset:

1. **Format your CSV file** with the following columns:
   ```
   date,consumption,temperature
   2019-01-01,5000,15.5
   2019-02-01,5200,16.2
   ...
   ```

2. **Upload via Streamlit**:
   - Uncheck "Use Sample Dataset" in the sidebar
   - Click "Browse files" to upload your CSV

3. **Requirements**:
   - Date column in YYYY-MM-DD format
   - Numerical consumption and temperature values
   - No missing values (or handle them first)

## 📊 Performance Metrics Explained

- **MAE**: Average absolute difference between predicted and actual values (lower is better)
- **RMSE**: Square root of average squared differences (penalizes large errors, lower is better)
- **MAPE**: Average percentage error (easy to interpret, lower is better)
- **R² Score**: Proportion of variance explained by model (0-1, higher is better)

## 🔍 Model Selection Tips

1. **Start Simple**: Begin with SARIMA (1,1,1)(1,1,1,12)
2. **Check Diagnostics**: Ensure residuals are white noise
3. **Add Exogenous**: Use SARIMAX if temperature correlation is strong
4. **Tune Parameters**: Adjust based on ACF/PACF plots
5. **Compare Models**: Lower AIC/BIC indicates better fit

## 🛠️ Troubleshooting

### Common Issues:

**1. Model Convergence Error**
- Try different parameter values
- Reduce parameter complexity
- Check for outliers in data

**2. Poor Forecast Accuracy**
- Increase training data
- Add more exogenous variables
- Try different seasonal periods

**3. Slow Performance**
- Reduce forecast periods
- Use simpler parameter values
- Close other applications

## 📚 Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Modeling**: Statsmodels (SARIMAX)
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Scikit-learn

## 🎓 Learning Resources

- [Statsmodels SARIMAX Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [Time Series Analysis with Python](https://www.statsmodels.org/stable/tsa.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

This project is created for educational purposes as part of a time series forecasting course.

## 👥 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📧 Support

For questions or issues, please create an issue in the repository or contact the project maintainer.

---

**Built with ❤️ using Streamlit and SARIMAX**
