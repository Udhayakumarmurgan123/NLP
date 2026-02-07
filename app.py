import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.preprocessing import load_data, prepare_data, check_stationarity
from models.sarimax_model import train_sarima, train_sarimax, forecast_future, evaluate_model

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("⚡ Seasonal Energy Consumption Forecasting Using SARIMAX")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Model Configuration")
    
    # Data Upload
    st.subheader("1. Data Source")
    use_sample = st.checkbox("Use Sample Dataset", value=True)
    
    if not use_sample:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    else:
        uploaded_file = None
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("2. Model Selection")
    model_type = st.selectbox(
        "Choose Model",
        ["SARIMA", "SARIMAX (with Temperature)"],
        index=1
    )
    
    st.markdown("---")
    
    # SARIMA Parameters
    st.subheader("3. Model Parameters")
    
    with st.expander("Non-Seasonal Parameters (p,d,q)", expanded=True):
        p = st.slider("AR order (p)", 0, 5, 1)
        d = st.slider("Differencing (d)", 0, 2, 1)
        q = st.slider("MA order (q)", 0, 5, 1)
    
    with st.expander("Seasonal Parameters (P,D,Q,s)", expanded=True):
        P = st.slider("Seasonal AR (P)", 0, 3, 1)
        D = st.slider("Seasonal Diff (D)", 0, 2, 1)
        Q = st.slider("Seasonal MA (Q)", 0, 3, 1)
        s = st.selectbox("Seasonal Period (s)", [12, 4, 7], index=0)
    
    st.markdown("---")
    
    # Forecast Settings
    st.subheader("4. Forecast Settings")
    forecast_periods = st.slider("Forecast Periods (months)", 6, 36, 12)
    
    st.markdown("---")
    
    # Train button
    train_button = st.button("🚀 Train Model & Forecast", type="primary", use_container_width=True)

# Main content
try:
    # Load data
    if use_sample or uploaded_file is not None:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file if not use_sample else None)
            df_processed, train_data, test_data = prepare_data(df, test_size=0.2)
        
        # Display data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Training Set", len(train_data))
        with col3:
            st.metric("Test Set", len(test_data))
        with col4:
            st.metric("Features", df.shape[1])
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Data Overview", 
            "🔍 Exploratory Analysis", 
            "🎯 Model Results",
            "📊 Diagnostics",
            "🔮 Future Forecast"
        ])
        
        # Tab 1: Data Overview
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Time series plot
            st.subheader("Energy Consumption Over Time")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df['consumption'], color='#1f77b4', linewidth=2)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
            ax.set_title('Historical Energy Consumption', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Tab 2: Exploratory Analysis
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Seasonal Decomposition")
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                decomposition = seasonal_decompose(df['consumption'], model='additive', period=12)
                
                fig, axes = plt.subplots(4, 1, figsize=(10, 10))
                decomposition.observed.plot(ax=axes[0], title='Observed', color='#1f77b4')
                decomposition.trend.plot(ax=axes[1], title='Trend', color='#ff7f0e')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='#2ca02c')
                decomposition.resid.plot(ax=axes[3], title='Residual', color='#d62728')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("Correlation Analysis")
                
                # Monthly pattern
                df['month'] = df.index.month
                monthly_avg = df.groupby('month')['consumption'].mean()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                monthly_avg.plot(kind='bar', color='#1f77b4', ax=ax)
                ax.set_xlabel('Month', fontsize=12)
                ax.set_ylabel('Average Consumption', fontsize=12)
                ax.set_title('Average Consumption by Month', fontsize=14, fontweight='bold')
                ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Temperature vs Consumption
                st.subheader("Temperature vs Consumption")
                fig, ax = plt.subplots(figsize=(10, 5))
                scatter = ax.scatter(df['temperature'], df['consumption'], 
                                   c=df.index.month, cmap='viridis', alpha=0.6, s=50)
                ax.set_xlabel('Temperature (°C)', fontsize=12)
                ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
                ax.set_title('Consumption vs Temperature', fontsize=14, fontweight='bold')
                plt.colorbar(scatter, ax=ax, label='Month')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Correlation matrix
                st.subheader("Correlation Matrix")
                corr_data = df[['consumption', 'temperature']].corr()
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=1, ax=ax, vmin=-1, vmax=1)
                ax.set_title('Feature Correlation', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Stationarity test
            st.subheader("Stationarity Check (ADF Test)")
            adf_result = check_stationarity(df['consumption'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ADF Statistic", f"{adf_result['ADF Statistic']:.4f}")
            with col2:
                st.metric("P-Value", f"{adf_result['P-Value']:.4f}")
            with col3:
                is_stationary = "✅ Yes" if adf_result['P-Value'] < 0.05 else "❌ No"
                st.metric("Stationary", is_stationary)
            
            st.info("💡 **Interpretation**: p-value < 0.05 indicates the series is stationary")
        
        # Tab 3: Model Results
        with tab3:
            if train_button:
                with st.spinner("Training model... This may take a few moments."):
                    order = (p, d, q)
                    seasonal_order = (P, D, Q, s)
                    
                    if model_type == "SARIMA":
                        model_fit, predictions, conf_int = train_sarima(
                            train_data, test_data, order, seasonal_order
                        )
                        exog_test = None
                    else:
                        # Prepare exogenous variables
                        train_idx = int(len(df_processed) * 0.8)
                        exog_train = df_processed['temperature'].iloc[:train_idx]
                        exog_test = df_processed['temperature'].iloc[train_idx:]
                        
                        model_fit, predictions, conf_int = train_sarimax(
                            train_data, test_data, exog_train, exog_test, order, seasonal_order
                        )
                    
                    # Store in session state
                    st.session_state['model_fit'] = model_fit
                    st.session_state['predictions'] = predictions
                    st.session_state['conf_int'] = conf_int
                    st.session_state['train_data'] = train_data
                    st.session_state['test_data'] = test_data
                    st.session_state['exog_test'] = exog_test
                    st.session_state['df_processed'] = df_processed
                
                st.success("✅ Model trained successfully!")
            
            if 'model_fit' in st.session_state:
                model_fit = st.session_state['model_fit']
                predictions = st.session_state['predictions']
                conf_int = st.session_state['conf_int']
                train_data = st.session_state['train_data']
                test_data = st.session_state['test_data']
                
                # Model Summary
                st.subheader("Model Summary")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.text(str(model_fit.summary()))
                
                with col2:
                    st.subheader("Model Information")
                    st.write(f"**Model Type**: {model_type}")
                    st.write(f"**Order**: {order}")
                    st.write(f"**Seasonal Order**: {seasonal_order}")
                    st.write(f"**AIC**: {model_fit.aic:.2f}")
                    st.write(f"**BIC**: {model_fit.bic:.2f}")
                
                # Performance Metrics
                st.subheader("Performance Metrics")
                metrics = evaluate_model(test_data, predictions)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{metrics['MAE']:.2f} kWh")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.2f} kWh")
                with col3:
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                with col4:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                
                # Forecast Plot
                st.subheader("Test Set Predictions vs Actual")
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Plot training data
                ax.plot(train_data.index, train_data, label='Training Data', 
                       color='#1f77b4', linewidth=2)
                
                # Plot test data
                ax.plot(test_data.index, test_data, label='Actual Test Data', 
                       color='#2ca02c', linewidth=2)
                
                # Plot predictions
                ax.plot(test_data.index, predictions, label='Predictions', 
                       color='#ff7f0e', linewidth=2, linestyle='--')
                
                # Plot confidence intervals
                ax.fill_between(test_data.index, 
                               conf_int.iloc[:, 0], 
                               conf_int.iloc[:, 1], 
                               alpha=0.2, color='#ff7f0e', label='95% Confidence Interval')
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
                ax.set_title('Model Predictions on Test Set', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("👆 Click 'Train Model & Forecast' button in the sidebar to see results")
        
        # Tab 4: Diagnostics
        with tab4:
            if 'model_fit' in st.session_state:
                model_fit = st.session_state['model_fit']
                
                st.subheader("Residual Diagnostics")
                
                fig = plt.figure(figsize=(14, 10))
                
                # Residuals plot
                ax1 = plt.subplot(2, 2, 1)
                residuals = model_fit.resid
                ax1.plot(residuals, color='#1f77b4')
                ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
                ax1.set_title('Residuals Over Time', fontweight='bold')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Residuals')
                ax1.grid(True, alpha=0.3)
                
                # Histogram
                ax2 = plt.subplot(2, 2, 2)
                ax2.hist(residuals, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
                ax2.set_title('Residuals Distribution', fontweight='bold')
                ax2.set_xlabel('Residuals')
                ax2.set_ylabel('Frequency')
                ax2.axvline(x=0, color='r', linestyle='--', linewidth=1)
                ax2.grid(True, alpha=0.3)
                
                # Q-Q plot
                ax3 = plt.subplot(2, 2, 3)
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=ax3)
                ax3.set_title('Q-Q Plot', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # ACF of residuals
                ax4 = plt.subplot(2, 2, 4)
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(residuals, lags=40, ax=ax4)
                ax4.set_title('ACF of Residuals', fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Ljung-Box Test
                st.subheader("Ljung-Box Test for Residuals")
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("LB Statistic", f"{lb_test['lb_stat'].values[0]:.4f}")
                with col2:
                    st.metric("P-Value", f"{lb_test['lb_pvalue'].values[0]:.4f}")
                
                if lb_test['lb_pvalue'].values[0] > 0.05:
                    st.success("✅ Residuals appear to be white noise (p > 0.05)")
                else:
                    st.warning("⚠️ Residuals may have remaining autocorrelation (p < 0.05)")
            else:
                st.info("👆 Train the model first to see diagnostics")
        
        # Tab 5: Future Forecast
        with tab5:
            if 'model_fit' in st.session_state:
                model_fit = st.session_state['model_fit']
                df_processed = st.session_state['df_processed']
                
                st.subheader(f"Forecast Next {forecast_periods} Months")
                
                with st.spinner("Generating forecast..."):
                    # Prepare exogenous data for future forecast if SARIMAX
                    if model_type == "SARIMAX (with Temperature)":
                        # Generate future temperature (seasonal pattern + noise)
                        last_date = df_processed.index[-1]
                        future_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq='MS')[1:]
                        
                        # Use historical average temperature pattern
                        avg_temp_by_month = df_processed.groupby(df_processed.index.month)['temperature'].mean()
                        future_temp = [avg_temp_by_month[date.month] + np.random.normal(0, 2) 
                                      for date in future_dates]
                        exog_future = pd.Series(future_temp, index=future_dates)
                    else:
                        exog_future = None
                    
                    future_forecast, future_conf_int = forecast_future(
                        model_fit, forecast_periods, exog_future
                    )
                
                # Forecast plot
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Historical data (last 24 months)
                historical = df['consumption'].iloc[-24:]
                ax.plot(historical.index, historical, label='Historical Data', 
                       color='#1f77b4', linewidth=2)
                
                # Future forecast
                ax.plot(future_forecast.index, future_forecast, label='Forecast', 
                       color='#ff7f0e', linewidth=2, linestyle='--')
                
                # Confidence intervals
                ax.fill_between(future_forecast.index, 
                               future_conf_int.iloc[:, 0], 
                               future_conf_int.iloc[:, 1], 
                               alpha=0.2, color='#ff7f0e', label='95% Confidence Interval')
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
                ax.set_title(f'Future Forecast ({forecast_periods} Months)', 
                           fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Forecast table
                st.subheader("Forecast Values")
                forecast_df = pd.DataFrame({
                    'Date': future_forecast.index.strftime('%Y-%m'),
                    'Forecasted Consumption': future_forecast.values.round(2),
                    'Lower Bound': future_conf_int.iloc[:, 0].values.round(2),
                    'Upper Bound': future_conf_int.iloc[:, 1].values.round(2)
                })
                st.dataframe(forecast_df, use_container_width=True)
                
                # Download forecast
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast as CSV",
                    data=csv,
                    file_name=f"energy_forecast_{forecast_periods}months.csv",
                    mime="text/csv"
                )
            else:
                st.info("👆 Train the model first to generate future forecasts")
    
    else:
        st.info("👈 Please upload a dataset or use the sample dataset from the sidebar")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Seasonal Energy Consumption Forecasting System</strong></p>
        <p>Built with Streamlit | Powered by SARIMAX</p>
    </div>
    """, unsafe_allow_html=True)
