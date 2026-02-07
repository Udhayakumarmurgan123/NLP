# 🎯 COMPLETE INSTALLATION & USAGE GUIDE

## 📦 What You Have

A complete, production-ready **Energy Consumption Forecasting** application with:
- ✅ Beautiful Streamlit UI
- ✅ SARIMA & SARIMAX models
- ✅ Interactive visualizations
- ✅ Sample dataset included
- ✅ Comprehensive diagnostics
- ✅ Future forecasting capabilities

---

## 🚀 INSTALLATION STEPS

### Step 1: Extract the Project

1. Download the `energy_forecasting_project` folder
2. Extract to your desired location (e.g., `C:\Projects\` or `~/Projects/`)

### Step 2: Open in VS Code

```bash
# Navigate to project directory
cd energy_forecasting_project

# Open in VS Code
code .
```

### Step 3: Set Up Python Environment

#### Option A: Automated Setup (Recommended)

**Windows:**
```bash
# Double-click setup.bat
# OR run in terminal:
setup.bat
```

**macOS/Linux:**
```bash
# Make script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

#### Option B: Manual Setup

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 4: Validate Installation

```bash
# Run validation script
python validate.py
```

This will check:
- ✅ Python version
- ✅ Required packages
- ✅ Project structure
- ✅ Data loading
- ✅ Model functionality

---

## 🎮 RUNNING THE APPLICATION

### Start the App

```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run Streamlit
streamlit run app.py
```

The app will automatically open at: **http://localhost:8501**

### Using Different Port

```bash
streamlit run app.py --server.port 8502
```

---

## 📊 HOW TO USE THE APPLICATION

### First-Time Usage (Recommended)

1. **Keep "Use Sample Dataset" checked** ✅
2. **Select Model**: "SARIMAX (with Temperature)"
3. **Parameters** (use defaults):
   - Non-Seasonal: (1,1,1)
   - Seasonal: (1,1,1,12)
4. **Click**: "🚀 Train Model & Forecast"
5. **Explore all 5 tabs**:
   - 📈 Data Overview
   - 🔍 Exploratory Analysis
   - 🎯 Model Results
   - 📊 Diagnostics
   - 🔮 Future Forecast

### Understanding Each Tab

#### 1️⃣ Data Overview
- **Purpose**: Understand your dataset
- **What to look for**: 
  - Number of records
  - Overall trend
  - Any unusual patterns

#### 2️⃣ Exploratory Analysis
- **Purpose**: Deep dive into patterns
- **What to look for**:
  - Seasonal patterns (monthly variations)
  - Temperature correlation
  - Stationarity (p-value < 0.05 is good)

#### 3️⃣ Model Results
- **Purpose**: Evaluate model performance
- **What to look for**:
  - Low MAPE (< 10% is excellent)
  - High R² (> 0.8 is good)
  - Predictions close to actual values

#### 4️⃣ Diagnostics
- **Purpose**: Check model quality
- **What to look for**:
  - Residuals centered around 0
  - Normal distribution (bell curve)
  - No patterns in residuals (white noise)

#### 5️⃣ Future Forecast
- **Purpose**: Generate predictions
- **What you get**:
  - Visual forecast with confidence intervals
  - Table with exact values
  - **Download button** for CSV export

---

## 🎨 CUSTOMIZING THE MODEL

### Parameter Tuning Guide

#### Non-Seasonal Parameters (p,d,q)
- **Start with**: (1,1,1)
- **If forecasts are too smooth**: Increase p or q
- **If data isn't stationary**: Increase d (max 2)

#### Seasonal Parameters (P,D,Q,s)
- **Start with**: (1,1,1,12) for monthly data
- **s=12**: Monthly seasonality
- **s=4**: Quarterly seasonality
- **Increase P or Q**: For stronger seasonal patterns

### Model Selection

**Use SARIMA when**:
- You only have consumption data
- Temperature data is unreliable
- You want simpler model

**Use SARIMAX when**:
- You have temperature data
- External factors are important
- You want better accuracy

---

## 📁 USING YOUR OWN DATA

### Data Format Required

Your CSV file must have these columns:
```csv
date,consumption,temperature
2019-01-01,5000,15.5
2019-02-01,5200,16.2
2019-03-01,4800,18.7
...
```

### Requirements:
- ✅ Date in YYYY-MM-DD format
- ✅ Numeric consumption values
- ✅ Numeric temperature values
- ✅ No missing values
- ✅ At least 24 months of data (recommended)

### Upload Steps:
1. **Uncheck** "Use Sample Dataset"
2. **Click** "Browse files"
3. **Select** your CSV file
4. **Verify** data loads correctly in Data Overview tab
5. **Proceed** with model training

---

## 💡 TIPS FOR BEST RESULTS

### 1. Data Quality
- **More data = better forecasts** (minimum 24 months)
- **Remove outliers** before modeling
- **Check for missing values**

### 2. Model Training
- **Start simple**: Use default parameters first
- **Check diagnostics**: Ensure residuals are white noise
- **Compare models**: Try both SARIMA and SARIMAX

### 3. Interpretation
- **MAPE < 5%**: Excellent forecast
- **MAPE 5-10%**: Good forecast
- **MAPE 10-20%**: Acceptable forecast
- **MAPE > 20%**: Poor forecast (adjust parameters)

### 4. Forecasting
- **Short-term (1-6 months)**: Most accurate
- **Medium-term (6-12 months)**: Good accuracy
- **Long-term (12+ months)**: Less reliable

---

## 🔧 TROUBLESHOOTING

### Issue: Port Already in Use
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID [PID_NUMBER] /F

# macOS/Linux
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run app.py --server.port 8502
```

### Issue: Module Not Found
```bash
# Ensure virtual environment is activated
# Then reinstall
pip install -r requirements.txt
```

### Issue: Model Won't Train
- **Check**: Parameter values aren't too high
- **Try**: Simpler parameters like (1,0,1)(1,0,1,12)
- **Verify**: Data has enough observations

### Issue: Poor Forecasts
- **Check diagnostics**: Residuals should be white noise
- **Try different parameters**: Increase/decrease p,q,P,Q
- **Add exogenous variables**: Use SARIMAX with temperature

---

## 📚 PROJECT FILES OVERVIEW

```
energy_forecasting_project/
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── README.md                   # Full documentation
├── QUICK_START.md              # Quick reference
├── validate.py                 # Validation script
├── setup.sh / setup.bat        # Setup scripts
├── .streamlit/config.toml      # UI configuration
├── data/
│   └── energy_consumption.csv  # Sample dataset
├── utils/
│   └── preprocessing.py        # Data utilities
└── models/
    └── sarimax_model.py        # Model functions
```

---

## 🎓 NEXT STEPS

### For Your Project Submission:

1. ✅ **Run the app** and explore all features
2. ✅ **Take screenshots** of each tab
3. ✅ **Train multiple models** with different parameters
4. ✅ **Compare results** (SARIMA vs SARIMAX)
5. ✅ **Generate forecasts** and download CSV
6. ✅ **Document findings** in your report

### Key Metrics to Report:
- Model parameters used
- AIC/BIC values
- MAE, RMSE, MAPE, R²
- Forecast accuracy
- Residual diagnostics results

---

## 📞 GETTING HELP

### Resources:
- 📖 **README.md**: Comprehensive documentation
- 📖 **QUICK_START.md**: Quick reference guide
- 📖 **PROJECT_STRUCTURE.md**: Detailed file structure

### Common Questions:

**Q: Can I use hourly data?**
A: Yes, but change seasonal period `s` to 24 or 168

**Q: Can I add more exogenous variables?**
A: Yes, modify the code to include additional variables

**Q: How do I improve accuracy?**
A: Try different parameters, add more data, or include more exogenous variables

---

## ✅ SUCCESS CHECKLIST

Before submitting your project, ensure:

- [ ] App runs without errors
- [ ] All 5 tabs are functional
- [ ] Model trains successfully
- [ ] Diagnostics look good (residuals are white noise)
- [ ] Forecasts are reasonable
- [ ] You can download forecast CSV
- [ ] You have screenshots for documentation
- [ ] You understand the results

---

## 🎉 YOU'RE READY!

This is a **complete, professional-grade** forecasting application.

**To start:**
```bash
streamlit run app.py
```

**Happy Forecasting! 📊⚡**

---

*For detailed technical information, see README.md*
*For quick commands, see QUICK_START.md*
