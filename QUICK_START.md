# 🚀 Quick Start Guide

## For VS Code Users

### 1. Open the Project in VS Code

```bash
# Open VS Code in the project directory
code .
```

### 2. Set Up Python Interpreter

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Python: Select Interpreter"
3. Choose the virtual environment you created (should show `./venv/bin/python`)

### 3. Install Extensions (Recommended)

- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Jupyter** (Microsoft) - if you want to test code in notebooks

### 4. Run the Application

#### Option A: Using VS Code Terminal
1. Open terminal: `Ctrl+` ` (backtick) or Terminal → New Terminal
2. Activate virtual environment (if not already activated):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```
3. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

#### Option B: Using VS Code Tasks
1. Press `Ctrl+Shift+P`
2. Type "Tasks: Run Task"
3. Select "Run Streamlit App" (if configured)

### 5. Access the Application

- The app will open automatically in your default browser
- If not, navigate to: `http://localhost:8501`

## Testing the Application

### Using Sample Data (Recommended for First Time)

1. Keep "Use Sample Dataset" checked in the sidebar
2. Select model type: "SARIMAX (with Temperature)"
3. Use default parameters or adjust:
   - Non-Seasonal: (1,1,1)
   - Seasonal: (1,1,1,12)
4. Click "🚀 Train Model & Forecast"
5. Explore all tabs:
   - Data Overview
   - Exploratory Analysis
   - Model Results
   - Diagnostics
   - Future Forecast

### Using Custom Data

1. Uncheck "Use Sample Dataset"
2. Upload your CSV file (format: date, consumption, temperature)
3. Follow the same steps as above

## Keyboard Shortcuts in Streamlit

- `R` - Rerun the app
- `C` - Clear cache
- `Ctrl+S` - Save and auto-rerun (if watching files)

## Troubleshooting

### Port Already in Use
```bash
# Kill the process using port 8501
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run app.py --server.port 8502
```

### Module Not Found Error
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt
```

### Streamlit Not Updating
```bash
# Clear cache
streamlit cache clear

# Or press 'C' in the running app
```

## Development Tips

### VS Code Debugging

1. Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": [
                "run",
                "app.py"
            ]
        }
    ]
}
```

2. Press `F5` to start debugging

### Live Editing

- Streamlit watches for file changes
- Save any `.py` file to see updates immediately
- Click "Rerun" or press `R` to refresh

### Best Practices

1. **Keep virtual environment activated** while working
2. **Test with sample data first** before using real data
3. **Save your model parameters** if you find good combinations
4. **Export forecasts** using the download button for documentation
5. **Check diagnostics tab** to ensure model quality

## Next Steps

1. ✅ Familiarize yourself with the UI
2. ✅ Test different parameter combinations
3. ✅ Compare SARIMA vs SARIMAX performance
4. ✅ Generate and download forecasts
5. ✅ Prepare your final project report with screenshots

## Need Help?

- Check `README.md` for detailed documentation
- Review error messages in the terminal
- Inspect residual diagnostics for model improvement
- Adjust parameters if forecasts are poor

---

**Happy Forecasting! 📊⚡**
