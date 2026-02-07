"""
Validation Script for Energy Forecasting Project
Run this script to ensure all components are working correctly
"""

import sys
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    print("=" * 60)
    print("CHECKING PYTHON VERSION")
    print("=" * 60)
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python version is too old. Please upgrade to Python 3.8+")
        return False

def check_packages():
    """Check if all required packages are installed"""
    print("\n" + "=" * 60)
    print("CHECKING REQUIRED PACKAGES")
    print("=" * 60)
    
    packages = {
        'streamlit': '1.31.0',
        'pandas': '2.1.4',
        'numpy': '1.26.3',
        'matplotlib': '3.8.2',
        'seaborn': '0.13.1',
        'statsmodels': '0.14.1',
        'sklearn': '1.4.0',
        'scipy': '1.12.0'
    }
    
    all_installed = True
    
    for package, min_version in packages.items():
        try:
            if package == 'sklearn':
                module = importlib.import_module('sklearn')
            else:
                module = importlib.import_module(package)
            
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package:<15} {version}")
        except ImportError:
            print(f"❌ {package:<15} NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_project_structure():
    """Check if all project files exist"""
    print("\n" + "=" * 60)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 60)
    
    import os
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'data/energy_consumption.csv',
        'utils/__init__.py',
        'utils/preprocessing.py',
        'models/__init__.py',
        'models/sarimax_model.py'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist

def test_data_loading():
    """Test if data can be loaded"""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    try:
        from utils.preprocessing import load_data, prepare_data
        
        # Test loading sample data
        df = load_data()
        print(f"✅ Sample data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Test data preparation
        df_processed, train_data, test_data = prepare_data(df, test_size=0.2)
        print(f"✅ Data split: {len(train_data)} train, {len(test_data)} test")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False

def test_model_import():
    """Test if model modules can be imported"""
    print("\n" + "=" * 60)
    print("TESTING MODEL IMPORTS")
    print("=" * 60)
    
    try:
        from models.sarimax_model import (
            train_sarima, 
            train_sarimax, 
            forecast_future, 
            evaluate_model
        )
        print("✅ All model functions imported successfully")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {str(e)}")
        return False

def test_basic_model():
    """Test if basic model can be trained"""
    print("\n" + "=" * 60)
    print("TESTING BASIC MODEL TRAINING")
    print("=" * 60)
    
    try:
        from utils.preprocessing import load_data, prepare_data
        from models.sarimax_model import train_sarima, evaluate_model
        
        # Load data
        df = load_data()
        df_processed, train_data, test_data = prepare_data(df, test_size=0.2)
        
        # Train simple model
        print("Training SARIMA(1,1,1)(1,1,1,12)...")
        model_fit, predictions, conf_int = train_sarima(
            train_data, 
            test_data, 
            order=(1,1,1),
            seasonal_order=(1,1,1,12)
        )
        
        # Evaluate
        metrics = evaluate_model(test_data, predictions)
        
        print(f"✅ Model trained successfully")
        print(f"   MAE: {metrics['MAE']:.2f}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   R²: {metrics['R2']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ENERGY FORECASTING PROJECT VALIDATOR" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Required Packages", check_packages()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Imports", test_model_import()))
    results.append(("Basic Model Training", test_basic_model()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All checks passed! Your project is ready to run.")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("   Try running: pip install -r requirements.txt")
    
    print("\n")

if __name__ == "__main__":
    main()
