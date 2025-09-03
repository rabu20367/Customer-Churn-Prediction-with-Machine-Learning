"""
Simple system test to verify all components work correctly.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from data_engine import DataEngine
        from model_engine import ModelEngine
        from intelligence_engine import IntelligenceEngine
        from config import Config
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_engine():
    """Test data engine functionality."""
    try:
        from data_engine import DataEngine
        import pandas as pd
        
        # Create sample data
        sample_data = pd.DataFrame({
            'customerID': ['CUST001'],
            'tenure': [24],
            'MonthlyCharges': [70.0],
            'TotalCharges': [1680.0],
            'Contract': ['Month-to-month'],
            'InternetService': ['DSL'],
            'PhoneService': ['Yes'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['No'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'PaymentMethod': ['Electronic check'],
            'Churn': ['No']
        })
        
        data_engine = DataEngine()
        X, y = data_engine.process(sample_data)
        
        print(f"✅ Data engine test passed: {X.shape[1]} features created")
        return True
    except Exception as e:
        print(f"❌ Data engine test failed: {e}")
        return False

def test_config():
    """Test configuration."""
    try:
        from config import Config
        
        print(f"✅ Config test passed: Model path = {Config.MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running system tests...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config),
        ("Data Engine Test", test_data_engine),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
