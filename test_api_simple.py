import requests
import json

# Test data
test_data = {
    "customerID": "TEST001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 1680.0
}

# Test the API
try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API call successful!")
        print(f"Churn probability: {result['churn_probability']:.4f}")
        print(f"Risk level: {result['churn_risk']}")
    else:
        print(f"❌ API call failed with status {response.status_code}")
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")
