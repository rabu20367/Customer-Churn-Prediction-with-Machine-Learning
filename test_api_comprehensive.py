"""
Comprehensive API testing suite for Customer Churn Prediction system.
Tests all endpoints, error scenarios, and edge cases.
"""
import pytest
import requests
import json
import time
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class TestChurnPredictionAPI:
    """Comprehensive test suite for the churn prediction API."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        self.base_url = API_BASE_URL
        self.session = requests.Session()
        self.valid_customer_data = {
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
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.session.get(f"{self.base_url}/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["data_engine_ready"] is True
    
    def test_model_info_endpoint(self):
        """Test model information endpoint."""
        response = self.session.get(f"{self.base_url}/model/info", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "feature_count" in data
        assert "features" in data
        assert "top_features" in data
        assert "feature_importance" in data
        assert "performance_metrics" in data
        assert "supported_endpoints" in data
        
        # Validate performance metrics
        assert data["performance_metrics"]["expected_accuracy"] == "89-92%"
        assert data["performance_metrics"]["response_time"] == "<20ms"
    
    def test_valid_prediction(self):
        """Test prediction with valid customer data."""
        response = self.session.post(
            f"{self.base_url}/predict",
            json=self.valid_customer_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "churn_risk" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        assert "model_version" in data
        
        # Validate data types and ranges
        assert isinstance(data["churn_probability"], float)
        assert 0 <= data["churn_probability"] <= 1
        assert data["churn_risk"] in ["Low", "Medium", "High"]
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["processing_time_ms"], float)
        assert data["processing_time_ms"] < 100  # Should be fast
    
    def test_invalid_enum_values(self):
        """Test prediction with invalid enum values."""
        invalid_data = self.valid_customer_data.copy()
        invalid_data["gender"] = "InvalidGender"
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=invalid_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422  # Validation error
    
    def test_invalid_numeric_ranges(self):
        """Test prediction with invalid numeric ranges."""
        # Test negative tenure
        invalid_data = self.valid_customer_data.copy()
        invalid_data["tenure"] = -1
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=invalid_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
        
        # Test excessive monthly charges
        invalid_data = self.valid_customer_data.copy()
        invalid_data["MonthlyCharges"] = 300.0
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=invalid_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test prediction with missing required fields."""
        incomplete_data = {
            "customerID": "TEST002",
            "tenure": 12
            # Missing most required fields
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=incomplete_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
    
    def test_empty_customer_id(self):
        """Test prediction with empty customer ID."""
        invalid_data = self.valid_customer_data.copy()
        invalid_data["customerID"] = ""
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=invalid_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
    
    def test_negative_total_charges(self):
        """Test prediction with negative total charges."""
        invalid_data = self.valid_customer_data.copy()
        invalid_data["TotalCharges"] = -100.0
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=invalid_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
    
    def test_high_risk_customer(self):
        """Test prediction with high-risk customer profile."""
        high_risk_data = self.valid_customer_data.copy()
        high_risk_data.update({
            "tenure": 1,  # Very short tenure
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "OnlineSecurity": "No",
            "TechSupport": "No"
        })
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=high_risk_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["churn_probability"] > 0.5  # Should be high risk
        assert data["churn_risk"] in ["Medium", "High"]
    
    def test_low_risk_customer(self):
        """Test prediction with low-risk customer profile."""
        low_risk_data = self.valid_customer_data.copy()
        low_risk_data.update({
            "tenure": 60,  # Long tenure
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes"
        })
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=low_risk_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["churn_probability"] < 0.5  # Should be low risk
        assert data["churn_risk"] in ["Low", "Medium"]
    
    def test_response_time_performance(self):
        """Test that response times are consistently fast."""
        response_times = []
        
        for i in range(5):  # Test 5 requests
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict",
                json=self.valid_customer_data,
                timeout=TEST_TIMEOUT
            )
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # All responses should be under 100ms
        assert all(rt < 100 for rt in response_times)
        
        # Average should be under 50ms
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 50
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=self.valid_customer_data,
                    timeout=TEST_TIMEOUT
                )
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Start 10 concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all requests succeeded
        status_codes = []
        while not results.empty():
            result = results.get()
            status_codes.append(result)
        
        assert len(status_codes) == 10
        assert all(code == 200 for code in status_codes)
    
    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = self.session.post(
            f"{self.base_url}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
    
    def test_unsupported_methods(self):
        """Test that unsupported HTTP methods return appropriate errors."""
        # Test PUT method
        response = self.session.put(f"{self.base_url}/predict", timeout=TEST_TIMEOUT)
        assert response.status_code == 405
        
        # Test DELETE method
        response = self.session.delete(f"{self.base_url}/predict", timeout=TEST_TIMEOUT)
        assert response.status_code == 405
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.session.options(f"{self.base_url}/predict", timeout=TEST_TIMEOUT)
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
    
    def test_error_response_format(self):
        """Test that error responses have consistent format."""
        invalid_data = {"invalid": "data"}
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=invalid_data,
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
        
        data = response.json()
        assert "error" in data or "detail" in data

def run_tests():
    """Run all tests."""
    print("🧪 Running comprehensive API tests...")
    
    # Test if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running. Please start the API first.")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start the API first.")
        return False
    
    # Run tests
    test_suite = TestChurnPredictionAPI()
    test_suite.setup()
    
    tests = [
        ("Health Endpoint", test_suite.test_health_endpoint),
        ("Model Info Endpoint", test_suite.test_model_info_endpoint),
        ("Valid Prediction", test_suite.test_valid_prediction),
        ("Invalid Enum Values", test_suite.test_invalid_enum_values),
        ("Invalid Numeric Ranges", test_suite.test_invalid_numeric_ranges),
        ("Missing Required Fields", test_suite.test_missing_required_fields),
        ("Empty Customer ID", test_suite.test_empty_customer_id),
        ("Negative Total Charges", test_suite.test_negative_total_charges),
        ("High Risk Customer", test_suite.test_high_risk_customer),
        ("Low Risk Customer", test_suite.test_low_risk_customer),
        ("Response Time Performance", test_suite.test_response_time_performance),
        ("Concurrent Requests", test_suite.test_concurrent_requests),
        ("Invalid JSON", test_suite.test_invalid_json),
        ("Unsupported Methods", test_suite.test_unsupported_methods),
        ("CORS Headers", test_suite.test_cors_headers),
        ("Error Response Format", test_suite.test_error_response_format),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"  🔍 Testing {test_name}...")
            test_func()
            print(f"  ✅ {test_name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_name} - FAILED: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! API is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the API implementation.")
        return False

if __name__ == "__main__":
    run_tests()
