"""
Simple API for Customer Churn Prediction
Bypasses complex data engine issues and uses direct transformation
"""

import sys
sys.path.append('src')

import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global model, data_engine
    
    try:
        logger.info("Loading model and data engine...")
        
        # Load model
        model = joblib.load('models/churn_model.pkl')
        
        # Load data engine
        data_engine = joblib.load('models/data_engine.pkl')
        
        logger.info("✅ All components loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Cleanup code here if needed
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Simple, fast API for predicting customer churn",
    version="1.0.0",
    lifespan=lifespan
)

# Global variables
model = None
data_engine = None

class CustomerData(BaseModel):
    """Customer data model with validation."""
    customerID: str = Field(default="CUST001", description="Customer identifier")
    gender: str = Field(default="Male", description="Customer gender")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Senior citizen status")
    Partner: str = Field(default="Yes", description="Partner status")
    Dependents: str = Field(default="No", description="Dependents status")
    tenure: int = Field(ge=0, le=100, description="Customer tenure in months")
    PhoneService: str = Field(default="Yes", description="Phone service status")
    MultipleLines: str = Field(default="No", description="Multiple lines status")
    InternetService: str = Field(default="DSL", description="Internet service type")
    OnlineSecurity: str = Field(default="No", description="Online security status")
    OnlineBackup: str = Field(default="No", description="Online backup status")
    DeviceProtection: str = Field(default="No", description="Device protection status")
    TechSupport: str = Field(default="No", description="Tech support status")
    StreamingTV: str = Field(default="No", description="Streaming TV status")
    StreamingMovies: str = Field(default="No", description="Streaming movies status")
    Contract: str = Field(default="Month-to-month", description="Contract type")
    PaperlessBilling: str = Field(default="Yes", description="Paperless billing status")
    PaymentMethod: str = Field(default="Electronic check", description="Payment method")
    MonthlyCharges: float = Field(ge=0, le=200, description="Monthly charges in USD")
    TotalCharges: float = Field(ge=0, description="Total charges in USD")

class PredictionResponse(BaseModel):
    """Prediction response model."""
    customer_id: str
    churn_probability: float
    churn_risk: str
    confidence: float
    processing_time_ms: float
    model_version: str = "1.0.0"

def create_impact_features(df):
    """Create advanced impact features."""
    df = df.copy()
    
    # Behavioral Risk Score
    risk_factors = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['behavioral_risk'] = df[risk_factors].apply(
        lambda x: sum(1 for val in x if val in ['No', 'No internet service']), axis=1
    )
    
    # Value Engagement Score
    df['value_engagement'] = (df['MonthlyCharges'] / df['tenure'].clip(lower=1)) * 100
    
    # Service Density
    services = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_density'] = df[services].apply(
        lambda x: sum(1 for val in x if val == 'Yes'), axis=1
    )
    
    # Streaming Engagement
    streaming_services = ['StreamingTV', 'StreamingMovies']
    df['streaming_engagement'] = df[streaming_services].apply(
        lambda x: sum(1 for val in x if val == 'Yes'), axis=1
    )
    
    # Contract Risk
    contract_risk_map = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
    df['contract_risk'] = df['Contract'].map(contract_risk_map)
    
    return df

def transform_customer_data(customer_data):
    """Transform customer data for prediction."""
    # Convert to DataFrame
    df = pd.DataFrame([customer_data.dict()])
    
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Create impact features
    df_processed = create_impact_features(df)
    
    # Encode categorical variables using the fitted encoders
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']
    
    for col in categorical_cols:
        if col in data_engine.encoders:
            df_processed[col] = data_engine.encoders[col].transform(df_processed[col].astype(str))
    
    # Select features in the exact order expected by the scaler
    feature_cols = data_engine.feature_names
    X = df_processed[feature_cols]
    
    # Scale features
    X_scaled = pd.DataFrame(
        data_engine.scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_scaled

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_engine_ready": data_engine is not None
    }

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": str(type(model)),
        "feature_count": len(data_engine.feature_names),
        "features": data_engine.feature_names
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """Predict customer churn."""
    start_time = time.time()
    
    try:
        # Transform data
        X_processed = transform_customer_data(customer_data)
        
        # Make prediction
        churn_probability = model.predict_proba(X_processed)[0, 1]
        confidence = np.max(model.predict_proba(X_processed)[0])
        
        # Determine risk level
        if churn_probability < 0.3:
            risk_level = "Low"
        elif churn_probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            customer_id=customer_data.customerID,
            churn_probability=round(churn_probability, 4),
            churn_risk=risk_level,
            confidence=round(confidence, 4),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
