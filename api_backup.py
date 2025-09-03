"""
Ultra-high-performance FastAPI with sub-20ms response times.
Optimized for production deployment and scalability.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import time
import logging
import os
import sys
import joblib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_engine import DataEngine
from model_engine import ModelEngine
from intelligence_engine import IntelligenceEngine
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Ultra-fast churn prediction with AI-powered insights by Hasibur",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
data_engine: Optional[DataEngine] = None
model_engine: Optional[ModelEngine] = None
intelligence_engine: Optional[IntelligenceEngine] = None

# Pydantic models
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

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    customers: List[CustomerData] = Field(description="List of customers to predict")

class PredictionResponse(BaseModel):
    """Prediction response model."""
    customer_id: str
    churn_probability: float
    churn_risk: str
    confidence: float
    explanation: str
    retention_strategy: str
    personalized_email: str
    critical_factors: List[Dict[str, Any]]
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[Dict[str, Any]]
    total_customers: int
    processing_time_ms: float
    avg_time_per_customer_ms: float

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global data_engine, model_engine, intelligence_engine
    
    try:
        logger.info("Initializing components...")
        
        # Load fitted data engine
        data_engine = joblib.load('models/data_engine.pkl')
        
        # Initialize model engine
        model_engine = ModelEngine()
        model_engine.load_model(Config.MODEL_PATH)
        
        # Initialize intelligence engine
        intelligence_engine = IntelligenceEngine(
            model_engine.model,
            data_engine.scaler,
            data_engine.feature_names
        )
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API by Hasibur",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_engine is not None and model_engine.model is not None,
        "data_engine_ready": data_engine is not None,
        "intelligence_engine_ready": intelligence_engine is not None
    }

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_engine.get_model_info()

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """
    Predict customer churn with AI-powered insights.
    Optimized for sub-20ms response times.
    """
    start_time = time.time()
    
    try:
        # Convert to dict
        data_dict = customer_data.dict()
        
        # Process data using the working transformation logic
        customer_df = pd.DataFrame([data_dict])
        logger.info(f"Input DataFrame columns: {customer_df.columns.tolist()}")
        
        # Handle missing values
        customer_df['TotalCharges'] = pd.to_numeric(customer_df['TotalCharges'], errors='coerce')
        customer_df['TotalCharges'] = customer_df['TotalCharges'].fillna(customer_df['TotalCharges'].median())
        
        # Create impact features
        df_processed = create_impact_features(customer_df)
        
        # Encode categorical variables using the fitted encoders
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in data_engine.encoders:
                df_processed[col] = data_engine.encoders[col].transform(df_processed[col].astype(str))
        
        # Select features in the exact order expected by the scaler (exclude customerID)
        feature_cols = [col for col in data_engine.feature_names if col in df_processed.columns]
        X = df_processed[feature_cols]
        
        # Scale features
        X_processed = pd.DataFrame(
            data_engine.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        logger.info(f"Processed DataFrame columns: {X_processed.columns.tolist()}")
        
        # Make prediction
        churn_probability = model_engine.model.predict_proba(X_processed)[0, 1]
        confidence = np.max(model_engine.model.predict_proba(X_processed)[0])
        
        # Determine risk level
        if churn_probability > 0.7:
            risk_level = "High"
        elif churn_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate insights
        insights = intelligence_engine.generate_complete_insight(data_dict, churn_probability)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            customer_id=customer_data.customerID,
            churn_probability=float(churn_probability),
            churn_risk=risk_level,
            confidence=float(confidence),
            explanation=insights['explanation'],
            retention_strategy=insights['retention_strategy'],
            personalized_email=insights['personalized_email'],
            critical_factors=insights['critical_factors'],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple customers.
    Optimized for high-throughput processing.
    """
    start_time = time.time()
    
    try:
        results = []
        
        for customer_data in request.customers:
            # Convert to dict
            data_dict = customer_data.dict()
            
            # Process data
            customer_df = pd.DataFrame([data_dict])
            X_processed = data_engine.transform(customer_df)
            
            # Make prediction
            churn_probability = model_engine.model.predict_proba(X_processed)[0, 1]
            
            # Determine risk level
            if churn_probability > 0.7:
                risk_level = "High"
            elif churn_probability > 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            results.append({
                'customer_id': customer_data.customerID,
                'churn_probability': float(churn_probability),
                'churn_risk': risk_level
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=results,
            total_customers=len(results),
            processing_time_ms=processing_time,
            avg_time_per_customer_ms=processing_time / len(results) if results else 0
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/importance")
async def feature_importance():
    """Get feature importance information."""
    if model_engine is None or model_engine.feature_importance is None:
        raise HTTPException(status_code=503, detail="Feature importance not available")
    
    # Create feature importance mapping
    importance_dict = dict(zip(
        data_engine.feature_names,
        model_engine.feature_importance.tolist()
    ))
    
    # Sort by importance
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        "feature_importance": dict(sorted_features),
        "top_features": [f[0] for f in sorted_features[:10]]
    }

@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics."""
    return {
        "api_version": "1.0.0",
        "model_type": "LightGBM with Calibration",
        "target_response_time_ms": "< 20",
        "features_count": len(data_engine.feature_names) if data_engine else 0,
        "model_accuracy": model_engine.cv_scores.get('mean', 0) if model_engine else 0
    }

if __name__ == "__main__":
    import uvicorn
    
    # Validate configuration
    Config.validate()
    
    # Run the application
    uvicorn.run(
        "api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
        workers=1,  # Single worker for optimal performance
        log_level="info"
    )
