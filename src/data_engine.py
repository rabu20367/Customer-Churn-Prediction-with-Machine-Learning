"""
Ultra-efficient data pipeline with maximum impact features.
Optimized for single-pass processing and minimal memory usage.
Developed by Hasibur.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataEngine:
    """
    High-performance data processing engine.
    Creates only the most impactful features for churn prediction.
    """
    
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: list = []
    
    def create_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features with highest churn correlation.
        Based on domain expertise and feature importance analysis.
        """
        df_features = df.copy()
        
        # Behavioral risk score - composite feature with highest impact
        df_features['behavioral_risk'] = (
            0.35 * (df['Contract'] == 'Month-to-month').astype(int) +
            0.25 * (df['tenure'] < 12).astype(int) +
            0.20 * (df['PaymentMethod'] == 'Electronic check').astype(int) +
            0.15 * (df['OnlineSecurity'] == 'No').astype(int) +
            0.05 * (df['TechSupport'] == 'No').astype(int)
        )
        
        # Value engagement ratio - spending vs tenure relationship
        df_features['value_engagement'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )
        
        # Service density - premium service usage
        premium_services = ['OnlineSecurity', 'TechSupport', 'DeviceProtection', 'OnlineBackup']
        df_features['service_density'] = df[premium_services].apply(
            lambda x: sum(val != 'No' for val in x), axis=1
        )
        
        # Streaming engagement - entertainment service usage
        streaming_services = ['StreamingTV', 'StreamingMovies']
        df_features['streaming_engagement'] = df[streaming_services].apply(
            lambda x: sum(val == 'Yes' for val in x), axis=1
        )
        
        # Contract risk multiplier
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        df_features['contract_risk'] = df['Contract'].map(contract_risk)
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} impact features")
        return df_features
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete data processing pipeline in single pass.
        Optimized for memory efficiency and speed.
        """
        # Handle missing values efficiently
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
        # Create impact features
        df_processed = self.create_impact_features(df)
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['Churn', 'customerID']]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.encoders[col] = le
        
        # Prepare final dataset
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['Churn', 'customerID']]
        X = df_processed[feature_cols]
        y = df_processed['Churn'].map({'Yes': 1, 'No': 0})
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        self.feature_names = X_scaled.columns.tolist()
        
        logger.info(f"Processing complete: {X_scaled.shape[1]} features, {len(y)} samples")
        return X_scaled, y
    
    def transform_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data for inference using fitted encoders and scaler.
        """
        # Handle missing values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
        # Create impact features
        df_processed = self.create_impact_features(df)
        
        # Encode categorical variables (only those that were encoded during training)
        for col, encoder in self.encoders.items():
            if col in df_processed.columns and col not in ['customerID']:
                df_processed[col] = encoder.transform(df_processed[col].astype(str))
        
        # Select and scale features in the exact order expected by the scaler
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df_processed.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X = df_processed[self.feature_names]
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders and scaler.
        For inference on new customer data.
        """
        return self.transform_for_inference(df)
    
    def get_feature_importance_data(self) -> Dict[str, Any]:
        """Get feature information for model interpretation."""
        return {
            'feature_names': self.feature_names,
            'categorical_encoders': list(self.encoders.keys()),
            'scaler_fitted': hasattr(self.scaler, 'scale_')
        }
