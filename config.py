"""
Configuration management for the churn prediction system.
Developed by Hasibur.
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration management."""
    
    # API Keys
    GROQ_API_KEY: Optional[str] = os.getenv('GROQ_API_KEY')
    
    # Model Configuration
    MODEL_PATH: str = 'models/churn_model.pkl'
    OPTUNA_TRIALS: int = 100
    
    # API Configuration
    API_HOST: str = '0.0.0.0'
    API_PORT: int = 8000
    
    # Data Configuration
    DATA_PATH: str = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.GROQ_API_KEY:
            print("Warning: GROQ_API_KEY not set. LLM features will be disabled.")
            return False
        return True
