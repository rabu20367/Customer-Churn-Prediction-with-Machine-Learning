"""
Complete training pipeline with automated optimization.
One-command model training and evaluation.
"""
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_engine import DataEngine
from model_engine import ModelEngine
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str = None) -> pd.DataFrame:
    """Load customer churn dataset."""
    if file_path is None:
        file_path = Config.DATA_PATH
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        logger.info("Please download the dataset from Kaggle and place it in the data/ folder")
        logger.info("Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully: {df.shape}")
    return df

def train_model():
    """Complete model training pipeline."""
    logger.info("🚀 Starting model training pipeline...")
    
    # 1. Load and process data
    logger.info("📊 Loading and processing data...")
    df = load_data()
    
    data_engine = DataEngine()
    X, y = data_engine.process(df)
    
    logger.info(f"Processed data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Churn rate: {y.mean():.2%}")
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Train model
    logger.info("🤖 Training optimized model...")
    model_engine = ModelEngine(n_trials=Config.OPTUNA_TRIALS)
    
    # Optimize hyperparameters
    logger.info("🔧 Optimizing hyperparameters...")
    model_engine.optimize_hyperparameters(X_train, y_train)
    
    # Train final model
    logger.info("🎯 Training final model...")
    model_engine.train_final_model(X_train, y_train)
    
    # 4. Evaluate model
    logger.info("📈 Evaluating model performance...")
    evaluation_results = model_engine.evaluate_model(X_test, y_test)
    
    # Print results
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Test AUC Score: {evaluation_results['auc_score']:.4f}")
    logger.info(f"CV AUC Score: {model_engine.cv_scores['mean']:.4f} (+/- {model_engine.cv_scores['std'] * 2:.4f})")
    
    # Classification report
    classification_rep = evaluation_results['classification_report']
    logger.info("\nClassification Report:")
    logger.info(f"Precision: {classification_rep['1']['precision']:.4f}")
    logger.info(f"Recall: {classification_rep['1']['recall']:.4f}")
    logger.info(f"F1-Score: {classification_rep['1']['f1-score']:.4f}")
    
    # Top features
    if evaluation_results['feature_importance']:
        logger.info("\nTop 10 Most Important Features:")
        sorted_features = sorted(
            evaluation_results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            logger.info(f"{i:2d}. {feature}: {importance:.4f}")
    
    # 5. Save model and data engine
    logger.info("💾 Saving model and components...")
    model_engine.save_model(Config.MODEL_PATH)
    
    # Save data engine
    os.makedirs('models', exist_ok=True)
    joblib.dump(data_engine, 'models/data_engine.pkl', compress=3)
    
    logger.info("✅ Training pipeline completed successfully!")
    logger.info(f"Model saved to: {Config.MODEL_PATH}")
    logger.info(f"Data engine saved to: models/data_engine.pkl")
    
    return model_engine, data_engine, evaluation_results

def main():
    """Main training function."""
    try:
        # Validate configuration
        Config.validate()
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Train model
        model_engine, data_engine, results = train_model()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Test AUC Score: {results['auc_score']:.4f}")
        print(f"🎯 CV AUC Score: {model_engine.cv_scores['mean']:.4f}")
        print(f"⚡ Model Type: LightGBM with Calibration")
        print(f"🔧 Features: {len(data_engine.feature_names)}")
        print(f"💾 Model saved to: {Config.MODEL_PATH}")
        print("\n🚀 Ready for deployment by Hasibur!")
        print("Run: python api.py")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
