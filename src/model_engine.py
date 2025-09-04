"""
Ultra-optimized LightGBM training with aggressive hyperparameter tuning.
Single-model focus for maximum performance and minimal complexity.
Developed by Hasibur.
"""
try:
    import optuna
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, classification_report
    import joblib
except ImportError as e:
    print(f"Import error in model_engine: {e}")
    # These imports are handled at runtime
import logging
from typing import Dict, Any, Tuple, Optional
import os

logger = logging.getLogger(__name__)

class ModelEngine:
    """
    High-performance model training engine.
    Focuses exclusively on LightGBM for maximum efficiency.
    """
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
        self.model: Optional[CalibratedClassifierCV] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_scores: Dict[str, float] = {}
        self.feature_importance: Optional[np.ndarray] = None
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Aggressive hyperparameter optimization using Optuna.
        Optimized for both speed and accuracy.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 15),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            
            # Use stratified CV for better performance estimation
            cv_scores = cross_val_score(
                model, X, y, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Create study with pruning for efficiency
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        logger.info(f"Best parameters found: {self.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        
        return self.best_params
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
        """
        Train final model with calibrated probabilities.
        Uses isotonic calibration for better probability estimates.
        """
        # Train base model with best parameters
        base_model = lgb.LGBMClassifier(**self.best_params)
        base_model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = base_model.feature_importances_
        
        # Calibrate probabilities using isotonic regression
        self.model = CalibratedClassifierCV(
            base_model, 
            method='isotonic', 
            cv=3
        )
        self.model.fit(X, y)
        
        # Calculate final CV score
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        self.cv_scores = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        logger.info(f"Final model CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        Returns detailed performance metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_final_model first.")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'auc_score': auc_score,
            'classification_report': classification_rep,
            'feature_importance': dict(zip(
                X_test.columns, 
                self.feature_importance
            )) if self.feature_importance is not None else None
        }
        
        logger.info(f"Test AUC Score: {auc_score:.4f}")
        
        return evaluation_results
    
    def save_model(self, model_path: str = 'models/churn_model.pkl') -> None:
        """
        Save model and metadata with compression.
        Includes all necessary components for inference.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path, compress=3)
        
        # Save metadata
        metadata = {
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'model_type': 'CalibratedClassifierCV',
            'base_model': 'LGBMClassifier'
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path, compress=3)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str = 'models/churn_model.pkl') -> CalibratedClassifierCV:
        """
        Load trained model and metadata.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_params = metadata.get('best_params')
            self.cv_scores = metadata.get('cv_scores')
            self.feature_importance = np.array(metadata.get('feature_importance')) if metadata.get('feature_importance') else None
        
        logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        """
        return {
            'model_type': type(self.model).__name__ if self.model else None,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'feature_count': len(self.feature_importance) if self.feature_importance is not None else 0,
            'top_features': self._get_top_features(10) if self.feature_importance is not None else None
        }
    
    def _get_top_features(self, n: int = 10) -> list:
        """
        Get top N most important features with names.
        """
        if self.feature_importance is None:
            return []
        
        # Get top feature indices
        top_indices = np.argsort(self.feature_importance)[-n:][::-1]
        
        # Return feature names if available, otherwise indices
        if hasattr(self, 'feature_names') and self.feature_names:
            return [self.feature_names[i] for i in top_indices]
        else:
            return top_indices.tolist()
    
    def set_feature_names(self, feature_names: list):
        """Set feature names for better interpretability."""
        self.feature_names = feature_names
