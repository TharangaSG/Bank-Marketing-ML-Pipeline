"""
Model training module for the Bank Marketing ML Pipeline.
Supports training, cross-validation, and model saving.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.model_evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Manages the training process for sklearn models."""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        logger.info(f"ModelTrainer initialized with {cv_folds}-fold CV")
        
    def train(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, float]:
        """
        Train a model and perform cross-validation.
        
        Args:
            model: Sklearn model instance
            X_train: Training features
            y_train: Training target
            
        Returns:
            Tuple of (trained_model, cv_score)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL TRAINING")
        logger.info(f"{'='*60}")
        
        # Cross validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        logger.info(f"Performing {self.cv_folds}-fold cross validation...")
        
        # Use ROC AUC for imbalanced datasets
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
        mean_cv_score = cv_scores.mean()
        
        logger.info(f"CV ROC AUC Scores: {cv_scores}")
        logger.info(f"Mean CV ROC AUC: {mean_cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Fit on full training data
        logger.info(f"Training final model on full training set ({len(X_train)} samples)...")
        model.fit(X_train, y_train)
        
        # Get training metrics
        train_pred = model.predict(X_train)
        if hasattr(model, "predict_proba"):
            train_proba = model.predict_proba(X_train)[:, 1]
        else:
            train_proba = None
            
        evaluator = ModelEvaluator(model, model.__class__.__name__)
        train_metrics = evaluator.calculate_metrics(y_train, train_pred, train_proba)
        
        logger.info("Training Set Metrics:")
        for metric, value in train_metrics.items():
            if isinstance(value, float):
                logger.info(f"  • {metric}: {value:.4f}")
        
        logger.info(f"✓ MODEL TRAINING COMPLETE")
        logger.info(f"{'='*60}\n")
        
        return model, mean_cv_score
