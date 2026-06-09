"""
Model evaluation module for the Bank Marketing ML Pipeline.
Computes various classification metrics.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates trained models and calculates performance metrics."""
    
    def __init__(self, model: Any, model_name: str):
        self.model = model
        self.model_name = model_name
        logger.info(f"ModelEvaluator initialized for {model_name}")
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                logger.warning("Could not calculate ROC AUC score")
                
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL EVALUATION - {self.model_name}")
        logger.info(f"{'='*60}")
        
        logger.info(f"Evaluating on {len(X_test)} test samples...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None
            
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        logger.info("Test Set Metrics:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix' and isinstance(value, float):
                logger.info(f"  • {metric}: {value:.4f}")
                
        # Generate classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"\nClassification Report:\n{report}")
        
        # Feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': X_test.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Important Features:")
            for idx, row in feature_imp.head(10).iterrows():
                logger.info(f"  • {row['feature']}: {row['importance']:.4f}")
                
            metrics['feature_importance'] = feature_imp.to_dict('records')
            
        logger.info(f"✓ MODEL EVALUATION COMPLETE")
        logger.info(f"{'='*60}\n")
        
        return metrics
