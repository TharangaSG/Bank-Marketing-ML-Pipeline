"""
MLflow utilities for the Bank Marketing ML Pipeline.
Handles MLflow tracking, parameter/metric logging, and model registry.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLflowTracker:
    """Manages MLflow tracking lifecycle and operations."""
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize the MLflow tracker.
        
        Args:
            tracking_uri: URI for the MLflow tracking server
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.active_run = None
        
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Configure MLflow with tracking URI and experiment name."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Check if experiment exists, if not create it
            client = MlflowClient(tracking_uri=self.tracking_uri)
            experiment = client.get_experiment_by_name(self.experiment_name)
            
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None):
        """Start an MLflow run."""
        if run_name:
            run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.active_run = mlflow.start_run(run_name=run_name)
        
        if tags:
            mlflow.set_tags(tags)
            
        logger.info(f"Started MLflow run: {self.active_run.info.run_id}")
        return self.active_run
        
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the active run."""
        if not self.active_run:
            logger.warning("No active MLflow run to log params to.")
            return
            
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters to MLflow")
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to the active run."""
        if not self.active_run:
            logger.warning("No active MLflow run to log metrics to.")
            return
            
        # Ensure values are castable to float
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                clean_metrics[k] = float(v)
                
        mlflow.log_metrics(clean_metrics)
        logger.info(f"Logged {len(clean_metrics)} metrics to MLflow")

    def log_model(self, model: Any, artifact_path: str = "model", registered_model_name: Optional[str] = None):
        """Log sklearn model to MLflow."""
        if not self.active_run:
            logger.warning("No active MLflow run to log model to.")
            return
            
        logger.info(f"Logging model to MLflow (artifact_path: {artifact_path})...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info("Model successfully logged to MLflow")
        
    def end_run(self):
        """End the active MLflow run."""
        if self.active_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
            self.active_run = None
