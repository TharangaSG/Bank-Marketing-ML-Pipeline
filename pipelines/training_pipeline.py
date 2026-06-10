"""
Bank Marketing ML Pipeline - Model Training Pipeline
"""

import os
import sys
import yaml
import logging
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_building import ModelBuilderFactory
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.mlflow_utils import MLflowTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def training_pipeline():
    config = load_config()
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Initialize MLflow
    mlflow_config = config.get('mlflow', {})
    tracker = None
    if mlflow_config:
        tracker = MLflowTracker(
            tracking_uri=mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db'),
            experiment_name=mlflow_config.get('experiment_name', 'bank_marketing')
        )
        tracker.start_run(run_name=mlflow_config.get('run_name_prefix', 'run') + '_training',
                         tags={"model_type": config['model']['model_type']})
    
    # 1. Load processed data
    logger.info("Loading processed data...")
    try:
        X_train = pd.read_csv(config['data_paths']['X_train'])
        X_test = pd.read_csv(config['data_paths']['X_test'])
        y_train = pd.read_csv(config['data_paths']['Y_train']).iloc[:, 0]
        y_test = pd.read_csv(config['data_paths']['Y_test']).iloc[:, 0]
        logger.info(f"Loaded X_train shape: {X_train.shape}")
    except FileNotFoundError:
        logger.error("Processed data not found. Please run the data pipeline first.")
        return
        
    # 2. Configure model
    model_type = config['model']['model_type']
    model_params = config['model']['sklearn_model_types'].get(model_type, {})
    
    if tracker:
        tracker.log_params(model_params)
        tracker.log_params({"cv_folds": config['training']['cv_folds']})
    
    logger.info(f"Building {model_type} model...")
    builder = ModelBuilderFactory.get_builder(model_type, **model_params)
    model = builder.build_model()
    
    # 3. Train model
    trainer = ModelTrainer(
        cv_folds=config['training']['cv_folds'],
        random_state=config['training']['random_state']
    )
    
    trained_model, cv_score = trainer.train(model, X_train, y_train)
    
    # 4. Save model
    os.makedirs(config['data_paths']['train_artifacts_dir'], exist_ok=True)
    model_path = config['model']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    builder.save_model(model_path)
    
    # 5. Evaluate model
    evaluator = ModelEvaluator(trained_model, model_type)
    metrics = evaluator.evaluate(X_test, y_test)
    
    # 6. Log metrics and model to MLflow
    if tracker:
        # log metrics (excluding complex objects like confusion matrix or feature importances list)
        simple_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        simple_metrics["cv_mean_roc_auc"] = cv_score
        tracker.log_metrics(simple_metrics)
        
        reg_name = mlflow_config.get('model_registry_name') if mlflow_config.get('register_model') else None
        tracker.log_model(trained_model, registered_model_name=reg_name)
        tracker.end_run()
    
    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    training_pipeline()
