"""
Bank Marketing ML Pipeline - Inference Pipeline
"""

import os
import sys
import yaml
import logging
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_inference import ModelInference
import src.gcs_utils as gcs_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def inference_pipeline():
    config = load_config()
    
    logger.info("=" * 80)
    logger.info("STARTING INFERENCE PIPELINE")
    logger.info("=" * 80)
    
    try:
        model_path = config['model']['model_path']
        preprocessors_dir = config['data_paths']['data_artifacts_dir']
        
        # In this simplistic pipeline, X_test is already preprocessed.
        # In a real deployment, inference pipeline receives raw JSON and preprocesses it.
        # For demonstration, we load X_test as batch data.
        data_path = config['inference']['data_path']
        
        logger.info("Initializing inference engine...")
        inference = ModelInference(model_path, preprocessors_dir)
        
        logger.info(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        sample_size = config['inference']['sample_size']
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
            
        logger.info(f"Running inference on {len(df)} records...")
        results = inference.predict(df)
        
        save_path = config['inference']['save_path']
        gcs_utils.ensure_dir(gcs_utils.dirname(save_path))
        results.to_csv(save_path, index=False)
        logger.info(f"Predictions saved to {save_path}")
        
        logger.info("=" * 80)
        logger.info("INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}")
        raise

if __name__ == "__main__":
    inference_pipeline()
