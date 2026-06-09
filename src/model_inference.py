"""
Model inference module for the Bank Marketing ML Pipeline.
Handles loading models, preprocessing data, and making predictions.
"""

import logging
import joblib
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    """Class to handle data preprocessing and predictions for inference."""
    
    def __init__(self, model_path: str, preprocessors_dir: str):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model
            preprocessors_dir: Directory containing saved encoders and scalers
        """
        self.model_path = model_path
        self.preprocessors_dir = preprocessors_dir
        self.model = None
        self.scaler = None
        self.encoders = {}
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load model and preprocessing artifacts."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            scaler_path = os.path.join(self.preprocessors_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
                
            encoders_path = os.path.join(self.preprocessors_dir, 'encoders.pkl')
            if os.path.exists(encoders_path):
                self.encoders = joblib.load(encoders_path)
                logger.info(f"Loaded encoders from {encoders_path}")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise

    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data exactly as done during training.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame ready for inference
        """
        df = data.copy()
        
        # This needs to exactly mirror the training data pipeline
        # For a robust implementation, these steps should use the saved encoders
        # Or ideally a Scikit-Learn Pipeline would contain everything
        
        logger.warning("preprocess_input: Implement specific preprocessing based on loaded artifacts.")
        # E.g.
        # if 'job' in df.columns and 'job' in self.encoders:
        #    ...
        
        return df

    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
        """
        Make predictions on input data.
        
        Args:
            input_data: Single record dictionary or DataFrame
            
        Returns:
            Predictions (dict for single record, DataFrame for batch)
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
            is_single = True
        else:
            df = input_data.copy()
            is_single = False
            
        # Optional: preprocess input here if the model expects scaled/encoded data
        # For pipeline models, preprocessing is built-in
        # processed_df = self.preprocess_input(df)
        
        logger.info(f"Making predictions for {len(df)} records")
        predictions = self.model.predict(df)
        
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df)[:, 1]
        else:
            probabilities = [None] * len(predictions)
            
        if is_single:
            return {
                'prediction': int(predictions[0]),
                'probability': float(probabilities[0]) if probabilities[0] is not None else None
            }
        else:
            results = df.copy()
            results['prediction'] = predictions
            results['probability'] = probabilities
            return results
