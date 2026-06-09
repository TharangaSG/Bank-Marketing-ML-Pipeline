"""
Bank Marketing ML Pipeline - Data Processing Pipeline
"""

import os
import sys
import yaml
import logging
import pandas as pd
import joblib
from typing import Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion import DataIngestorFactory
from src.handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, UnknownValueHandler
from src.outlier_detection import OutlierDetector, IQROutlierDetection
from src.feature_encoding import BinaryEncodingStrategy, NominalEncodingStrategy, OrdinalEncodingStrategy
from src.feature_scaling import StandardScalingStrategy
from src.data_splitter import StratifiedTrainTestSplitStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def data_pipeline(force_rebuild: bool = False):
    """Run the end-to-end data processing pipeline."""
    config = load_config()
    
    logger.info("=" * 80)
    logger.info("STARTING DATA PIPELINE")
    logger.info("=" * 80)
    
    # Ensure directories exist
    os.makedirs(config['data_paths']['processed_dir'], exist_ok=True)
    os.makedirs(config['data_paths']['data_artifacts_dir'], exist_ok=True)
    
    # 1. Ingestion
    data_path = config['data_paths']['raw_data']
    ingestor = DataIngestorFactory.get_ingestor(data_path)
    df = ingestor.ingest(data_path, sep=config['data']['delimiter'])
    
    # Drop irrelevant columns if any
    cols_to_drop = config['columns'].get('drop_columns', [])
    if cols_to_drop:
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        logger.info(f"Dropped columns: {cols_to_drop}")

    # 2. Handle missing/unknown values
    unknown_cols = ['job', 'education', 'contact', 'poutcome']
    unknown_handler = UnknownValueHandler(columns=unknown_cols, replacement_strategy='mode')
    df = unknown_handler.handle(df)
    
    # 3. Outlier detection and handling
    outlier_cols = config['columns']['outlier_columns']
    outlier_detector = OutlierDetector(IQROutlierDetection(threshold=config['outlier_detection']['iqr_multiplier']))
    df = outlier_detector.handle_outliers(df, outlier_cols, method=config['outlier_detection']['handling_method'])
    
    # 4. Target encoding
    target_col = config['columns']['target']
    df[target_col] = df[target_col].map({'yes': 1, 'no': 0})
    logger.info(f"Encoded target '{target_col}'")
    
    # 5. Feature Encoding
    # Binary
    binary_cols = config['feature_encoding']['binary_columns']
    binary_encoder = BinaryEncodingStrategy(binary_cols)
    df = binary_encoder.encode(df)
    
    # Ordinal
    ordinal_mappings = config['feature_encoding']['ordinal_mappings']
    ordinal_encoder = OrdinalEncodingStrategy(ordinal_mappings)
    df = ordinal_encoder.encode(df)
    
    # Nominal (One-Hot)
    nominal_cols = config['feature_encoding']['nominal_columns']
    nominal_encoder = NominalEncodingStrategy(nominal_cols, drop_first=True)
    df = nominal_encoder.encode(df)
    
    # Save encoders for inference
    encoders = {
        'binary': binary_encoder.encoder_dicts,
        'nominal': nominal_encoder.get_encoder_dicts(),
        'ordinal': ordinal_mappings
    }
    joblib.dump(encoders, os.path.join(config['data_paths']['data_artifacts_dir'], 'encoders.pkl'))
    
    # 6. Feature Scaling
    numeric_cols = config['feature_scaling']['columns_to_scale']
    scaler_strategy = StandardScalingStrategy()
    df = scaler_strategy.scale(df, numeric_cols)
    
    # Save scaler
    joblib.dump(scaler_strategy.get_scaler(), os.path.join(config['data_paths']['data_artifacts_dir'], 'scaler.pkl'))
    
    # 7. Data Splitting
    splitter = StratifiedTrainTestSplitStrategy(
        test_size=config['data_splitting']['test_size'],
        random_state=config['data_splitting']['random_state']
    )
    X_train, X_test, y_train, y_test = splitter.split_data(df, target_col)
    
    # Save processed data
    logger.info("Saving processed datasets...")
    X_train.to_csv(config['data_paths']['X_train'], index=False)
    X_test.to_csv(config['data_paths']['X_test'], index=False)
    y_train.to_csv(config['data_paths']['Y_train'], index=False)
    y_test.to_csv(config['data_paths']['Y_test'], index=False)
    
    logger.info("=" * 80)
    logger.info("DATA PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    data_pipeline()
