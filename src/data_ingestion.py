"""
Data ingestion module for Bank Marketing ML Pipeline.
Supports CSV data loading with comprehensive logging and validation.
Follows the Factory pattern from the sample project, with PySpark compatibility.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd
import numpy as np

# Manual PySpark availability flag
PYSPARK_AVAILABLE = True

if PYSPARK_AVAILABLE:
    try:
        from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
    except ImportError:
        PYSPARK_AVAILABLE = False
        SparkDataFrame = None
        SparkSession = None
else:
    SparkDataFrame = None
    SparkSession = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    """Abstract base class for data ingestion supporting both pandas and PySpark."""
    
    def __init__(self, spark: Optional['SparkSession'] = None):
        if PYSPARK_AVAILABLE and spark:
            self.spark = spark
        else:
            self.spark = None

    @abstractmethod
    def ingest(self, file_path: str, **options) -> Union[pd.DataFrame, 'SparkDataFrame']:
        pass


class DataIngestorCSV(DataIngestor):
    """CSV data ingestion implementation supporting both pandas and PySpark."""
    
    def ingest(self, file_path: str, **options) -> Union[pd.DataFrame, 'SparkDataFrame']:
        if not PYSPARK_AVAILABLE or self.spark is None:
            return self._ingest_pandas(file_path, **options)
        else:
            return self._ingest_pyspark(file_path, **options)
            
    def _ingest_pandas(self, file_path: str, **options) -> pd.DataFrame:
        start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📥 DATA INGESTION - CSV (PANDAS)")
        logger.info(f"{'='*60}")
        logger.info(f"🐼 Engine: Pandas")
        logger.info(f"🔗 Source: {file_path}")
        
        try:
            df = pd.read_csv(file_path, **options)
            load_time = time.time() - start_time
            
            logger.info(f"✅ CSV INGESTION COMPLETED")
            logger.info(f"📊 Dataset metrics:")
            logger.info(f"  • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load CSV data from {file_path}: {str(e)}")
            raise

    def _ingest_pyspark(self, file_path: str, **options) -> 'SparkDataFrame':
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - CSV (PySpark)")
        logger.info(f"{'='*60}")
        
        try:
            csv_options = {
                "header": "true",
                "inferSchema": "true",
                "sep": options.get("sep", ",")
            }
            df = self.spark.read.options(**csv_options).csv(file_path)
            
            row_count = df.count()
            logger.info(f"✓ Successfully loaded CSV data using PySpark - Shape: ({row_count}, {len(df.columns)})")
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load CSV data via PySpark: {str(e)}")
            raise


class DataIngestorFactory:
    """Factory class to create appropriate data ingestor based on file type."""
    
    @staticmethod
    def get_ingestor(file_path: str, spark: Optional['SparkSession'] = None) -> DataIngestor:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return DataIngestorCSV(spark)
        else:
            raise ValueError(f"Unsupported file type for PySpark compatible ingestor: {file_extension}")
