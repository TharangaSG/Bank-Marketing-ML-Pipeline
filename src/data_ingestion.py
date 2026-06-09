"""
Data ingestion module for Bank Marketing ML Pipeline.
Supports CSV data loading with comprehensive logging and validation.
Follows the Factory pattern from the sample project.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    """Abstract base class for data ingestion."""
    
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest data from the specified path.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            pandas DataFrame
        """
        pass


class DataIngestorCSV(DataIngestor):
    """CSV data ingestion implementation with comprehensive logging."""
    
    def ingest(self, file_path: str, **options) -> pd.DataFrame:
        """Ingest CSV data using pandas."""
        start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📥 DATA INGESTION - CSV (PANDAS)")
        logger.info(f"{'='*60}")
        logger.info(f"🐼 Engine: Pandas")
        logger.info(f"🔗 Source: {file_path}")
        logger.info(f"⚙️ Options: {options if options else 'Default pandas options'}")
        
        try:
            df = pd.read_csv(file_path, **options)
            
            # Enhanced logging with detailed metrics
            load_time = time.time() - start_time
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            
            logger.info(f"✅ CSV INGESTION COMPLETED")
            logger.info(f"📊 Dataset metrics:")
            logger.info(f"  • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            logger.info(f"  • Memory usage: {memory_mb:.2f} MB")
            logger.info(f"  • Load time: {load_time:.2f} seconds")
            logger.info(f"  • Throughput: {df.shape[0]/load_time:,.0f} rows/second")
            logger.info(f"🔍 Column overview:")
            logger.info(f"  • Numeric: {len(df.select_dtypes(include=[np.number]).columns)} columns")
            logger.info(f"  • Text/Object: {len(df.select_dtypes(include=['object']).columns)} columns")
            logger.info(f"  • Sample columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load CSV data from {file_path}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorExcel(DataIngestor):
    """Excel data ingestion implementation."""
    
    def ingest(self, file_path: str, sheet_name: Optional[str] = None, **options) -> pd.DataFrame:
        """Ingest Excel data using pandas."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📥 DATA INGESTION - EXCEL (PANDAS)")
        logger.info(f"{'='*60}")
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **options)
            logger.info(f"✅ Excel ingestion completed - Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"✗ Failed to load Excel data: {str(e)}")
            raise


class DataIngestorFactory:
    """Factory class to create appropriate data ingestor based on file type."""
    
    @staticmethod
    def get_ingestor(file_path: str) -> DataIngestor:
        """
        Get appropriate data ingestor based on file extension.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataIngestor: Appropriate ingestor instance
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return DataIngestorCSV()
        elif file_extension in ['.xlsx', '.xls']:
            return DataIngestorExcel()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
