"""
Outlier detection strategies for the Bank Marketing ML Pipeline.
Uses IQR-based detection with Strategy pattern from sample project.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OutlierDetectionStrategy(ABC):
    """Abstract base class for outlier detection strategies."""
    
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Detect outliers in specified columns."""
        pass
    
    @abstractmethod
    def get_outlier_bounds(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """Get outlier bounds for specified columns."""
        pass


class IQROutlierDetection(OutlierDetectionStrategy):
    """IQR-based outlier detection strategy."""
    
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
        logger.info(f"Initialized IQROutlierDetection with threshold: {threshold}")
    
    def get_outlier_bounds(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """Calculate outlier bounds using IQR method."""
        bounds = {}
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            
            bounds[col] = (lower_bound, upper_bound)
            
            logger.info(f"  Column '{col}': Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
            logger.info(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return bounds
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        logger.info(f"\n{'='*60}")
        logger.info(f"OUTLIER DETECTION - IQR METHOD")
        logger.info(f"{'='*60}")
        logger.info(f"Starting IQR outlier detection for columns: {columns}")
        
        bounds = self.get_outlier_bounds(df, columns)
        result_df = df.copy()
        total_outliers = 0
        
        for col in columns:
            lower_bound, upper_bound = bounds[col]
            outlier_mask = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
            result_df[f"{col}_outlier"] = outlier_mask
            
            outlier_count = outlier_mask.sum()
            outlier_percentage = (outlier_count / len(result_df)) * 100
            
            logger.info(f"  ✓ {col}: Found {outlier_count} outliers ({outlier_percentage:.2f}%)")
            total_outliers += outlier_count
        
        logger.info(f"\n{'='*60}")
        logger.info(f'✓ OUTLIER DETECTION COMPLETE - Total outlier instances: {total_outliers}')
        logger.info(f"{'='*60}\n")
        
        return result_df


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    """Z-Score based outlier detection strategy."""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        logger.info(f"Initialized ZScoreOutlierDetection with threshold: {threshold}")
    
    def get_outlier_bounds(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """Calculate outlier bounds using Z-Score method."""
        bounds = {}
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            lower = mean - self.threshold * std
            upper = mean + self.threshold * std
            bounds[col] = (lower, upper)
        return bounds
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Detect outliers using Z-Score method."""
        result_df = df.copy()
        for col in columns:
            mean = result_df[col].mean()
            std = result_df[col].std()
            z_scores = np.abs((result_df[col] - mean) / std)
            result_df[f"{col}_outlier"] = z_scores > self.threshold
        return result_df


class OutlierDetector:
    """Main outlier detector class that uses different strategies."""
    
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy
        logger.info(f"OutlierDetector initialized with strategy: {strategy.__class__.__name__}")
    
    def detect_outliers(self, df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
        """Detect outliers in selected columns."""
        logger.info(f"Detecting outliers in {len(selected_columns)} columns")
        return self._strategy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df: pd.DataFrame, selected_columns: List[str], 
                       method: str = 'remove', min_outliers: int = 2) -> pd.DataFrame:
        """
        Handle outliers using specified method.
        
        Args:
            df: DataFrame
            selected_columns: Columns to check for outliers
            method: 'remove' to drop outlier rows, 'cap' to clip at bounds
            min_outliers: Minimum outlier columns to trigger row removal
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"OUTLIER HANDLING - {method.upper()}")
        logger.info(f"{'='*60}")
        
        initial_rows = len(df)
        
        if method == 'remove':
            df_with_outliers = self.detect_outliers(df, selected_columns)
            outlier_columns = [f"{col}_outlier" for col in selected_columns]
            
            outlier_count = df_with_outliers[outlier_columns].sum(axis=1)
            mask = outlier_count < min_outliers
            cleaned_df = df_with_outliers[mask].drop(columns=outlier_columns)
            
            rows_removed = initial_rows - len(cleaned_df)
            removal_pct = (rows_removed / initial_rows * 100) if initial_rows > 0 else 0
            
            logger.info(f"✓ Removed {rows_removed} rows with {min_outliers}+ outliers ({removal_pct:.2f}%)")
            logger.info(f"✓ Remaining rows: {len(cleaned_df)}")
            
        elif method == 'cap':
            bounds = self._strategy.get_outlier_bounds(df, selected_columns)
            cleaned_df = df.copy()
            
            for col in selected_columns:
                lower_bound, upper_bound = bounds[col]
                capped = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                n_capped = (cleaned_df[col] != capped).sum()
                cleaned_df[col] = capped
                
                if n_capped > 0:
                    logger.info(f"  ✓ Capped {n_capped} values in '{col}' to [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            logger.info(f"✓ Capped outliers at IQR bounds for {len(selected_columns)} columns")
            
        else:
            raise ValueError(f"Unknown outlier handling method: {method}")
        
        logger.info(f"{'='*60}\n")
        return cleaned_df
