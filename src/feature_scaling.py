"""
Feature scaling strategies for the Bank Marketing ML Pipeline.
Supports StandardScaler and MinMaxScaler.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureScalingStrategy(ABC):
    """Abstract base class for feature scaling strategies."""
    
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Scale specified columns in the DataFrame."""
        pass
    
    @abstractmethod
    def get_scaler(self):
        """Return the fitted scaler object."""
        pass


class StandardScalingStrategy(FeatureScalingStrategy):
    """Standard scaling (z-score normalization) strategy."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns_scaled = []
        logger.info("StandardScalingStrategy initialized")
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Apply standard scaling to specified columns."""
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE SCALING - STANDARD (Z-SCORE)")
        logger.info(f"{'='*60}")
        
        df_scaled = df.copy()
        available_cols = [col for col in columns_to_scale if col in df_scaled.columns]
        self.columns_scaled = available_cols
        
        logger.info(f"Scaling {len(available_cols)} columns: {available_cols}")
        
        # Log before scaling
        logger.info(f"\nStatistics BEFORE scaling:")
        for col in available_cols:
            stats = df_scaled[col].describe()
            logger.info(f"  {col}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, "
                       f"Min={stats['min']:.2f}, Max={stats['max']:.2f}")
        
        # Fit and transform
        df_scaled[available_cols] = self.scaler.fit_transform(df_scaled[available_cols])
        
        # Log after scaling
        logger.info(f"\nStatistics AFTER scaling:")
        for col in available_cols:
            stats = df_scaled[col].describe()
            logger.info(f"  {col}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
                       f"Min={stats['min']:.4f}, Max={stats['max']:.4f}")
        
        logger.info(f"\n✓ STANDARD SCALING COMPLETE - {len(available_cols)} columns processed")
        logger.info(f"{'='*60}\n")
        
        return df_scaled
    
    def get_scaler(self):
        return self.scaler


class MinMaxScalingStrategy(FeatureScalingStrategy):
    """Min-Max scaling strategy to scale features to [0, 1] range."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.columns_scaled = []
        logger.info("MinMaxScalingStrategy initialized")
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Apply Min-Max scaling to specified columns."""
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE SCALING - MIN-MAX")
        logger.info(f"{'='*60}")
        
        df_scaled = df.copy()
        available_cols = [col for col in columns_to_scale if col in df_scaled.columns]
        self.columns_scaled = available_cols
        
        logger.info(f"Scaling {len(available_cols)} columns: {available_cols}")
        
        # Log before
        logger.info(f"\nStatistics BEFORE scaling:")
        for col in available_cols:
            logger.info(f"  {col}: Min={df_scaled[col].min():.2f}, Max={df_scaled[col].max():.2f}")
        
        # Fit and transform
        df_scaled[available_cols] = self.scaler.fit_transform(df_scaled[available_cols])
        
        # Log after
        logger.info(f"\nStatistics AFTER scaling:")
        for col in available_cols:
            logger.info(f"  {col}: Min={df_scaled[col].min():.4f}, Max={df_scaled[col].max():.4f}")
        
        logger.info(f"\n✓ MIN-MAX SCALING COMPLETE - {len(available_cols)} columns processed")
        logger.info(f"{'='*60}\n")
        
        return df_scaled
    
    def get_scaler(self):
        return self.scaler
