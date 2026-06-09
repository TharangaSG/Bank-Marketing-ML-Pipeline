"""
Feature encoding strategies for the Bank Marketing ML Pipeline.
Supports one-hot, label, and ordinal encoding.
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEncodingStrategy(ABC):
    """Abstract base class for feature encoding strategies."""
    
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode features in the DataFrame."""
        pass


class BinaryEncodingStrategy(FeatureEncodingStrategy):
    """
    Binary encoding strategy for yes/no columns.
    Maps 'yes' -> 1, 'no' -> 0.
    """
    
    def __init__(self, binary_columns: List[str]):
        self.binary_columns = binary_columns
        self.encoder_dicts = {}
        logger.info(f"BinaryEncodingStrategy initialized for columns: {binary_columns}")
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply binary encoding (yes=1, no=0)."""
        logger.info(f"\n{'='*60}")
        logger.info(f"BINARY ENCODING")
        logger.info(f"{'='*60}")
        
        df_encoded = df.copy()
        
        for col in self.binary_columns:
            if col not in df_encoded.columns:
                logger.warning(f"  ⚠ Column '{col}' not found")
                continue
            
            unique_vals = df_encoded[col].unique()
            logger.info(f"  {col}: unique values = {unique_vals}")
            
            mapping = {'yes': 1, 'no': 0}
            df_encoded[col] = df_encoded[col].map(mapping)
            self.encoder_dicts[col] = mapping
            
            logger.info(f"  ✓ Encoded '{col}': yes→1, no→0")
        
        logger.info(f"✓ BINARY ENCODING COMPLETE")
        logger.info(f"{'='*60}\n")
        return df_encoded


class NominalEncodingStrategy(FeatureEncodingStrategy):
    """
    Nominal encoding strategy using one-hot encoding.
    Creates binary columns for each category.
    """
    
    def __init__(self, nominal_columns: List[str], drop_first: bool = False):
        self.nominal_columns = nominal_columns
        self.drop_first = drop_first
        self.encoder_dicts = {}
        logger.info(f"NominalEncodingStrategy initialized for columns: {nominal_columns}")
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding to nominal columns."""
        logger.info(f"\n{'='*60}")
        logger.info(f"ONE-HOT ENCODING")
        logger.info(f"{'='*60}")
        
        df_encoded = df.copy()
        
        for col in self.nominal_columns:
            if col not in df_encoded.columns:
                logger.warning(f"  ⚠ Column '{col}' not found")
                continue
            
            unique_values = sorted(df_encoded[col].dropna().unique())
            logger.info(f"  {col}: {len(unique_values)} unique values")
            
            # Save encoder info
            self.encoder_dicts[col] = {
                'categories': list(unique_values),
                'encoder_type': 'one_hot'
            }
            
            # Create one-hot columns
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=self.drop_first)
            # Ensure integer type
            dummies = dummies.astype(int)
            
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            
            created_cols = list(dummies.columns)
            logger.info(f"  ✓ Created {len(created_cols)} binary columns for '{col}'")
        
        logger.info(f"✓ ONE-HOT ENCODING COMPLETE")
        logger.info(f"{'='*60}\n")
        return df_encoded
    
    def get_encoder_dicts(self) -> Dict:
        """Get the encoder dictionaries for all columns."""
        return self.encoder_dicts


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    """
    Ordinal encoding strategy with custom ordering.
    Maps categorical values to ordered numeric values.
    """
    
    def __init__(self, ordinal_mappings: Dict[str, Dict[str, int]]):
        self.ordinal_mappings = ordinal_mappings
        logger.info(f"OrdinalEncodingStrategy initialized for columns: {list(ordinal_mappings.keys())}")
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ordinal encoding."""
        logger.info(f"\n{'='*60}")
        logger.info(f"ORDINAL ENCODING")
        logger.info(f"{'='*60}")
        
        df_encoded = df.copy()
        
        for col, mapping in self.ordinal_mappings.items():
            if col not in df_encoded.columns:
                logger.warning(f"  ⚠ Column '{col}' not found")
                continue
            
            logger.info(f"  {col}: Mapping = {mapping}")
            
            # Before encoding distribution
            before_dist = df_encoded[col].value_counts().to_dict()
            logger.info(f"  Before: {before_dist}")
            
            df_encoded[col] = df_encoded[col].map(mapping)
            
            # Handle unmapped values
            unmapped = df_encoded[col].isnull().sum()
            if unmapped > 0:
                logger.warning(f"  ⚠ {unmapped} unmapped values in '{col}', filling with -1")
                df_encoded[col] = df_encoded[col].fillna(-1).astype(int)
            else:
                df_encoded[col] = df_encoded[col].astype(int)
            
            after_dist = df_encoded[col].value_counts().to_dict()
            logger.info(f"  ✓ Encoded '{col}': {after_dist}")
        
        logger.info(f"✓ ORDINAL ENCODING COMPLETE")
        logger.info(f"{'='*60}\n")
        return df_encoded


class TargetEncodingStrategy(FeatureEncodingStrategy):
    """
    Encode the target variable y: 'yes' -> 1, 'no' -> 0.
    """
    
    def __init__(self, target_column: str = 'y'):
        self.target_column = target_column
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode the target variable."""
        df_encoded = df.copy()
        
        if self.target_column in df_encoded.columns:
            mapping = {'yes': 1, 'no': 0}
            df_encoded[self.target_column] = df_encoded[self.target_column].map(mapping)
            logger.info(f"  ✓ Target '{self.target_column}' encoded: yes→1, no→0")
        
        return df_encoded
