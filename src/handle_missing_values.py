"""
Missing value handling strategies for the Bank Marketing ML Pipeline.
Uses Strategy pattern from the sample project.
In the bank dataset, "unknown" values in categorical columns are treated as missing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Union

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MissingValueHandlingStrategy(ABC):
    """Abstract base class for missing value handling strategies."""
    
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """Strategy to drop rows with missing values in critical columns."""
    
    def __init__(self, critical_columns: List[str] = None):
        self.critical_columns = critical_columns or []
        logger.info(f"Initialized DropMissingValuesStrategy for columns: {self.critical_columns}")

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values in critical columns."""
        initial_count = len(df)
        
        if self.critical_columns:
            df_cleaned = df.dropna(subset=self.critical_columns)
        else:
            df_cleaned = df.dropna()
        
        final_count = len(df_cleaned)
        n_dropped = initial_count - final_count
        
        logger.info(f"✓ Dropped {n_dropped} rows with missing values")
        logger.info(f"  • Initial rows: {initial_count}")
        logger.info(f"  • Final rows: {final_count}")
        
        return df_cleaned


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to fill missing values using various methods.
    Supports mean/median/mode filling.
    """
    
    def __init__(
        self, 
        method: str = 'mean', 
        fill_value: Optional[Union[str, float, int]] = None, 
        relevant_column: Optional[str] = None
    ):
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values based on the configured strategy."""
        df_filled = df.copy()
        
        if self.relevant_column:
            col = self.relevant_column
            missing_before = df_filled[col].isnull().sum()
            
            if missing_before == 0:
                logger.info(f"  ✓ No missing values in {col}")
                return df_filled
            
            if self.method == 'mean':
                fill_val = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(fill_val)
                logger.info(f"  ✓ Filled {missing_before} missing values in '{col}' with mean: {fill_val:.2f}")
                
            elif self.method == 'median':
                fill_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(fill_val)
                logger.info(f"  ✓ Filled {missing_before} missing values in '{col}' with median: {fill_val:.2f}")
                
            elif self.method == 'mode':
                fill_val = df_filled[col].mode()[0]
                df_filled[col] = df_filled[col].fillna(fill_val)
                logger.info(f"  ✓ Filled {missing_before} missing values in '{col}' with mode: {fill_val}")
                
            elif self.method == 'constant' and self.fill_value is not None:
                df_filled[col] = df_filled[col].fillna(self.fill_value)
                logger.info(f"  ✓ Filled {missing_before} missing values in '{col}' with constant: {self.fill_value}")
                
            else:
                raise ValueError(f"Invalid method '{self.method}' or missing fill_value")
        else:
            # Fill all columns based on type
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            categorical_cols = df_filled.select_dtypes(include=['object']).columns
            
            for col in numeric_cols:
                missing = df_filled[col].isnull().sum()
                if missing > 0:
                    if self.method in ['mean', 'median']:
                        fill_val = df_filled[col].mean() if self.method == 'mean' else df_filled[col].median()
                        df_filled[col] = df_filled[col].fillna(fill_val)
                        logger.info(f"  ✓ Filled {missing} values in '{col}' with {self.method}: {fill_val:.2f}")
            
            for col in categorical_cols:
                missing = df_filled[col].isnull().sum()
                if missing > 0:
                    fill_val = df_filled[col].mode()[0]
                    df_filled[col] = df_filled[col].fillna(fill_val)
                    logger.info(f"  ✓ Filled {missing} values in '{col}' with mode: {fill_val}")
        
        return df_filled


class UnknownValueHandler(MissingValueHandlingStrategy):
    """
    Special strategy for the Bank dataset: replace 'unknown' categorical values 
    with the mode of that column (excluding 'unknown').
    """
    
    def __init__(self, columns: List[str] = None, replacement_strategy: str = 'mode'):
        self.columns = columns or []
        self.replacement_strategy = replacement_strategy
        logger.info(f"Initialized UnknownValueHandler for columns: {self.columns}")
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace 'unknown' values with appropriate fill values."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 HANDLING 'UNKNOWN' VALUES")
        logger.info(f"{'='*60}")
        
        df_handled = df.copy()
        total_replaced = 0
        
        for col in self.columns:
            if col not in df_handled.columns:
                logger.warning(f"  ⚠ Column '{col}' not found in DataFrame")
                continue
            
            unknown_count = (df_handled[col] == 'unknown').sum()
            
            if unknown_count == 0:
                logger.info(f"  ✓ No 'unknown' values in '{col}'")
                continue
            
            if self.replacement_strategy == 'mode':
                # Calculate mode excluding 'unknown'
                non_unknown = df_handled[df_handled[col] != 'unknown'][col]
                fill_val = non_unknown.mode()[0] if len(non_unknown.mode()) > 0 else 'unknown'
                df_handled[col] = df_handled[col].replace('unknown', fill_val)
                logger.info(f"  ✓ Replaced {unknown_count} 'unknown' values in '{col}' with mode: '{fill_val}'")
            elif self.replacement_strategy == 'keep':
                logger.info(f"  ✓ Keeping {unknown_count} 'unknown' values in '{col}' (will be encoded)")
            
            total_replaced += unknown_count
        
        logger.info(f"\n✅ UNKNOWN VALUE HANDLING COMPLETE")
        logger.info(f"  • Total 'unknown' values processed: {total_replaced}")
        logger.info(f"{'='*60}\n")
        
        return df_handled
