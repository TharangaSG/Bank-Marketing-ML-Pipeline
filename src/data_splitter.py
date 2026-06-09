"""
Data splitting strategies for the Bank Marketing ML Pipeline.
Supports simple and stratified train-test splits.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplittingStrategy(ABC):
    """Abstract base class for data splitting strategies."""
    
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Simple random train-test split strategy."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        logger.info(f"SimpleTrainTestSplitStrategy initialized with test_size={test_size}")

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform simple random train-test split."""
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA SPLITTING - SIMPLE")
        logger.info(f"{'='*60}")
        logger.info(f"Target column: '{target_column}'")
        logger.info(f"Total samples: {len(df)}, Features: {len(df.columns) - 1}")
        
        Y = df[target_column]
        X = df.drop(columns=[target_column])
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"✓ Data split completed:")
        logger.info(f"  • Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  • Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        logger.info(f"  • Features: {X_train.shape[1]}")
        logger.info(f"{'='*60}\n")
        
        return X_train, X_test, Y_train, Y_test


class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    """Stratified train-test split strategy to maintain class distribution."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        logger.info(f"StratifiedTrainTestSplitStrategy initialized with test_size={test_size}")

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform stratified train-test split."""
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA SPLITTING - STRATIFIED")
        logger.info(f"{'='*60}")
        logger.info(f"Target column: '{target_column}'")
        logger.info(f"Total samples: {len(df)}, Features: {len(df.columns) - 1}")
        
        # Log target distribution
        target_dist = df[target_column].value_counts()
        logger.info(f"\nTarget Variable Distribution:")
        for value, count in target_dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {value}: {count:,} ({pct:.1f}%)")
        
        Y = df[target_column]
        X = df.drop(columns=[target_column])
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state, stratify=Y
        )
        
        # Log split distributions
        logger.info(f"\nSplit Results:")
        logger.info(f"  ✓ Training set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  ✓ Test set: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        logger.info(f"\nTarget Distribution in Training Set:")
        for value, count in Y_train.value_counts().items():
            logger.info(f"  {value}: {count:,} ({count/len(Y_train)*100:.1f}%)")
        
        logger.info(f"\nTarget Distribution in Test Set:")
        for value, count in Y_test.value_counts().items():
            logger.info(f"  {value}: {count:,} ({count/len(Y_test)*100:.1f}%)")
        
        logger.info(f"\n✓ STRATIFIED DATA SPLITTING COMPLETE")
        logger.info(f"{'='*60}\n")
        
        return X_train, X_test, Y_train, Y_test


class DataSplitter:
    """Main data splitter class that uses different strategies."""
    
    def __init__(self, strategy: DataSplittingStrategy):
        self.strategy = strategy
        logger.info(f"DataSplitter initialized with strategy: {strategy.__class__.__name__}")
    
    def split(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data using the configured strategy."""
        return self.strategy.split_data(df, target_column)
