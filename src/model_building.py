"""
Model building module for the Bank Marketing ML Pipeline.
"""

import os
import joblib
import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseModelBuilder(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs
    
    @abstractmethod
    def build_model(self):
        pass
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save.")
        import src.gcs_utils as gcs_utils
        gcs_utils.save_artifact(self.model, filepath)
    
    def load_model(self, filepath: str):
        import src.gcs_utils as gcs_utils
        self.model = gcs_utils.load_artifact(filepath)
        return self.model


class RandomForestModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        defaults = {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5,
                    'min_samples_leaf': 2, 'random_state': 42, 'class_weight': 'balanced', 'n_jobs': 1}
        defaults.update(kwargs)
        super().__init__('RandomForest', **defaults)

    def build_model(self):
        self.model = RandomForestClassifier(**self.model_params)
        return self.model


class XGBoostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        defaults = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                    'random_state': 42, 'scale_pos_weight': 7, 'eval_metric': 'logloss'}
        defaults.update(kwargs)
        super().__init__('XGBoost', **defaults)

    def build_model(self):
        self.model = XGBClassifier(**self.model_params)
        return self.model


class GradientBoostingModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        defaults = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}
        defaults.update(kwargs)
        super().__init__('GradientBoosting', **defaults)

    def build_model(self):
        self.model = GradientBoostingClassifier(**self.model_params)
        return self.model


class LogisticRegressionModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        defaults = {'random_state': 42, 'max_iter': 1000, 'class_weight': 'balanced'}
        defaults.update(kwargs)
        super().__init__('LogisticRegression', **defaults)

    def build_model(self):
        self.model = LogisticRegression(**self.model_params)
        return self.model


class ModelBuilderFactory:
    _builders = {
        'random_forest': RandomForestModelBuilder,
        'gradient_boosting': GradientBoostingModelBuilder,
        'xgboost': XGBoostModelBuilder,
        'logistic_regression': LogisticRegressionModelBuilder,
    }
    
    @classmethod
    def get_builder(cls, model_type: str, **kwargs) -> BaseModelBuilder:
        if model_type not in cls._builders:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._builders[model_type](**kwargs)
