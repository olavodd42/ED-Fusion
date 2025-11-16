"""
Módulos de processamento de dados MIMIC-IV-ED
"""

from .mimic_loader import MIMICLoader
from .preprocessing import EDPreprocessor
from .feature_engineering import FeatureEngineer
from .labeling import OutcomeLabeler
from .notes_loader import NotesLoader
from .text_preprocessing import TextPreprocessor, TextConfig
from .text_integration import TextTabularIntegrator, FusionStrategy, create_stratified_splits

__all__ = [
    'MIMICLoader',
    'EDPreprocessor', 
    'FeatureEngineer',
    'OutcomeLabeler',
    'NotesLoader',
    'TextPreprocessor',
    'TextConfig',
    'TextTabularIntegrator',
    'FusionStrategy',
    'create_stratified_splits'
]