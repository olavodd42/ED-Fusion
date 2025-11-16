"""
Integração de dados textuais com tabulares
"""
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Estratégias de fusão texto-tabular"""
    CONCATENATE = "concatenate"  # Concatenar embeddings
    ATTENTION = "attention"       # Cross-attention
    HIERARCHICAL = "hierarchical" # Hierárquico


class TextTabularIntegrator:
    """
    Integra dados textuais com features tabulares.
    """
    
    def __init__(self, strategy: FusionStrategy = FusionStrategy.CONCATENATE):
        self.strategy = strategy
        
    def associate_notes_to_stays(self,
                                 df_tabular: pd.DataFrame,
                                 notes: pd.DataFrame,
                                 strategy: str = 'priority') -> pd.DataFrame:
        """
        Associa notas aos ED stays.
        
        Args:
            df_tabular: Features tabulares
            notes: Notas pré-processadas
            strategy: 'first' | 'last' | 'priority' | 'concat'
        """
        
        logger.info(f"🔗 Associando notas aos stays (strategy={strategy})...")
        
        if strategy == 'priority':
            # Prioridade: discharge > radiology > nursing
            priority_order = ['Discharge summary', 'Radiology', 'Nursing']
            
            # Ordenar por prioridade
            notes['priority'] = notes['note_type'].map(
                {t: i for i, t in enumerate(priority_order)}
            ).fillna(999)
            
            # Pegar nota de maior prioridade por hadm_id
            notes_agg = notes.sort_values('priority').groupby(
                ['subject_id', 'hadm_id']
            ).first().reset_index()
            
        elif strategy == 'first':
            notes_agg = notes.sort_values('note_time').groupby(
                ['subject_id', 'hadm_id']
            ).first().reset_index()
            
        elif strategy == 'last':
            notes_agg = notes.sort_values('note_time').groupby(
                ['subject_id', 'hadm_id']
            ).last().reset_index()
            
        elif strategy == 'concat':
            # Concatenar todas as notas
            notes_agg = notes.groupby(['subject_id', 'hadm_id']).agg({
                'cleaned_text': lambda x: ' [SEP] '.join(x),
                'note_type': lambda x: ', '.join(set(x))
            }).reset_index()
        
        # Merge com tabular
        df_integrated = df_tabular.merge(
            notes_agg[['subject_id', 'hadm_id', 'cleaned_text', 'note_type']],
            on=['subject_id', 'hadm_id'],
            how='left'
        )
        
        logger.info(f"✓ {df_integrated['cleaned_text'].notna().sum():,} stays com texto")
        
        return df_integrated
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Estatísticas da integração"""
        
        has_text = df['cleaned_text'].notna()
        
        stats = {
            'total_stays': len(df),
            'with_text': has_text.sum(),
            'text_coverage': has_text.mean(),
            'avg_text_length': df.loc[has_text, 'cleaned_text'].str.len().mean(),
            'tabular_features': len([c for c in df.columns if not c.startswith('note')])
        }
        
        if 'note_type' in df.columns:
            stats['by_category'] = df[has_text]['note_type'].value_counts().to_dict()
        
        return stats
    
    def create_multimodal_dataset(self,
                                  df: pd.DataFrame,
                                  text_column: str = 'cleaned_text',
                                  outcome_column: str = 'critical_outcome') -> pd.DataFrame:
        """
        Cria dataset final multi-modal.
        """
        
        logger.info("🎨 Criando dataset multi-modal...")
        
        # Selecionar features
        feature_cols = [c for c in df.columns if c.startswith('triage_') or c.startswith('lab_')]
        
        dataset = df[[
            'subject_id', 'hadm_id', 'stay_id',
            text_column, outcome_column
        ] + feature_cols].copy()
        
        # Flags
        dataset['has_text'] = dataset[text_column].notna()
        dataset = dataset.rename(columns={outcome_column: 'outcome'})
        
        logger.info(f"✓ Dataset criado: {dataset.shape}")
        
        return dataset


def create_stratified_splits(df: pd.DataFrame,
                             outcome_col: str = 'outcome',
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1,
                             test_ratio: float = 0.1,
                             random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Cria splits estratificados.
    """
    
    from sklearn.model_selection import train_test_split
    
    # Train + val/test
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df[outcome_col],
        random_state=random_state
    )
    
    # Val / test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df[outcome_col],
        random_state=random_state
    )
    
    logger.info(f"✓ Splits criados: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }