"""
Pré-processamento de texto clínico
"""
import pandas as pd
import numpy as np
import re
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextConfig:
    """Configurações de pré-processamento de texto"""
    max_tokens_per_segment: int = 512
    min_text_length: int = 50
    remove_special_chars: bool = False
    lowercase: bool = False


class TextPreprocessor:
    """
    Pré-processador de texto clínico.
    
    Operações:
    - Remoção de PHI markers
    - Limpeza de formatação
    - Segmentação de notas longas
    """
    
    def __init__(self, config: TextConfig):
        self.config = config
        
    def clean_text(self, text: str) -> str:
        """Limpa texto de uma nota"""
        
        if pd.isna(text) or not text:
            return ""
        
        # Remover markers PHI comuns
        text = re.sub(r'\[\*\*[^\]]+\*\*\]', ' [REDACTED] ', text)
        
        # Remover múltiplos espaços
        text = re.sub(r'\s+', ' ', text)
        
        # Remover caracteres especiais se configurado
        if self.config.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\:\;\-]', '', text)
        
        # Lowercase se configurado
        if self.config.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def segment_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Segmenta texto em chunks de tamanho máximo.
        Aproximação: 1 token ~= 4 caracteres
        """
        
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return [text]
        
        # Dividir por sentenças
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= max_chars:
                current_segment += " " + sentence
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def preprocess_dataframe(self, 
                            df: pd.DataFrame,
                            text_column: str = 'text') -> pd.DataFrame:
        """Pré-processa DataFrame de notas"""
        
        logger.info(f"🧹 Pré-processando {len(df):,} notas...")
        
        df = df.copy()
        
        # Limpar texto
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Estatísticas
        df['original_length'] = df[text_column].str.len()
        df['cleaned_length'] = df['cleaned_text'].str.len()
        df['is_valid'] = df['cleaned_length'] >= self.config.min_text_length
        
        logger.info(f"✓ Notas válidas: {df['is_valid'].sum():,}/{len(df):,}")
        
        return df