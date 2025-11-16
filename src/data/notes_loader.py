"""
MIMIC-IV Clinical Notes Loader
Carrega e processa notas clínicas do MIMIC-IV-Note
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotesLoader:
    """
    Carrega notas clínicas do MIMIC-IV-Note com filtros temporais e por categoria.
    
    Notas disponíveis:
    - discharge.csv: Resumos de alta hospitalar
    - radiology.csv: Laudos de radiologia
    
    Arquitetura permite adicionar mais tipos facilmente.
    """
    
    # Categorias de notas suportadas
    SUPPORTED_CATEGORIES = {
        'discharge': {
            'file': 'discharge.csv',
            'description': 'Discharge Summaries',
            'priority': 1
        },
        'radiology': {
            'file': 'radiology.csv', 
            'description': 'Radiology Reports',
            'priority': 2
        }
    }
    
    def __init__(self, data_root: str = '../data/raw'):
        self.data_root = Path(data_root)
        self.notes_cache = {}
        
        # Validar existência dos arquivos
        self._validate_files()
    
    def _validate_files(self):
        """Valida se os arquivos de notas existem."""
        available = []
        missing = []
        
        for cat, info in self.SUPPORTED_CATEGORIES.items():
            filepath = self.data_root / info['file']
            if filepath.exists():
                available.append(cat)
            else:
                missing.append(f"{cat} ({filepath})")
        
        if available:
            logger.info(f"✓ Categorias disponíveis: {', '.join(available)}")
        
        if missing:
            logger.warning(f"⚠️  Arquivos não encontrados:")
            for m in missing:
                logger.warning(f"    - {m}")
    
    def load_notes(
        self,
        categories: Optional[List[str]] = None,
        subject_ids: Optional[List[int]] = None,
        hadm_ids: Optional[List[int]] = None,
        chunksize: int = 10000
    ) -> pd.DataFrame:
        """
        Carrega notas em chunks para economizar memória.
        """
        if categories is None:
            categories = list(self.SUPPORTED_CATEGORIES.keys())
        
        all_notes = []
        
        for cat in categories:
            filepath = self.data_root / self.SUPPORTED_CATEGORIES[cat]['file']
            
            if not filepath.exists():
                logger.warning(f"⏭️ Pulando {cat}: arquivo não encontrado")
                continue
            
            logger.info(f"📖 Carregando {cat} notes em chunks de {chunksize:,}...")
            
            # CORREÇÃO: Primeiro ler sample para detectar colunas disponíveis
            sample = pd.read_csv(filepath, nrows=5)
            available_cols = sample.columns.tolist()
            
            logger.debug(f"   Colunas disponíveis: {available_cols}")
            
            # Definir colunas desejadas com prioridade
            desired_cols = {
                'subject_id': ('int32', False),
                'hadm_id': ('Int32', False),
                'charttime': (None, True),   # ⭐ TIMESTAMP PRINCIPAL
                'chartdate': (None, True),   # ⭐ TIMESTAMP ALTERNATIVO
                'storetime': (None, True),   # ⭐ TIMESTAMP BACKUP
                'text': (str, False),
                'note_id': (str, False),
                'note_type': (str, False),
                'note_seq': ('Int32', False)
            }
            
            # Selecionar apenas colunas que existem
            usecols = []
            dtype_map = {}
            parse_dates = []
            
            for col, (dtype, is_date) in desired_cols.items():
                if col in available_cols:
                    usecols.append(col)
                    if dtype:
                        dtype_map[col] = dtype
                    if is_date:
                        parse_dates.append(col)
            
            logger.info(f"   Carregando colunas: {usecols}")
            if parse_dates:
                logger.info(f"   Parseando datas: {parse_dates}")
            
            chunks_processed = 0
            rows_kept = 0
            
            # Ler em chunks
            chunk_iterator = pd.read_csv(
                filepath, 
                usecols=usecols, 
                dtype=dtype_map,
                parse_dates=parse_dates if parse_dates else None,
                chunksize=chunksize, 
                low_memory=False
            )
            
            for chunk in chunk_iterator:
                # Aplicar filtros no chunk
                if subject_ids is not None:
                    chunk = chunk[chunk['subject_id'].isin(subject_ids)]
                
                if hadm_ids is not None:
                    chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]
                
                # Remover notas vazias
                chunk = chunk[chunk['text'].notna() & (chunk['text'].str.len() > 0)]
                
                if len(chunk) > 0:
                    chunk['note_category'] = cat
                    chunk['category_priority'] = self.SUPPORTED_CATEGORIES[cat]['priority']
                    all_notes.append(chunk)
                    rows_kept += len(chunk)
                
                chunks_processed += 1
                if chunks_processed % 10 == 0:
                    logger.info(f"   Processados {chunks_processed} chunks, {rows_kept:,} notas mantidas")
            
            logger.info(f"✓ {cat}: {rows_kept:,} notas carregadas")
        
        if not all_notes:
            raise ValueError("Nenhuma nota foi carregada")
        
        combined = pd.concat(all_notes, ignore_index=True)
        
        # Log de colunas finais
        logger.info(f"📋 Colunas no dataset final: {combined.columns.tolist()}")
        
        # Verificar timestamps disponíveis
        time_cols = ['charttime', 'chartdate', 'storetime']
        available_time_cols = [col for col in time_cols if col in combined.columns]
        
        if available_time_cols:
            logger.info(f"✓ Timestamps disponíveis: {available_time_cols}")
            for col in available_time_cols:
                non_null = combined[col].notna().sum()
                pct = non_null / len(combined) * 100
                logger.info(f"  - {col}: {non_null:,} ({pct:.1f}%) não-nulos")
        else:
            logger.warning("⚠️ NENHUM timestamp disponível nas notas!")
        
        logger.info(f"✅ Total: {len(combined):,} notas")
        
        return combined
    
    def filter_temporal(
        self,
        notes_df: pd.DataFrame,
        edstays_df: pd.DataFrame,
        time_buffer_hours: float = 0
        ) -> pd.DataFrame:
            """
            Filtra notas que estão disponíveis ANTES ou DURANTE o ED stay.
            
            Critério: charttime <= (ED discharge time + buffer)
            """
            logger.info("⏱️ Aplicando filtro temporal nas notas...")
            
            # Verificar se temos timestamps
            if 'charttime' not in notes_df.columns and 'chartdate' not in notes_df.columns:
                logger.warning("⚠️ Sem timestamps disponíveis, pulando filtro temporal")
                return notes_df
            
            # Usar charttime se disponível, senão usar chartdate
            time_col = 'charttime' if 'charttime' in notes_df.columns else 'chartdate'
            
            # Verificar se hadm_id existe em ambos dataframes
            if 'hadm_id' not in notes_df.columns:
                logger.error("❌ Coluna 'hadm_id' não encontrada nas notas")
                return notes_df
            
            if 'hadm_id' not in edstays_df.columns:
                logger.error("❌ Coluna 'hadm_id' não encontrada em edstays")
                return notes_df
            
            # Remover notas sem hadm_id (não podem ser vinculadas)
            notes_with_hadm = notes_df[notes_df['hadm_id'].notna()].copy()
            initial_count = len(notes_with_hadm)
            
            logger.info(f"  - Notas com hadm_id válido: {initial_count:,}")
            
            # Merge com ED stays para obter timestamps
            df = notes_with_hadm.merge(
                edstays_df[['subject_id', 'hadm_id', 'intime', 'outtime']],
                on=['subject_id', 'hadm_id'],
                how='inner'
            )
            
            merge_count = len(df)
            logger.info(f"  - Notas após merge com ED stays: {merge_count:,}")
            
            if merge_count == 0:
                logger.warning("⚠️ Nenhuma nota após merge! Verificar compatibilidade de IDs")
                return pd.DataFrame()
            
            # Converter para datetime
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df['intime'] = pd.to_datetime(df['intime'], errors='coerce')
            df['outtime'] = pd.to_datetime(df['outtime'], errors='coerce')
            
            # Remover linhas com timestamps inválidos
            df = df.dropna(subset=[time_col, 'intime', 'outtime'])
            
            logger.info(f"  - Notas com timestamps válidos: {len(df):,}")
            
            # Calcular tempo limite (ED discharge + buffer)
            df['time_limit'] = df['outtime'] + pd.Timedelta(hours=time_buffer_hours)
            
            # Filtrar: nota deve ser registrada antes do tempo limite
            df['is_available'] = df[time_col] <= df['time_limit']
            df_filtered = df[df['is_available']].copy()
            
            # Calcular tempo relativo (minutos desde admissão ED)
            df_filtered['time_from_admission_min'] = (
                (df_filtered[time_col] - df_filtered['intime']).dt.total_seconds() / 60
            )
            
            # Limpar colunas auxiliares temporárias
            df_filtered = df_filtered.drop(columns=['time_limit', 'is_available', 'intime', 'outtime'])
            
            filtered_count = len(df_filtered)
            filtered_pct = filtered_count / initial_count * 100 if initial_count > 0 else 0
            
            logger.info(f"✓ Filtro temporal: {initial_count:,} → {filtered_count:,} notas "
                    f"({filtered_pct:.1f}% mantidas)")
            
            return df_filtered
    
    def get_notes_for_stay(
        self,
        subject_id: int,
        hadm_id: int,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Busca todas as notas de um ED stay específico.
        """
        notes = self.load_notes(
            categories=categories,
            subject_ids=[subject_id],
            hadm_ids=[hadm_id]
        )
        
        # Ordenar por timestamp se disponível
        if 'charttime' in notes.columns:
            notes = notes.sort_values('charttime')
        elif 'chartdate' in notes.columns:
            notes = notes.sort_values('chartdate')
        
        return notes
    
    def get_stats(self, notes_df: pd.DataFrame) -> Dict:
        """
        Gera estatísticas descritivas das notas.
        """
        stats = {
            'total_notes': len(notes_df),
            'unique_patients': notes_df['subject_id'].nunique(),
        }
        
        # Admissions (se disponível)
        if 'hadm_id' in notes_df.columns:
            stats['unique_admissions'] = notes_df['hadm_id'].nunique()
        
        # Por categoria
        if 'note_category' in notes_df.columns:
            stats['by_category'] = notes_df['note_category'].value_counts().to_dict()
        
        # Tamanho do texto
        stats['avg_text_length'] = notes_df['text'].str.len().mean()
        stats['median_text_length'] = notes_df['text'].str.len().median()
        
        # Intervalo de datas (se disponível)
        if 'charttime' in notes_df.columns:
            stats['date_range'] = {
                'earliest': notes_df['charttime'].min(),
                'latest': notes_df['charttime'].max()
            }
        elif 'chartdate' in notes_df.columns:
            stats['date_range'] = {
                'earliest': notes_df['chartdate'].min(),
                'latest': notes_df['chartdate'].max()
            }
        
        return stats
    
    def inspect_file_structure(self, category: str) -> pd.DataFrame:
        """
        Utilitário para inspecionar estrutura de um arquivo de notas.
        Útil para debug.
        """
        filepath = self.data_root / self.SUPPORTED_CATEGORIES[category]['file']
        
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        # Ler apenas primeiras linhas
        sample = pd.read_csv(filepath, nrows=10)
        
        print(f"\n{'='*60}")
        print(f"ESTRUTURA: {category}")
        print(f"{'='*60}")
        print(f"Arquivo: {filepath}")
        print(f"\nColunas ({len(sample.columns)}):")
        for col in sample.columns:
            dtype = sample[col].dtype
            non_null = sample[col].notna().sum()
            print(f"  - {col:20s} | {str(dtype):15s} | {non_null}/10 não-nulos")
        
        print(f"\nPrimeiras linhas:")
        print(sample.head(3))
        
        return sample