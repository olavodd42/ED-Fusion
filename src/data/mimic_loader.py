"""
Módulo otimizado para carregar dados do MIMIC-IV
Versão com estratégias avançadas para labevents
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import logging
import numpy as np
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICLoader:
    """Carrega dados do MIMIC-IV de forma extremamente otimizada"""
    
    def __init__(self, data_root: str = "./data/raw"):
        self.data_root = Path(data_root)
        self.data = {}
        
    def load_all(self, 
                 load_labs: bool = False, 
                 load_vitals: bool = True,
                 sample_size: Optional[int] = None,
                 lab_strategy: str = 'filtered') -> Dict[str, pd.DataFrame]:
        """
        Carrega datasets essenciais com estratégias otimizadas.
        
        Parâmetros:
            load_labs: bool -> Se True, carrega labevents
            load_vitals: bool -> Se True, carrega vitalsign
            sample_size: Optional[int] -> Amostragem de ED stays
            lab_strategy: str -> 'sample' | 'filtered' | 'full'
                - 'sample': 10M registros (testes rápidos)
                - 'filtered': Apenas 68 labs relevantes (recomendado)
                - 'full': Tudo (requer 32GB+ RAM)
        """
        logger.info("="*60)
        logger.info("🚀 MIMIC-IV Loader Otimizado v2.0")
        logger.info("="*60)
        
        # Core
        self.data['patients'] = self._load_csv_optimized(
            'core/patients.csv',
            usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod'],
            sample_size=sample_size
        )
        logger.info(f"✓ Pacientes: {len(self.data['patients']):,}")
        
        # ED stays
        self.data['edstays'] = self._load_csv_optimized(
            'ed/edstays.csv',
            sample_size=sample_size
        )
        logger.info(f"✓ ED stays: {len(self.data['edstays']):,}")
        
        # Obter IDs para filtragem
        stay_ids = self.data['edstays']['stay_id'].unique() if sample_size else None
        subject_ids = self.data['edstays']['subject_id'].unique()
        
        # Triage
        self.data['triage'] = self._load_csv_optimized(
            'ed/triage.csv',
            filter_col='stay_id' if stay_ids is not None else None,
            filter_values=stay_ids
        )
        logger.info(f"✓ Triage: {len(self.data['triage']):,}")
        
        # Vitals
        if load_vitals:
            self.data['vitalsign'] = self._load_vitalsign_optimized(stay_ids)
        else:
            self.data['vitalsign'] = pd.DataFrame()
        
        # Diagnosis & Medrecon
        self.data['diagnosis'] = self._load_csv_optimized(
            'ed/diagnosis.csv',
            filter_col='stay_id' if stay_ids is not None else None,
            filter_values=stay_ids
        )
        logger.info(f"✓ Diagnosis: {len(self.data['diagnosis']):,}")
        
        self.data['medrecon'] = self._load_csv_optimized(
            'ed/medrecon.csv',
            filter_col='stay_id' if stay_ids is not None else None,
            filter_values=stay_ids
        )
        logger.info(f"✓ Medrecon: {len(self.data['medrecon']):,}")
        
        # Hospital
        self.data['admissions'] = self._load_csv_optimized(
            'hosp/admissions.csv',
            filter_col='subject_id',
            filter_values=subject_ids
        )
        logger.info(f"✓ Admissions: {len(self.data['admissions']):,}")
        
        hadm_ids = self.data['admissions']['hadm_id'].unique()
        self.data['diagnoses_icd'] = self._load_csv_optimized(
            'hosp/diagnoses_icd.csv',
            filter_col='hadm_id',
            filter_values=hadm_ids,
            usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version']
        )
        logger.info(f"✓ Diagnoses ICD: {len(self.data['diagnoses_icd']):,}")
        
        # Lab dictionary
        self.data['d_labitems'] = self._load_csv_optimized('hosp/d_labitems.csv')
        logger.info(f"✓ Lab items dict: {len(self.data['d_labitems']):,}")
        
        # Lab events - ESTRATÉGIAS OTIMIZADAS
        if load_labs:
            logger.info("\n" + "="*60)
            logger.info(f"🧪 CARREGANDO LABEVENTS - Estratégia: {lab_strategy.upper()}")
            logger.info("="*60)
            
            if lab_strategy == 'sample':
                logger.info("📊 Modo SAMPLE: Primeiros 10M registros")
                self.data['labevents'] = self._load_labevents_optimized(
                    subject_ids,
                    max_rows=10_000_000
                )
            
            elif lab_strategy == 'filtered':
                try:
                    from .lab_config import ALL_RELEVANT_ITEMIDS
                    logger.info(f"🎯 Modo FILTERED: {len(ALL_RELEVANT_ITEMIDS)} itemids relevantes")
                    self.data['labevents'] = self._load_labevents_optimized(
                        subject_ids,
                        itemid_filter=ALL_RELEVANT_ITEMIDS
                    )
                except ImportError:
                    logger.warning("⚠️  lab_config não encontrado, usando modo sample")
                    self.data['labevents'] = self._load_labevents_optimized(
                        subject_ids,
                        max_rows=10_000_000
                    )
            
            else:  # 'full'
                logger.warning("⚠️  Modo FULL: Isso usará 30GB+ de RAM!")
                self.data['labevents'] = self._load_labevents_optimized(subject_ids)
        else:
            self.data['labevents'] = pd.DataFrame()
        
        # ICU
        self.data['icustays'] = self._load_csv_optimized(
            'icu/icustays.csv',
            filter_col='subject_id',
            filter_values=subject_ids
        )
        logger.info(f"✓ ICU stays: {len(self.data['icustays']):,}")
        
        # Resumo final
        logger.info("\n" + "="*60)
        self._print_memory_usage()
        logger.info("="*60)
        logger.info("✅ Carregamento concluído com sucesso!")
        
        return self.data
    
    def _load_labevents_optimized(self, 
                                  subject_ids: pd.Series, 
                                  max_rows: Optional[int] = None,
                                  itemid_filter: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Carrega labevents com MÁXIMA otimização
        """
        filepath = 'hosp/labevents.csv'
        full_path = self.data_root / filepath
        
        if not full_path.exists():
            logger.warning(f"⚠️  {filepath} não encontrado")
            return pd.DataFrame()
        
        try:
            # Converter para set (busca O(1))
            subject_ids_set = set(subject_ids)
            itemid_set = set(itemid_filter) if itemid_filter else None
            
            # Colunas essenciais
            usecols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 
                       'valuenum', 'valueuom']
            
            # Tipos explícitos
            dtype = {
                'subject_id': 'int32',
                'hadm_id': 'float32',
                'itemid': 'int32',
                'valuenum': 'float32'
            }
            
            chunks = []
            total_rows = 0
            chunk_size = 1_000_000
            start_time = pd.Timestamp.now()
            
            # Processar em chunks
            for i, chunk in enumerate(pd.read_csv(
                full_path,
                chunksize=chunk_size,
                usecols=usecols,
                dtype=dtype,
                low_memory=False,
                parse_dates=['charttime']
            )):
                # Filtro 1: subject_id
                mask = chunk['subject_id'].isin(subject_ids_set)
                chunk_filtered = chunk[mask]
                
                # Filtro 2: itemid (se especificado)
                if itemid_set is not None:
                    chunk_filtered = chunk_filtered[
                        chunk_filtered['itemid'].isin(itemid_set)
                    ]
                
                # Filtro 3: Remove NaN em valuenum
                chunk_filtered = chunk_filtered.dropna(subset=['valuenum'])
                
                if len(chunk_filtered) > 0:
                    chunks.append(chunk_filtered.copy())
                    total_rows += len(chunk_filtered)
                    
                    # Log a cada 10 chunks
                    if (i + 1) % 10 == 0:
                        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                        rate = total_rows / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"   Chunk {i+1:3d}: {total_rows:,} labs | "
                            f"Taxa: {rate:,.0f} rows/s"
                        )
                
                # Liberar memória
                del chunk, chunk_filtered
                
                # Limite de rows
                if max_rows and total_rows >= max_rows:
                    logger.info(f"🛑 Limite de {max_rows:,} registros atingido")
                    break
                
                # Garbage collection periódico
                if (i + 1) % 50 == 0:
                    gc.collect()
            
            # Concatenar
            if chunks:
                logger.info("🔄 Concatenando chunks...")
                df = pd.concat(chunks, ignore_index=True)
                
                # Otimizar tipos finais
                df = self._optimize_dtypes(df)
                
                # Estatísticas
                elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                mem_mb = df.memory_usage(deep=True).sum() / 1024**2
                
                logger.info("\n📊 Estatísticas de Labevents:")
                logger.info(f"   Total registros: {len(df):,}")
                logger.info(f"   Subject_ids únicos: {df['subject_id'].nunique():,}")
                logger.info(f"   Itemids únicos: {df['itemid'].nunique():,}")
                logger.info(f"   Memória: {mem_mb:.1f} MB")
                logger.info(f"   Tempo: {elapsed:.1f}s")
                logger.info(f"   Taxa: {len(df)/elapsed:,.0f} rows/s")
                
                return df
            else:
                logger.warning("⚠️  Nenhum labevents encontrado após filtros")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar labevents: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _load_vitalsign_optimized(self, stay_ids: Optional[pd.Series]) -> pd.DataFrame:
        """Carrega vitalsign de forma otimizada"""
        filepath = 'ed/vitalsign.csv'
        full_path = self.data_root / filepath
        
        if not full_path.exists():
            logger.warning(f"⚠️  {filepath} não encontrado")
            return pd.DataFrame()
        
        try:
            usecols = ['subject_id', 'stay_id', 'charttime', 
                      'temperature', 'heartrate', 'resprate', 'o2sat', 
                      'sbp', 'dbp', 'rhythm', 'pain']
            
            df = pd.read_csv(full_path, usecols=usecols, low_memory=False)
            
            if stay_ids is not None:
                df = df[df['stay_id'].isin(stay_ids)]
            
            df = self._optimize_dtypes(df)
            logger.info(f"✓ Vitalsign: {len(df):,}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar vitalsign: {str(e)}")
            return pd.DataFrame()
    
    def _load_csv_optimized(self, 
                           filepath: str, 
                           usecols: Optional[List[str]] = None,
                           sample_size: Optional[int] = None,
                           filter_col: Optional[str] = None,
                           filter_values: Optional[pd.Series] = None) -> pd.DataFrame:
        """Carrega CSV otimizado"""
        full_path = self.data_root / filepath
        
        if not full_path.exists():
            return pd.DataFrame()
        
        try:
            kwargs = {
                'low_memory': False,
                'na_values': ['', 'NA', 'NULL', 'null', 'nan']
            }
            
            if usecols:
                kwargs['usecols'] = usecols
            if sample_size:
                kwargs['nrows'] = sample_size
            
            df = pd.read_csv(full_path, **kwargs)
            
            if filter_col and filter_values is not None and filter_col in df.columns:
                df = df[df[filter_col].isin(filter_values)].copy()
            
            df = self._optimize_dtypes(df)
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar {filepath}: {str(e)}")
            return pd.DataFrame()
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Otimiza tipos de dados"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'int64':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            elif col_type == 'float64':
                df[col] = df[col].astype(np.float32)
            
            elif col_type == 'object':
                num_unique = df[col].nunique()
                num_total = len(df[col])
                
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
        
        return df
    
    def _print_memory_usage(self):
        """Imprime uso de memória"""
        logger.info("\n📊 Uso de Memória:")
        total_mb = 0
        
        for name, df in self.data.items():
            if not df.empty:
                mem_mb = df.memory_usage(deep=True).sum() / 1024**2
                total_mb += mem_mb
                logger.info(f"   {name:20s}: {mem_mb:7.1f} MB ({len(df):,} registros)")
        
        logger.info(f"   {'-'*20}  {'-'*7}")
        logger.info(f"   {'TOTAL':20s}: {total_mb:7.1f} MB\n")
    
    def get_data_summary(self) -> pd.DataFrame:
        """Retorna resumo dos datasets"""
        summary = []
        for name, df in self.data.items():
            if not df.empty:
                summary.append({
                    'Dataset': name,
                    'Registros': f"{len(df):,}",
                    'Colunas': len(df.columns),
                    'Memória (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}"
                })
        return pd.DataFrame(summary)