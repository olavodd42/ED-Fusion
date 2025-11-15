"""
Criação de labels para os outcomes
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class OutcomeLabeler:
    """
    Cria labels para:
    1. Critical Outcome: morte hospitalar OU transferência ICU em 12h
    2. Lengthened ED Stay: ED LOS > 24 horas
    Parâmetros:
        * df: pd.DataFrame - DataFrame pré-processado com dados do ED
        * icu_stays: pd.DataFrame - DataFrame com dados de ICU stays
    Métodos:
        * create_all_labels(): Cria todos os labels de outcome.
    """
    
    def __init__(self, df: pd.DataFrame, icu_stays: pd.DataFrame) -> None:
        self.df = df
        self.icu_stays = icu_stays
        
    def create_all_labels(self) -> pd.DataFrame:
        """
        Cria todos os labels.
        Retorno:
            * pd.DataFrame - DataFrame com labels adicionados.
        """

        logger.info("\n🏷️  Criando labels de outcome...")
        
        df = self.df.copy()
        
        # Label 1: Critical Outcome
        df = self._label_critical_outcome(df)
        
        # Label 2: Lengthened ED Stay
        df = self._label_lengthened_stay(df)
        
        # Estatísticas
        self._print_label_statistics(df)
        
        return df
    
    def _label_critical_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Critical Outcome = morte hospitalar OU transferência ICU em 12h.
        Parâmetros:
            * df: pd.DataFrame -> DataFrame a ser rotulado.
        Retorno:
            * pd.DataFrame -> DataFrame com label adicionado.
        """
        # Morte hospitalar
        df['hospital_death'] = (df['hospital_expire_flag'] == 1).astype(int)
        
        # Transferência ICU em 12h
        df['icu_transfer_12h'] = 0
        
        # Merge com ICU stays
        icu_early = self.icu_stays.copy()
        icu_early['intime_icu'] = pd.to_datetime(icu_early['intime'])
        
        for idx, row in df.iterrows():
            stay_id = row['stay_id']
            ed_intime = row['intime']
            
            # Buscar ICU admissions para este stay
            patient_icu = icu_early[
                (icu_early['subject_id'] == row['subject_id']) &
                (icu_early['hadm_id'] == row['hadm_id'])
            ]
            
            if not patient_icu.empty:
                # Calcular tempo até ICU
                time_to_icu = (patient_icu['intime_icu'].min() - ed_intime).total_seconds() / 3600
                
                if 0 <= time_to_icu <= 12:
                    df.loc[idx, 'icu_transfer_12h'] = 1
        
        # Critical outcome = morte OU ICU em 12h
        df['critical_outcome'] = (
            (df['hospital_death'] == 1) | 
            (df['icu_transfer_12h'] == 1)
        ).astype(int)
        
        return df
    
    def _label_lengthened_stay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lengthened ED Stay = ED LOS > 24 horas.
        Parâmetros:
            * df: pd.DataFrame -> DataFrame a ser rotulado.
        Retorno:
            * pd.DataFrame -> DataFrame com label adicionado.
        """
        if 'ed_los_hours' in df.columns:
            df['lengthened_stay'] = (df['ed_los_hours'] > 24).astype(int)
        else:
            logger.warning("⚠ ED LOS não disponível para calcular lengthened stay")
            df['lengthened_stay'] = 0
        
        return df
    
    def _print_label_statistics(self, df: pd.DataFrame) -> None:
        """
        Imprime estatísticas dos labels.
        Parâmetros:
            * df: pd.DataFrame -> DataFrame com labels.
        """
        logger.info("\n📊 Distribuição dos Labels:")
        logger.info("="*50)
        
        # Critical Outcome
        if 'critical_outcome' in df.columns:
            n_critical = df['critical_outcome'].sum()
            pct_critical = n_critical / len(df) * 100
            logger.info(f"Critical Outcome:")
            logger.info(f"  Total: {n_critical} ({pct_critical:.2f}%)")
            
            if 'hospital_death' in df.columns:
                n_death = df['hospital_death'].sum()
                logger.info(f"  - Morte hospitalar: {n_death}")
            
            if 'icu_transfer_12h' in df.columns:
                n_icu = df['icu_transfer_12h'].sum()
                logger.info(f"  - ICU em 12h: {n_icu}")
        
        # Lengthened Stay
        if 'lengthened_stay' in df.columns:
            n_lengthened = df['lengthened_stay'].sum()
            pct_lengthened = n_lengthened / len(df) * 100
            logger.info(f"\nLengthened ED Stay (>24h):")
            logger.info(f"  Total: {n_lengthened} ({pct_lengthened:.2f}%)")
        
        logger.info("="*50)