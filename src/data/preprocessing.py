"""
Preprocessamento e filtros conforme especificações do artigo ED-Copilot
Versão robusta que verifica existência de colunas
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class EDPreprocessor:
    """
    Aplica filtros do artigo ED-Copilot:
    1. Pacientes admitidos (hospitalizados)
    2. Idade ≥ 18 anos
    3. Com informações de triagem completas
    4. Remove testes duplicados
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.filtered_data = {}
        self._validate_input_data()
        
    def _validate_input_data(self):
        """Valida que os datasets essenciais existem"""
        required = ['edstays', 'patients']
        missing = [k for k in required if k not in self.data or self.data[k].empty]
        
        if missing:
            raise ValueError(f"Datasets obrigatórios faltando: {missing}")
        
        logger.info(f"✓ Datasets carregados: {list(self.data.keys())}")
    
    def apply_filters(self) -> pd.DataFrame:
        """Aplica todos os filtros sequencialmente"""
        logger.info("\n" + "="*60)
        logger.info("INICIANDO PREPROCESSAMENTO")
        logger.info("="*60)
        
        # 1. Merge inicial de dados ED + pacientes
        df = self._merge_initial_data()
        logger.info(f"\n1️⃣  Dados iniciais: {len(df)} ED stays")
        
        # 2. Filtrar pacientes admitidos
        df = self._filter_admitted_patients(df)
        logger.info(f"2️⃣  Após filtrar admitidos: {len(df)} stays")
        
        # 3. Filtrar idade >= 18
        df = self._filter_adult_patients(df)
        logger.info(f"3️⃣  Após filtrar adultos: {len(df)} stays")
        
        # 4. Filtrar triagem completa
        df = self._filter_complete_triage(df)
        logger.info(f"4️⃣  Após filtrar triagem completa: {len(df)} stays")
        
        # 5. Processar timestamps
        df = self._process_timestamps(df)
        
        # 6. Calcular ED LOS
        df = self._calculate_ed_los(df)
        
        # 7. Adicionar flags úteis
        df = self._add_derived_features(df)
        
        logger.info(f"\n✅ Preprocessamento concluído: {len(df)} registros finais")
        logger.info("="*60 + "\n")
        
        self.filtered_data['main'] = df
        return df
    
    def _get_available_columns(self, df: pd.DataFrame, 
                               desired_cols: List[str]) -> List[str]:
        """Retorna apenas colunas que existem no DataFrame"""
        return [col for col in desired_cols if col in df.columns]
    
    def _merge_initial_data(self) -> pd.DataFrame:
        """Merge ED stays com pacientes e admissions de forma robusta"""
        df = self.data['edstays'].copy()
        
        logger.info(f"   ED stays inicial: {df.shape}")
        logger.info(f"   Colunas ED stays: {df.columns.tolist()}")
        
        # ===== DEBUG: Verificar patients =====
        logger.info(f"\n   Patients shape: {self.data['patients'].shape}")
        logger.info(f"   Patients colunas: {self.data['patients'].columns.tolist()}")
        
        # Verificar se gender existe em patients
        if 'gender' in self.data['patients'].columns:
            gender_dist = self.data['patients']['gender'].value_counts()
            logger.info(f"   Gender em patients: {dict(gender_dist)}")
        else:
            logger.warning("   ⚠️  'gender' NÃO existe em patients!")
        
        # ===== MERGE COM PATIENTS =====
        # Definir colunas desejadas (verificar existência)
        desired_patient_cols = [
            'subject_id', 'gender', 'anchor_age', 'anchor_year', 
            'anchor_year_group', 'dod'
        ]
        
        available_patient_cols = self._get_available_columns(
            self.data['patients'], 
            desired_patient_cols
        )
        
        # subject_id é obrigatório
        if 'subject_id' not in available_patient_cols:
            raise ValueError("Coluna 'subject_id' ausente em patients")
        
        logger.info(f"   Colunas disponíveis em patients: {available_patient_cols}")
        
        patients = self.data['patients'][available_patient_cols].copy()
        
        # DEBUG: Verificar dados antes do merge
        logger.info(f"\n   Antes do merge:")
        logger.info(f"   - df shape: {df.shape}")
        logger.info(f"   - patients shape: {patients.shape}")
        logger.info(f"   - subject_ids em df: {df['subject_id'].nunique()}")
        logger.info(f"   - subject_ids em patients: {patients['subject_id'].nunique()}")
        
        # MERGE
        df = df.merge(patients, on='subject_id', how='left')
        
        logger.info(f"\n   Após merge com patients:")
        logger.info(f"   - df shape: {df.shape}")
        logger.info(f"   - Colunas: {df.columns.tolist()}")
        
        # Verificar se gender foi mergeado
        if 'gender' in df.columns:
            gender_after = df['gender'].value_counts()
            logger.info(f"   ✓ Gender após merge: {dict(gender_after)}")
            missing_gender = df['gender'].isna().sum()
            if missing_gender > 0:
                logger.warning(f"   ⚠️  {missing_gender} registros sem gender após merge")
        elif 'gender_x' in df.columns:
            df['gender'] = df['gender_x']
            gender_after = df['gender'].value_counts()
            logger.info(f"   ✓ Gender após merge: {dict(gender_after)}")
            missing_gender = df['gender'].isna().sum()
        elif 'gender_y' in df.columns:
            df['gender'] = df['gender_y']
            gender_after = df['gender'].value_counts()
            logger.info(f"   ✓ Gender após merge: {dict(gender_after)}")
            missing_gender = df['gender'].isna().sum()
        else:
            logger.warning("   ⚠️  'gender' NÃO está no DataFrame após merge!")
            logger.warning("   ⚠️  Definindo gender como 'U'")
            df['gender'] = 'U'
        
        # ===== MERGE COM ADMISSIONS =====
        if 'admissions' in self.data and not self.data['admissions'].empty:
            desired_adm_cols = [
                'subject_id', 'hadm_id', 'admittime', 'dischtime', 
                'deathtime', 'hospital_expire_flag', 'admission_type', 
                'admission_location', 'discharge_location'
            ]
            
            available_adm_cols = self._get_available_columns(
                self.data['admissions'],
                desired_adm_cols
            )
            
            if 'subject_id' in available_adm_cols and 'hadm_id' in available_adm_cols:
                admissions = self.data['admissions'][available_adm_cols].copy()
                df = df.merge(
                    admissions, 
                    on=['subject_id', 'hadm_id'], 
                    how='left', 
                    suffixes=('', '_adm')
                )
                logger.info(f"   ✓ Merge com admissions: {len(available_adm_cols)} colunas")
        
        # ===== MERGE COM TRIAGE =====
        if 'triage' in self.data and not self.data['triage'].empty:
            triage = self.data['triage'].copy()
            
            # Verificar se stay_id existe em ambos
            if 'stay_id' in df.columns and 'stay_id' in triage.columns:
                df = df.merge(triage, on='stay_id', how='left', suffixes=('', '_triage'))
                logger.info(f"   ✓ Merge com triage: {triage.shape[1]} colunas")
            else:
                logger.warning("   ⚠️  Não foi possível fazer merge com triage (stay_id ausente)")
        
        logger.info(f"   → Shape final após merges: {df.shape}")
        
        return df
    
    def _filter_admitted_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtrar apenas pacientes que foram admitidos ao hospital"""
        initial_count = len(df)
        
        if 'hadm_id' not in df.columns:
            logger.warning("   ⚠️  'hadm_id' não encontrado, pulando filtro de admissão")
            return df
        
        # Pacientes com hadm_id válido (foram admitidos)
        df = df[df['hadm_id'].notna()].copy()
        
        filtered_count = initial_count - len(df)
        logger.info(f"   → Removidos {filtered_count} stays não admitidos")
        
        return df
    
    def _filter_adult_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtrar pacientes com idade >= 18 anos"""
        initial_count = len(df)
        
        # Verificar campos necessários
        if 'anchor_age' not in df.columns:
            logger.warning("   ⚠️  'anchor_age' não encontrado, pulando filtro de idade")
            return df
        
        # Processar intime se existir
        if 'intime' in df.columns:
            df['intime'] = pd.to_datetime(df['intime'], errors='coerce')
        
        # Calcular idade na admissão ED
        if 'anchor_year' in df.columns and 'intime' in df.columns:
            # Garantir que anchor_year é numérico
            df['anchor_year'] = pd.to_numeric(df['anchor_year'], errors='coerce')
            
            # Calcular diferença de anos
            df['years_from_anchor'] = (
                df['intime'].dt.year - df['anchor_year']
            ).astype('Int64')
            
            df['age_at_ed'] = df['anchor_age'] + df['years_from_anchor']
        else:
            # Se não temos anchor_year, usar anchor_age diretamente
            logger.warning("   ⚠️  Usando anchor_age diretamente (anchor_year não disponível)")
            df['age_at_ed'] = df['anchor_age']
        
        # Filtrar >= 18 anos e <= 120 (valores válidos)
        df = df[
            (df['age_at_ed'] >= 18) & 
            (df['age_at_ed'] <= 120)
        ].copy()
        
        filtered_count = initial_count - len(df)
        logger.info(f"   → Removidos {filtered_count} pacientes < 18 anos")
        
        return df
    
    def _filter_complete_triage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtrar apenas registros com triagem completa"""
        initial_count = len(df)
        
        # Campos obrigatórios de triagem (conforme paper)
        required_triage_fields = [
            'temperature',
            'heartrate', 
            'resprate',
            'o2sat',
            'sbp',  # systolic blood pressure
            'dbp',  # diastolic blood pressure
            'acuity'
        ]
        
        # Verificar quais campos estão disponíveis
        available_fields = [f for f in required_triage_fields if f in df.columns]
        
        if len(available_fields) == 0:
            logger.warning("   ⚠️  Nenhum campo de triagem encontrado!")
            return df
        
        # Estratégia: Aceitar registros com pelo menos 5 dos 7 campos
        df['triage_completeness'] = df[available_fields].notna().sum(axis=1)
        min_required = min(5, len(available_fields))
        
        df = df[df['triage_completeness'] >= min_required].copy()
        
        filtered_count = initial_count - len(df)
        logger.info(f"   → Removidos {filtered_count} stays com triagem incompleta")
        logger.info(f"   → Campos de triagem disponíveis: {', '.join(available_fields)}")
        
        return df
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processar e validar timestamps"""
        timestamp_cols = ['intime', 'outtime', 'admittime', 'dischtime', 'deathtime']
        
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _calculate_ed_los(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular ED Length of Stay em horas"""
        if 'intime' not in df.columns or 'outtime' not in df.columns:
            logger.warning("   ⚠️  Campos de tempo ausentes, ED LOS não calculado")
            return df
        
        # Calcular LOS
        df['ed_los_hours'] = (
            df['outtime'] - df['intime']
        ).dt.total_seconds() / 3600
        
        # Filtrar valores inválidos
        df.loc[df['ed_los_hours'] < 0, 'ed_los_hours'] = np.nan
        df.loc[df['ed_los_hours'] > 168, 'ed_los_hours'] = np.nan  # > 7 dias
        
        # Estatísticas
        valid_los = df['ed_los_hours'].dropna()
        if len(valid_los) > 0:
            logger.info(f"\n📊 ED Length of Stay (horas):")
            logger.info(f"   Média: {valid_los.mean():.2f}h")
            logger.info(f"   Mediana: {valid_los.median():.2f}h")
            logger.info(f"   Min: {valid_los.min():.2f}h")
            logger.info(f"   Max: {valid_los.max():.2f}h")
            
            stays_24h = (valid_los > 24).sum()
            logger.info(f"   Stays > 24h: {stays_24h} ({stays_24h/len(valid_los)*100:.1f}%)")
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features derivadas úteis"""
        
        # Flag de lengthened ED stay (> 24h)
        if 'ed_los_hours' in df.columns:
            df['lengthened_ed_stay'] = (df['ed_los_hours'] > 24).astype(int)
        
        # Flag de mortalidade hospitalar
        if 'hospital_expire_flag' in df.columns:
            df['hospital_death'] = df['hospital_expire_flag'].fillna(0).astype(int)
        elif 'deathtime' in df.columns:
            df['hospital_death'] = df['deathtime'].notna().astype(int)
        
        # Faixa etária
        if 'age_at_ed' in df.columns:
            df['age_group'] = pd.cut(
                df['age_at_ed'],
                bins=[0, 30, 60, 90, 120],
                labels=['18-30', '31-60', '61-90', '90+']
            )
        
        return df
    
    def get_filter_summary(self) -> pd.DataFrame:
        """Retorna resumo dos dados filtrados"""
        if 'main' not in self.filtered_data:
            logger.warning("Dados não processados ainda. Execute apply_filters() primeiro.")
            return pd.DataFrame()
        
        df = self.filtered_data['main']
        
        summary = {
            'Total registros': len(df),
            'Pacientes únicos': df['subject_id'].nunique() if 'subject_id' in df.columns else 'N/A',
        }
        
        # Adicionar métricas condicionalmente
        if 'age_at_ed' in df.columns:
            summary['Idade média (anos)'] = f"{df['age_at_ed'].mean():.1f}"
            summary['Idade min-max'] = f"{df['age_at_ed'].min():.0f}-{df['age_at_ed'].max():.0f}"
        
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            summary['Sexo masculino (%)'] = f"{(gender_counts.get('M', 0) / len(df) * 100):.1f}"
            summary['Sexo feminino (%)'] = f"{(gender_counts.get('F', 0) / len(df) * 100):.1f}"
        
        if 'ed_los_hours' in df.columns:
            summary['ED LOS média (h)'] = f"{df['ed_los_hours'].mean():.2f}"
            summary['ED LOS mediana (h)'] = f"{df['ed_los_hours'].median():.2f}"
        
        if 'lengthened_ed_stay' in df.columns:
            lengthened = (df['lengthened_ed_stay'] == 1).sum()
            summary['Lengthened ED Stay (%)'] = f"{lengthened / len(df) * 100:.1f}"
        
        if 'hospital_death' in df.columns:
            deaths = (df['hospital_death'] == 1).sum()
            summary['Mortalidade hospitalar (%)'] = f"{deaths / len(df) * 100:.2f}"
        
        return pd.DataFrame([summary]).T.rename(columns={0: 'Valor'})
    
    def get_column_availability(self) -> pd.DataFrame:
        """Retorna quais colunas estão disponíveis em cada dataset"""
        availability = {}
        
        for name, df in self.data.items():
            if not df.empty:
                availability[name] = {
                    'Registros': len(df),
                    'Colunas': len(df.columns),
                    'Colunas disponíveis': ', '.join(df.columns[:10]) + '...' if len(df.columns) > 10 else ', '.join(df.columns)
                }
        
        return pd.DataFrame(availability).T


class FeatureEngineer:
    """
    Engenharia de features para o modelo ED-Copilot
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_triage_features(self) -> pd.DataFrame:
        """Cria features de triagem normalizadas"""
        df = self.df.copy()
        
        # Features vitais
        vital_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
        
        for col in vital_cols:
            if col in df.columns:
                # Normalização Z-score
                df[f'{col}_zscore'] = (
                    df[col] - df[col].mean()
                ) / df[col].std()
                
                # Flag de valores anormais (> 2 desvios padrão)
                df[f'{col}_abnormal'] = (abs(df[f'{col}_zscore']) > 2).astype(int)
        
        return df
    
    def create_temporal_features(self) -> pd.DataFrame:
        """Cria features temporais"""
        df = self.df.copy()
        
        if 'intime' in df.columns:
            df['hour_of_day'] = df['intime'].dt.hour
            df['day_of_week'] = df['intime'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour_of_day'] >= 20) | (df['hour_of_day'] <= 6)).astype(int)
        
        return df
    
    def create_risk_scores(self) -> pd.DataFrame:
        """Cria scores de risco simples"""
        df = self.df.copy()
        
        # Score de risco baseado em sinais vitais anormais
        abnormal_cols = [c for c in df.columns if c.endswith('_abnormal')]
        if abnormal_cols:
            df['vital_risk_score'] = df[abnormal_cols].sum(axis=1)
        
        return df


class OutcomeLabeler:
    """
    Cria labels de outcome conforme paper:
    1. Critical outcome (morte ou ICU em 12h)
    2. Lengthened ED stay (> 24h)
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], df_main: pd.DataFrame):
        self.data = data
        self.df_main = df_main
    
    def create_labels(self) -> pd.DataFrame:
        """Cria ambos os labels"""
        df = self.df_main.copy()
        
        # Label 1: Critical Outcome
        df = self._label_critical_outcome(df)
        
        # Label 2: Lengthened ED Stay
        df = self._label_lengthened_stay(df)
        
        # Estatísticas
        self._print_label_statistics(df)
        
        return df
    
    def _label_critical_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Critical outcome = morte hospitalar OU transferência ICU em 12h
        """
        df['critical_outcome'] = 0
        
        # Morte hospitalar
        if 'hospital_death' in df.columns:
            df.loc[df['hospital_death'] == 1, 'critical_outcome'] = 1
        
        # ICU transfer em 12h
        if 'icustays' in self.data and not self.data['icustays'].empty:
            icu = self.data['icustays'].copy()
            icu['intime_icu'] = pd.to_datetime(icu['intime'])
            
            # Merge com df
            df_icu = df.merge(
                icu[['subject_id', 'hadm_id', 'intime_icu']],
                on=['subject_id', 'hadm_id'],
                how='left'
            )
            
            # Calcular tempo até ICU
            if 'outtime' in df_icu.columns:
                df_icu['time_to_icu_hours'] = (
                    df_icu['intime_icu'] - df_icu['outtime']
                ).dt.total_seconds() / 3600
                
                # ICU em 12h
                df.loc[
                    (df_icu['time_to_icu_hours'] >= 0) & 
                    (df_icu['time_to_icu_hours'] <= 12),
                    'critical_outcome'
                ] = 1
        
        return df
    
    def _label_lengthened_stay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lengthened ED stay = ED LOS > 24h
        """
        if 'ed_los_hours' in df.columns:
            df['lengthened_ed_stay'] = (df['ed_los_hours'] > 24).astype(int)
        else:
            df['lengthened_ed_stay'] = 0
        
        return df
    
    def _print_label_statistics(self, df: pd.DataFrame):
        """Imprime estatísticas dos labels"""
        logger.info("\n" + "="*60)
        logger.info("ESTATÍSTICAS DOS LABELS")
        logger.info("="*60)
        
        total = len(df)
        
        if 'critical_outcome' in df.columns:
            critical = (df['critical_outcome'] == 1).sum()
            logger.info(f"Critical Outcome: {critical} ({critical/total*100:.2f}%)")
        
        if 'lengthened_ed_stay' in df.columns:
            lengthened = (df['lengthened_ed_stay'] == 1).sum()
            logger.info(f"Lengthened ED Stay: {lengthened} ({lengthened/total*100:.2f}%)")
        
        logger.info("="*60 + "\n")