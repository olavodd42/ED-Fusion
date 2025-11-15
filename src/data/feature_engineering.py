"""
Engenharia de features conforme especificações do artigo ED-Copilot
Versão otimizada e robusta
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Extrai e processa features para o modelo ED-Copilot:
    
    **Triage Features (9 variáveis)**:
    - Demografia: idade, sexo
    - Sinais vitais: FC, PA, FR, temperatura, SpO2
    - Clínica: ESI acuity, dor
    - Chief complaint (texto)
    
    **Laboratory Features (68 testes em 12 grupos)**:
    - CBC, CHEM, COAG, UA, LACTATE, LFTs, LIPASE, LYTES, 
      BLOOD_GAS, CARDIO, TOX, INFLAM
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 lab_data: Optional[pd.DataFrame] = None, 
                 lab_items: Optional[pd.DataFrame] = None):
        """
        Parâmetros:
            df: DataFrame preprocessado (após EDPreprocessor)
            lab_data: DataFrame de labevents (opcional)
            lab_items: DataFrame d_labitems (opcional)
        """
        self.df = df.copy()
        self.lab_data = lab_data if lab_data is not None else pd.DataFrame()
        self.lab_items = lab_items if lab_items is not None else pd.DataFrame()
        
        # Importar mapeamento de labs
        self._load_lab_config()
        
        # Validar dados
        self._validate_data()
    
    def _load_lab_config(self):
        """Carrega configuração de laboratórios"""
        try:
            from .lab_config import (
                RELEVANT_LAB_ITEMIDS, 
                ITEMID_TO_GROUP,
                GROUP_TIME_COSTS
            )
            self.lab_groups = RELEVANT_LAB_ITEMIDS
            self.itemid_to_group = ITEMID_TO_GROUP
            self.group_time_costs = GROUP_TIME_COSTS
            logger.info("✓ Configuração de labs carregada")
        except ImportError:
            logger.warning("⚠️  lab_config.py não encontrado, usando defaults")
            self._create_default_lab_config()
    
    def _create_default_lab_config(self):
        """Cria configuração padrão se lab_config não existir"""
        # Configuração mínima (você deve usar lab_config.py completo)
        self.lab_groups = {
            'CBC': [51221, 51300, 51222],  # Exemplos
            'CHEM': [51006, 50912, 50983],
            # ... adicionar outros grupos
        }
        
        self.itemid_to_group = {}
        for group, itemids in self.lab_groups.items():
            for itemid in itemids:
                self.itemid_to_group[itemid] = group
        
        # Time-costs do paper (Tabela 7)
        self.group_time_costs = {
            'CBC': 30, 'CHEM': 60, 'COAG': 48, 'UA': 40,
            'LACTATE': 4, 'LFTS': 104, 'LIPASE': 100, 'LYTES': 89,
            'BLOOD_GAS': 12, 'CARDIO': 122, 'TOX': 70, 'INFLAM': 178
        }
    
    def _validate_data(self):
        """Valida integridade dos dados"""
        if self.df.empty:
            raise ValueError("DataFrame principal está vazio")
        
        required_cols = ['stay_id', 'subject_id']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias faltando: {missing}")
        
        logger.info(f"✓ DataFrame validado: {len(self.df):,} registros")
    
    # ========== TRIAGE FEATURES ==========
    
    def extract_triage_features(self) -> pd.DataFrame:
        """
        Extrai 9 features de triagem conforme Tabela 7 do paper:
        
        1. Age
        2. Gender
        3. Heart Rate (FC)
        4. Respiratory Rate (FR)
        5. Systolic BP
        6. Diastolic BP
        7. Temperature
        8. SpO2
        9. ESI Acuity
        
        Retorna:
            DataFrame com features de triagem extraídas e normalizadas
        """
        logger.info("\n" + "="*60)
        logger.info("🏥 EXTRAÇÃO DE TRIAGE FEATURES")
        logger.info("="*60)
        
        df = self.df.copy()
        
        # 1. DEMOGRAFIA
        if 'age_at_ed' in df.columns:
            df['triage_age'] = pd.to_numeric(df['age_at_ed'], errors='coerce')
            logger.info(f"  ✓ Age: média {df['triage_age'].mean():.1f} anos")
        else:
            logger.warning("  ⚠️  'age_at_ed' não encontrado")
            df['triage_age'] = np.nan
        
        if 'gender' in df.columns:
            if df['gender'].dtype.name == 'category':
                df['gender'] = df['gender'].astype(str)
            df['triage_gender_male'] = (df['gender'] == 'M').astype(int)
            df['triage_gender_female'] = (df['gender'] == 'F').astype(int)
            male_pct = df['triage_gender_male'].mean() * 100
            logger.info(f"  ✓ Gender: {male_pct:.1f}% masculino")
        else:
            logger.warning("  ⚠️  'gender' não encontrado")
            df['triage_gender_male'] = 0
            df['triage_gender_female'] = 0
        
        # 2. SINAIS VITAIS
        vital_mappings = {
            'triage_heart_rate': ['heartrate', 'heart_rate'],
            'triage_respiratory_rate': ['resprate', 'resp_rate', 'respiratory_rate'],
            'triage_sbp': ['sbp', 'systolic_bp'],
            'triage_dbp': ['dbp', 'diastolic_bp'],
            'triage_temperature': ['temperature', 'temp'],
            'triage_spo2': ['o2sat', 'spo2', 'oxygen_saturation']
        }
        
        for target_col, possible_sources in vital_mappings.items():
            value_set = False
            for source_col in possible_sources:
                if source_col in df.columns:
                    df[target_col] = pd.to_numeric(df[source_col], errors='coerce')
                    valid_pct = df[target_col].notna().sum() / len(df) * 100
                    logger.info(f"  ✓ {target_col}: {valid_pct:.1f}% válidos")
                    value_set = True
                    break
            
            if not value_set:
                logger.warning(f"  ⚠️  {target_col} não encontrado")
                df[target_col] = np.nan
        
        # 3. CLÍNICA
        if 'acuity' in df.columns:
            df['triage_acuity'] = pd.to_numeric(df['acuity'], errors='coerce')
            logger.info(f"  ✓ ESI Acuity: média {df['triage_acuity'].mean():.2f}")
        else:
            logger.warning("  ⚠️  'acuity' não encontrado")
            df['triage_acuity'] = np.nan
        
        # 4. DOR (opcional, não está nos 9 principais)
        if 'pain' in df.columns:
            df['triage_pain'] = pd.to_numeric(df['pain'], errors='coerce')
        
        # 5. CHIEF COMPLAINT (manter como texto para processamento posterior)
        if 'chiefcomplaint' in df.columns:
            # df['triage_chief_complaint'] = df['chiefcomplaint'].fillna('')
            if df['chiefcomplaint'].dtype.name == 'category':
                df['triage_chief_complaint'] = df['chiefcomplaint'].astype(str).replace('nan', '')
            else:
                df['triage_chief_complaint'] = df['chiefcomplaint'].fillna('').astype(str)
            logger.info(f"  ✓ Chief Complaint: {(df['triage_chief_complaint'] != '').sum()} com texto")
        
        # ESTATÍSTICAS FINAIS
        triage_cols = [c for c in df.columns if c.startswith('triage_')]
        logger.info(f"\n✅ Total de features de triagem: {len(triage_cols)}")
        
        # Completude por feature
        completeness = {}
        for col in triage_cols:
            if df[col].dtype in ['float64', 'int64']:
                completeness[col] = df[col].notna().sum() / len(df) * 100
        
        if completeness:
            logger.info("\n📊 Completude das features:")
            for col, pct in sorted(completeness.items(), key=lambda x: -x[1]):
                logger.info(f"  {col:30s}: {pct:5.1f}%")
        
        return df
    
    # ========== LABORATORY FEATURES ==========
    
    def extract_lab_features(self, 
                           time_window_hours: int = 12,
                           use_first_value: bool = True) -> pd.DataFrame:
        """
        Extrai features de laboratório nos primeiros X horas do ED.
        Agrupa em 12 categorias conforme Tabela 7 do paper.
        
        Parâmetros:
            time_window_hours: Janela temporal (padrão 12h)
            use_first_value: Se True, usa primeiro valor de cada lab
        
        Retorna:
            DataFrame com features de laboratório agregadas por grupo
        """
        logger.info("\n" + "="*60)
        logger.info("🔬 EXTRAÇÃO DE LABORATORY FEATURES")
        logger.info("="*60)
        
        if self.lab_data.empty:
            logger.warning("⚠️  Dados de laboratório não disponíveis")
            return self._add_empty_lab_features(self.df)
        
        # 1. Filtrar labs na janela temporal
        labs_filtered = self._filter_labs_by_time(time_window_hours)
        
        if labs_filtered.empty:
            logger.warning("⚠️  Nenhum lab encontrado na janela temporal")
            return self._add_empty_lab_features(self.df)
        
        # 2. Mapear itemids para grupos
        labs_filtered = self._assign_lab_groups(labs_filtered)
        
        # 3. Agregar por grupo (otimizado com pandas groupby)
        lab_features = self._aggregate_labs_efficient(labs_filtered, use_first_value)
        
        # 4. Merge com DataFrame principal
        df_result = self.df.merge(lab_features, on='stay_id', how='left')
        
        # 5. Preencher valores ausentes (labs não solicitados)
        df_result = self._fill_missing_labs(df_result)
        
        # 6. Adicionar flags de utilização
        df_result = self._add_lab_utilization_flags(df_result)
        
        logger.info(f"\n✅ Features de laboratório extraídas")
        
        return df_result
    
    def _filter_labs_by_time(self, hours: int) -> pd.DataFrame:
        """Filtra labs na janela temporal do ED"""
        
        # Merge com edstays para ter timestamps
        labs = self.lab_data.merge(
            self.df[['stay_id', 'subject_id', 'hadm_id', 'intime']],
            on=['subject_id', 'hadm_id'],
            how='inner'
        )
        
        # Converter timestamps
        labs['charttime'] = pd.to_datetime(labs['charttime'], errors='coerce')
        labs['intime'] = pd.to_datetime(labs['intime'], errors='coerce')
        
        # Remover registros com timestamps inválidos
        labs = labs.dropna(subset=['charttime', 'intime'])
        
        # Calcular tempo desde entrada no ED
        labs['hours_from_ed'] = (
            labs['charttime'] - labs['intime']
        ).dt.total_seconds() / 3600
        
        # Filtrar janela
        labs = labs[
            (labs['hours_from_ed'] >= 0) & 
            (labs['hours_from_ed'] <= hours)
        ].copy()
        
        logger.info(f"  → {len(labs):,} labs na janela de {hours}h")
        logger.info(f"  → {labs['stay_id'].nunique():,} ED stays com labs")
        
        return labs
    
    def _assign_lab_groups(self, labs: pd.DataFrame) -> pd.DataFrame:
        """Atribui grupo a cada lab baseado em itemid"""
        
        # Mapear itemid -> grupo
        labs['lab_group'] = labs['itemid'].map(self.itemid_to_group)
        
        # Remover labs que não pertencem a nenhum grupo relevante
        labs = labs[labs['lab_group'].notna()].copy()
        
        logger.info(f"  → {len(labs):,} labs dos 68 testes relevantes")
        
        # Estatísticas por grupo
        group_counts = labs['lab_group'].value_counts()
        logger.info("\n  📊 Distribuição por grupo:")
        for group, count in group_counts.head(5).items():
            pct = count / len(labs) * 100
            logger.info(f"    {group:15s}: {count:6,} ({pct:5.1f}%)")
        
        return labs
    
    def _aggregate_labs_efficient(self, labs: pd.DataFrame, 
                                  use_first: bool = True) -> pd.DataFrame:
        """
        Agrega labs por grupo de forma eficiente usando pandas groupby
        """
        logger.info("\n  🔄 Agregando labs por grupo...")
        
        # Preparar dados
        labs['valuenum'] = pd.to_numeric(labs['valuenum'], errors='coerce')
        
        # Agrupar por stay_id e lab_group
        if use_first:
            # Usar primeiro valor (ordem cronológica)
            labs = labs.sort_values(['stay_id', 'lab_group', 'hours_from_ed'])
            agg_labs = labs.groupby(['stay_id', 'lab_group']).first().reset_index()
        else:
            # Usar média de valores
            agg_labs = labs.groupby(['stay_id', 'lab_group']).agg({
                'valuenum': 'mean',
                'hours_from_ed': 'min'  # Tempo do primeiro teste
            }).reset_index()
        
        # Pivot para formato wide (uma coluna por grupo)
        lab_features = agg_labs.pivot(
            index='stay_id',
            columns='lab_group',
            values='valuenum'
        ).reset_index()
        
        # Renomear colunas
        lab_features.columns = [
            f'lab_{col}_value' if col != 'stay_id' else col 
            for col in lab_features.columns
        ]
        
        # Adicionar tempos
        lab_times = agg_labs.pivot(
            index='stay_id',
            columns='lab_group',
            values='hours_from_ed'
        ).reset_index()
        
        lab_times.columns = [
            f'lab_{col}_time' if col != 'stay_id' else col 
            for col in lab_times.columns
        ]
        
        # Merge valores e tempos
        lab_features = lab_features.merge(lab_times, on='stay_id', how='left')
        
        logger.info(f"  ✓ {len(lab_features):,} stays com labs agregados")
        
        return lab_features
    
    def _add_empty_lab_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas vazias de labs quando não há dados"""
        for group in self.lab_groups.keys():
            df[f'lab_{group}_value'] = np.nan
            df[f'lab_{group}_time'] = np.nan
            df[f'lab_{group}_ordered'] = 0
        
        return df
    
    def _fill_missing_labs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenche valores ausentes de labs não solicitados"""
        
        lab_value_cols = [c for c in df.columns if c.startswith('lab_') and c.endswith('_value')]
        
        # NaN significa que o teste não foi solicitado
        # Não imputar - manter NaN para o modelo saber que não foi feito
        
        return df
    
    def _add_lab_utilization_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona flags indicando quais grupos foram solicitados"""
        
        for group in self.lab_groups.keys():
            value_col = f'lab_{group}_value'
            flag_col = f'lab_{group}_ordered'
            
            if value_col in df.columns:
                df[flag_col] = df[value_col].notna().astype(int)
        
        # Calcular total de grupos solicitados
        ordered_cols = [c for c in df.columns if c.endswith('_ordered')]
        df['total_lab_groups_ordered'] = df[ordered_cols].sum(axis=1)
        
        logger.info(f"\n  📊 Utilização de labs:")
        logger.info(f"    Média de grupos por paciente: {df['total_lab_groups_ordered'].mean():.2f}")
        logger.info(f"    Pacientes sem labs: {(df['total_lab_groups_ordered'] == 0).sum()}")
        
        return df
    
    # ========== TIME-COST CALCULATION ==========
    
    def calculate_lab_time_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula time-cost total de laboratório por paciente.
        Usa time-costs do paper (Tabela 7).
        
        Estratégias:
        1. Sequential: Soma de todos os time-costs
        2. Parallel: Máximo time-cost (processamento paralelo)
        3. Mixed: Combinação realista
        """
        logger.info("\n" + "="*60)
        logger.info("⏱️  CALCULANDO TIME-COSTS")
        logger.info("="*60)
        
        df = df.copy()
        
        # Inicializar colunas
        df['lab_time_cost_sequential'] = 0.0
        df['lab_time_cost_parallel'] = 0.0
        df['lab_time_cost_mixed'] = 0.0
        
        # Para cada grupo
        for group, time_cost in self.group_time_costs.items():
            ordered_col = f'lab_{group}_ordered'
            
            if ordered_col in df.columns:
                ordered_mask = df[ordered_col] == 1
                
                # Sequential (soma)
                df.loc[ordered_mask, 'lab_time_cost_sequential'] += time_cost
                
                # Parallel (máximo) - calcular depois
                
                # Mixed (média ponderada: 70% parallel + 30% sequential)
                # Aproximação mais realista
        
        # Calcular parallel (máximo time-cost dos grupos solicitados)
        for idx, row in df.iterrows():
            max_cost = 0
            for group, time_cost in self.group_time_costs.items():
                if row.get(f'lab_{group}_ordered', 0) == 1:
                    max_cost = max(max_cost, time_cost)
            
            df.loc[idx, 'lab_time_cost_parallel'] = max_cost
        
        # Mixed strategy
        df['lab_time_cost_mixed'] = (
            0.7 * df['lab_time_cost_parallel'] + 
            0.3 * df['lab_time_cost_sequential']
        )
        
        # Estatísticas
        logger.info(f"\n📊 Time-costs médios (minutos):")
        logger.info(f"  Sequential: {df['lab_time_cost_sequential'].mean():.1f}")
        logger.info(f"  Parallel: {df['lab_time_cost_parallel'].mean():.1f}")
        logger.info(f"  Mixed: {df['lab_time_cost_mixed'].mean():.1f}")
        
        # Converter para horas para comparar com ED LOS
        df['lab_time_cost_hours'] = df['lab_time_cost_mixed'] / 60
        
        # Comparar com ED LOS se disponível
        if 'ed_los_hours' in df.columns:
            # Proporção do LOS gasta em labs
            df['lab_time_pct_of_los'] = (
                df['lab_time_cost_hours'] / df['ed_los_hours'] * 100
            )
            
            avg_pct = df['lab_time_pct_of_los'].mean()
            logger.info(f"\n  ⏱️  Labs representam {avg_pct:.1f}% do ED LOS em média")
        
        return df
    
    # ========== FEATURE SUMMARY ==========
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retorna resumo de todas as features extraídas"""
        
        summary = []
        
        # Triage features
        triage_cols = [c for c in df.columns if c.startswith('triage_')]
        for col in triage_cols:
            if df[col].dtype in ['float64', 'int64']:
                summary.append({
                    'Feature': col,
                    'Tipo': 'Triage',
                    'Completude (%)': df[col].notna().sum() / len(df) * 100,
                    'Média': df[col].mean(),
                    'Std': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max()
                })
        
        # Lab features
        lab_value_cols = [c for c in df.columns if c.startswith('lab_') and c.endswith('_value')]
        for col in lab_value_cols:
            summary.append({
                'Feature': col,
                'Tipo': 'Laboratory',
                'Completude (%)': df[col].notna().sum() / len(df) * 100,
                'Média': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max()
            })
        
        return pd.DataFrame(summary)
    
    def save_features(self, df: pd.DataFrame, output_path: str = '../data/processed/features.parquet'):
        """Salva features extraídas"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False, compression='snappy')
        
        file_size = output_path.stat().st_size / 1024**2
        logger.info(f"\n💾 Features salvas em: {output_path}")
        logger.info(f"   Tamanho: {file_size:.1f} MB")
        logger.info(f"   Registros: {len(df):,}")
        logger.info(f"   Features: {len(df.columns)}")