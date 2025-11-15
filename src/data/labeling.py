"""
Criação de labels para os outcomes conforme paper ED-Copilot
Versão otimizada sem loops
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OutcomeLabeler:
    """
    Cria labels de outcome conforme paper ED-Copilot (Tabela 1):
    
    **Label 1: Critical Outcome (9.67% positivos)**
    - Morte hospitalar (inpatient mortality) OU
    - Transferência para ICU em 12h
    
    **Label 2: Lengthened ED Stay (6.90% positivos)**
    - ED LOS > 24 horas
    
    Parâmetros:
        * df: pd.DataFrame -> DataFrame principal com ED stays processados
        * data_dict: Optional[Dict[str, pd.DataFrame]] -> Dict com 'admissions', 'icustays', etc.
    Métodos:
        * create_all_labels() -> pd.DataFrame: Cria todos os labels e retorna DataFrame com labels adicionados
        * get_label_summary(df) -> pd.DataFrame: Retorna resumo dos labels em formato DataFrame
        * analyze_label_correlations(df) -> Dict: Analisa correlações entre labels e features
        * save_labels(df, output_path) -> None: Salva dados com labels no caminho especificado
    """
    
    def __init__(self, df: pd.DataFrame, data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        self.df = df.copy()
        self.data_dict = data_dict if data_dict is not None else {}
        
        # Validar dados
        self._validate_data()
    
    def _validate_data(self):
        """Valida integridade dos dados de entrada"""
        if self.df.empty:
            raise ValueError("DataFrame está vazio")
        
        required_cols = ['stay_id', 'subject_id']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias faltando: {missing}")
        
        logger.info(f"✓ OutcomeLabeler inicializado: {len(self.df):,} registros")
    
    def create_all_labels(self) -> pd.DataFrame:
        """
        Cria todos os labels de outcome.
        
        Retorna:
            DataFrame com labels adicionados:
            - critical_outcome (0/1)
            - lengthened_ed_stay (0/1)
            - hospital_death (0/1)
            - icu_transfer_12h (0/1)
        """
        logger.info("\n" + "="*60)
        logger.info("🏷️  CRIAÇÃO DE OUTCOME LABELS")
        logger.info("="*60)
        
        df = self.df.copy()
        
        # Label 1: Critical Outcome
        df = self._label_critical_outcome(df)
        
        # Label 2: Lengthened ED Stay
        df = self._label_lengthened_stay(df)
        
        # Estatísticas finais
        self._print_label_statistics(df)
        
        # Validar distribuição
        self._validate_label_distribution(df)
        
        logger.info("\n✅ Labels criados com sucesso!")
        
        return df
    
    # ========== CRITICAL OUTCOME ==========
    
    def _label_critical_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Critical Outcome = Morte Hospitalar OU Transferência ICU em 12h
        
        Conforme paper:
        - Inpatient mortality: 467 (1.44%)
        - ICU transfer in 12h: 2894 (8.94%)
        - Critical outcome: 3129 (9.67%)
        
        Nota: Há overlap entre morte e ICU
        """
        logger.info("\n1️⃣  Criando label: Critical Outcome")
        
        # Inicializar componentes
        df['hospital_death'] = 0
        df['icu_transfer_12h'] = 0
        
        # Componente 1: Morte Hospitalar
        df = self._label_hospital_death(df)
        
        # Componente 2: Transferência ICU em 12h
        df = self._label_icu_transfer(df)
        
        # Label final: OR lógico
        df['critical_outcome'] = (
            (df['hospital_death'] == 1) | 
            (df['icu_transfer_12h'] == 1)
        ).astype(int)
        
        # Estatísticas
        n_death = df['hospital_death'].sum()
        n_icu = df['icu_transfer_12h'].sum()
        n_critical = df['critical_outcome'].sum()
        pct_critical = n_critical / len(df) * 100
        
        logger.info(f"  ✓ Morte hospitalar: {n_death:,} ({n_death/len(df)*100:.2f}%)")
        logger.info(f"  ✓ ICU em 12h: {n_icu:,} ({n_icu/len(df)*100:.2f}%)")
        logger.info(f"  ✓ Critical Outcome: {n_critical:,} ({pct_critical:.2f}%)")
        
        return df
    
    def _label_hospital_death(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica morte hospitalar através de múltiplas fontes
        """
        # Fonte 1: hospital_expire_flag (de admissions)
        if 'hospital_expire_flag' in df.columns:
            df['hospital_death'] = df['hospital_expire_flag'].fillna(0).astype(int)
            n_from_flag = df['hospital_death'].sum()
            logger.info(f"    → {n_from_flag:,} mortes via hospital_expire_flag")
        
        # Fonte 2: deathtime (de admissions)
        elif 'deathtime' in df.columns:
            df['hospital_death'] = df['deathtime'].notna().astype(int)
            n_from_deathtime = df['hospital_death'].sum()
            logger.info(f"    → {n_from_deathtime:,} mortes via deathtime")
        
        # Fonte 3: dod (date of death de patients)
        elif 'dod' in df.columns and 'dischtime' in df.columns:
            # Morte durante hospitalização (dod ≈ dischtime)
            df['dod'] = pd.to_datetime(df['dod'], errors='coerce')
            df['dischtime'] = pd.to_datetime(df['dischtime'], errors='coerce')
            
            # Se morreu no mesmo dia da alta, considerar morte hospitalar
            df['hospital_death'] = (
                df['dod'].notna() & 
                (df['dod'] - df['dischtime']).dt.days.abs() <= 1
            ).astype(int)
            
            n_from_dod = df['hospital_death'].sum()
            logger.info(f"    → {n_from_dod:,} mortes via dod")
        
        else:
            logger.warning("    ⚠️  Nenhuma fonte de mortalidade disponível")
            df['hospital_death'] = 0
        
        return df
    
    def _label_icu_transfer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica transferência para ICU em 12h após entrada no ED
        Usa merge otimizado (sem loops)
        """
        if 'icustays' not in self.data_dict or self.data_dict['icustays'].empty:
            logger.warning("    ⚠️  Dados de ICU não disponíveis")
            df['icu_transfer_12h'] = 0
            return df
        
        # Preparar dados de ICU
        icu = self.data_dict['icustays'].copy()
        icu['intime_icu'] = pd.to_datetime(icu['intime'], errors='coerce')
        
        # Preparar dados de ED
        df['intime_ed'] = pd.to_datetime(df['intime'], errors='coerce')
        df['outtime_ed'] = pd.to_datetime(df['outtime'], errors='coerce')
        
        # Merge ED com ICU por subject_id e hadm_id
        df_with_icu = df.merge(
            icu[['subject_id', 'hadm_id', 'intime_icu']],
            on=['subject_id', 'hadm_id'],
            how='left'
        )
        
        # Calcular tempo desde saída do ED até entrada na ICU
        # (se ICU entry está entre ED exit e 12h depois)
        df_with_icu['hours_to_icu'] = (
            df_with_icu['intime_icu'] - df_with_icu['outtime_ed']
        ).dt.total_seconds() / 3600
        
        # ICU transfer em 12h = entrada na ICU entre 0 e 12h após saída do ED
        icu_12h_mask = (
            df_with_icu['hours_to_icu'].notna() &
            (df_with_icu['hours_to_icu'] >= 0) &
            (df_with_icu['hours_to_icu'] <= 12)
        )
        
        # Para múltiplas ICU admissions, agrupar por stay_id
        icu_transfers = df_with_icu[icu_12h_mask].groupby('stay_id').size()
        
        # Mapear de volta ao DataFrame original
        df['icu_transfer_12h'] = df['stay_id'].map(icu_transfers).fillna(0).astype(int)
        df['icu_transfer_12h'] = (df['icu_transfer_12h'] > 0).astype(int)
        
        n_icu = df['icu_transfer_12h'].sum()
        logger.info(f"    → {n_icu:,} transferências ICU em 12h")
        
        # Cleanup
        if 'hours_to_icu' in df.columns:
            df = df.drop(columns=['hours_to_icu'])
        
        return df
    
    # ========== LENGTHENED ED STAY ==========
    
    def _label_lengthened_stay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lengthened ED Stay = ED LOS > 24 horas
        
        Conforme paper: 2232 (6.90%) dos casos
        """
        logger.info("\n2️⃣  Criando label: Lengthened ED Stay")
        
        if 'ed_los_hours' not in df.columns:
            # Tentar calcular se temos intime e outtime
            if 'intime' in df.columns and 'outtime' in df.columns:
                df['intime'] = pd.to_datetime(df['intime'], errors='coerce')
                df['outtime'] = pd.to_datetime(df['outtime'], errors='coerce')
                
                df['ed_los_hours'] = (
                    df['outtime'] - df['intime']
                ).dt.total_seconds() / 3600
                
                logger.info("    → ED LOS calculado a partir de timestamps")
            else:
                logger.warning("    ⚠️  ED LOS não disponível")
                df['lengthened_ed_stay'] = 0
                return df
        
        # Label: LOS > 24h
        df['lengthened_ed_stay'] = (df['ed_los_hours'] > 24).astype(int)
        
        # Estatísticas
        n_lengthened = df['lengthened_ed_stay'].sum()
        pct_lengthened = n_lengthened / len(df) * 100
        
        logger.info(f"  ✓ Lengthened Stay: {n_lengthened:,} ({pct_lengthened:.2f}%)")
        
        # Estatísticas adicionais de ED LOS
        los_stats = df['ed_los_hours'].describe()
        logger.info(f"\n  📊 Estatísticas de ED LOS:")
        logger.info(f"    Média: {los_stats['mean']:.2f}h")
        logger.info(f"    Mediana: {los_stats['50%']:.2f}h")
        logger.info(f"    P75: {los_stats['75%']:.2f}h")
        logger.info(f"    P95: {df['ed_los_hours'].quantile(0.95):.2f}h")
        
        return df
    
    # ========== ESTATÍSTICAS E VALIDAÇÃO ==========
    
    def _print_label_statistics(self, df: pd.DataFrame):
        """Imprime estatísticas detalhadas dos labels"""
        logger.info("\n" + "="*60)
        logger.info("📊 ESTATÍSTICAS DOS LABELS")
        logger.info("="*60)
        
        total = len(df)
        
        # Critical Outcome
        if 'critical_outcome' in df.columns:
            n_critical = df['critical_outcome'].sum()
            pct_critical = n_critical / total * 100
            
            logger.info(f"\n1️⃣  Critical Outcome: {n_critical:,} ({pct_critical:.2f}%)")
            
            if 'hospital_death' in df.columns:
                n_death = df['hospital_death'].sum()
                pct_death = n_death / total * 100
                logger.info(f"    ├─ Morte hospitalar: {n_death:,} ({pct_death:.2f}%)")
            
            if 'icu_transfer_12h' in df.columns:
                n_icu = df['icu_transfer_12h'].sum()
                pct_icu = n_icu / total * 100
                logger.info(f"    └─ ICU em 12h: {n_icu:,} ({pct_icu:.2f}%)")
            
            # Overlap
            if 'hospital_death' in df.columns and 'icu_transfer_12h' in df.columns:
                both = ((df['hospital_death'] == 1) & (df['icu_transfer_12h'] == 1)).sum()
                logger.info(f"    → Overlap (morte + ICU): {both:,}")
        
        # Lengthened ED Stay
        if 'lengthened_ed_stay' in df.columns:
            n_lengthened = df['lengthened_ed_stay'].sum()
            pct_lengthened = n_lengthened / total * 100
            
            logger.info(f"\n2️⃣  Lengthened ED Stay: {n_lengthened:,} ({pct_lengthened:.2f}%)")
        
        # Crosstab
        if 'critical_outcome' in df.columns and 'lengthened_ed_stay' in df.columns:
            logger.info("\n📋 Tabela Cruzada:")
            crosstab = pd.crosstab(
                df['critical_outcome'],
                df['lengthened_ed_stay'],
                margins=True
            )
            logger.info(f"\n{crosstab}")
        
        logger.info("\n" + "="*60)
    
    def _validate_label_distribution(self, df: pd.DataFrame):
        """
        Valida se distribuição está próxima dos valores do paper
        
        Paper (Tabela 1):
        - Critical outcome: 9.67%
        - Lengthened ED stay: 6.90%
        """
        logger.info("\n🔍 Validação vs. Paper:")
        
        # Critical outcome
        if 'critical_outcome' in df.columns:
            pct_critical = df['critical_outcome'].mean() * 100
            paper_pct = 9.67
            diff = pct_critical - paper_pct
            
            status = "✓" if abs(diff) < 5 else "⚠️"
            logger.info(f"  {status} Critical Outcome: {pct_critical:.2f}% (paper: {paper_pct:.2f}%, diff: {diff:+.2f}%)")
        
        # Lengthened stay
        if 'lengthened_ed_stay' in df.columns:
            pct_lengthened = df['lengthened_ed_stay'].mean() * 100
            paper_pct = 6.90
            diff = pct_lengthened - paper_pct
            
            status = "✓" if abs(diff) < 5 else "⚠️"
            logger.info(f"  {status} Lengthened ED Stay: {pct_lengthened:.2f}% (paper: {paper_pct:.2f}%, diff: {diff:+.2f}%)")
        
        logger.info("\n💡 Diferenças são esperadas devido a:")
        logger.info("  - Filtros aplicados (admitidos, adultos, triagem completa)")
        logger.info("  - Versão do MIMIC-IV")
        logger.info("  - Definições específicas de cada instituição")
    
    def get_label_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna resumo dos labels em formato DataFrame
        
        Útil para documentação e análise
        """
        summary = []
        
        # Critical Outcome
        if 'critical_outcome' in df.columns:
            n = df['critical_outcome'].sum()
            pct = n / len(df) * 100
            
            summary.append({
                'Label': 'Critical Outcome',
                'N Positivos': n,
                '% Positivos': pct,
                'N Negativos': len(df) - n,
                '% Negativos': 100 - pct,
                'Paper %': 9.67
            })
        
        # Lengthened ED Stay
        if 'lengthened_ed_stay' in df.columns:
            n = df['lengthened_ed_stay'].sum()
            pct = n / len(df) * 100
            
            summary.append({
                'Label': 'Lengthened ED Stay',
                'N Positivos': n,
                '% Positivos': pct,
                'N Negativos': len(df) - n,
                '% Negativos': 100 - pct,
                'Paper %': 6.90
            })
        
        return pd.DataFrame(summary)
    
    def analyze_label_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Analisa correlações entre labels e features
        """
        logger.info("\n🔗 Analisando correlações...")
        
        results = {}
        
        # Features para análise
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir IDs e os próprios labels
        exclude = ['stay_id', 'subject_id', 'hadm_id', 'critical_outcome', 
                  'lengthened_ed_stay', 'hospital_death', 'icu_transfer_12h']
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        
        # Correlação com Critical Outcome
        if 'critical_outcome' in df.columns and numeric_cols:
            corr_critical = df[numeric_cols + ['critical_outcome']].corr()['critical_outcome'].abs()
            top_critical = corr_critical.sort_values(ascending=False).head(11)[1:]  # Excluir auto
            
            results['critical_outcome'] = top_critical.to_dict()
            
            logger.info("\n  Top 5 features correlacionadas com Critical Outcome:")
            for feat, corr in list(top_critical.items())[:5]:
                logger.info(f"    {feat:40s}: {corr:.4f}")
        
        # Correlação com Lengthened Stay
        if 'lengthened_ed_stay' in df.columns and numeric_cols:
            corr_lengthened = df[numeric_cols + ['lengthened_ed_stay']].corr()['lengthened_ed_stay'].abs()
            top_lengthened = corr_lengthened.sort_values(ascending=False).head(11)[1:]
            
            results['lengthened_ed_stay'] = top_lengthened.to_dict()
            
            logger.info("\n  Top 5 features correlacionadas com Lengthened ED Stay:")
            for feat, corr in list(top_lengthened.items())[:5]:
                logger.info(f"    {feat:40s}: {corr:.4f}")
        
        return results
    
    def save_labels(self, df: pd.DataFrame, output_path: str = '../data/processed/labeled_data.parquet'):
        """Salva dados com labels"""
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False, compression='snappy')
        
        file_size = output_path.stat().st_size / 1024**2
        logger.info(f"\n💾 Dados com labels salvos em: {output_path}")
        logger.info(f"   Tamanho: {file_size:.1f} MB")
        logger.info(f"   Registros: {len(df):,}")