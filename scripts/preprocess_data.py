"""
Script principal de preprocessamento dos dados MIMIC-IV-ED
"""
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import yaml
import logging
from src.data import MIMICLoader, EDPreprocessor, FeatureEngineer, OutcomeLabeler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = './configs/default.yaml') -> dict:
    """
    Carrega arquivo de configuração.
    Parâmetros:
        * config_path: Caminho para o arquivo de configuração YAML.
    Retorno:
        * dict: Dicionário com as configurações carregadas.
    """
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Pipeline completo de preprocessamento"""
    logger.info("="*80)
    logger.info("ED-COPILOT: PIPELINE DE PREPROCESSAMENTO")
    logger.info("="*80)
    
    # 1. Carregar configuração
    config = load_config()
    data_root = config['paths']['raw_data']
    processed_path = Path(config['paths']['processed_data'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Carregar dados brutos
    logger.info("\n📂 ETAPA 1: Carregamento de Dados")
    loader = MIMICLoader(data_root=data_root)
    
    # Carregar sem labevents inicialmente (otimização de memória)
    data = loader.load_all(load_labs=False)
    logger.info("💡 Labevents não carregado para economizar memória")
    logger.info("   (será carregado sob demanda durante feature engineering)")
    
    # Mostrar resumo
    summary = loader.get_data_summary()
    logger.info(f"\n{summary.to_string()}\n")
    
    # 3. Aplicar filtros e preprocessamento
    logger.info("\n🔧 ETAPA 2: Preprocessamento")
    preprocessor = EDPreprocessor(data)
    df_filtered = preprocessor.apply_filters()
    
    # Mostrar resumo dos filtros
    filter_summary = preprocessor.get_filter_summary()
    logger.info(f"\n{filter_summary.to_string()}\n")
    
    # 4. Engenharia de features
    logger.info("\n⚙️  ETAPA 3: Engenharia de Features")
    
    # Features de triagem (não precisa de labs)
    logger.info("Extraindo features de triagem...")
    engineer = FeatureEngineer(
        df=df_filtered,
        lab_data=pd.DataFrame(),  # Vazio por enquanto
        lab_items=data['d_labitems']
    )
    df_features = engineer.extract_triage_features()
    
    # Features de laboratório (carregar labs sob demanda se necessário)
    extract_lab_features = config.get('features', {}).get('extract_labs', False)
    
    if extract_lab_features:
        logger.info("⏳ Carregando labevents para extração de features...")
        data['labevents'] = loader._load_csv('hosp/labevents.csv', optimize_dtypes=True)
        
        engineer.lab_data = data['labevents']
        df_features = engineer.extract_lab_features(time_window_hours=12, calculate_costs=True)
    else:
        logger.info("⏭️  Features de laboratório não extraídas (extract_labs=False no config)")
    
    # 5. Criar labels
    logger.info("\n🏷️  ETAPA 4: Criação de Labels")
    labeler = OutcomeLabeler(df_features, data['icustays'])
    df_final = labeler.create_all_labels()
    
    # 6. Salvar dados processados
    logger.info("\n💾 ETAPA 5: Salvando Dados Processados")
    
    output_file = processed_path / 'ed_copilot_processed.parquet'
    df_final.to_parquet(output_file, index=False)
    logger.info(f"✓ Dados salvos em: {output_file}")
    logger.info(f"  → Shape: {df_final.shape}")
    logger.info(f"  → Tamanho: {output_file.stat().st_size / 1024**2:.2f} MB")
    
    # Salvar também resumo estatístico
    stats_file = processed_path / 'preprocessing_stats.csv'
    stats = {
        'total_records': len(df_final),
        'unique_patients': df_final['subject_id'].nunique(),
        'critical_outcome_pct': df_final['critical_outcome'].mean() * 100,
        'lengthened_stay_pct': df_final['lengthened_stay'].mean() * 100,
        'mean_age': df_final['age'].mean(),
        'mean_ed_los': df_final['ed_los_hours'].mean()
    }
    pd.DataFrame([stats]).to_csv(stats_file, index=False)
    logger.info(f"✓ Estatísticas salvas em: {stats_file}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ PREPROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("="*80)


if __name__ == "__main__":
    main()