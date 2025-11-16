import sys
sys.path.append('src')

import pandas as pd
from pathlib import Path

def inspect_data():
    print("🔍 Inspecionando dados brutos...\n")
    
    # Carregar um dos arquivos processados
    train_path = Path('C:/Users/OlavoDefendiDalberto/Projetos/ED-Fusion2/data/processed/multimodal_train.parquet')
    
    if not train_path.exists():
        print(f"❌ Arquivo não encontrado: {train_path}")
        return
    
    df = pd.read_parquet(train_path)
    
    print("=" * 80)
    print("📊 INFO GERAL")
    print("=" * 80)
    print(f"Número de linhas: {len(df)}")
    print(f"Número de colunas: {len(df.columns)}")
    print(f"\n🔢 Primeiras colunas:\n{df.columns.tolist()[:20]}")
    
    print("\n" + "=" * 80)
    print("📋 EXEMPLO DE 1 LINHA COMPLETA")
    print("=" * 80)
    first_row = df.iloc[0]
    for col, val in first_row.items():
        if pd.notna(val):  # Mostrar apenas valores não-nulos
            print(f"{col:30s} = {val}")
    
    print("\n" + "=" * 80)
    print("🧪 TRIAGE FEATURES (esperadas)")
    print("=" * 80)
    expected_triage = [
        'age', 'gender', 'heartrate', 'resprate', 
        'o2sat', 'sbp', 'dbp', 'temperature',
        'acuity', 'chiefcomplaint', 'pain'
    ]
    
    for feat in expected_triage:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            print(f"✅ {feat:20s} - {non_null}/{len(df)} valores ({non_null/len(df)*100:.1f}%)")
        else:
            print(f"❌ {feat:20s} - COLUNA NÃO ENCONTRADA")
    
    print("\n" + "=" * 80)
    print("🧬 LABORATÓRIO FEATURES (amostra)")
    print("=" * 80)
    lab_samples = ['hemoglobin', 'wbc', 'creatinine', 'sodium', 'glucose']
    
    for feat in lab_samples:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            print(f"✅ {feat:20s} - {non_null}/{len(df)} valores ({non_null/len(df)*100:.1f}%)")
        else:
            print(f"❌ {feat:20s} - COLUNA NÃO ENCONTRADA")
    
    print("\n" + "=" * 80)
    print("🎯 LABELS")
    print("=" * 80)
    if 'critical_outcome' in df.columns:
        print(f"✅ critical_outcome: {df['critical_outcome'].sum()} positivos ({df['critical_outcome'].mean()*100:.1f}%)")
    if 'lengthened_stay' in df.columns:
        print(f"✅ lengthened_stay: {df['lengthened_stay'].sum()} positivos ({df['lengthened_stay'].mean()*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("💡 RECOMENDAÇÃO")
    print("=" * 80)
    
    # Detectar features que realmente existem
    possible_age_cols = [c for c in df.columns if 'age' in c.lower()]
    possible_gender_cols = [c for c in df.columns if 'gender' in c.lower() or 'sex' in c.lower()]
    
    if possible_age_cols:
        print(f"🔍 Possíveis colunas de idade: {possible_age_cols}")
    if possible_gender_cols:
        print(f"🔍 Possíveis colunas de sexo: {possible_gender_cols}")
    
    # Salvar mapeamento sugerido
    print("\n📝 Crie um arquivo 'column_mapping.json' com o mapeamento correto:")
    print("Exemplo:")
    print('''{
    "age": "anchor_age",
    "gender": "gender",
    "heartrate": "heart_rate",
    ...
}''')

if __name__ == "__main__":
    inspect_data()