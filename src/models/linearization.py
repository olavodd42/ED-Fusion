"""
Linearização de features para ED-Copilot.
Adaptado para estrutura MIMIC-IV processada.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import ast

class FeatureLinearizer:
    """
    Lineariza features tabulares (triage + labs) em sequências de texto.
    Métodos:
        * linearize_triage: str -> Lineariza informações de triagem.
        * get_performed_groups: List[Dict] -> Identifica grupos de laboratório realizados.
        * linearize_lab_group: str -> Lineariza um grupo de laboratório.
        * linearize_full_sequence: str -> Lineariza sequência completa (triage + labs).
        * create_training_examples: pd.DataFrame -> Cria exemplos de treinamento a partir do Data
    """
    
    def __init__(self):
        # Mapeamento de colunas triage
        self.triage_features = {
            'age': 'triage_age',
            'heart_rate': 'triage_heart_rate',
            'respiratory_rate': 'triage_respiratory_rate',
            'sbp': 'triage_sbp',
            'dbp': 'triage_dbp',
            'temperature': 'triage_temperature',
            'spo2': 'triage_spo2',
            'acuity': 'triage_acuity',
            'pain': 'triage_pain',
            'chief_complaint': 'triage_chief_complaint'
        }
        
        # Grupos de laboratório
        self.lab_groups = [
            'CBC', 'CHEM', 'COAG', 'UA', 'LACTATE', 
            'LFTS', 'LIPASE', 'LYTES', 'CARDIO', 
            'BLOOD_GAS', 'INFLAM'
        ]
        
        # Time-costs (do paper)
        self.time_costs = {
            'CBC': 30,
            'CHEM': 60,
            'COAG': 48,
            'UA': 40,
            'LACTATE': 4,
            'LFTS': 104,
            'LIPASE': 100,
            'LYTES': 89,
            'BLOOD_GAS': 12,
            'CARDIO': 122,
            'INFLAM': 178
        }
    
    def linearize_triage(self, row: pd.Series) -> str:
        """
        Lineariza informações de triagem.
        
        Formato: "feature1: value1 | feature2: value2 | ... | [EOS]"
        Parâmetros:
            * row: pd.Series -> Linha do DataFrame com dados do paciente.
        Retorna:
            * str -> Sequência linearizada de triagem.
        """

        parts = []
        
        # Gender (one-hot → string)
        if row.get('triage_gender_male', 0) == 1:
            parts.append("gender: Male")
        elif row.get('triage_gender_female', 0) == 1:
            parts.append("gender: Female")
        
        # Outras features
        for logical_name, col_name in self.triage_features.items():
            if col_name in row.index and pd.notna(row[col_name]):
                value = row[col_name]
                
                # Formatação especial
                if logical_name == 'chief_complaint':
                    # Limitar tamanho e remover pipe
                    value = str(value).replace('|', ';')[:100]
                elif logical_name in ['temperature', 'sbp', 'dbp', 'spo2']:
                    # Arredondar valores numéricos
                    value = f"{float(value):.1f}"
                
                parts.append(f"{logical_name}: {value}")
        
        if not parts:
            return "[EOS]"
        
        return " | ".join(parts) + " | [EOS]"
    
    def get_performed_groups(self, row: pd.Series) -> List[Dict]:
        """
        Identifica grupos realizados e retorna em ordem temporal.
        
        Parâmetros:
            * row: pd.Series -> Linha do DataFrame com dados do paciente.
        Retorna:
            * List[Dict] -> Lista de dicionários com info dos grupos realizados.
        """
        performed = []
        
        for group in self.lab_groups:
            ordered_col = f'lab_{group}_ordered'
            value_col = f'lab_{group}_value'
            time_col = f'lab_{group}_time'
            
            # Verificar se foi ordenado
            if row.get(ordered_col, 0) == 1:
                group_info = {
                    'group': group,
                    'value': row.get(value_col, np.nan),
                    'time': row.get(time_col, np.nan),
                    'time_cost': self.time_costs.get(group, 60)
                }
                performed.append(group_info)
        
        # Ordenar por timestamp (se disponível)
        performed_with_time = [g for g in performed if pd.notna(g['time'])]
        performed_without_time = [g for g in performed if pd.isna(g['time'])]
        
        if performed_with_time:
            performed_with_time.sort(key=lambda x: x['time'])
        
        return performed_with_time + performed_without_time
    
    def linearize_lab_group(self, group_info: Dict) -> str:
        """
        Lineariza um grupo de laboratório.
        Parâmetros:
            * group_info: Dict -> Dicionário com info do grupo (group, value).
        Retorna:
            * str -> Sequência linearizada do grupo de laboratório.
        """
        group = group_info['group']
        value = group_info['value']
        
        if pd.notna(value):
            # Se value for string (ex: dict serializado), tentar parsear
            if isinstance(value, str):
                try:
                    # Tentar desserializar
                    value_dict = ast.literal_eval(value)
                    if isinstance(value_dict, dict):
                        # Múltiplos valores
                        parts = [f"{k}: {v}" for k, v in value_dict.items()]
                        return f"{group}: " + " | ".join(parts) + " | [EOS]"
                except:
                    pass
            
            # Valor único
            return f"{group}: {value} | [EOS]"
        else:
            # Sem valor disponível, apenas indicar grupo
            return f"{group}: ordered | [EOS]"
    
    def linearize_full_sequence(self, row: pd.Series) -> Dict:
        """
        Lineariza sequência completa: triage + labs.
        Parâmetros:
            * row: pd.Series -> Linha do DataFrame com dados do paciente.
        
        Retorna:
            * Dict -> Dicionário com:
            {
                'text_sequence': str,
                'lab_groups': List[str],
                'time_costs': List[int],
                'eos_positions': List[int],
                'total_time_cost': int
            }
        """

        # 1. Triage
        triage_text = self.linearize_triage(row)
        sequence_parts = [triage_text]
        eos_positions = [len(triage_text.split())]
        
        # 2. Labs em ordem temporal
        performed = self.get_performed_groups(row)
        
        lab_groups = []
        time_costs = []
        
        for group_info in performed:
            lab_text = self.linearize_lab_group(group_info)
            sequence_parts.append(lab_text)
            
            lab_groups.append(group_info['group'])
            time_costs.append(group_info['time_cost'])
            
            eos_positions.append(
                eos_positions[-1] + len(lab_text.split())
            )
        
        return {
            'text_sequence': " ".join(sequence_parts),
            'lab_groups': lab_groups,
            'time_costs': time_costs,
            'eos_positions': eos_positions,
            'total_time_cost': sum(time_costs),
            'num_labs': len(lab_groups)
        }
    
    def create_training_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria exemplos de treinamento a partir do DataFrame.
        Parâmetros:
            * df: pd.DataFrame -> DataFrame com dados dos pacientes.
        Retorna:
            * pd.DataFrame -> DataFrame com exemplos linearizados.
        """
        
        examples = []
        
        print("🔄 Linearizando sequências...")
        for idx, row in df.iterrows():
            linearized = self.linearize_full_sequence(row)
            
            examples.append({
                'subject_id': row.get('subject_id'),
                'hadm_id': row.get('hadm_id'),
                'stay_id': row.get('stay_id'),
                'text_sequence': linearized['text_sequence'],
                'lab_groups': linearized['lab_groups'],
                'time_costs': linearized['time_costs'],
                'eos_positions': linearized['eos_positions'],
                'total_time_cost': linearized['total_time_cost'],
                'num_labs': linearized['num_labs'],
                # Labels
                'outcome': row.get('outcome', 0),
                'critical_outcome': row.get('critical_outcome', row.get('outcome', 0)),
                'lengthened_stay': row.get('lengthened_stay', 0),
                # Texto clínico (se disponível)
                'clinical_text': row.get('cleaned_text', None),
                'has_text': pd.notna(row.get('cleaned_text'))
            })
            
            if (idx + 1) % 10000 == 0:
                print(f"   Processados {idx+1:,} / {len(df):,}")
        
        result = pd.DataFrame(examples)
        
        print(f"\n✅ Linearização completa!")
        print(f"   - Total: {len(result):,} exemplos")
        print(f"   - Com labs: {(result['num_labs'] > 0).sum():,} ({(result['num_labs'] > 0).mean()*100:.1f}%)")
        print(f"   - Com texto: {result['has_text'].sum():,} ({result['has_text'].mean()*100:.1f}%)")
        
        return result


# Teste
if __name__ == "__main__":
    # Carregar amostra
    df = pd.read_parquet('data/processed/multimodal_train.parquet')
    print(f"📋 Dataset: {len(df):,} linhas")
    print(f"📊 Colunas: {df.columns.tolist()[:10]}...")
    
    # Linearizar
    linearizer = FeatureLinearizer()
    result = linearizer.create_training_examples(df.head(100))
    
    # Mostrar exemplo
    print("\n" + "="*80)
    print("📝 EXEMPLO DE SEQUÊNCIA LINEARIZADA")
    print("="*80)
    
    # Pegar exemplo com labs
    example = result[result['num_labs'] > 0].iloc[0]
    
    print(f"\nStay ID: {example['stay_id']}")
    print(f"Grupos: {example['lab_groups']}")
    print(f"Time-cost: {example['total_time_cost']} min")
    print(f"Outcome: {example['outcome']}")
    print(f"\nSequência ({len(example['text_sequence'].split())} tokens):")
    print(example['text_sequence'][:800] + "...")
    
    if example['has_text']:
        print(f"\nTexto clínico disponível: {len(example['clinical_text'])} chars")