import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import ast

class EDCopilotDataset(Dataset):
    """
    Dataset para ED-Copilot.
    Parâmetros:
        * data_path: str -> Caminho para o arquivo Parquet processado.
        * tokenizer_name: str -> Nome do tokenizer pré-treinado.
        * max_length: int -> Comprimento máximo da sequência tokenizada.
    
    Retorna:
        * Dataset PyTorch com tokenizações e labels:
            - input_ids: torch.Tensor -> IDs dos tokens.
            - attention_mask: torch.Tensor -> Máscara de atenção.
            - lab_group_labels: torch.Tensor -> Labels dos grupos de exames laboratoriais.
            - critical_outcome: torch.Tensor -> Label de desfecho crítico.
            - lengthened_stay: torch.Tensor -> Label de permanência prolongada.
            - num_labs: torch.Tensor -> Número de exames laboratoriais.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "microsoft/BioGPT",
        max_length: int = 656  # Conforme paper
    ):
        self.df = pd.read_parquet(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Adiciona token [EOS] se não existir
        if '[EOS]' not in self.tokenizer.vocab:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[EOS]']})
        
        # Mapeamento de grupos para IDs
        self.group_to_id = {
            'CBC': 0, 'CHEM': 1, 'COAG': 2, 'UA': 3,
            'LACTATE': 4, 'LFTs': 5, 'LIPASE': 6, 'LYTES': 7,
            'BLOOD_GAS': 8, 'CARDIO': 9, 'TOX': 10, 'INFLAM': 11
        }
        self.num_groups = len(self.group_to_id)
        
    def __len__(self) -> int:
        """Retorna o número de exemplos no dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Obtém um exemplo do dataset.
        Parâmetros:
            * idx: int -> Índice do exemplo.
        Retorna:
            * dict -> Dicionário com tensores para o exemplo.
        """
        row = self.df.iloc[idx]
        
        # Tokenizar sequência
        text_sequence = row['text_sequence']
        encoding = self.tokenizer(
            text_sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels de grupos (autoregressive)
        lab_groups = ast.literal_eval(row['lab_groups']) if isinstance(row['lab_groups'], str) else row['lab_groups']
        lab_group_ids = [self.group_to_id[g] for g in lab_groups]
        
        # Pad com -100 (ignorado pela loss)
        lab_group_labels = torch.full((self.num_groups,), -100, dtype=torch.long)
        for i, gid in enumerate(lab_group_ids[:self.num_groups]):
            lab_group_labels[i] = gid
        
        # Labels de desfecho (classificação binária)
        critical_outcome = torch.tensor(row['critical_outcome'], dtype=torch.float32)
        lengthened_stay = torch.tensor(row['lengthened_stay'], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'lab_group_labels': lab_group_labels,
            'critical_outcome': critical_outcome,
            'lengthened_stay': lengthened_stay,
            'num_labs': torch.tensor(len(lab_group_ids), dtype=torch.long)
        }