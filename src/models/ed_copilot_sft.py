import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class EDCopilotSFT(nn.Module):
    """
    ED-Copilot com Supervised Fine-Tuning.
    
    Arquitetura:
        - Backbone: BioGPT (345M params)
        - MLP φ: prediz próximo grupo de lab (12 classes)
        - MLP ψ: prediz desfechos (2 tarefas binárias)'
    Parâmetros:
        * model_name: str -> Nome do modelo pré-treinado.
        * num_lab_groups: int -> Número de grupos de exames laboratoriais.
        * hidden_size: int -> Tamanho das camadas ocultas das MLPs.
        * dropout: float -> Taxa de dropout nas MLPs.
    Métodos:
        * forward: dict -> Forward pass.
        * compute_loss: dict -> Calcula loss combinado.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BioGPT",
        num_lab_groups: int = 12,
        hidden_size: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Backbone LM
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Resize embeddings se adicionamos [EOS]
        # (isso será feito no script de treinamento)
        
        # MLP φ: predição de próximo grupo
        self.lab_predictor = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_lab_groups)
        )
        
        # MLP ψ: predição de desfechos (2 tarefas)
        self.outcome_predictor = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)  # critical_outcome, lengthened_stay
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Parâmetros:
            * input_ids: torch.Tensor -> IDs dos tokens [batch, seq_len].
            * attention_mask: torch.Tensor -> Máscara de atenção [batch, seq_len].
        Retorna:
            * dict -> Dicionário com saídas:
                - hidden_states: torch.Tensor -> Estados ocultos do backbone.
                - lab_logits: torch.Tensor -> Logits para predição de grupos de lab.
                - outcome_logits: torch.Tensor -> Logits para predição de desfechos.
        """
        # Extrair hidden states do backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Predição de próximo grupo em cada posição [EOS]
        lab_logits = self.lab_predictor(hidden_states)  # [batch, seq_len, num_lab_groups]
        
        # Predição de desfecho usando último [EOS]
        # (na prática, vamos identificar a posição do último [EOS] válido)
        last_hidden = hidden_states[:, -1, :]  # Simplificado: usar último token
        outcome_logits = self.outcome_predictor(last_hidden)  # [batch, 2]
        
        return {
            'hidden_states': hidden_states,
            'lab_logits': lab_logits,
            'outcome_logits': outcome_logits
        }
    
    def compute_loss(self, outputs, batch, class_weight=10.0):
        """
        Calcula loss combinado: L_lab + L_y
        
        Parâmetros:
            * outputs: dict -> Saídas do forward pass.
            * batch: dict -> Batch de dados com labels.
            * class_weight: float -> Peso para classe positiva na BCE.
        Retorna:
            * dict -> Dicionário com losses:
                - loss: torch.Tensor -> Loss total.
                - loss_lab: float -> Loss de predição de grupos de lab.
                - loss_outcome: float -> Loss de predição de desfechos.
        """
        
        lab_logits = outputs['lab_logits']
        outcome_logits = outputs['outcome_logits']
        
        # Loss autoregressive para labs
        # Simplificado: usar apenas primeira predição
        lab_pred = lab_logits[:, 0, :]  # [batch, num_lab_groups]
        lab_target = batch['lab_group_labels'][:, 0]  # [batch]
        
        # Ignorar padding (-100)
        valid_mask = lab_target != -100
        if valid_mask.sum() > 0:
            loss_lab = nn.functional.cross_entropy(
                lab_pred[valid_mask],
                lab_target[valid_mask]
            )
        else:
            loss_lab = torch.tensor(0.0, device=lab_pred.device)
        
        # Loss para desfechos (BCE com peso para classe positiva)
        critical_target = batch['critical_outcome'].unsqueeze(1)
        lengthened_target = batch['lengthened_stay'].unsqueeze(1)
        targets = torch.cat([critical_target, lengthened_target], dim=1)
        
        # Calcular pesos dinamicamente
        pos_weight = torch.tensor([class_weight, class_weight], device=targets.device)
        loss_outcome = nn.functional.binary_cross_entropy_with_logits(
            outcome_logits,
            targets,
            pos_weight=pos_weight
        )
        
        # Loss total
        total_loss = loss_lab + loss_outcome
        
        return {
            'loss': total_loss,
            'loss_lab': loss_lab.item(),
            'loss_outcome': loss_outcome.item()
        }