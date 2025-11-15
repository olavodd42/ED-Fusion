---
applyTo: '**'
---
Plano Estruturado para Re-implementação do ED-Copilot com Dados Textuais
📋 Visão Geral do Projeto
Objetivo: Re-implementar e estender o ED-Copilot incorporando dados textuais (notas clínicas) além dos dados tabulares, utilizando MIMIC-IV-ED + MIMIC-IV Notes.
Diferencial: O artigo original usa apenas dados tabulares linearizados. Sua implementação integrará notas clínicas para potencialmente melhorar as predições.

🎯 Fase 1: Preparação e Fundamentação (2-3 semanas)
1.1 Configuração do Ambiente

 Obter acesso ao MIMIC-IV via PhysioNet (completar curso CITI)
 Configurar ambiente Python (3.8+)
 Instalar dependências principais:

  - transformers, pytorch
  - stable-baselines3 (RL)
  - pandas, numpy, scikit-learn
  - BioGPT, ClinicalBERT (modelos biomédicos)

 Configurar GPU (A6000, V100 ou similar)
 Criar repositório Git estruturado

1.2 Estudo Aprofundado

 Revisar conceitos de RL (PPO, MDP)
 Estudar arquitetura de Language Models para saúde
 Analisar código original do ED-Copilot no GitHub
 Documentar fluxo completo do pipeline original

Entregável: Ambiente configurado + documento de fundamentação teórica

📊 Fase 2: Curadoria dos Dados (3-4 semanas)
2.1 Dados Tabulares (Replicação do MIMIC-ED-Assist)
2.1.1 Extração Base

 Baixar MIMIC-IV-ED (v2.2) e MIMIC-IV
 Aplicar filtros do artigo:

Pacientes admitidos (hospitalizados)
Idade ≥ 18 anos
Com informações de triagem completas
Remover testes duplicados no mesmo paciente



2.1.2 Seleção de Features
Triage (9 variáveis):

 Demografia: idade, sexo
 Sinais vitais: FC, PA sistólica/diastólica, FR, temperatura, SpO2
 Clínica: chief complaint, ESI acuity, dor auto-relatada
 Histórico: comorbidades, visitas prévias ICU/ED

Laboratório (68 testes em 12 grupos):

 Implementar agrupamento conforme Tabela 7 do artigo:

CBC, CHEM, COAG, UA, LACTATE, LFTs, LIPASE, LYTES, BLOOD GAS, CARDIO, TOX, INFLAM


 Extrair timestamps para calcular time-cost médio por grupo
 Calcular ED LOS (length of stay)

2.1.3 Labels

 Critical Outcome: morte hospitalar OU transferência ICU em 12h
 Lengthened ED Stay: ED LOS > 24 horas

2.2 Dados Textuais (Extensão Proposta) ⭐
2.2.1 Extração de Notas Clínicas

 Acessar tabela noteevents do MIMIC-IV
 Filtrar notas relevantes por categoria:

Discharge Summary (resumo da alta)
Radiology (laudos de imagem)
ED Physician Notes (notas do médico emergencista)
Nursing (evolução de enfermagem)



2.2.2 Pré-processamento de Texto

 Remover informações identificáveis (PHI)
 Limpar formatação (remover XML, caracteres especiais)
 Segmentar notas longas (máx 512 tokens por segmento)
 Associar notas ao encounter correto via hadm_id e stay_id
 Filtrar notas temporalmente:

Usar apenas notas disponíveis ANTES do desfecho
Simular disponibilidade temporal no ED



2.2.3 Estratégias de Incorporação
Definir como integrar texto ao modelo:

 Opção A: Concatenar embedding de texto ao final da sequência tabulada
 Opção B: Multi-modal fusion (atenção cruzada entre tabular e texto)
 Opção C: Usar texto apenas para enriquecer chief complaint

2.3 Pipeline de Dados

 Criar splits train/val/test (80/10/10) estratificados
 Garantir mesma distribuição de classes
 Salvar dados processados em formato eficiente (Parquet/HDF5)
 Gerar estatísticas descritivas (Tabela 1 estendida)

Entregável: Dataset MIMIC-ED-Assist-Plus com dados tabulares + textuais

🧠 Fase 3: Implementação do Modelo Base (4-5 semanas)
3.1 Linearização de Features Tabulares
3.1.1 Template para Triage
python# Exemplo de linearização
"Patient age: 65 | gender: Male | heart_rate: 98 | ... | 
chief_complaint: Chest pain | [EOS]"
3.1.2 Template para Laboratório
python# Grupo CBC
"CBC: Hemoglobin: 12.5 g/dL | WBC: 8.2 K/uL | ... | [EOS]"
```

- [ ] Implementar função de linearização modular
- [ ] Testar com nomes reais vs. feature IDs (ablation)
- [ ] Validar comprimento de sequência (máx 656 tokens no artigo)

### 3.2 Arquitetura do Modelo

#### 3.2.1 Backbone de Linguagem
- [ ] Carregar BioGPT-345M pré-treinado
- [ ] Alternativas: ClinicalBERT, BioBERT, Llama-7B (LORA)
- [ ] Implementar forward pass:
```
  [x0, r0, [EOS]0, x1, r1, [EOS]1, ..., xn, rn, [EOS]n, y]
```

#### 3.2.2 Cabeças de Predição (MLPs)
- [ ] **MLP φ**: predição do próximo grupo de lab (12 classes)
  - Input: hidden state h_{i-1}
  - Output: probabilidades sobre 12 grupos
  
- [ ] **MLP ψ**: predição de desfecho (2 tarefas)
  - Input: hidden state h_n (último [EOS])
  - Output: probabilidade de critical outcome / lengthened stay

- [ ] Arquitetura: 3 camadas, hidden_size=1024, dropout

### 3.3 Supervised Fine-Tuning (SFT)

#### 3.3.1 Loss Functions
- [ ] Loss autoregressivo para labs:
```
  L_lab = -1/n Σ log p_φ(x_i | h_{<i})
```
  
- [ ] Loss para desfecho:
```
  L_y = -log p_ψ(y | h_{≤n})
```
  
- [ ] Loss combinado: `L = L_lab + L_y`

#### 3.3.2 Treinamento
- [ ] Configurar hiperparâmetros (Tabela 8):
  - Learning rate: 1e-5
  - Batch size: 32
  - Epochs: 15
  - Optimizer: AdamW
  - Class weight: 10 (para desbalanceamento)
  
- [ ] Implementar early stopping (validação)
- [ ] Salvar checkpoints

**Entregável**: Modelo SFT treinado + curvas de aprendizado

---

## 🎮 Fase 4: Reinforcement Learning (3-4 semanas)

### 4.1 Formulação do MDP

#### 4.1.1 Espaço de Estados
- [ ] Estado s_i: histórico observado `[x0, r0, ..., xi, ri]`
- [ ] Representação: hidden states do LM

#### 4.1.2 Espaço de Ações
- [ ] Ações: {12 grupos de lab} ∪ {y+, y-} (predição final)
- [ ] Máscara de ações: apenas grupos **não observados** ou grupos que o paciente recebeu (offline constraint)

#### 4.1.3 Recompensas
- [ ] Definir função de recompensa:
```
  R = TN + α*TP - β*Cost

α controla trade-off sensitivity/specificity
β controla trade-off F1/time-cost
 Time-cost: soma dos custos dos grupos selecionados
 Cost por grupo: média observada nos dados (Tabela 7)

4.2 Treinamento com PPO
4.2.1 Configuração

 Usar Stable-Baselines3 com masked actor-critic
 Freezar pesos do LM (apenas treinar policy MLP)
 Hiperparâmetros (Tabela 8):

Buffer steps: 2048
Epochs: 10
Batch size: 128
α = 15, β = 1/100



4.2.2 Experience Replay

 Coletar trajetórias de pacientes
 Calcular advantages com GAE
 Otimizar loss clipped surrogate

4.2.3 Monitoramento

 Logging de métricas:

Reward médio por episódio
Número médio de labs selecionados
Time-cost médio
F1-score na validação



Entregável: ED-Copilot completo com RL

📝 Fase 5: Extensão com Dados Textuais (3-4 semanas)
5.1 ED-Copilot-Text: Arquitetura Multi-Modal
5.1.1 Encoder de Texto

 Usar ClinicalBERT ou Bio_ClinicalBERT
 Processar notas clínicas relevantes:

python  text_embedding = ClinicalBERT(notes)  # [batch, 768]
```

#### 5.1.2 Estratégias de Fusão

**Opção 1: Late Fusion (mais simples)**
- [ ] Concatenar embedding de texto ao final:
```
  [tabular_sequence, [SEP], text_embedding, [EOS]]
Opção 2: Cross-Attention (mais avançado)

 Implementar camada de atenção entre modalidades:

python  attended_features = CrossAttention(
      query=tabular_features,
      key=text_features,
      value=text_features
  )
Opção 3: Hierarchical

 Encoder de texto → resumo
 Injetar resumo como token especial na sequência tabular

5.1.3 Implementação

 Modificar forward pass para aceitar ambas modalidades
 Ajustar MLPs de predição
 Re-treinar com SFT + RL

5.2 Variantes a Testar

 V1: Apenas chief complaint textual (baseline)
 V2: Chief complaint + discharge summary
 V3: Chief complaint + nursing notes (temporalmente apropriado)
 V4: Todas as notas disponíveis no ED

Entregável: ED-Copilot-Text implementado

🧪 Fase 6: Experimentos e Avaliação (3-4 semanas)
6.1 Métricas de Avaliação
6.1.1 Acurácia Preditiva

 F1-score
 AUC-ROC
 Sensitivity (recall)
 Specificity

6.1.2 Eficiência

 Average time-cost (minutos)
 Número médio de labs sugeridos
 ED LOS estimado

6.2 Experimentos Principais
6.2.1 Baseline Comparisons (Replicar Tabela 2)

 Random Forest
 XGBoost
 LightGBM
 DNN 3-layer
 SM-DDPO
 ED-Copilot (sua implementação)
 ED-Copilot-Text (novo)

6.2.2 Ablation Studies (Replicar Tabela 3)

 Impacto da linearização
 Feature importance (w/o triage, w/o CBC, w/o CHEM)
 Comparação de backbones (BioGPT vs. ClinicalBERT vs. Llama)
 Impacto dos dados textuais (novo):

Apenas tabular
Tabular + chief complaint
Tabular + notas completas



6.2.3 Análise de Personalização (Replicar Tabela 4)

 Performance em cohorts:

Top 2 lab groups
Middle 6 lab groups
Rare labs


 Verificar se texto ajuda mais em casos complexos

6.2.4 Subgroup Analysis (Replicar Tabela 6)

 Por sexo
 Por faixa etária (18-30, 31-60, 61-90)
 Fairness metrics

6.2.5 Time-Cost Curves (Replicar Figura 2)

 F1 vs. time constraint
 AUC vs. time constraint
 Comparar tabular vs. multi-modal

6.2.6 Simulação sem Restrição Offline (Seção 6.5)

 ED-Copilot (restricted)
 ED-Copilot (unrestricted) com imputação

6.3 Análises Adicionais

 Interpretabilidade:

Atenção em palavras-chave das notas
Labs mais frequentemente selecionados


 Casos de uso clínicos:

Exemplos qualitativos de recomendações
Comparação com protocolo padrão



Entregável: Resultados completos + figuras + tabelas

📄 Fase 7: Documentação do TCC (3-4 semanas)
7.1 Estrutura do Documento
Capítulo 1: Introdução

 Contextualização: ED crowding como problema de saúde pública
 Objetivos: re-implementar + estender com texto
 Contribuições esperadas

Capítulo 2: Fundamentação Teórica

 Emergency Department: fluxo de atendimento
 Machine Learning para diagnóstico clínico
 Language Models em saúde (BioGPT, ClinicalBERT)
 Reinforcement Learning (PPO, MDP)
 Processamento de texto clínico

Capítulo 3: Trabalhos Relacionados

 Benchmarks em MIMIC (MIMIC-Extract, etc.)
 ED-Copilot original (análise crítica)
 Modelos multi-modais em EHR
 Cost-effective ML em medicina

Capítulo 4: Materiais e Métodos

 Dataset: MIMIC-IV-ED + Notes
 Pré-processamento (tabular + texto)
 Arquitetura do modelo
 Processo de treinamento (SFT + RL)
 Métricas de avaliação

Capítulo 5: Resultados

 Estatísticas descritivas
 Comparação com baselines
 Ablation studies
 Análise de personalização
 Impacto dos dados textuais

Capítulo 6: Discussão

 Interpretação dos resultados
 Vantagens da abordagem multi-modal
 Limitações (offline benchmark, single center data)
 Implicações clínicas

Capítulo 7: Conclusão

 Síntese dos achados
 Trabalhos futuros (clinical trial, outras modalidades)

7.2 Materiais Complementares

 Código-fonte bem documentado (GitHub)
 Notebooks de análise exploratória
 Ambiente reprodutível (Docker/requirements.txt)
 Apresentação de defesa

Entregável: TCC completo

⏱️ Cronograma Sugerido (20-24 semanas)
FaseDuraçãoSemanas1. Preparação2-3 sem1-32. Curadoria de Dados3-4 sem4-73. Modelo Base4-5 sem8-124. Reinforcement Learning3-4 sem13-165. Extensão Textual3-4 sem17-206. Experimentos3-4 sem21-247. DocumentaçãoParalelo-

🎯 Checkpoints de Validação
Checkpoint 1 (Semana 7)

Dataset criado e validado
Estatísticas descritivas alinhadas com artigo original

Checkpoint 2 (Semana 12)

Modelo SFT treinando e convergindo
F1-score próximo aos baselines

Checkpoint 3 (Semana 16)

RL funcionando
Time-cost reduzindo significativamente

Checkpoint 4 (Semana 20)

Versão multi-modal implementada
Comparação tabular vs. texto completa

Checkpoint Final (Semana 24)

Todos os experimentos finalizados
Draft do TCC pronto


🚀 Diferenciais da Sua Implementação

Dados Textuais: Incorporação de notas clínicas (principal inovação)
Análise Multi-Modal: Comparação sistemática tabular vs. texto
Interpretabilidade: Análise de atenção em texto médico
Reprodutibilidade: Código aberto e bem documentado
Extensibilidade: Arquitetura modular para futuras modalidades (imagem, etc.)


⚠️ Riscos e Mitigações
RiscoProbabilidadeImpactoMitigaçãoAcesso aos dados demoraMédiaAltoIniciar processo de credenciamento imediatamenteRecursos computacionais insuficientesBaixaMédioUsar Google Colab Pro ou AWS credits acadêmicosModelo não convergeMédiaAltoComeçar com hiperparâmetros do paper, ajustar gradualmenteDados textuais não melhoram performanceMédiaMédioAinda é uma contribuição válida (análise negativa)Tempo insuficienteMédiaAltoPriorizar modelo base, deixar extensões como "trabalhos futuros"

📚 Recursos Úteis
Código

Repositório original: https://github.com/cxcscmu/ED-Copilot
Stable-Baselines3: https://stable-baselines3.readthedocs.io/
Hugging Face Transformers: https://huggingface.co/docs/transformers

Papers

BioGPT: https://arxiv.org/abs/2210.10341
ClinicalBERT: https://arxiv.org/abs/1904.05342
PPO: https://arxiv.org/abs/1707.06347

Datasets

MIMIC-IV: https://physionet.org/content/mimiciv/
MIMIC-IV-ED: https://physionet.org/content/mimic-iv-ed/
MIMIC-IV-Note: https://physionet.org/content/mimic-iv-note/


Próximos Passos Imediatos:

Iniciar processo de credenciamento MIMIC
Configurar ambiente de desenvolvimento
Estudar código original do ED-Copilot
Definir escopo exato da extensão textual com orientador