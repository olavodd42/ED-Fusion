# Dados do ED-Copilot

## Como obter os dados

### MIMIC-IV
1. Acesse: https://physionet.org/content/mimiciv/
2. Complete o treinamento CITI
3. Faça download dos módulos:
   - `hosp/` - Hospital data
   - `ed/` - Emergency Department data
   - `icu/` - ICU data

### Estrutura esperada
```
data/
├── raw/
│   ├── mimic-iv-ed/
│   ├── mimic-iv-hosp/
│   └── mimic-iv-icu/
├── processed/
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
└── interim/
```

## Notas importantes
- **NÃO commite dados reais do MIMIC**
- Os dados são protegidos por HIPAA
- Use apenas para fins de pesquisa aprovados
