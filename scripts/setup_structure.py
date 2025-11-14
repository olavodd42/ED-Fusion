"""
Script para criar a estrutura completa do projeto ED-Copilot
"""
import os
from pathlib import Path

def create_structure():
    """Cria a estrutura de diretórios e arquivos básicos"""
    
    # Estrutura de diretórios
    directories = [
        # Data
        "data/raw",
        "data/processed",
        "data/interim",
        
        # Source code
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        
        # Scripts
        "scripts",
        
        # Configs
        "configs",
        
        # Tests
        "tests",
        
        # Results
        "results/figures",
        "results/tables",
        "results/checkpoints",
        
        # Docs
        "docs",
        
        # Paper
        "paper/figures",
        
        # Notebooks (já existe)
        "notebooks"
    ]
    
    # Criar diretórios
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Criado: {directory}/")
    
    # Criar arquivos __init__.py
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Criado: {init_file}")
    
    # Criar READMEs
    readme_contents = {
        "data/README.md": """# Dados do ED-Copilot

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
""",
        
        "paper/README.md": """# TCC - ED-Copilot

Documentos relacionados ao Trabalho de Conclusão de Curso.

## Estrutura
- `main.tex` - Documento principal (LaTeX)
- `references.bib` - Referências bibliográficas
- `figures/` - Figuras para o documento
"""
    }
    
    for filepath, content in readme_contents.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Criado: {filepath}")
    
    print("\n✅ Estrutura criada com sucesso!")
    print("\n📁 Estrutura do projeto:")
    print_tree(".", prefix="", max_depth=2)

def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Imprime árvore de diretórios"""
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
        dirs = [e for e in entries if e.is_dir() and not e.name.startswith('.')]
        
        for i, entry in enumerate(dirs):
            is_last = i == len(dirs) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{entry.name}/")
            
            extension = "    " if is_last else "│   "
            print_tree(entry, prefix + extension, max_depth, current_depth + 1)
    except PermissionError:
        pass

if __name__ == "__main__":
    create_structure()