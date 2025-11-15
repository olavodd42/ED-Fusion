#!/bin/bash
# Script para iniciar Jupyter Notebook no container

echo "🚀 Iniciando Jupyter Notebook..."
echo "📍 Acesse: http://localhost:8888"
echo ""

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
