# Criar arquivo Dockerfile com suporte a GPU
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Instalar Python 3.10 e dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Criar link simbólico para python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Atualizar pip
RUN pip install --upgrade pip

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configurar Jupyter
RUN mkdir -p /root/.jupyter && \
    jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

# Expor porta do Jupyter
EXPOSE 8888

CMD ["/bin/bash"]