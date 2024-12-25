FROM ubuntu:latest

# Atualizar e instalar dependências
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configurar o diretório de trabalho
WORKDIR /app

# Criar e ativar um ambiente virtual
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Instalar dependências do Langflow e do Componente Gemini
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clonar o repositório do Langflow
RUN git clone https://github.com/langflow-ai/langflow.git /app/langflow

# Instalar o Langflow a partir do repositório clonado
RUN pip install --no-cache-dir /app/langflow

# Configurar a pasta para componentes customizados
COPY custom_components/ /app/langflow/src/backend/langflow/custom_components/

# Definir a chave secreta do Langflow (alterar no docker-compose ou no Portainer)
ENV LANGFLOW_SECRET_KEY="CHANGE_THIS_TO_A_REAL_SECRET"

# Expor a porta 7860
EXPOSE 7860

# Comando para iniciar o Langflow
CMD ["langflow", "--host", "0.0.0.0", "--port", "7860"]
