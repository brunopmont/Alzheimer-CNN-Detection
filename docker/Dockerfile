# 1. Imagem base oficial da NVIDIA com CUDA 12.1 e ferramentas de desenvolvimento
# Usamos 'devel' em vez de 'runtime' para garantir que bibliotecas como Antspy compilem se necessário
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. Configurações de ambiente para evitar interação durante a instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo

ENV CACHE_BUST=v2

# 3. Instalar dependências do sistema e repositório para Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# 4. Instalar Python 3.12 e ferramentas de pip
RUN apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip

# 5. Garantir que 'python' e 'pip' apontem para a versão 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 6. Definir diretório de trabalho
WORKDIR /app

# 7. Instalar PyTorch PRIMEIRO (Devido à URL específica do CUDA 12.1)
# Isso evita conflitos de dependência com o TensorFlow depois
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 8. Copiar o requirements (sem as linhas do torch)
COPY requirements_limpo.txt .

# 9. Instalar o resto das dependências (TensorFlow, Antspy, etc.)
RUN pip install --no-cache-dir -r requirements_limpo.txt

# 10. Comando padrão ao iniciar o container
CMD ["python3.12"]