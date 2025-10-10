FROM python:3.10-slim

# Instala dependências do sistema necessárias para EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    poppler-utils \
    build-essential \       # gcc, g++, make, etc.
    cmake \                 # necessário para algumas extensões PyTorch
    ninja-build \           # Ninja para compilar rapidamente as extensões
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

EXPOSE 5000
ENV FLASK_ENV=production

CMD ["python", "main.py"]
