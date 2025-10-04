FROM python:3.10-slim

# Instala dependências do sistema (inclui pacotes necessários para o EasyOCR)
RUN apt-get update && apt-get install -y \
    libgl1 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Define a porta que o Railway vai expor
EXPOSE 5000

# Define a variável de ambiente para evitar o modo debug
ENV FLASK_ENV=production

# Comando para iniciar o servidor Flask
CMD ["python", "main.py"]
