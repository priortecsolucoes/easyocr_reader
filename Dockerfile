FROM python:3.10-slim

# Instala dependências do sistema (inclui poppler para pdf2image)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY . .

# Porta para Railway
EXPOSE 8080

# Comando de inicialização
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
