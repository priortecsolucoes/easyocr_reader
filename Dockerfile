FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    poppler-utils \
    build-essential \  
    cmake \            
    ninja-build \      
    git \              
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_ENV=production
ENV HF_HOME=/root/.cache/huggingface  

EXPOSE 5000

CMD ["python", "main.py"]
