import io
import time
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import easyocr
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

# Inicializa EasyOCR
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

# Inicializa Donut
print("🔄 Inicializando Donut (documento completo, metade inferior)...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("✅ Donut carregado com sucesso!")

# Função auxiliar para Donut na metade inferior
def run_donut_bottom_half(pil_image: Image.Image, percent_bottom: float = 0.5) -> str:
    width, height = pil_image.size
    crop_start = int(height * (1 - percent_bottom))
    bottom_crop = pil_image.crop((0, crop_start, width, height)).convert("RGB")
    bottom_crop = bottom_crop.resize((512, 512))  # redimensionamento para Donut

    inputs = processor(images=bottom_crop, return_tensors="pt", task_prompt="<s_text>").to(device)
    outputs = model.generate(**inputs, max_length=1024)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return text.strip()

@app.route('/upload-png', methods=['POST'])
def upload_png():
    start = time.time()
    try:
        password = request.form.get('password', None)
        if password != EXPECTED_PASSWORD:
            return jsonify({'error': 'Senha inválida ou não fornecida.'}), 401

        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado.'}), 400

        file = request.files['file']

        keywords_str = request.for_
