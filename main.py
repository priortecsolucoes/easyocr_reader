import io
import time
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

# -------------------------------
# 🔄 Inicialização dos modelos OCR
# -------------------------------
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

print("🔄 Inicializando TrOCR (manuscrito)...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("✅ TrOCR carregado com sucesso!")

# -------------------------------
# Função auxiliar para ler com TrOCR na parte inferior
# -------------------------------
def run_trocr_bottom_half(pil_image: Image.Image, percent_bottom: float = 0.35) -> str:
    """Executa o TrOCR apenas na parte inferior da imagem, com pré-processamento."""
    width, height = pil_image.size
    crop_start = int(height * (1 - percent_bottom))
    bottom_crop = pil_image.crop((0, crop_start, width, height))

    # Pré-processamento
    bottom_crop = bottom_crop.convert("L")  # escala de cinza
    bottom_crop = bottom_crop.filter(ImageFilter.MedianFilter(size=3))
    bottom_crop = ImageEnhance.Contrast(bottom_crop).enhance(2.0)
    bottom_crop = ImageEnhance.Sharpness(bottom_crop).enhance(1.5)

    # Executa TrOCR
    pixel_values = processor(images=bottom_crop, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=128)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# -------------------------------
# Endpoint principal
# -------------------------------
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

        keywords_str = request.form.get('keywords', None)
        percent_str = request.form.get('percent', '0.25')
        percent_trocr_str = request.form.get('percent_trocr', '0.35')  # fração inferior para TrOCR

        # -------------------------------
        # Parâmetros
        # -------------------------------
        keywords = []
        if keywords_str and keywords_str.strip():
            keywords = [k.strip().upper() for k in keywords_str.split(',') if k.strip()]

        try:
            percent = float(percent_str)
            if not (0 < percent <= 1):
                raise ValueError
            percent_trocr = float(percent_trocr_str)
            if not (0 < percent_trocr <= 1):
                raise ValueError
        except Exception:
            return jsonify({'error': 'Percentual inválido. Use um valor entre 0 e 1.'}), 400

        # -------------------------------
        # Carrega imagem
        # -------------------------------
        img = Image.open(file.stream).convert("RGB")
        width, height = img.size

        # -------------------------------
        # 🔹 Leitura com EasyOCR
        # -------------------------------
        if not keywords:
            image_np = np.array(img)
            easy_text = reader.readtext(image_np, detail=0, paragraph=True)
        else:
            top_crop = img.crop((0, 0, width, int(height * percent)))
            image_top_np = np.array(top_crop)
            partial_text = reader.readtext(image_top_np, detail=0, paragraph=True)
            partial_text_joined = " ".join(partial_text).upper()

            if any(keyword in partial_text_joined for keyword in keywords):
                bottom_crop = img.crop((0, int(height * percent), width, height))
                image_bottom_np = np.array(bottom_crop)
                rest_text = reader.readtext(image_bottom_np, detail=0, paragraph=True)
                easy_text = partial_text + rest_text
            else:
                total_time = time.time() - start
                print("⏹ Documento não identificado pelas palavras-chave fornecidas.")
                return jsonify({
                    'status': 'not_identified',
                    'message': 'Documento não identificado pelas palavras-chave fornecidas no início.',
                    'ocr_result': partial_text,
                    'time': round(total_time, 2)
                })

        easy_text_joined = " ".join(easy_text)

        # -----------------------------
