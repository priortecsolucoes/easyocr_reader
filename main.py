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

print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

print("🔄 Inicializando Donut...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("✅ Donut carregado com sucesso!")

def run_donut_bottom(pil_image: Image.Image, bottom_percent: float = 0.35) -> str:
    width, height = pil_image.size
    start_crop = int(height * (1 - bottom_percent))
    bottom_crop = pil_image.crop((0, start_crop, width, height))
    bottom_crop = bottom_crop.convert("RGB").resize((480, 480))
    
    pixel_values = processor(bottom_crop, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=1024)
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        task_prompt="<s_doc>"
    )[0]
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
        keywords_str = request.form.get('keywords', None)
        percent_str = request.form.get('percent', '0.25')
        percent_donut_str = request.form.get('percent_donut', '0.50')

        keywords = []
        if keywords_str and keywords_str.strip():
            keywords = [k.strip().upper() for k in keywords_str.split(',') if k.strip()]

        try:
            percent = float(percent_str)
            percent_donut = float(percent_donut_str)
            if not (0 < percent <= 1) or not (0 < percent_donut <= 1):
                raise ValueError
        except Exception:
            return jsonify({'error': 'Percentual inválido. Use um valor entre 0 e 1.'}), 400

        img = Image.open(file.stream).convert("RGB")
        width, height = img.size

        # Leitura EasyOCR
        if not keywords:
            easy_text = reader.readtext(np.array(img), detail=0, paragraph=True)
        else:
            top_crop = img.crop((0, 0, width, int(height * percent)))
            partial_text = reader.readtext(np.array(top_crop), detail=0, paragraph=True)
            partial_text_joined = " ".join(partial_text).upper()

            if any(keyword in partial_text_joined for keyword in keywords):
                bottom_crop = img.crop((0, int(height * percent), width, height))
                rest_text = reader.readtext(np.array(bottom_crop), detail=0, paragraph=True)
                easy_text = partial_text + rest_text
            else:
                total_time = time.time() - start
                return jsonify({
                    'status': 'not_identified',
                    'message': 'Documento não identificado pelas palavras-chave fornecidas no início.',
                    'ocr_result': partial_text,
                    'time': round(total_time, 2)
                })

        easy_text_joined = " ".join(easy_text)

        # Leitura Donut na parte inferior
        donut_text = run_donut_bottom(img, bottom_percent=percent_donut)

        combined_result = easy_text_joined.strip() + "\n---\n" + donut_text.strip()
        total_time = time.time() - start

        return jsonify({
            'status': 'success',
            'easyocr_text': easy_text_joined,
            'donut_text': donut_text,
            'ocr_result': combined_result,
            'time': round(total_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
