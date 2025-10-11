import io
import time
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import easyocr

from doctr.models import ocr_predictor
from doctr.io import DocumentFile

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

# -------------------------------
# Inicializando EasyOCR
# -------------------------------
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

# -------------------------------
# Inicializando docTR
# -------------------------------
print("🔄 Inicializando docTR OCR...")
doctr_model = ocr_predictor(pretrained=True)
print("✅ docTR carregado com sucesso!")

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

        # -------------------------------
        # Verifica palavras-chave e percentual
        # -------------------------------
        keywords = []
        if keywords_str and keywords_str.strip():
            keywords = [k.strip().upper() for k in keywords_str.split(',') if k.strip()]
        try:
            percent = float(percent_str)
            if not (0 < percent <= 1):
                raise ValueError
        except Exception:
            return jsonify({'error': 'Percentual inválido. Use um valor entre 0 e 1.'}), 400

        # -------------------------------
        # Carrega imagem
        # -------------------------------
        img = Image.open(file.stream).convert("RGB")
        width, height = img.size

        # -------------------------------
        # Leitura EasyOCR (igual antes)
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
                return jsonify({
                    'status': 'not_identified',
                    'message': 'Documento não identificado pelas palavras-chave fornecidas no início.',
                    'ocr_result_easyocr': partial_text,
                    'time': round(total_time, 2)
                })

        # -------------------------------
        # Leitura docTR (documento inteiro)
        # -------------------------------
        doc = DocumentFile.from_images([img])  # Passa lista de PIL.Image
        doctr_result = doctr_model(doc)
        doctr_text_str = doctr_result.render().strip()

        total_time = time.time() - start
        return jsonify({
            'status': 'success',
            'ocr_result_easyocr': easy_text,
            'ocr_result_doctr': doctr_text_str,
            'time': round(total_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------
# Health check
# -------------------------------
@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
