import io
import time
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import easyocr

from doctr.models import ocr_predictor

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

# -------------------------------
# 🔄 Inicialização EasyOCR
# -------------------------------
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

# -------------------------------
# 🔄 Inicialização docTR (Doctr)
# -------------------------------
print("🔄 Inicializando docTR (manuscritos)...")
doc_tr_model = ocr_predictor(pretrained=True)
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
        # Parâmetros
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
        # Leitura EasyOCR
        # -------------------------------
        image_np = np.array(img)
        easy_text = reader.readtext(image_np, detail=0, paragraph=True)

        # -------------------------------
        # Leitura docTR
        # -------------------------------
        # Executa em **toda imagem** ou somente na parte inferior se quiser limitar:
        # crop_start = int(height * 0.65)
        # img_bottom = img.crop((0, crop_start, width, height))
        doc_tr_result = doc_tr_model([img])[0]  # retorna objeto Document
        doc_tr_text_lines = [line.value for block in doc_tr_result.blocks for line in block.lines]
        doc_tr_text = " ".join(doc_tr_text_lines)

        total_time = time.time() - start
        return jsonify({
            'status': 'success',
            'easyocr_text': easy_text,
            'doctr_text': doc_tr_text,
            'time': round(total_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
