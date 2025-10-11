import io
import time
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import easyocr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# 🔒 Senha esperada
EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

# 🧠 Inicializa EasyOCR
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

# 🧠 Inicializa docTR
print("🔄 Inicializando docTR...")
doctr_model = ocr_predictor(pretrained=True)
print("✅ docTR carregado com sucesso!")

@app.route('/upload-png', methods=['POST'])
def upload_png():
    start = time.time()
    try:
        # 🔐 Validação de senha
        password = request.form.get('password', None)
        if password != EXPECTED_PASSWORD:
            return jsonify({'error': 'Senha inválida ou não fornecida.'}), 401

        # 📄 Validação de arquivo enviado
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado.'}), 400

        file = request.files['file']
        keywords_str = request.form.get('keywords', None)
        percent_str = request.form.get('percent', '0.25')

        # 🔠 Processa palavras-chave e percentual
        keywords = []
        if keywords_str is not None and keywords_str.strip():
            keywords = [k.strip().upper() for k in keywords_str.split(',') if k.strip()]
        try:
            percent = float(percent_str)
            if not (0 < percent <= 1):
                raise ValueError
        except Exception:
            return jsonify({'error': 'Percentual inválido. Use um valor entre 0 e 1.'}), 400

        # 📸 Lê imagem
        img_bytes = file.read()
        file.stream.seek(0)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = img.size

        # =========================
        #  🔍 Leitura completa (EasyOCR)
        # =========================
        image_np = np.array(img)
        easy_text = reader.readtext(image_np, detail=0, paragraph=True)

        # =========================
        #  📖 Leitura com docTR
        # =========================
        doc = DocumentFile.from_images(image_np)
        doctr_result = doctr_model(doc)
        doctr_text = doctr_result.render()
        doctr_text_str = doctr_text.strip()

        # =========================
        #  🔎 Filtro por palavras-chave (igual ao original)
        # =========================
        if not keywords:
            total_time = time.time() - start
            return jsonify({
                'status': 'success',
                'ocr_result_easyocr': easy_text,
                'ocr_result_doctr': doctr_text_str,
                'time': round(total_time, 2)
            })

        # Leitura parcial (parte superior)
        top_crop = img.crop((0, 0, width, int(height * percent)))
        image_top_np = np.array(top_crop)
        partial_text = reader.readtext(image_top_np, detail=0, paragraph=True)
        partial_text_joined = " ".join(partial_text).upper()

        if any(keyword in partial_text_joined for keyword in keywords):
            bottom_crop = img.crop((0, int(height * percent), width, height))
            image_bottom_np = np.array(bottom_crop)
            rest_text = reader.readtext(image_bottom_np, detail=0, paragraph=True)
            full_text = partial_text + rest_text
            total_time = time.time() - start
            return jsonify({
                'status': 'success',
                'ocr_result_easyocr': full_text,
                'ocr_result_doctr': doctr_text_str,
                'time': round(total_time, 2)
            })
        else:
            total_time = time.time() - start
            return jsonify({
                'status': 'not_identified',
                'message': 'Documento não identificado pelas palavras-chave fornecidas no início.',
                'ocr_result_easyocr': partial_text,
                'ocr_result_doctr': doctr_text_str,
                'time': round(total_time, 2)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
