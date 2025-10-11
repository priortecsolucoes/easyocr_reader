import io
import time
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import easyocr
from doctr.models import ocr as doctr_ocr
from doctr.io import DocumentFile

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

# -------------------------------
# Inicializando OCRs
# -------------------------------
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

print("🔄 Inicializando docTR (manuscritos)...")
ocr_model = doctr_ocr(pretrained=True)
print("✅ docTR carregado com sucesso!")

# -------------------------------
# Função para docTR na parte inferior
# -------------------------------
def run_doctr_bottom_half(pil_image: Image.Image, percent_bottom: float = 0.35) -> str:
    width, height = pil_image.size
    crop_start = int(height * (1 - percent_bottom))
    bottom_crop = pil_image.crop((0, crop_start, width, height))

    # Converte para bytes, docTR precisa
    img_bytes = io.BytesIO()
    bottom_crop.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    doc = DocumentFile.from_images([Image.open(img_bytes)])
    result = ocr_model(doc)
    # Junta todos os textos detectados
    text = " ".join([el.value for page in result.pages for block in page.blocks for line in block.lines for el in line.elements])
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
        percent_doctr_str = request.form.get('percent_doctr', '0.35')

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
            percent_doctr = float(percent_doctr_str)
            if not (0 < percent_doctr <= 1):
                raise ValueError
        except Exception:
            return jsonify({'error': 'Percentual inválido. Use um valor entre 0 e 1.'}), 400

        img = Image.open(file.stream).convert("RGB")
        width, height = img.size

        # -------------------------------
        # Leitura EasyOCR
        # -------------------------------
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
                    'easyocr_result': partial_text,
                    'time': round(total_time, 2)
                })

        easy_text_joined = " ".join(easy_text)

        # -------------------------------
        # Leitura docTR (manuscritos parte inferior)
        # -------------------------------
        doctr_text = run_doctr_bottom_half(img, percent_bottom=percent_doctr)

        total_time = time.time() - start
        return jsonify({
            'status': 'success',
            'easyocr_result': easy_text,
            'doctr_result': doctr_text,
            'time': round(total_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
