import io
import time
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import easyocr
import torch
from doctr.documents import DocumentFile
from doctr.models import ocr as doctr_ocr

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

# -------------------------------
# Inicialização EasyOCR
# -------------------------------
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

# -------------------------------
# Inicialização docTR
# -------------------------------
print("🔄 Inicializando docTR (manuscritos na parte inferior)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
doctr_model = doctr_ocr(pretrained=True).to(device)
print("✅ docTR carregado com sucesso!")

def run_doctr_bottom_half(pil_image: Image.Image, percent_bottom: float = 0.35):
    width, height = pil_image.size
    crop_start = int(height * (1 - percent_bottom))
    bottom_crop = pil_image.crop((0, crop_start, width, height))
    
    # Converte em DocumentFile (lista de imagens)
    doc_file = DocumentFile.from_images([bottom_crop])
    
    # Processa com docTR
    result = doctr_model(doc_file)
    
    # Extrai texto
    texts = [block.value for page in result.pages for block in page.blocks]
    return " ".join(texts).strip()

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
        percent_bottom_str = request.form.get('percent_bottom', '0.35')

        keywords = []
        if keywords_str and keywords_str.strip():
            keywords = [k.strip().upper() for k in keywords_str.split(',') if k.strip()]
        try:
            percent = float(percent_str)
            percent_bottom = float(percent_bottom_str)
            if not (0 < percent <= 1) or not (0 < percent_bottom <= 1):
                raise ValueError
        except Exception:
            return jsonify({'error': 'Percentual inválido. Use um valor entre 0 e 1.'}), 400

        img = Image.open(file.stream).convert("RGB")
        width, height = img.size

        # -------------------------------
        # EasyOCR
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
                    'message': 'Documento não identificado pelas palavras-chave fornecidas.',
                    'ocr_result_easy': partial_text,
                    'ocr_result_doctr': "",
                    'time': round(total_time, 2)
                })

        easy_text_joined = " ".join(easy_text)

        # -------------------------------
        # docTR na parte inferior
        # -------------------------------
        doctr_text = run_doctr_bottom_half(img, percent_bottom=percent_bottom)

        total_time = time.time() - start
        return jsonify({
            'status': 'success',
            'ocr_result_easy': easy_text_joined,
            'ocr_result_doctr': doctr_text,
            'time': round(total_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
