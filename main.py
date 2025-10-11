import io
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import easyocr
import time
from doctr.models import ocr as doctr_ocr

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

print("🔄 Inicializando EasyOCR...")
easyocr_reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

print("🔄 Inicializando docTR OCR (manuscritos na parte inferior)...")
doctr_model = doctr_ocr(pretrained=True)
print("✅ docTR carregado com sucesso!")

def run_doctr_bottom_half(pil_image: Image.Image, percent_bottom: float = 0.35):
    width, height = pil_image.size
    crop_start = int(height * (1 - percent_bottom))
    bottom_crop = pil_image.crop((0, crop_start, width, height))
    
    # Processa com docTR diretamente uma lista de imagens
    result = doctr_model([bottom_crop])
    
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

        keywords = []
        if keywords_str is not None and keywords_str.strip():
            keywords = [k.strip().upper() for k in keywords_str.split(',') if k.strip()]

        try:
            percent = float(percent_str)
            if not (0 < percent <= 1):
                raise ValueError
        except Exception:
            return jsonify({'error': 'Percentual inválido. Use um valor entre 0 e 1.'}), 400

        img = Image.open(file.stream).convert("RGB")
        width, height = img.size

        # EasyOCR em todo o documento
        image_np = np.array(img)
        easyocr_text = easyocr_reader.readtext(image_np, detail=0, paragraph=True)

        # Se não houver palavras-chave, retorna apenas EasyOCR
        if not keywords:
            total_time = time.time() - start
            return jsonify({
                'status': 'success',
                'easyocr_result': easyocr_text,
                'doctr_bottom_text': "",
                'time': round(total_time, 2)
            })

        # Processa topo para verificação de palavras-chave
        top_crop = img.crop((0, 0, width, int(height * percent)))
        image_top_np = np.array(top_crop)
        partial_text = easyocr_reader.readtext(image_top_np, detail=0, paragraph=True)
        partial_text_joined = " ".join(partial_text).upper()

        if any(keyword in partial_text_joined for keyword in keywords):
            # Executa docTR na parte inferior
            doctr_text = run_doctr_bottom_half(img, percent_bottom=0.35)
            full_text = partial_text + easyocr_text
            ocr_time = time.time() - start
            return jsonify({
                'status': 'success',
                'easyocr_result': full_text,
                'doctr_bottom_text': doctr_text,
                'time': round(ocr_time, 2)
            })
        else:
            total_time = time.time() - start
            return jsonify({
                'status': 'not_identified',
                'message': 'Documento não identificado pelas palavras-chave fornecidas no início.',
                'easyocr_result': partial_text,
                'doctr_bottom_text': "",
                'time': round(total_time, 2)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
