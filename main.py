import io
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import easyocr
import time

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'], detect_network='dbnet18', recog_network='standard')

print("✅ EasyOCR carregado com sucesso!")

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

        # Verifica palavras-chave e percentual
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

        # Se não passar keywords ou for vazia, processa imagem inteira direto
        if not keywords:
            image_np = np.array(img)
            text = reader.readtext(image_np, detail=0, paragraph=True)
            total_time = time.time() - start
            print(f"✅ OCR completo executado sem filtro em {total_time:.2f}s")
            return jsonify({
                'status': 'success',
                'ocr_result': text,
                'time': round(total_time, 2)
            })

        # Caso contrário faz leitura parcial para verificar palavras-chave
        top_crop = img.crop((0, 0, width, int(height * percent)))
        image_top_np = np.array(top_crop)
        partial_text = reader.readtext(image_top_np, detail=0, paragraph=True)
        partial_text_joined = " ".join(partial_text).upper()

        if any(keyword in partial_text_joined for keyword in keywords):
            bottom_crop = img.crop((0, int(height * percent), width, height))
            image_bottom_np = np.array(bottom_crop)
            rest_text = reader.readtext(image_bottom_np, detail=0, paragraph=True)
            full_text = partial_text + rest_text
            ocr_time = time.time() - start
            print(f"✅ OCR completo fracionado concluído em {ocr_time:.2f}s")
            return jsonify({
                'status': 'success',
                'ocr_result': full_text,
                'time': round(ocr_time, 2)
            })
        else:
            total_time = time.time() - start
            print("⏹ Documento não identificado pelas palavras-chave fornecidas.")
            return jsonify({
                'status': 'not_identified',
                'message': 'Documento não identificado pelas palavras-chave fornecidas no início.',
                'ocr_result': partial_text,
                'time': round(total_time, 2)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
