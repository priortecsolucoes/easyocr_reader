import io
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import easyocr
import time

app = Flask(__name__)

# Carrega o modelo OCR apenas uma vez no startup
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'], recog_network='latin_g2', gpu=False)
print("✅ EasyOCR carregado com sucesso!")


@app.route('/upload-png', methods=['POST'])
def upload_png():
    start = time.time()
    try:
        # Verifica se o arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado.'}), 400

        file = request.files['file']

        # Lê e converte imagem
        img = Image.open(file.stream).convert("RGB")
        image_np = np.array(img)

        # Executa OCR
        ocr_start = time.time()
        text = reader.readtext(image_np, detail=0, paragraph=True)
        ocr_time = time.time() - ocr_start

        total_time = time.time() - start
        print(f"✅ OCR concluído em {ocr_time:.2f}s | Tempo total {total_time:.2f}s")

        return jsonify({
            'status': 'success',
            'ocr_result': text,
            'time': round(total_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
